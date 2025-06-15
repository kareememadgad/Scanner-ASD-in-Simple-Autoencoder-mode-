import torch
from torch import nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        
    def forward(self, x):
        attention = F.relu(self.conv1(x))
        attention = torch.sigmoid(self.conv2(attention))
        return x * attention

class ConditionalAENet(nn.Module):
    def __init__(self, input_dim=128, block_size=313, num_domains=2, num_attributes=2):
        super(ConditionalAENet, self).__init__()
        self.input_dim = input_dim
        self.block_size = block_size
        self.num_domains = num_domains
        self.num_attributes = num_attributes
        
        # Initialize covariance matrices
        self.cov_source = None
        self.cov_target = None

        # Memory-efficient encoder
        self.shared_encoder = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            AttentionBlock(32),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            AttentionBlock(64),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            AttentionBlock(128),
            nn.MaxPool2d(2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            AttentionBlock(256),
            nn.MaxPool2d(2)
        )
        
        self.encoded_size = (input_dim // 16) * (block_size // 16) * 256

        # Memory-efficient attribute encoders
        self.attribute_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.encoded_size, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
            ) for _ in range(num_attributes)
        ])

        # Memory-efficient attribute classifier
        self.attribute_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_attributes)
        )

        # Modified decoder to exactly match saved model architecture
        self.decoder = nn.Sequential(
            nn.Linear(64, 40960),  # Input size 64
            nn.Unflatten(1, (256, 8, 20)),  # Adjusted dimensions
            
            # First block
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            AttentionBlock(128),
            
            # Second block
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            AttentionBlock(64),
            
            # Third block
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            AttentionBlock(32),
            
            # Final block
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, output_padding=(1, 1)),
            nn.Tanh()
        )

        # Initialize prototypes
        self.prototypes = nn.Parameter(torch.randn(num_attributes, 256) * 0.02)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def update_covariance(self, source_z, target_z):
        """Update covariance matrices for source and target domains."""
        with torch.no_grad():  # Add no_grad to save memory
            source_z_centered = source_z - source_z.mean(dim=0)
            target_z_centered = target_z - target_z.mean(dim=0)
            
            reg = 1e-6 * torch.eye(source_z.size(1), device=source_z.device)
            self.cov_source = torch.matmul(source_z_centered.t(), source_z_centered) / (source_z.size(0) - 1) + reg
            self.cov_target = torch.matmul(target_z_centered.t(), target_z_centered) / (target_z.size(0) - 1) + reg

    def encode(self, x):
        x = self.shared_encoder(x)
        x = x.view(x.size(0), -1)
        z_list = [enc(x) for enc in self.attribute_encoders]
        z_stack = torch.stack(z_list, dim=1)
        return z_stack

    def forward(self, x, attribute_label=None):
        z_stack = self.encode(x)
        
        z_mean = z_stack.mean(dim=1)
        attribute_logits = self.attribute_classifier(z_mean)
        
        if attribute_label is not None:
            idx = attribute_label.view(-1, 1, 1).expand(-1, 1, z_stack.size(2))
            z = z_stack.gather(1, idx).squeeze(1)
        else:
            pred = attribute_logits.argmax(dim=1)
            idx = pred.view(-1, 1, 1).expand(-1, 1, z_stack.size(2))
            z = z_stack.gather(1, idx).squeeze(1)
        
        # Project z to 64 dimensions before decoder
        z = z[:, :64]  # Take first 64 dimensions
        
        recon = self.decoder(z)
        
        if recon.shape != x.shape:
            recon = F.interpolate(recon, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        temperature = 0.1
        proto_sim = F.cosine_similarity(z.unsqueeze(1), self.prototypes[:, :64].unsqueeze(0), dim=2) / temperature
        
        return recon, z, proto_sim, attribute_logits
