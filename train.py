import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from datasets.dcase_dcase202x_t2_loader import DCASE202XT2Loader
from networks.dcase2023t2_ae.network import ConditionalAENet
from datasets.dcase2023t2.dataset import DCASE2023T2Dataset

from tqdm import tqdm
import os
import argparse
import sys
from torch.optim.lr_scheduler import OneCycleLR

# original lib
import common as com
from networks.models import Models
from datasets.loader_common import get_machine_type_dict

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################

def calculate_mahalanobis_distance(z, cov):
    # z: batch_size x latent_dim
    # cov: latent_dim x latent_dim
    z_centered = z - z.mean(dim=0)
    
    # Add small epsilon to diagonal for numerical stability
    eps = 1e-6
    cov = cov + torch.eye(cov.size(0), device=cov.device) * eps
    
    # Calculate inverse using SVD for better numerical stability
    U, S, V = torch.svd(cov)
    S_inv = torch.diag(1.0 / (S + eps))
    inv_cov = torch.mm(torch.mm(V, S_inv), U.t())
    
    # Calculate Mahalanobis distance
    mahalanobis = torch.sum(z_centered @ inv_cov * z_centered, dim=1)
    return mahalanobis

def calculate_covariance(z):
    # z: batch_size x latent_dim
    z_centered = z - z.mean(dim=0)
    cov = torch.matmul(z_centered.t(), z_centered) / (z.size(0) - 1)
    return cov

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ConditionalAENet(
        input_dim=args.input_dim,
        block_size=args.block_size,
        num_domains=2
    ).to(device)
    
    # Create datasets
    source_dataset = DCASE2023T2Dataset(
        root_dir=args.dataset_dir,
        machine_type=args.machine_type,
        split='train',
        domain='source'
    )
    
    target_dataset = DCASE2023T2Dataset(
        root_dir=args.dataset_dir,
        machine_type=args.machine_type,
        split='train',
        domain='target'
    )
    
    # Create data loaders
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # Create progress bar
        pbar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)))
        
        for (source_data, _), (target_data, _) in pbar:
            # Move data to device
            source_data = source_data.to(device)
            target_data = target_data.to(device)

            # Ensure channel dimension exists
            if source_data.ndim == 3:
                source_data = source_data.unsqueeze(1)
            if target_data.ndim == 3:
                target_data = target_data.unsqueeze(1)

            # Forward pass for source domain
            source_recon, source_z, source_proto, source_logits = model(source_data)
            
            # Forward pass for target domain
            target_recon, target_z, target_proto, target_logits = model(target_data)
            
            # Resize reconstructed output to match input dimensions
            source_recon = F.interpolate(source_recon, size=source_data.shape[2:], mode='bilinear', align_corners=False)
            target_recon = F.interpolate(target_recon, size=target_data.shape[2:], mode='bilinear', align_corners=False)
            
            # Reconstruction loss (L1 for better stability)
            source_recon_loss = F.l1_loss(source_recon.squeeze(1), source_data.squeeze(1))
            target_recon_loss = F.l1_loss(target_recon.squeeze(1), target_data.squeeze(1))
            recon_loss = source_recon_loss + target_recon_loss
            
            # Prototype loss (negative cosine similarity)
            source_proto_loss = -torch.mean(source_proto)
            target_proto_loss = -torch.mean(target_proto)
            proto_loss = source_proto_loss + target_proto_loss
            
            # Domain adaptation loss (MMD)
            mmd_loss = maximum_mean_discrepancy(source_z, target_z)
            
            # Covariance alignment loss
            model.update_covariance(source_z, target_z)
            cov_loss = F.mse_loss(model.cov_source, model.cov_target)
            
            # Total loss
            loss = (
                recon_loss +
                0.1 * proto_loss +
                0.1 * mmd_loss +
                0.05 * cov_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_description(f'Epoch {epoch+1}/{args.epochs}')
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        avg_loss = total_loss / len(source_loader)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/{args.machine_type}_model.pth')
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'models/{args.machine_type}_checkpoint_{epoch+1}.pth')
        
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}')

def maximum_mean_discrepancy(source, target):
    """Calculate Maximum Mean Discrepancy (MMD) between source and target domains."""
    kernels = [1, 2, 4, 8, 16]
    xx, yy, zz = 0, 0, 0
    
    for kernel in kernels:
        # RBF kernel
        xx += torch.exp(-torch.sum((source.unsqueeze(1) - source.unsqueeze(0)) ** 2, dim=2) / kernel)
        yy += torch.exp(-torch.sum((target.unsqueeze(1) - target.unsqueeze(0)) ** 2, dim=2) / kernel)
        zz += torch.exp(-torch.sum((source.unsqueeze(1) - target.unsqueeze(0)) ** 2, dim=2) / kernel)
    
    return torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(zz)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data/dcase2024t2/dev_data/raw')
    parser.add_argument('--machine_type', type=str, default='Scanner')
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--block_size', type=int, default=313)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=10)
    args = parser.parse_args()

    # Verify data directory exists
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    
    # Verify machine type directory exists
    machine_dir = os.path.join(args.dataset_dir, args.machine_type)
    if not os.path.exists(machine_dir):
        raise FileNotFoundError(f"Machine type directory not found: {machine_dir}")
    
    print(f"Training on {args.machine_type} data from {machine_dir}")
    train(args)

if __name__ == '__main__':
    main()
