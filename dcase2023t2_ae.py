import os
import sys
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import scipy
from sklearn import metrics
import csv
from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd
import json
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
import random
import torch.nn as nn
import gc

from networks.base_model import BaseModel
from networks.dcase2023t2_ae.network import ConditionalAENet
from networks.criterion.mahala import cov_v, loss_function_mahala, calc_inv_cov
from tools.plot_anm_score import AnmScoreFigData
from tools.plot_loss_curve import csv_to_figdata

def calculate_pauc(y_true, y_score, max_fpr=0.1):
    """Calculate partial AUC up to max_fpr."""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    p_auc = metrics.auc(fpr, tpr) / max_fpr
    return p_auc

def calculate_metrics(y_true, y_score, threshold=0.5):
    """Calculate all required metrics for a given domain."""
    # Convert scores to binary predictions using threshold
    y_pred = (y_score >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    pauc = calculate_pauc(y_true, y_score)
    
    return {
        'AUC': auc,
        'pAUC': pauc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

class DCASE2023T2AE(BaseModel):
    def __init__(self, args, train, test):
        super().__init__(
            args=args,
            train=train,
            test=test
        )
        parameter_list = [{"params":self.model.parameters()}]
        self.optimizer = optim.AdamW(parameter_list, lr=self.args.learning_rate, weight_decay=1e-4)
        
        # Memory-efficient learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Early stopping parameters
        self.patience = 10
        self.min_delta = 0.001
        self.best_val_loss = float('inf')
        self.counter = 0
        
        # Training epochs
        self.max_epochs = getattr(self.args, 'max_epochs', 100)
        
        # Reduced augmentation parameters
        self.freq_mask_param = 32
        self.time_mask_param = 100
        self.num_freq_masks = 2
        self.num_time_masks = 2
        self.noise_std = 0.01
        
        # Loss weights
        self.lambda_recon = 1.0
        self.lambda_proto = 0.1
        self.lambda_cls = 0.05
        self.lambda_consistency = 0.02
        
        # Batch accumulation steps
        self.accumulation_steps = 4  # Accumulate gradients for 4 batches
        
        self.mse_score_distr_file_path = self.model_dir/f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mse.pickle"
        self.mahala_score_distr_file_path = self.model_dir/f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mahala.pickle"

    def apply_frequency_masking(self, x):
        batch_size, channels, freq, time = x.shape
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(1, freq - f))
            x[:, :, f0:f0 + f, :] = 0
        return x

    def apply_time_masking(self, x):
        batch_size, channels, freq, time = x.shape
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(1, time - t))
            x[:, :, :, t0:t0 + t] = 0
        return x

    def apply_gaussian_noise(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def apply_amplitude_scaling(self, x):
        scale = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
        return x * scale

    def apply_time_shift(self, x, max_shift=20):
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=3)

    def augment_data(self, x):
        x = self.apply_frequency_masking(x)
        x = self.apply_time_masking(x)
        x = self.apply_gaussian_noise(x)
        x = self.apply_amplitude_scaling(x)
        x = self.apply_time_shift(x)
        return x

    def init_model(self):
        self.block_size = self.data.height
        return ConditionalAENet(input_dim=self.data.input_dim, block_size=self.block_size, num_attributes=2)

    def get_log_header(self):
        self.column_heading_list=[
                ["loss"],
                ["val_loss"],
                ["recon_loss"], 
                ["recon_loss_source", "recon_loss_target"],
                ["proto_loss"],
                ["cls_loss"],
                ["consistency_loss"]
        ]
        return "loss,val_loss,recon_loss,recon_loss_source,recon_loss_target,proto_loss,cls_loss,consistency_loss"
    
    def train(self, epoch):
        if epoch <= self.epoch:
            return
            
        torch.autograd.set_detect_anomaly(True)
        train_loss = 0
        train_recon_loss = 0
        train_proto_loss = 0
        train_cls_loss = 0
        train_consistency_loss = 0
        latent_vectors = []
        attribute_labels = []
        y_pred = []
        train_loader = self.train_loader
        n_clusters = 2
        pseudo_labels = None
        warmup_epochs = 3

        self.model.train()
        self.optimizer.zero_grad()  # Zero gradients at start
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data = batch[0].to(self.device).float()
            if data.shape[0] <= 1:
                continue
                
            # Apply data augmentation
            aug_data = self.augment_data(data)
            
            # If you have real attribute labels, use them; else use pseudo_labels
            if pseudo_labels is not None:
                attr_label = torch.tensor(pseudo_labels[batch_idx*data.shape[0]:(batch_idx+1)*data.shape[0]], dtype=torch.long, device=self.device)
            else:
                attr_label = torch.zeros(data.shape[0], dtype=torch.long, device=self.device)
            
            # Forward pass on original data
            recon_orig, z_orig, proto_sim_orig, attr_logits_orig = self.model(data, attr_label)
            
            # Forward pass on augmented data
            recon_aug, z_aug, proto_sim_aug, attr_logits_aug = self.model(aug_data, attr_label)
            
            # Calculate losses
            recon_loss = F.mse_loss(recon_orig, data, reduction='mean')
            proto_loss = 1 - F.cosine_similarity(z_orig, self.model.prototypes[attr_label], dim=1).mean()
            cls_loss = F.cross_entropy(attr_logits_orig, attr_label)
            consistency_loss = F.mse_loss(z_orig, z_aug, reduction='mean')
            
            # Total loss
            loss = (self.lambda_recon * recon_loss + 
                   self.lambda_proto * proto_loss + 
                   self.lambda_cls * cls_loss +
                   self.lambda_consistency * consistency_loss)
            
            # Scale loss by accumulation steps
            loss = loss / self.accumulation_steps
            loss.backward()
            
            # Update weights after accumulation steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            train_loss += float(loss) * self.accumulation_steps
            train_recon_loss += float(recon_loss)
            train_proto_loss += float(proto_loss)
            train_cls_loss += float(cls_loss)
            train_consistency_loss += float(consistency_loss)
            
            # Collect latent vectors for clustering
            latent_vectors.append(z_orig.detach().cpu().numpy())
            attribute_labels.append(attr_label.detach().cpu().numpy())
            y_pred.append(loss.item() * self.accumulation_steps)
            
            # Clear memory
            del recon_orig, z_orig, proto_sim_orig, attr_logits_orig
            del recon_aug, z_aug, proto_sim_aug, attr_logits_aug
            torch.cuda.empty_cache()
            
            if batch_idx % self.args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() * self.accumulation_steps:.6f}')
                
        # After warmup, cluster latent features and assign pseudo-labels
        if epoch >= warmup_epochs:
            latent_vectors_np = np.concatenate(latent_vectors, axis=0)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(latent_vectors_np)
            pseudo_labels = kmeans.labels_
            print(f"Pseudo-labels assigned for epoch {epoch}")
            
        # Clear memory
        del latent_vectors, attribute_labels, y_pred
        gc.collect()
        torch.cuda.empty_cache()

    def calc_valid_mahala_score(self, data, y_pred, inv_cov_source, inv_cov_target):
        data = data.to(self.device).float()
        recon_data, _ = self.model(data)
        loss_source, num = loss_function_mahala(
            recon_x=recon_data,
            x=data,
            block_size=self.block_size,
            cov=inv_cov_source,
            use_precision=True,
            reduction=False
        )
        loss_source = self.loss_reduction(score=self.loss_reduction_1d(loss_source), n_loss=num)

        loss_target, num = loss_function_mahala(
            recon_x=recon_data,
            x=data,
            block_size=self.block_size,
            cov=inv_cov_target,
            use_precision=True,
            reduction=False
        )
        loss_target = self.loss_reduction(score=self.loss_reduction_1d(loss_target), n_loss=num)
        y_pred.append(min(loss_target.item(), loss_source.item()))
        return y_pred

    def loss_reduction_1d(self, score):
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        return torch.sum(score) / n_loss

    def loss_fn(self,recon_x, x):
        ### MSE loss ###
        loss = F.mse_loss(recon_x, x.view(recon_x.shape), reduction="none")
        return loss

    def test(self):
        self.model.eval()
        test_loader = self.test_loader
        all_scores = []
        all_labels = []
        all_domains = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                data = batch[0].to(self.device).float()
                labels = batch[1].to(self.device) if len(batch) > 1 else None
                domains = batch[2].to(self.device) if len(batch) > 2 else None
                
                # Use most probable attribute branch
                recon, z, proto_sim, attr_logits = self.model(data)
                
                # Anomaly score: combine multiple metrics
                recon_error = F.mse_loss(recon, data, reduction='none').mean(dim=(1,2,3)).cpu().numpy()
                proto_score = 1 - proto_sim.max(dim=1)[0].cpu().numpy()
                
                # Add Mahalanobis distance score
                if self.model.cov_source is not None and self.model.cov_target is not None:
                    inv_cov_source = torch.inverse(self.model.cov_source)
                    inv_cov_target = torch.inverse(self.model.cov_target)
                    
                    z_centered = z - z.mean(dim=0)
                    mahala_source = torch.sqrt(torch.sum(torch.matmul(z_centered, inv_cov_source) * z_centered, dim=1))
                    mahala_target = torch.sqrt(torch.sum(torch.matmul(z_centered, inv_cov_target) * z_centered, dim=1))
                    
                    mahala_score = torch.min(mahala_source, mahala_target).cpu().numpy()
                else:
                    mahala_score = np.zeros_like(recon_error)
                
                # Combine scores
                anomaly_score = (0.5 * recon_error + 
                               0.3 * proto_score + 
                               0.2 * mahala_score)
                
                all_scores.extend(anomaly_score)
                if labels is not None:
                    all_labels.extend(labels.cpu().numpy())
                if domains is not None:
                    all_domains.extend(domains.cpu().numpy())
                
                # Clear memory
                del recon, z, proto_sim, attr_logits
                torch.cuda.empty_cache()
        
        # Convert to numpy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels) if all_labels else None
        all_domains = np.array(all_domains) if all_domains else None
        
        # Save anomaly scores to Excel
        scores_df = pd.DataFrame({
            'anomaly_score': all_scores,
            'label': all_labels if all_labels is not None else np.zeros_like(all_scores),
            'domain': all_domains if all_domains is not None else np.zeros_like(all_scores)
        })
        scores_file = os.path.join(self.model_dir, f'anomaly_scores_{self.args.model}_{self.args.dataset}.xlsx')
        scores_df.to_excel(scores_file, index=False)
        print(f"Anomaly scores saved to {scores_file}")
        
        # Calculate and save metrics if labels are available
        if all_labels is not None and all_domains is not None:
            source_mask = (all_domains == 0)
            target_mask = (all_domains == 1)
            
            source_metrics = calculate_metrics(all_labels[source_mask], all_scores[source_mask])
            target_metrics = calculate_metrics(all_labels[target_mask], all_scores[target_mask])
            
            metrics_data = {
                'Metric': ['AUC', 'pAUC', 'Precision', 'Recall', 'F1'],
                'Source': [source_metrics['AUC'], source_metrics['pAUC'], 
                          source_metrics['Precision'], source_metrics['Recall'], 
                          source_metrics['F1']],
                'Target': [target_metrics['AUC'], target_metrics['pAUC'], 
                          target_metrics['Precision'], target_metrics['Recall'], 
                          target_metrics['F1']]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_file = os.path.join(self.model_dir, f'evaluation_metrics_{self.args.model}_{self.args.dataset}.xlsx')
            metrics_df.to_excel(metrics_file, index=False)
            print(f"Evaluation metrics saved to {metrics_file}")
            
            print("\nTest Results:")
            print("Source Domain:")
            for metric, value in source_metrics.items():
                print(f"{metric}: {value:.4f}")
            print("\nTarget Domain:")
            for metric, value in target_metrics.items():
                print(f"{metric}: {value:.4f}")

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def to_python_type(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj
