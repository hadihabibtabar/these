import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import logging
from pathlib import Path
import wandb

class DataAugmentation:
    """Data augmentation methods for tabular/embedding inputs."""
    
    def __init__(self, 
                 feature_dropout_prob: float = 0.1,
                 feature_mask_prob: float = 0.1,
                 noise_std: float = 0.01):
        self.feature_dropout_prob = feature_dropout_prob
        self.feature_mask_prob = feature_mask_prob
        self.noise_std = noise_std
    
    def feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop features with probability p."""
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.feature_dropout_prob))
        return x * mask
    
    def feature_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask features with probability p."""
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.feature_mask_prob))
        return x * mask
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to the input."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation."""
        augmentations = [
            self.feature_dropout,
            self.feature_masking,
            self.add_noise
        ]
        aug_fn = np.random.choice(augmentations)
        return aug_fn(x)

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        return F.normalize(x, dim=1)

class ContrastiveLoss(nn.Module):
    """NT-Xent (Normalized Temperature-Scaled Cross Entropy) loss."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss between two sets of projections.
        Args:
            z1: First set of projections (N x D)
            z2: Second set of projections (N x D)
        """
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        N = z1.shape[0]
        similarity = torch.matmul(z1, z2.T) / self.temperature
        
        # Labels are the diagonal elements
        labels = torch.arange(N, device=z1.device)
        
        # Compute loss
        loss = F.cross_entropy(similarity, labels)
        return loss

class SSLModule:
    """Self-supervised learning module for pretraining Transformer encoders."""
    
    def __init__(self,
                 encoder: nn.Module,
                 input_dim: int,
                 projection_dim: int = 128,
                 temperature: float = 0.07,
                 feature_dropout_prob: float = 0.1,
                 feature_mask_prob: float = 0.1,
                 noise_std: float = 0.01,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.encoder = encoder
        self.device = device
        self.encoder.to(device)
        
        # Initialize augmentation and projection
        self.augmentation = DataAugmentation(
            feature_dropout_prob=feature_dropout_prob,
            feature_mask_prob=feature_mask_prob,
            noise_std=noise_std
        )
        
        self.projection = ProjectionHead(
            input_dim=input_dim,
            output_dim=projection_dim
        ).to(device)
        
        self.criterion = ContrastiveLoss(temperature=temperature)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.projection.parameters()),
            lr=1e-3
        )
        
        # Initialize tensorboard
        self.writer = SummaryWriter(
            log_dir=os.path.join('runs', f'ssl_pretraining_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        )

class SSLPretrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = ContrastiveLoss()
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=self.log_dir / "ssl_pretraining.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="mmoe-ssl-pretraining")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x1, x2) in enumerate(train_loader):
            x1, x2 = x1.to(self.device), x2.to(self.device)
            
            # Forward pass
            z1 = self.model(x1)
            z2 = self.model(x2)
            
            # Compute loss
            loss = self.criterion(z1, z2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logging.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                if self.use_wandb:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx
                    })
        
        avg_loss = total_loss / len(train_loader)
        return {"loss": avg_loss}
    
    def pretrain(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        save_every: int = 5
    ):
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            metrics = self.train_epoch(train_loader, epoch)
            
            # Logging
            logging.info(f"Epoch {epoch} completed. Average loss: {metrics['loss']:.4f}")
            if self.use_wandb:
                wandb.log({
                    "epoch_loss": metrics['loss'],
                    "epoch": epoch
                })
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = self.checkpoint_dir / f"ssl_checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path, epoch, metrics)
            
            # Save best model
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                best_model_path = self.checkpoint_dir / "ssl_best_model.pt"
                self.save_checkpoint(best_model_path, epoch, metrics)
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float]
    ):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)
        logging.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']

# Example usage:
"""
# Initialize SSL module
ssl_module = SSLModule(
    encoder=transformer_encoder,  # Your TransformerExpert instance
    input_dim=embedding_dim * num_fields,
    projection_dim=128
)

# Pretrain on unlabeled data
metrics = ssl_module.pretrain(
    train_loader=unlabeled_loader,
    num_epochs=10,
    save_path='pretrained_encoder.pt'
)

# Load pretrained encoder for downstream tasks
ssl_module.load_encoder('pretrained_encoder.pt')
""" 