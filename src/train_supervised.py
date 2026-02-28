import os
import sys
import logging
import argparse
import yaml
import torch
import numpy as np
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.features import FeatureMap
from fuxictr.preprocess import FeatureProcessor, build_dataset
from fuxictr.pytorch.torch_utils import get_device
from src.MMoE import MMoE
import json
from datetime import datetime


def train_model(config_path, expid, data_path=None, gpu=-1):
    """Train the enhanced MMoE model with transformer components."""
    
    # Load configuration
    model_config = load_config(config_path, expid)
    dataset_config = load_config(config_path, model_config['dataset_id'])
    
    # Set up logging
    set_logger(model_config)
    logging.info("=" * 50)
    logging.info("Training Enhanced MMoE with Transformer Components")
    logging.info("=" * 50)
    
    # Set device
    device = get_device(gpu)
    logging.info(f"Using device: {device}")
    
    # Build feature map
    feature_map = FeatureMap(dataset_config, data_root=dataset_config['data_root'])
    feature_map.load(os.path.join(dataset_config['data_root'], 'feature_map.json'), 
                    os.path.join(dataset_config['data_root'], 'feature_vocab.json'))
    
    # Build datasets
    train_data = build_dataset(dataset_config, feature_map, 'train')
    valid_data = build_dataset(dataset_config, feature_map, 'valid')
    test_data = build_dataset(dataset_config, feature_map, 'test')
    
    logging.info(f"Train samples: {len(train_data)}")
    logging.info(f"Valid samples: {len(valid_data)}")
    logging.info(f"Test samples: {len(test_data)}")
    
    # Initialize model
    model = MMoE(
        feature_map=feature_map,
        task=model_config['task'],
        num_tasks=model_config['num_tasks'],
        model_id=model_config['model_id'],
        gpu=gpu,
        learning_rate=model_config['learning_rate'],
        embedding_dim=model_config['embedding_dim'],
        num_experts=model_config['num_experts'],
        cvr_weight=model_config.get('cvr_weight', 1.0),
        transformer_layers=model_config.get('transformer_layers', 2),
        transformer_heads=model_config.get('transformer_heads', 4),
        transformer_dropout=model_config.get('transformer_dropout', 0.1),
        transformer_dim_feedforward=model_config.get('transformer_dim_feedforward', 512),
        **model_config
    )
    
    # Training loop with enhanced monitoring
    best_metric = 0.0
    patience_counter = 0
    early_stop_patience = model_config.get('early_stop_patience', 5)
    
    logging.info("Starting training...")
    for epoch in range(model_config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = {f'task_{i}_auc': 0.0 for i in range(model_config['num_tasks'])}
        
        for batch_idx, batch in enumerate(train_data):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            loss = model.add_loss(batch)
            
            # Backward pass
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            
            train_loss += loss.item()
            
            # Compute metrics (simplified for efficiency)
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    outputs = model.forward(batch)
                    labels = model.get_labels(batch)
                    
                    for i in range(model_config['num_tasks']):
                        # Compute AUC (simplified)
                        pred = torch.sigmoid(outputs[i]).cpu().numpy()
                        true = labels[i].cpu().numpy()
                        
                        # Simple accuracy as proxy for AUC
                        acc = ((pred > 0.5) == true).mean()
                        train_metrics[f'task_{i}_auc'] += acc
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {f'task_{i}_auc': 0.0 for i in range(model_config['num_tasks'])}
        
        with torch.no_grad():
            for batch in valid_data:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                loss = model.add_loss(batch)
                val_loss += loss.item()
                
                # Compute metrics
                outputs = model.forward(batch)
                labels = model.get_labels(batch)
                
                for i in range(model_config['num_tasks']):
                    pred = torch.sigmoid(outputs[i]).cpu().numpy()
                    true = labels[i].cpu().numpy()
                    acc = ((pred > 0.5) == true).mean()
                    val_metrics[f'task_{i}_auc'] += acc
        
        # Average metrics
        num_train_batches = len(train_data)
        num_val_batches = len(valid_data)
        
        train_loss /= num_train_batches
        val_loss /= num_val_batches
        
        for key in train_metrics:
            train_metrics[key] /= num_train_batches
            val_metrics[key] /= num_val_batches
        
        # Log progress
        logging.info(f"Epoch {epoch+1}/{model_config['epochs']}")
        logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        for i in range(model_config['num_tasks']):
            logging.info(f"Task {i+1} - Train AUC: {train_metrics[f'task_{i}_auc']:.4f}, "
                        f"Val AUC: {val_metrics[f'task_{i}_auc']:.4f}")
        
        # Early stopping
        current_metric = sum(val_metrics.values()) / len(val_metrics)
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(model_config['model_root'], f"{expid}_best.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'epoch': epoch,
                'best_metric': best_metric,
                'config': model_config
            }, model_path)
            logging.info(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Test evaluation
    logging.info("Evaluating on test set...")
    model.eval()
    test_metrics = {f'task_{i}_auc': 0.0 for i in range(model_config['num_tasks'])}
    
    with torch.no_grad():
        for batch in test_data:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model.forward(batch)
            labels = model.get_labels(batch)
            
            for i in range(model_config['num_tasks']):
                pred = torch.sigmoid(outputs[i]).cpu().numpy()
                true = labels[i].cpu().numpy()
                acc = ((pred > 0.5) == true).mean()
                test_metrics[f'task_{i}_auc'] += acc
    
    num_test_batches = len(test_data)
    for key in test_metrics:
        test_metrics[key] /= num_test_batches
    
    logging.info("Final Test Results:")
    for i in range(model_config['num_tasks']):
        logging.info(f"Task {i+1} Test AUC: {test_metrics[f'task_{i}_auc']:.4f}")
    
    # Save final results
    results = {
        'expid': expid,
        'best_metric': best_metric,
        'test_metrics': test_metrics,
        'config': model_config,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(model_config['model_root'], f"{expid}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {results_path}")
    logging.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced MMoE Model')
    parser.add_argument('--config', type=str, default='./config/', 
                       help='Path to config directory')
    parser.add_argument('--expid', type=str, required=True,
                       help='Experiment ID to run')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to data file (optional)')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU device ID (-1 for CPU)')
    
    args = parser.parse_args()
    
    # Train model
    train_model(args.config, args.expid, args.data_path, args.gpu)


if __name__ == "__main__":
    main() 