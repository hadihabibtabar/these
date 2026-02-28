import os
import sys
import logging
import argparse
import yaml
import torch
import numpy as np
import json
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from fuxictr.utils import load_config, set_logger
from fuxictr.features import FeatureMap
from fuxictr.preprocess import build_dataset
from fuxictr.pytorch.torch_utils import get_device
from src.MMoE import MMoE
import pandas as pd
from datetime import datetime


def evaluate_model(model_path, config_path, expid, gpu=-1):
    """Evaluate the enhanced MMoE model with comprehensive metrics."""
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['config']
    
    # Set up logging
    set_logger(model_config)
    logging.info("=" * 50)
    logging.info("Evaluating Enhanced MMoE with Transformer Components")
    logging.info("=" * 50)
    
    # Set device
    device = get_device(gpu)
    logging.info(f"Using device: {device}")
    
    # Load dataset configuration
    dataset_config = load_config(config_path, model_config['dataset_id'])
    
    # Build feature map
    feature_map = FeatureMap(dataset_config, data_root=dataset_config['data_root'])
    feature_map.load(os.path.join(dataset_config['data_root'], 'feature_map.json'), 
                    os.path.join(dataset_config['data_root'], 'feature_vocab.json'))
    
    # Build test dataset
    test_data = build_dataset(dataset_config, feature_map, 'test')
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
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Evaluation
    logging.info("Starting evaluation...")
    
    all_predictions = {f'task_{i}': [] for i in range(model_config['num_tasks'])}
    all_targets = {f'task_{i}': [] for i in range(model_config['num_tasks'])}
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model.forward(batch)
            labels = model.get_labels(batch)
            
            # Compute loss
            loss = model.add_loss(batch)
            total_loss += loss.item()
            
            # Store predictions and targets
            for i in range(model_config['num_tasks']):
                pred = torch.sigmoid(outputs[i]).cpu().numpy()
                true = labels[i].cpu().numpy()
                
                all_predictions[f'task_{i}'].extend(pred)
                all_targets[f'task_{i}'].extend(true)
            
            if batch_idx % 100 == 0:
                logging.info(f"Processed {batch_idx} batches...")
    
    # Compute comprehensive metrics
    metrics = {}
    task_names = ['CTR', 'CVR']  # Click-Through Rate, Conversion Rate
    
    for i in range(model_config['num_tasks']):
        task_key = f'task_{i}'
        predictions = np.array(all_predictions[task_key])
        targets = np.array(all_targets[task_key])
        
        # Binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Compute metrics
        task_metrics = {
            'auc': roc_auc_score(targets, predictions),
            'accuracy': accuracy_score(targets, binary_predictions),
            'precision': precision_score(targets, binary_predictions, zero_division=0),
            'recall': recall_score(targets, binary_predictions, zero_division=0),
            'f1_score': f1_score(targets, binary_predictions, zero_division=0),
            'logloss': -np.mean(targets * np.log(predictions + 1e-8) + 
                               (1 - targets) * np.log(1 - predictions + 1e-8)),
            'positive_rate': np.mean(targets),
            'prediction_mean': np.mean(predictions),
            'prediction_std': np.std(predictions)
        }
        
        metrics[task_names[i]] = task_metrics
        
        # Log task-specific results
        logging.info(f"\n{task_names[i]} Results:")
        logging.info(f"  AUC: {task_metrics['auc']:.4f}")
        logging.info(f"  Accuracy: {task_metrics['accuracy']:.4f}")
        logging.info(f"  Precision: {task_metrics['precision']:.4f}")
        logging.info(f"  Recall: {task_metrics['recall']:.4f}")
        logging.info(f"  F1-Score: {task_metrics['f1_score']:.4f}")
        logging.info(f"  LogLoss: {task_metrics['logloss']:.4f}")
        logging.info(f"  Positive Rate: {task_metrics['positive_rate']:.4f}")
        logging.info(f"  Prediction Mean: {task_metrics['prediction_mean']:.4f}")
        logging.info(f"  Prediction Std: {task_metrics['prediction_std']:.4f}")
    
    # Overall metrics
    overall_metrics = {
        'average_auc': np.mean([metrics[task]['auc'] for task in task_names]),
        'average_accuracy': np.mean([metrics[task]['accuracy'] for task in task_names]),
        'average_f1': np.mean([metrics[task]['f1_score'] for task in task_names]),
        'average_logloss': np.mean([metrics[task]['logloss'] for task in task_names]),
        'total_loss': total_loss / len(test_data)
    }
    
    logging.info(f"\nOverall Results:")
    logging.info(f"  Average AUC: {overall_metrics['average_auc']:.4f}")
    logging.info(f"  Average Accuracy: {overall_metrics['average_accuracy']:.4f}")
    logging.info(f"  Average F1-Score: {overall_metrics['average_f1']:.4f}")
    logging.info(f"  Average LogLoss: {overall_metrics['average_logloss']:.4f}")
    logging.info(f"  Total Loss: {overall_metrics['total_loss']:.4f}")
    
    # Save detailed results
    results = {
        'expid': expid,
        'model_path': model_path,
        'task_metrics': metrics,
        'overall_metrics': overall_metrics,
        'model_config': model_config,
        'evaluation_timestamp': datetime.now().isoformat(),
        'test_samples': len(test_data)
    }
    
    # Save results
    results_path = os.path.join(model_config['model_root'], f"{expid}_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"\nDetailed results saved to: {results_path}")
    
    # Create summary DataFrame
    summary_data = []
    for task in task_names:
        task_metrics = metrics[task]
        summary_data.append({
            'Task': task,
            'AUC': task_metrics['auc'],
            'Accuracy': task_metrics['accuracy'],
            'Precision': task_metrics['precision'],
            'Recall': task_metrics['recall'],
            'F1-Score': task_metrics['f1_score'],
            'LogLoss': task_metrics['logloss'],
            'Positive_Rate': task_metrics['positive_rate']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(model_config['model_root'], f"{expid}_evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    logging.info(f"Summary saved to: {summary_path}")
    logging.info("Evaluation completed!")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Enhanced MMoE Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='./config/',
                       help='Path to config directory')
    parser.add_argument('--expid', type=str, required=True,
                       help='Experiment ID')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU device ID (-1 for CPU)')
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_model(args.model_path, args.config, args.expid, args.gpu)
    
    # Print final summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.expid}")
    print(f"Average AUC: {results['overall_metrics']['average_auc']:.4f}")
    print(f"Average Accuracy: {results['overall_metrics']['average_accuracy']:.4f}")
    print(f"Average F1-Score: {results['overall_metrics']['average_f1']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main() 