#!/usr/bin/env python3
"""
Comprehensive Test Script for Enhanced MMoE Project
Runs the complete pipeline and collects metrics for comparison.
"""

import os
import sys
import time
import psutil
import torch
import json
import logging
from datetime import datetime
import subprocess
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def run_command_with_metrics(command, description):
    """Run a command and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        end_time = time.time()
        end_memory = get_memory_usage()
        
        duration = end_time - start_time
        peak_memory = max(start_memory, end_memory)
        
        print(f"Duration: {duration:.2f} seconds")
        print(f"Peak Memory: {peak_memory:.2f} GB")
        print(f"Return Code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return {
            'success': result.returncode == 0,
            'duration': duration,
            'peak_memory': peak_memory,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        print(f"Error running command: {e}")
        return {
            'success': False,
            'duration': time.time() - start_time,
            'peak_memory': get_memory_usage(),
            'error': str(e)
        }

def create_test_data():
    """Create a small test dataset for quick testing."""
    print("Creating test dataset...")
    
    # Create test data directory
    os.makedirs("data/test", exist_ok=True)
    
    # Create a small sample of the data for testing
    import pandas as pd
    import numpy as np
    
    # Read a small sample of the original data
    try:
        df = pd.read_csv("data/final/train_data_final.csv", nrows=10000)
        print(f"Loaded {len(df)} samples for testing")
        
        # Create test splits
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        # Save test data
        train_df.to_csv("data/test/train_data_test.csv", index=False)
        val_df.to_csv("data/test/valid_data_test.csv", index=False)
        test_df.to_csv("data/test/test_data_test.csv", index=False)
        
        print(f"Created test dataset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
    except Exception as e:
        print(f"Error creating test data: {e}")
        # Create synthetic data if original data is not available
        print("Creating synthetic test data...")
        
        n_samples = 1000
        n_features = 20
        
        # Create synthetic features
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        data = np.random.randn(n_samples, n_features)
        
        # Create labels
        ctr_labels = (np.random.random(n_samples) > 0.7).astype(int)
        cvr_labels = (np.random.random(n_samples) > 0.9).astype(int)
        
        df = pd.DataFrame(data, columns=feature_cols)
        df['is_clicked'] = ctr_labels
        df['is_installed'] = cvr_labels
        
        # Split data
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        # Save test data
        os.makedirs("data/test", exist_ok=True)
        train_df.to_csv("data/test/train_data_test.csv", index=False)
        val_df.to_csv("data/test/valid_data_test.csv", index=False)
        test_df.to_csv("data/test/test_data_test.csv", index=False)
        
        print(f"Created synthetic test dataset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

def update_config_for_test():
    """Update configuration files for testing."""
    print("Updating configuration for testing...")
    
    # Update dataset config for test data
    test_dataset_config = {
        'test_dataset': {
            'data_root': './data/test/',
            'data_format': 'csv',
            'train_data': './data/test/train_data_test.csv',
            'valid_data': './data/test/valid_data_test.csv',
            'test_data': './data/test/test_data_test.csv',
            'min_categr_count': 1,
            'feature_cols': [
                {
                    'name': [f'feature_{i}' for i in range(20)],
                    'active': True,
                    'dtype': 'float',
                    'type': 'numeric'
                }
            ],
            'label_col': [
                {'name': 'is_clicked', 'dtype': 'float'},
                {'name': 'is_installed', 'dtype': 'float'}
            ]
        }
    }
    
    # Update model config for testing
    test_model_config = {
        'MMoE_Test_v1': {
            'batch_norm': False,
            'batch_size': 256,
            'dataset_id': 'test_dataset',
            'debug_mode': False,
            'early_stop_patience': 2,
            'embedding_dim': 16,
            'embedding_regularizer': 1.0e-06,
            'epochs': 3,
            'eval_steps': None,
            'expert_hidden_units': [128, 128],
            'feature_config': None,
            'feature_specs': None,
            'gate_hidden_units': [64, 32],
            'group_id': None,
            'hidden_activations': 'relu',
            'learning_rate': 0.001,
            'loss': ['binary_crossentropy', 'binary_crossentropy'],
            'metrics': ['logloss', 'AUC'],
            'model': 'MMoE',
            'model_root': './checkpoints/',
            'monitor': 'AUC',
            'monitor_mode': 'max',
            'net_dropout': 0.1,
            'net_regularizer': 0,
            'num_experts': 4,
            'num_tasks': 2,
            'num_workers': 1,
            'optimizer': 'adam',
            'pickle_feature_encoder': True,
            'save_best_only': True,
            'seed': 2024,
            'shuffle': True,
            'task': ['binary_classification', 'binary_classification'],
            'tower_hidden_units': [64, 32],
            'use_features': None,
            'verbose': 1,
            'transformer_layers': 1,
            'transformer_heads': 2,
            'transformer_dropout': 0.1,
            'transformer_dim_feedforward': 128
        }
    }
    
    # Save updated configs
    import yaml
    
    with open('config/dataset_config.yaml', 'w') as f:
        yaml.dump(test_dataset_config, f, default_flow_style=False)
    
    with open('config/model_config.yaml', 'w') as f:
        yaml.dump(test_model_config, f, default_flow_style=False)
    
    print("Configuration updated for testing")

def run_complete_pipeline():
    """Run the complete pipeline and collect metrics."""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'python_version': sys.version,
            'torch_version': torch.__version__ if torch else 'N/A',
            'cuda_available': torch.cuda.is_available() if torch else False
        },
        'pipeline_steps': {}
    }
    
    print("Starting comprehensive project test...")
    print(f"Timestamp: {metrics['timestamp']}")
    
    # Step 1: Create test data
    print("\n" + "="*60)
    print("STEP 1: Creating test dataset")
    print("="*60)
    
    start_time = time.time()
    create_test_data()
    data_creation_time = time.time() - start_time
    metrics['pipeline_steps']['data_creation'] = {
        'duration': data_creation_time,
        'memory_usage': get_memory_usage()
    }
    
    # Step 2: Update configuration
    print("\n" + "="*60)
    print("STEP 2: Updating configuration")
    print("="*60)
    
    start_time = time.time()
    update_config_for_test()
    config_update_time = time.time() - start_time
    metrics['pipeline_steps']['config_update'] = {
        'duration': config_update_time,
        'memory_usage': get_memory_usage()
    }
    
    # Step 3: Test training
    print("\n" + "="*60)
    print("STEP 3: Testing training pipeline")
    print("="*60)
    
    training_result = run_command_with_metrics(
        "python src/train_supervised.py --config ./config/ --expid MMoE_Test_v1 --gpu -1",
        "Training Enhanced MMoE Model"
    )
    metrics['pipeline_steps']['training'] = training_result
    
    # Step 4: Test evaluation
    print("\n" + "="*60)
    print("STEP 4: Testing evaluation pipeline")
    print("="*60)
    
    if training_result['success']:
        evaluation_result = run_command_with_metrics(
            "python src/evaluate_model.py --model_path ./checkpoints/MMoE_Test_v1_best.pth --config ./config/ --expid MMoE_Test_v1 --gpu -1",
            "Evaluating Enhanced MMoE Model"
        )
        metrics['pipeline_steps']['evaluation'] = evaluation_result
    else:
        print("Skipping evaluation due to training failure")
        metrics['pipeline_steps']['evaluation'] = {'success': False, 'error': 'Training failed'}
    
    # Step 5: Test federated learning (if available)
    print("\n" + "="*60)
    print("STEP 5: Testing federated learning")
    print("="*60)
    
    federated_result = run_command_with_metrics(
        "python src/run_federated.py --config ./config/ --expid MMoE_Test_Federated --num_clients 2 --num_rounds 2 --iid --data_path ./data/test/train_data_test.csv",
        "Federated Learning Test"
    )
    metrics['pipeline_steps']['federated_learning'] = federated_result
    
    # Step 6: Generate report
    print("\n" + "="*60)
    print("STEP 6: Generating comparison report")
    print("="*60)
    
    report_result = run_command_with_metrics(
        "python scripts/generate_report.py",
        "Generating Performance Report"
    )
    metrics['pipeline_steps']['report_generation'] = report_result
    
    # Save metrics
    with open('test_results.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: test_results.json")
    
    # Print summary
    print("\nSUMMARY:")
    for step, result in metrics['pipeline_steps'].items():
        status = "✅ SUCCESS" if result.get('success', False) else "❌ FAILED"
        duration = result.get('duration', 0)
        print(f"{step}: {status} ({duration:.2f}s)")
    
    return metrics

def main():
    """Main function to run the complete test."""
    try:
        metrics = run_complete_pipeline()
        return metrics
    except Exception as e:
        print(f"Error in main test: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 
