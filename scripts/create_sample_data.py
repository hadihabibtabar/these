#!/usr/bin/env python3
"""
Create Sample Dataset for Testing Enhanced MMoE Models
Generates a smaller dataset for quick testing and validation.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def create_sample_dataset(input_file, output_dir, sample_size=10000):
    """Create a sample dataset for testing."""
    
    print(f"Creating sample dataset from {input_file}")
    print(f"Sample size: {sample_size}")
    
    # Read the original data
    df = pd.read_csv(input_file, sep='\t', nrows=sample_size*2)  # Read more to ensure we have enough after filtering
    
    # Ensure we have the required columns
    required_cols = ['is_clicked', 'is_installed']
    if not all(col in df.columns for col in required_cols):
        print("Warning: Required label columns not found. Creating dummy labels.")
        df['is_clicked'] = np.random.randint(0, 2, size=len(df))
        df['is_installed'] = np.random.randint(0, 2, size=len(df))
    
    # Create feature columns if they don't exist
    feature_cols = [f'f_{i}' for i in range(2, 80)]
    for col in feature_cols:
        if col not in df.columns:
            if col.startswith('f_'):
                # Categorical features
                df[col] = np.random.randint(0, 100, size=len(df))
            else:
                # Numerical features
                df[col] = np.random.randn(len(df))
    
    # Split the data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the datasets
    train_path = os.path.join(output_dir, 'train_data_sample.csv')
    val_path = os.path.join(output_dir, 'valid_data_sample.csv')
    test_path = os.path.join(output_dir, 'test_data_sample.csv')
    
    train_df.to_csv(train_path, sep='\t', index=False)
    val_df.to_csv(val_path, sep='\t', index=False)
    test_df.to_csv(test_path, sep='\t', index=False)
    
    print(f"Sample datasets created:")
    print(f"  Train: {len(train_df)} samples -> {train_path}")
    print(f"  Valid: {len(val_df)} samples -> {val_path}")
    print(f"  Test: {len(test_df)} samples -> {test_path}")
    
    return train_path, val_path, test_path


def create_feature_map(output_dir):
    """Create feature map for the sample dataset."""
    
    # Create feature map configuration
    feature_map_config = {
        'data_root': output_dir,
        'data_format': 'csv',
        'train_data': 'train_data_sample.csv',
        'valid_data': 'valid_data_sample.csv',
        'test_data': 'test_data_sample.csv',
        'min_categr_count': 3,
        'feature_cols': [
            {
                'name': [f'f_{i}' for i in range(2, 42)],
                'active': True,
                'dtype': 'str',
                'type': 'categorical'
            },
            {
                'name': [f'f_{i}' for i in range(42, 80)],
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
    
    # Save feature map
    import yaml
    feature_map_path = os.path.join(output_dir, 'feature_map.yaml')
    with open(feature_map_path, 'w') as f:
        yaml.dump(feature_map_config, f, default_flow_style=False)
    
    print(f"Feature map created: {feature_map_path}")
    return feature_map_path


def main():
    """Main function to create sample dataset."""
    
    # Input and output paths
    input_file = 'data/final/train_data_final.csv'
    output_dir = 'data/sample'
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        print("Creating synthetic dataset instead...")
        
        # Create synthetic dataset
        n_samples = 10000
        df = pd.DataFrame()
        
        # Create features
        for i in range(2, 80):
            if i < 42:
                df[f'f_{i}'] = np.random.randint(0, 100, size=n_samples)
            else:
                df[f'f_{i}'] = np.random.randn(n_samples)
        
        # Create labels
        df['is_clicked'] = np.random.randint(0, 2, size=n_samples)
        df['is_installed'] = np.random.randint(0, 2, size=n_samples)
        
        # Save synthetic dataset
        os.makedirs('data/synthetic', exist_ok=True)
        df.to_csv('data/synthetic/train_data_synthetic.csv', sep='\t', index=False)
        input_file = 'data/synthetic/train_data_synthetic.csv'
        output_dir = 'data/sample'
    
    # Create sample dataset
    train_path, val_path, test_path = create_sample_dataset(input_file, output_dir, sample_size=5000)
    
    # Create feature map
    feature_map_path = create_feature_map(output_dir)
    
    print("\nSample dataset creation completed!")
    print("You can now use this dataset for testing the enhanced models.")


if __name__ == "__main__":
    main() 