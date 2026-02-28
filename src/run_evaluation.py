import os
import argparse
import torch
import numpy as np
from typing import Dict, List
from fuxictr.utils import load_config
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader
from src.MMoE import MMoE
from src.ssl_module import SSLModule
from src.evaluate_federated_model import ModelEvaluator

def load_baseline_models(
    feature_map: FeatureMap,
    model_config: dict,
    device: str
) -> Dict[str, MMoE]:
    """Load baseline models for comparison."""
    models = {}
    
    # MMoE + MLP (original)
    mlp_config = model_config.copy()
    mlp_config['use_transformer'] = False
    models['MMoE + MLP'] = MMoE(feature_map=feature_map, **mlp_config).to(device)
    
    # MMoE + Transformer
    transformer_config = model_config.copy()
    transformer_config['use_transformer'] = True
    models['MMoE + Transformer'] = MMoE(feature_map=feature_map, **transformer_config).to(device)
    
    # MMoE + Transformer + SSL
    ssl_config = transformer_config.copy()
    ssl_config['use_ssl'] = True
    models['MMoE + Transformer + SSL'] = MMoE(feature_map=feature_map, **ssl_config).to(device)
    
    return models

def evaluate_baselines(
    models: Dict[str, MMoE],
    test_loader: H5DataLoader,
    evaluator: ModelEvaluator
) -> Dict[str, Dict[str, float]]:
    """Evaluate baseline models."""
    results = {}
    
    for model_name, model in models.items():
        metrics = evaluator.evaluate_model(model, test_loader, model_name)
        results[model_name] = metrics
    
    return results

def evaluate_federated_model(
    feature_map: FeatureMap,
    model_config: dict,
    test_loader: H5DataLoader,
    evaluator: ModelEvaluator,
    device: str
):
    """Evaluate the federated model and its clients."""
    # Load global model
    global_model = MMoE(feature_map=feature_map, **model_config).to(device)
    global_model.load_state_dict(torch.load('global_model.pt'))
    
    # Evaluate global model
    evaluator.evaluate_model(global_model, test_loader, 'MMoE + Transformer + SSL + FL')
    
    # Load and evaluate client models
    for client_id in range(model_config['num_clients']):
        client_model = MMoE(feature_map=feature_map, **model_config).to(device)
        client_model.load_state_dict(torch.load(f'client_{client_id}_model.pt'))
        evaluator.evaluate_model(
            client_model,
            test_loader,
            f'Client_{client_id}'
        )

def analyze_ssl_effect(
    feature_map: FeatureMap,
    model_config: dict,
    test_loader: H5DataLoader,
    evaluator: ModelEvaluator,
    device: str
):
    """Analyze the effect of SSL on learned representations."""
    # Load models
    model_without_ssl = MMoE(feature_map=feature_map, **model_config).to(device)
    model_without_ssl.load_state_dict(torch.load('model_without_ssl.pt'))
    
    model_with_ssl = MMoE(feature_map=feature_map, **model_config).to(device)
    model_with_ssl.load_state_dict(torch.load('model_with_ssl.pt'))
    
    # Initialize SSL module
    ssl_module = SSLModule(
        encoder=model_with_ssl.encoder,
        input_dim=model_config['embedding_dim'] * feature_map.num_fields
    ).to(device)
    
    # Analyze representations
    evaluator.analyze_ssl_representations(
        model_without_ssl,
        ssl_module,
        test_loader,
        save_path=os.path.join(evaluator.results_dir, 'ssl_analysis')
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--results_dir', type=str, default='evaluation_results', help='Directory to save results')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config and feature map
    params = load_config(args.config)
    feature_map = FeatureMap(params['dataset_id'], params['data_root'])
    feature_map_json = os.path.join(params['data_root'], params['dataset_id'], "feature_map.json")
    feature_map.load(feature_map_json, params)
    
    # Create test data loader
    test_loader = H5DataLoader(
        feature_map,
        stage="test",
        train_data=args.data_path,
        valid_data=args.data_path,
        **params
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        feature_map=feature_map,
        model_config=params,
        results_dir=args.results_dir
    )
    
    # Load and evaluate baseline models
    baseline_models = load_baseline_models(feature_map, params, device)
    baseline_results = evaluate_baselines(baseline_models, test_loader, evaluator)
    
    # Evaluate federated model
    evaluate_federated_model(feature_map, params, test_loader, evaluator, device)
    
    # Analyze SSL effect
    analyze_ssl_effect(feature_map, params, test_loader, evaluator, device)
    
    # Generate comparison table
    evaluator.generate_comparison_table(baseline_results)
    
    # Plot convergence
    evaluator.plot_convergence(
        save_path=os.path.join(args.results_dir, 'convergence.png')
    )
    
    # Save all results
    evaluator.save_results()

if __name__ == "__main__":
    main() 