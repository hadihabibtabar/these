import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from datetime import datetime
import logging
from src.MMoE import MMoE
from src.ssl_module import SSLModule
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader

class ModelEvaluator:
    def __init__(
        self,
        feature_map: FeatureMap,
        model_config: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        results_dir: str = "evaluation_results"
    ):
        self.feature_map = feature_map
        self.model_config = model_config
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(results_dir, 'evaluation.log')),
                logging.StreamHandler()
            ]
        )
        
        # Initialize results storage
        self.results = {
            'global_model': {},
            'client_models': {},
            'ssl_analysis': {},
            'convergence': {
                'train_loss': [],
                'val_auc': [],
                'val_accuracy': []
            }
        }

    def evaluate_model(
        self,
        model: MMoE,
        data_loader: H5DataLoader,
        model_name: str = "global_model"
    ) -> Dict[str, float]:
        """Evaluate model performance using multiple metrics."""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader.make_iterator():
                if isinstance(batch, tuple):
                    inputs, labels = batch
                else:
                    inputs = batch
                    labels = batch[1]  # Assuming labels are in second position
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                preds = outputs[f"{self.feature_map.labels[0]}_pred"]
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            "AUC": roc_auc_score(all_labels, all_preds),
            "LogLoss": log_loss(all_labels, all_preds),
            "Accuracy": accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        }
        
        self.results[model_name] = metrics
        logging.info(f"Evaluation results for {model_name}: {metrics}")
        
        return metrics

    def analyze_ssl_representations(
        self,
        model: MMoE,
        ssl_module: SSLModule,
        data_loader: H5DataLoader,
        save_path: Optional[str] = None
    ):
        """Analyze and visualize learned representations with and without SSL."""
        # Get embeddings without SSL
        model.eval()
        embeddings_no_ssl = []
        labels = []
        
        with torch.no_grad():
            for batch in data_loader.make_iterator():
                if isinstance(batch, tuple):
                    inputs, batch_labels = batch
                else:
                    inputs = batch
                    batch_labels = batch[1]
                
                inputs = inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Get embeddings from the encoder
                feature_emb = model.embedding_layer(inputs)
                expert_outputs = model.mmoe_layer(feature_emb.flatten(start_dim=1).unsqueeze(1))
                embeddings = expert_outputs[0]  # Use first task's embeddings
                
                embeddings_no_ssl.extend(embeddings.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
        
        # Get embeddings with SSL
        ssl_module.encoder.eval()
        embeddings_ssl = []
        
        with torch.no_grad():
            for batch in data_loader.make_iterator():
                if isinstance(batch, tuple):
                    inputs, _ = batch
                else:
                    inputs = batch
                
                inputs = inputs.to(self.device)
                embeddings = ssl_module.encoder(inputs)
                embeddings_ssl.extend(embeddings.cpu().numpy())
        
        # Visualize embeddings
        self._plot_embeddings(
            np.array(embeddings_no_ssl),
            np.array(labels),
            "Without SSL Pretraining",
            save_path=f"{save_path}_no_ssl.png" if save_path else None
        )
        
        self._plot_embeddings(
            np.array(embeddings_ssl),
            np.array(labels),
            "With SSL Pretraining",
            save_path=f"{save_path}_ssl.png" if save_path else None
        )
        
        # Store results
        self.results['ssl_analysis'] = {
            'embeddings_no_ssl': embeddings_no_ssl,
            'embeddings_ssl': embeddings_ssl,
            'labels': labels
        }

    def _plot_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: str,
        save_path: Optional[str] = None
    ):
        """Plot embeddings using t-SNE."""
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=labels,
            cmap='coolwarm',
            alpha=0.7
        )
        plt.colorbar(scatter)
        plt.title(f"t-SNE of Learned Representations - {title}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_convergence(
        self,
        save_path: Optional[str] = None
    ):
        """Plot convergence metrics over training rounds."""
        metrics = ['train_loss', 'val_auc', 'val_accuracy']
        titles = ['Training Loss', 'Validation AUC', 'Validation Accuracy']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            values = self.results['convergence'][metric]
            axes[idx].plot(range(1, len(values) + 1), values, marker='o')
            axes[idx].set_xlabel("Round")
            axes[idx].set_ylabel(title)
            axes[idx].set_title(title)
            axes[idx].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def generate_comparison_table(
        self,
        baseline_results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Generate comparison table of different model variants."""
        # Combine all results
        all_results = {**baseline_results, **self.results}
        
        # Create DataFrame
        df = pd.DataFrame.from_dict(all_results, orient='index')
        df.index.name = 'Model'
        
        # Save to CSV
        csv_path = os.path.join(self.results_dir, 'model_comparison.csv')
        df.to_csv(csv_path)
        
        # Generate markdown table
        markdown_table = df.to_markdown()
        with open(os.path.join(self.results_dir, 'model_comparison.md'), 'w') as f:
            f.write(markdown_table)
        
        return df

    def save_results(self):
        """Save all evaluation results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f'evaluation_results_{timestamp}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value.tolist() if isinstance(value, np.ndarray) else value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logging.info(f"Results saved to {results_path}")

def main():
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--results_dir', type=str, default='evaluation_results', help='Directory to save results')
    args = parser.parse_args()
    
    # Load config and feature map
    params = load_config(args.config)
    feature_map = FeatureMap(params['dataset_id'], params['data_root'])
    feature_map_json = os.path.join(params['data_root'], params['dataset_id'], "feature_map.json")
    feature_map.load(feature_map_json, params)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        feature_map=feature_map,
        model_config=params,
        results_dir=args.results_dir
    )
    
    # Load models and evaluate
    # ... (implement model loading and evaluation logic)
    
    # Save results
    evaluator.save_results()

if __name__ == "__main__":
    main() 