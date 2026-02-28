#!/usr/bin/env python3
"""
Report Generation Script for Enhanced MMoE Models
Generates comprehensive comparison reports for all trained model variants.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np


def load_results(checkpoints_dir):
    """Load all evaluation results from checkpoints directory."""
    results = {}
    
    for filename in os.listdir(checkpoints_dir):
        if filename.endswith('_evaluation_results.json'):
            expid = filename.replace('_evaluation_results.json', '')
            filepath = os.path.join(checkpoints_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    results[expid] = json.load(f)
                print(f"Loaded results for {expid}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return results


def create_summary_table(results):
    """Create a summary table of all model results."""
    summary_data = []
    
    for expid, result in results.items():
        overall_metrics = result['overall_metrics']
        task_metrics = result['task_metrics']
        
        # Extract metrics
        row = {
            'Model': expid,
            'Average_AUC': overall_metrics['average_auc'],
            'Average_Accuracy': overall_metrics['average_accuracy'],
            'Average_F1': overall_metrics['average_f1'],
            'Average_LogLoss': overall_metrics['average_logloss'],
            'CTR_AUC': task_metrics['CTR']['auc'],
            'CVR_AUC': task_metrics['CVR']['auc'],
            'CTR_Accuracy': task_metrics['CTR']['accuracy'],
            'CVR_Accuracy': task_metrics['CVR']['accuracy'],
            'CTR_F1': task_metrics['CTR']['f1_score'],
            'CVR_F1': task_metrics['CVR']['f1_score']
        }
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def create_visualizations(results, output_dir):
    """Create visualization plots for model comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    models = list(results.keys())
    metrics = ['average_auc', 'average_accuracy', 'average_f1']
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results[model]['overall_metrics'][metric] for model in models]
        
        axes[i].bar(models, values, color=sns.color_palette("husl", len(models)))
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create task-specific comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    tasks = ['CTR', 'CVR']
    metrics = ['auc', 'accuracy']
    
    for i, task in enumerate(tasks):
        for j, metric in enumerate(metrics):
            values = [results[model]['task_metrics'][task][metric] for model in models]
            
            axes[i, j].bar(models, values, color=sns.color_palette("husl", len(models)))
            axes[i, j].set_title(f'{task} {metric.upper()}')
            axes[i, j].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(results, output_dir):
    """Generate a comprehensive report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary table
    summary_df = create_summary_table(results)
    
    # Save summary table
    summary_path = os.path.join(output_dir, 'model_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Generate markdown report
    report_path = os.path.join(output_dir, 'model_comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Enhanced MMoE Model Comparison Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Performance Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Find best model for each metric
        best_auc = summary_df.loc[summary_df['Average_AUC'].idxmax()]
        best_accuracy = summary_df.loc[summary_df['Average_Accuracy'].idxmax()]
        best_f1 = summary_df.loc[summary_df['Average_F1'].idxmax()]
        
        f.write(f"- **Best Average AUC**: {best_auc['Model']} ({best_auc['Average_AUC']:.4f})\n")
        f.write(f"- **Best Average Accuracy**: {best_accuracy['Model']} ({best_accuracy['Average_Accuracy']:.4f})\n")
        f.write(f"- **Best Average F1-Score**: {best_f1['Model']} ({best_f1['Average_F1']:.4f})\n\n")
        
        f.write("## Model Architecture Analysis\n\n")
        
        # Analyze model types
        transformer_models = [m for m in models if 'Transformer' in m]
        federated_models = [m for m in models if 'Federated' in m]
        ssl_models = [m for m in models if 'SSL' in m]
        
        f.write(f"- **Transformer Models**: {len(transformer_models)} variants\n")
        f.write(f"- **Federated Models**: {len(federated_models)} variants\n")
        f.write(f"- **SSL Models**: {len(ssl_models)} variants\n\n")
        
        # Performance analysis by type
        if transformer_models:
            transformer_avg = summary_df[summary_df['Model'].isin(transformer_models)]['Average_AUC'].mean()
            f.write(f"- **Transformer Models Average AUC**: {transformer_avg:.4f}\n")
        
        if federated_models:
            federated_avg = summary_df[summary_df['Model'].isin(federated_models)]['Average_AUC'].mean()
            f.write(f"- **Federated Models Average AUC**: {federated_avg:.4f}\n")
        
        if ssl_models:
            ssl_avg = summary_df[summary_df['Model'].isin(ssl_models)]['Average_AUC'].mean()
            f.write(f"- **SSL Models Average AUC**: {ssl_avg:.4f}\n")
        
        f.write("\n## Recommendations\n\n")
        
        # Generate recommendations based on results
        best_overall = summary_df.loc[summary_df['Average_AUC'].idxmax()]
        
        f.write(f"### Best Overall Model: {best_overall['Model']}\n")
        f.write(f"- Average AUC: {best_overall['Average_AUC']:.4f}\n")
        f.write(f"- Average Accuracy: {best_overall['Average_Accuracy']:.4f}\n")
        f.write(f"- Average F1-Score: {best_overall['Average_F1']:.4f}\n\n")
        
        f.write("### Use Case Recommendations:\n\n")
        f.write("1. **For High Performance**: Use the best overall model\n")
        f.write("2. **For Privacy**: Use federated learning models\n")
        f.write("3. **For Efficiency**: Use transformer models with smaller architectures\n")
        f.write("4. **For Research**: Use SSL models for representation learning\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("Generated visualizations are available in the output directory:\n")
        f.write("- `model_comparison.png`: Overall metric comparison\n")
        f.write("- `task_comparison.png`: Task-specific performance\n")
        f.write("- `model_summary.csv`: Detailed metrics table\n")
    
    print(f"Report generated: {report_path}")
    print(f"Summary table: {summary_path}")
    print(f"Visualizations saved to: {output_dir}")


def main():
    """Main function to generate the report."""
    print("Generating Enhanced MMoE Model Comparison Report")
    print("=" * 50)
    
    # Load results
    checkpoints_dir = "./checkpoints"
    results = load_results(checkpoints_dir)
    
    if not results:
        print("No evaluation results found in checkpoints directory.")
        print("Please run training and evaluation first.")
        return
    
    print(f"Loaded results for {len(results)} models")
    
    # Generate report
    output_dir = "./reports"
    generate_report(results, output_dir)
    
    print("\nReport generation completed!")
    print(f"Check the '{output_dir}' directory for results.")


if __name__ == "__main__":
    main() 