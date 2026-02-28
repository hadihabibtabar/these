import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from src.ssl_module import SSLPretrainer
from src.train_supervised import SupervisedTrainer
from src.fl_server import start_server
from src.fl_client import start_client
from src.evaluate_model import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Train MMoE model with SSL and FL')
    parser.add_argument('--stage', type=str, required=True,
                      choices=['ssl', 'supervised', 'fl_server', 'fl_client', 'evaluate'],
                      help='Training stage to run')
    parser.add_argument('--client_id', type=int, default=0,
                      help='Client ID for federated learning')
    parser.add_argument('--num_tasks', type=int, default=2,
                      help='Number of tasks in multi-task learning')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory for logging')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory for model checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                      help='Whether to use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.log_dir).mkdir(exist_ok=True)
    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    
    # Initialize model (replace with your model)
    model = YourMMoEModel()  # Replace with your actual model
    model = model.to(args.device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if args.stage == 'ssl':
        # SSL pretraining
        pretrainer = SSLPretrainer(
            model=model,
            optimizer=optimizer,
            device=args.device,
            log_dir=f"{args.log_dir}/ssl",
            checkpoint_dir=f"{args.checkpoint_dir}/ssl",
            use_wandb=args.use_wandb
        )
        
        # Load your SSL data
        ssl_train_loader = YourSSLDataLoader()  # Replace with your data loader
        
        pretrainer.pretrain(
            train_loader=ssl_train_loader,
            num_epochs=args.num_epochs,
            save_every=5
        )
    
    elif args.stage == 'supervised':
        # Supervised fine-tuning
        trainer = SupervisedTrainer(
            model=model,
            optimizer=optimizer,
            device=args.device,
            num_tasks=args.num_tasks,
            log_dir=f"{args.log_dir}/supervised",
            checkpoint_dir=f"{args.checkpoint_dir}/supervised",
            use_wandb=args.use_wandb
        )
        
        # Load your supervised data
        train_loader = YourTrainDataLoader()  # Replace with your data loader
        val_loader = YourValDataLoader()  # Replace with your data loader
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            save_every=5,
            patience=10
        )
    
    elif args.stage == 'fl_server':
        # Start federated learning server
        start_server(
            model=model,
            num_clients=10,  # Adjust based on your setup
            num_rounds=100,  # Adjust based on your needs
            server_address="[::]:8080",
            log_dir=f"{args.log_dir}/fl_server",
            use_wandb=args.use_wandb
        )
    
    elif args.stage == 'fl_client':
        # Start federated learning client
        # Load client-specific data
        client_train_loader = YourClientTrainDataLoader()  # Replace with your data loader
        client_val_loader = YourClientValDataLoader()  # Replace with your data loader
        
        start_client(
            model=model,
            train_loader=client_train_loader,
            val_loader=client_val_loader,
            optimizer=optimizer,
            device=args.device,
            num_tasks=args.num_tasks,
            client_id=args.client_id,
            server_address="localhost:8080",
            log_dir=f"{args.log_dir}/fl_client_{args.client_id}"
        )
    
    elif args.stage == 'evaluate':
        # Evaluate model
        # Load test data
        test_loader = YourTestDataLoader()  # Replace with your data loader
        
        metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=args.device,
            num_tasks=args.num_tasks,
            log_dir=f"{args.log_dir}/evaluation",
            use_wandb=args.use_wandb
        )
        
        print("Evaluation metrics:", metrics)

if __name__ == '__main__':
    main() 