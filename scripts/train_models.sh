#!/bin/bash

# Enhanced MMoE Training Script
# This script trains multiple variants of the transformer-enhanced MMoE model

echo "=========================================="
echo "Training Enhanced MMoE Models"
echo "=========================================="

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Training configurations
MODELS=(
    "MMoE_Transformer_v1"
    "MMoE_Transformer_v2"
    "MMoE_Federated_v1"
    "MMoE_Federated_v2"
    "MMoE_SSL_v1"
)

# Train each model variant
for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Training $model"
    echo "=========================================="
    
    # Train centralized model
    if [[ $model == *"Federated"* ]]; then
        echo "Training federated model: $model"
        python src/run_federated.py \
            --config ./config/ \
            --expid $model \
            --num_clients 5 \
            --num_rounds 10 \
            --iid \
            --data_path ./data/final/train_data_final.csv
    else
        echo "Training centralized model: $model"
        python src/train_supervised.py \
            --config ./config/ \
            --expid $model \
            --gpu 0
    fi
    
    # Evaluate the trained model
    echo "Evaluating $model"
    python src/evaluate_model.py \
        --model_path ./checkpoints/${model}_best.pth \
        --config ./config/ \
        --expid $model \
        --gpu 0
    
    echo "Completed training and evaluation of $model"
    echo "=========================================="
done

echo "=========================================="
echo "All models trained and evaluated!"
echo "=========================================="

# Generate comparison report
echo "Generating comparison report..."
python scripts/generate_report.py

echo "Training script completed!" 