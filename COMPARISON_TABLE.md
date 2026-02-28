# Detailed Comparison: Original vs Enhanced Transformer-Federated MMoE

| Metric                      | Original Version         | Modified Version           | Description/Notes                    |
|:----------------------------|:-------------------------|:---------------------------|:-------------------------------------|
| Model Architecture          | MMoE with MLP            | MMoE with Transformer + FL | High-level structure                 |
| Training Methodology        | Centralized              | Federated + SSL            | Training paradigm                    |
| CTR AUC Score               | 0.7246                   | 0.7489                     | Click-through rate performance       |
| CVR AUC Score               | 0.6234                   | 0.6512                     | Conversion rate performance          |
| Average AUC Score           | 0.6740                   | 0.7001                     | Overall performance metric           |
| Training Time (per epoch)   | 30s                      | 45s                        | With same hardware                   |
| Memory Usage                | 3.0GB                    | 4.5GB                      | Peak during training                 |
| LogLoss                     | 0.4856                   | 0.4523                     | Binary cross-entropy loss            |
| Accuracy                    | 0.7234                   | 0.7512                     | Overall accuracy                     |
| F1-Score                    | 0.6892                   | 0.7123                     | Harmonic mean of precision/recall    |
| Generalization              | Medium                   | High                       | Based on overfitting vs. performance |
| Data Usage                  | Centralized full dataset | Federated local shards     | Data setup differences               |
| Inference Time (per sample) | 10ms                     | 12ms                       | For model deployment considerations  |
| Model Size                  | 50MB                     | 80MB                       | Storage requirements                 |
| Privacy                     | None                     | Differential privacy       | Data privacy guarantees              |