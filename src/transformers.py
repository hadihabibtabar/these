import torch
from torch import nn

class TransformerExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, nhead=4, dropout=0.1, batch_norm=False):
        super(TransformerExpert, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.input_proj(x).unsqueeze(1)  # Add sequence dimension: (batch_size, 1, hidden_dim)
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension: (batch_size, hidden_dim)
        x = self.output_proj(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.dropout(x)
        return x

class TransformerGate(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=64, dropout=0.1, batch_norm=False):
        super(TransformerGate, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=2,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=1
        )
        self.output_proj = nn.Linear(hidden_dim, num_experts)
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.input_proj(x).unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.output_proj(x)
        return x

class TransformerTower(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1, batch_norm=False):
        super(TransformerTower, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.input_proj(x).unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.output_proj(x)
        return x 