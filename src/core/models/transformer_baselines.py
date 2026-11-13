"""
Transformer-based baselines for Stage 8.
TransformerRegressor, GradeAwareTransformer, TransformerLSTM, GradeAwareTransformerLSTM.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_num_grades():
    with open(PROJECT_ROOT / "data" / "processed" / "grade_mapping.json") as f:
        return json.load(f)["num_grades"]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerRegressor(nn.Module):
    """Vanilla Transformer Encoder for multivariate regression."""

    def __init__(self, input_dim=6, seq_len=153, d_model=64, n_heads=4, num_layers=2,
                 dim_feedforward=128, dropout=0.2, output_dim=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x, return_features=False):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.out_norm(x)
        features = x.mean(dim=1)
        if return_features:
            return features
        out = F.gelu(self.fc1(features))
        out = self.fc_dropout(out)
        return self.fc2(out)


class GradeAwareTransformerRegressor(nn.Module):
    """Transformer + grade embedding."""

    def __init__(self, input_dim=6, seq_len=153, d_model=64, n_heads=4, num_layers=2,
                 dim_feedforward=128, dropout=0.2, output_dim=5, num_grades=None, grade_emb_dim=8):
        super().__init__()
        if num_grades is None:
            num_grades = _load_num_grades()
        self.encoder = TransformerRegressor(input_dim, seq_len, d_model, n_heads, num_layers,
                                            dim_feedforward, dropout, output_dim)
        self.grade_embed = nn.Embedding(num_grades, grade_emb_dim)
        self.fc1 = nn.Linear(d_model + grade_emb_dim, 64)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x, grade_ids):
        ts_features = self.encoder(x, return_features=True)
        grade_emb = self.grade_embed(grade_ids)
        combined = torch.cat([ts_features, grade_emb], dim=-1)
        out = F.gelu(self.fc1(combined))
        out = self.fc_dropout(out)
        return self.fc2(out)


class TransformerLSTMRegressor(nn.Module):
    """LSTM + Transformer hybrid encoder."""

    def __init__(self, input_dim=6, seq_len=153, lstm_hidden=64, d_model=64,
                 n_heads=4, num_layers=1, dim_feedforward=128, dropout=0.2, output_dim=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True)
        self.proj = nn.Linear(lstm_hidden, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x, return_features=False):
        x, _ = self.lstm(x)
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.out_norm(x)
        features = x.mean(dim=1)
        if return_features:
            return features
        out = F.gelu(self.fc1(features))
        out = self.fc_dropout(out)
        return self.fc2(out)


class GradeAwareTransformerLSTMRegressor(nn.Module):
    """LSTM + Transformer + grade embedding."""

    def __init__(self, input_dim=6, seq_len=153, lstm_hidden=64, d_model=64,
                 n_heads=4, num_layers=1, dim_feedforward=128, dropout=0.2, output_dim=5,
                 num_grades=None, grade_emb_dim=8):
        super().__init__()
        if num_grades is None:
            num_grades = _load_num_grades()
        self.encoder = TransformerLSTMRegressor(input_dim, seq_len, lstm_hidden, d_model,
                                                n_heads, num_layers, dim_feedforward, dropout, output_dim)
        self.grade_embed = nn.Embedding(num_grades, grade_emb_dim)
        self.fc1 = nn.Linear(d_model + grade_emb_dim, 64)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x, grade_ids):
        ts_features = self.encoder(x, return_features=True)
        grade_emb = self.grade_embed(grade_ids)
        combined = torch.cat([ts_features, grade_emb], dim=-1)
        out = F.gelu(self.fc1(combined))
        out = self.fc_dropout(out)
        return self.fc2(out)


# Configs
TRANSFORMER_CONFIGS = {
    "default": dict(d_model=64, n_heads=4, num_layers=2, dim_feedforward=128, dropout=0.2),
    "light": dict(d_model=32, n_heads=4, num_layers=1, dim_feedforward=64, dropout=0.3),
}
TRANSFORMER_LSTM_CONFIGS = {
    "default": dict(lstm_hidden=64, d_model=64, n_heads=4, num_layers=1, dim_feedforward=128, dropout=0.2),
    "light": dict(lstm_hidden=32, d_model=32, n_heads=4, num_layers=1, dim_feedforward=64, dropout=0.3),
}


def build_transformer(model_name, input_dim, seq_len=153, variant="default", output_dim=5):
    """Build a Transformer-based model."""
    if model_name == "transformer":
        cfg = TRANSFORMER_CONFIGS.get(variant, TRANSFORMER_CONFIGS["default"])
        return TransformerRegressor(input_dim=input_dim, seq_len=seq_len, output_dim=output_dim, **cfg)
    elif model_name == "grade_transformer":
        cfg = TRANSFORMER_CONFIGS.get(variant, TRANSFORMER_CONFIGS["default"])
        return GradeAwareTransformerRegressor(input_dim=input_dim, seq_len=seq_len, output_dim=output_dim, **cfg)
    elif model_name == "transformer_lstm":
        cfg = TRANSFORMER_LSTM_CONFIGS.get(variant, TRANSFORMER_LSTM_CONFIGS["default"])
        return TransformerLSTMRegressor(input_dim=input_dim, seq_len=seq_len, output_dim=output_dim, **cfg)
    elif model_name == "grade_transformer_lstm":
        cfg = TRANSFORMER_LSTM_CONFIGS.get(variant, TRANSFORMER_LSTM_CONFIGS["default"])
        return GradeAwareTransformerLSTMRegressor(input_dim=input_dim, seq_len=seq_len, output_dim=output_dim, **cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")
