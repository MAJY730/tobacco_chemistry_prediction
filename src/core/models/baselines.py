"""
Baseline models for tobacco chemical prediction.
Supports Ridge, RandomForest, XGBoost (ML) and LSTM, GRU, CNN+LSTM (DL).
"""

import warnings
import numpy as np

# --- ML models ---
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBRegressor = None

# --- DL models ---
import torch
import torch.nn as nn


# ============================================================
# Machine Learning Models
# ============================================================

def build_ridge(alpha=1.0):
    return MultiOutputRegressor(Ridge(alpha=alpha, random_state=42))


def build_random_forest(n_estimators=200, max_depth=20, n_jobs=-1):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=n_jobs,
    )


def build_xgboost(n_estimators=200, max_depth=6, learning_rate=0.1):
    if not HAS_XGBOOST:
        warnings.warn("XGBoost not installed, returning None")
        return None
    return MultiOutputRegressor(
        XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbosity=0,
        )
    )


def flatten_X(X):
    """Flatten (N, T, F) -> (N, T*F) for ML models."""
    return X.reshape(X.shape[0], -1)


# ============================================================
# Deep Learning Models
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        out = out[:, -1, :]            # last time step
        out = self.dropout(out)
        return self.fc(out)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=5, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=5, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(64, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, T, F) -> permute to (B, F, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # (B, T', 64)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


def build_lstm(input_size, hidden_size=64, output_size=5, dropout=0.2):
    return LSTMModel(input_size, hidden_size, output_size=output_size, dropout=dropout)


def build_gru(input_size, hidden_size=64, output_size=5, dropout=0.2):
    return GRUModel(input_size, hidden_size, output_size=output_size, dropout=dropout)


def build_cnn_lstm(input_size, hidden_size=64, output_size=5, dropout=0.2):
    return CNNLSTMModel(input_size, hidden_size, output_size=output_size, dropout=dropout)
