"""
Grade-aware baseline models for Stage 6 ablation.
GradeAwareMLP, GradeAwareLSTM, GradeAwareGRU, GradeAwareCNNLSTM.
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


class GradeAwareMLP(nn.Module):
    """MLP with weather statistics + grade embedding."""

    def __init__(self, input_dim=6, num_grades=None, grade_emb_dim=8, dropout=0.2, output_dim=5):
        super().__init__()
        if num_grades is None:
            num_grades = _load_num_grades()
        self.grade_embed = nn.Embedding(num_grades, grade_emb_dim)
        n_weather = input_dim * 4
        self.fc1 = nn.Linear(n_weather + grade_emb_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, grade_ids):
        B = x.size(0)
        mean_f = x.mean(dim=1)
        std_f = x.std(dim=1)
        max_f = x.max(dim=1).values
        min_f = x.min(dim=1).values
        stats = torch.cat([mean_f, std_f, max_f, min_f], dim=-1)
        g = self.grade_embed(grade_ids)
        h = torch.cat([stats, g], dim=-1)
        h = self.dropout(F.relu(self.fc1(h)))
        h = self.dropout(F.relu(self.fc2(h)))
        return self.fc3(h)


class GradeAwareLSTM(nn.Module):
    """LSTM + grade embedding."""

    def __init__(self, input_dim=6, num_grades=None, grade_emb_dim=8, hidden_size=64,
                 num_layers=1, dropout=0.2, output_dim=5):
        super().__init__()
        if num_grades is None:
            num_grades = _load_num_grades()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.grade_embed = nn.Embedding(num_grades, grade_emb_dim)
        self.fc1 = nn.Linear(hidden_size + grade_emb_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, grade_ids):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        g = self.grade_embed(grade_ids)
        h = torch.cat([h, g], dim=-1)
        h = self.dropout(F.relu(self.fc1(h)))
        return self.fc2(h)


class GradeAwareGRU(nn.Module):
    """GRU + grade embedding."""

    def __init__(self, input_dim=6, num_grades=None, grade_emb_dim=8, hidden_size=64,
                 num_layers=1, dropout=0.2, output_dim=5):
        super().__init__()
        if num_grades is None:
            num_grades = _load_num_grades()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.grade_embed = nn.Embedding(num_grades, grade_emb_dim)
        self.fc1 = nn.Linear(hidden_size + grade_emb_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, grade_ids):
        out, _ = self.gru(x)
        h = out[:, -1, :]
        g = self.grade_embed(grade_ids)
        h = torch.cat([h, g], dim=-1)
        h = self.dropout(F.relu(self.fc1(h)))
        return self.fc2(h)


class GradeAwareCNNLSTM(nn.Module):
    """CNN + LSTM + grade embedding."""

    def __init__(self, input_dim=6, num_grades=None, grade_emb_dim=8, hidden_size=64,
                 dropout=0.2, output_dim=5):
        super().__init__()
        if num_grades is None:
            num_grades = _load_num_grades()
        self.conv1 = nn.Conv1d(input_dim, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.grade_embed = nn.Embedding(num_grades, grade_emb_dim)
        self.fc1 = nn.Linear(hidden_size + grade_emb_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, grade_ids):
        x_c = x.permute(0, 2, 1)
        x_c = F.relu(self.conv1(x_c))
        x_c = F.relu(self.conv2(x_c))
        x_c = x_c.permute(0, 2, 1)
        out, _ = self.lstm(x_c)
        h = out[:, -1, :]
        g = self.grade_embed(grade_ids)
        h = torch.cat([h, g], dim=-1)
        h = self.dropout(F.relu(self.fc1(h)))
        return self.fc2(h)


def build_grade_aware_baseline(model_name, input_dim):
    builders = {
        "grade_mlp": GradeAwareMLP,
        "grade_lstm": GradeAwareLSTM,
        "grade_gru": GradeAwareGRU,
        "grade_cnn_lstm": GradeAwareCNNLSTM,
    }
    return builders[model_name](input_dim=input_dim)
