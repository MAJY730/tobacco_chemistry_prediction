"""TSMixer with a fixed 16-dimensional one-hot Grade representation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.top_ts_models import TSMixerRegressor


class OneHotGradeTSMixer(nn.Module):
    """Reuse the formal TSMixer encoder and replace Grade embedding with one-hot."""

    def __init__(
        self,
        input_dim: int = 7,
        num_grades: int = 16,
        seq_len: int = 153,
        d_model: int = 64,
        n_blocks: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        output_dim: int = 5,
    ) -> None:
        super().__init__()
        self.num_grades = num_grades
        self.tsmixer = TSMixerRegressor(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            n_blocks=n_blocks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            output_dim=output_dim,
        )
        self.fc1 = nn.Linear(d_model + num_grades, 64)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, climate: torch.Tensor, grade_ids: torch.Tensor) -> torch.Tensor:
        climate_features = self.tsmixer(climate, return_features=True)
        grade_onehot = F.one_hot(grade_ids, num_classes=self.num_grades).to(
            dtype=climate_features.dtype
        )
        combined = torch.cat([climate_features, grade_onehot], dim=-1)
        output = F.gelu(self.fc1(combined))
        output = self.fc_dropout(output)
        return self.fc2(output)
