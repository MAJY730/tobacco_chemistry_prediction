"""
Top-tier time series models for Stage 5.
TSMixer, PatchTST, and Grade-aware variants.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TSMixer
# ============================================================

class TSMixerBlock(nn.Module):
    """One TSMixer block: time mixing + feature mixing with residuals."""

    def __init__(self, seq_len, d_model, hidden_dim, dropout=0.2):
        super().__init__()
        # Time mixing: operate on (B, C, L)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_fc1 = nn.Linear(seq_len, hidden_dim)
        self.time_fc2 = nn.Linear(hidden_dim, seq_len)
        self.time_dropout = nn.Dropout(dropout)

        # Feature mixing: operate on (B, L, C)
        self.feat_norm = nn.LayerNorm(d_model)
        self.feat_fc1 = nn.Linear(d_model, hidden_dim)
        self.feat_fc2 = nn.Linear(hidden_dim, d_model)
        self.feat_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, d_model)

        # Time mixing: transpose -> (B, d_model, L) -> linear over L -> transpose back
        residual = x
        x_t = self.time_norm(x).transpose(1, 2)  # (B, d_model, L)
        x_t = self.time_dropout(F.gelu(self.time_fc1(x_t)))
        x_t = self.time_fc2(x_t)
        x_t = x_t.transpose(1, 2)  # (B, L, d_model)
        x = residual + x_t

        # Feature mixing
        residual = x
        x_f = self.feat_norm(x)
        x_f = self.feat_dropout(F.gelu(self.feat_fc1(x_f)))
        x_f = self.feat_fc2(x_f)
        x = residual + x_f

        return x


class TSMixerRegressor(nn.Module):
    """TSMixer for multivariate time series regression.

    Args:
        input_dim: number of input features (6 or 7)
        seq_len: sequence length (153)
        d_model: embedding dimension
        n_blocks: number of mixer blocks
        hidden_dim: hidden dim for mixing MLPs
        dropout: dropout rate
        output_dim: number of targets (5)
    """

    def __init__(self, input_dim=6, seq_len=153, d_model=64, n_blocks=3,
                 hidden_dim=128, dropout=0.2, output_dim=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            TSMixerBlock(seq_len, d_model, hidden_dim, dropout)
            for _ in range(n_blocks)
        ])
        self.pool_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x, return_features=False):
        # x: (B, L, input_dim)
        x = self.input_proj(x)  # (B, L, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.pool_norm(x)
        features = x.mean(dim=1)  # (B, d_model)
        if return_features:
            return features
        out = F.gelu(self.fc1(features))
        out = self.fc_dropout(out)
        return self.fc2(out)


# ============================================================
# PatchTST
# ============================================================

class PatchTSTRegressor(nn.Module):
    """PatchTST for multivariate time series regression.

    Args:
        input_dim: number of input features (6 or 7)
        seq_len: sequence length (153)
        patch_len: length of each patch
        stride: stride between patches
        d_model: transformer embedding dimension
        n_heads: attention heads
        n_layers: transformer encoder layers
        dim_feedforward: feedforward dim in transformer
        dropout: dropout rate
        output_dim: number of targets (5)
    """

    def __init__(self, input_dim=6, seq_len=153, patch_len=16, stride=8,
                 d_model=64, n_heads=4, n_layers=2, dim_feedforward=128,
                 dropout=0.2, output_dim=5):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim

        # Compute number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        patch_dim = patch_len * input_dim

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

        # FC head
        self.fc1 = nn.Linear(d_model, 64)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def _patchify(self, x):
        # x: (B, L, C) -> (B, num_patches, patch_len * C)
        B, L, C = x.shape
        patches = []
        for i in range(0, L - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :]  # (B, patch_len, C)
            patches.append(patch.reshape(B, -1))
        return torch.stack(patches, dim=1)  # (B, num_patches, patch_len*C)

    def forward(self, x, return_features=False):
        B, L, C = x.shape
        # Patchify
        patches = self._patchify(x)  # (B, num_patches, patch_len*C)
        # Embed
        x_emb = self.patch_embed(patches)  # (B, num_patches, d_model)
        x_emb = x_emb + self.pos_embed
        x_emb = self.embed_dropout(x_emb)
        # Transformer
        x_out = self.transformer(x_emb)  # (B, num_patches, d_model)
        x_out = self.out_norm(x_out)
        # Pool
        features = x_out.mean(dim=1)  # (B, d_model)
        if return_features:
            return features
        out = F.gelu(self.fc1(features))
        out = self.fc_dropout(out)
        return self.fc2(out)


# ============================================================
# Grade-Aware Wrappers
# ============================================================

class GradeAwareTSMixer(nn.Module):
    """TSMixer with grade embedding concatenated before FC head."""

    def __init__(self, input_dim=6, num_grades=16, grade_emb_dim=8, seq_len=153,
                 d_model=64, n_blocks=3, hidden_dim=128, dropout=0.2, output_dim=5):
        super().__init__()
        self.tsmixer = TSMixerRegressor(input_dim, seq_len, d_model, n_blocks,
                                        hidden_dim, dropout, output_dim)
        self.grade_embed = nn.Embedding(num_grades, grade_emb_dim)
        self.fc1 = nn.Linear(d_model + grade_emb_dim, 64)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x, grade_ids):
        ts_features = self.tsmixer(x, return_features=True)
        grade_emb = self.grade_embed(grade_ids)
        combined = torch.cat([ts_features, grade_emb], dim=-1)
        out = F.gelu(self.fc1(combined))
        out = self.fc_dropout(out)
        return self.fc2(out)


class GradeAwarePatchTST(nn.Module):
    """PatchTST with grade embedding concatenated before FC head."""

    def __init__(self, input_dim=6, num_grades=16, grade_emb_dim=8, seq_len=153,
                 patch_len=16, stride=8, d_model=64, n_heads=4, n_layers=2,
                 dim_feedforward=128, dropout=0.2, output_dim=5):
        super().__init__()
        self.patchtst = PatchTSTRegressor(input_dim, seq_len, patch_len, stride,
                                          d_model, n_heads, n_layers,
                                          dim_feedforward, dropout, output_dim)
        self.grade_embed = nn.Embedding(num_grades, grade_emb_dim)
        self.fc1 = nn.Linear(d_model + grade_emb_dim, 64)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x, grade_ids):
        ts_features = self.patchtst(x, return_features=True)
        grade_emb = self.grade_embed(grade_ids)
        combined = torch.cat([ts_features, grade_emb], dim=-1)
        out = F.gelu(self.fc1(combined))
        out = self.fc_dropout(out)
        return self.fc2(out)


# ============================================================
# Builder functions
# ============================================================

MODEL_CONFIGS = {
    "tsmixer_default": dict(d_model=64, n_blocks=3, hidden_dim=128, dropout=0.2),
    "tsmixer_light": dict(d_model=32, n_blocks=2, hidden_dim=64, dropout=0.3),
    "patchtst_default": dict(patch_len=16, stride=8, d_model=64, n_heads=4,
                              n_layers=2, dim_feedforward=128, dropout=0.2),
    "patchtst_light": dict(patch_len=16, stride=8, d_model=32, n_heads=2,
                            n_layers=1, dim_feedforward=64, dropout=0.3),
}


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_model(model_name, input_dim, seq_len=153, variant="default",
                num_grades=None, output_dim=5):
    """Build model by name and variant."""
    base_name = model_name.replace("grade_", "")

    key = f"{base_name}_{variant}"
    cfg = MODEL_CONFIGS.get(key, MODEL_CONFIGS.get(f"{base_name}_default", {}))

    if model_name == "tsmixer":
        return TSMixerRegressor(input_dim=input_dim, seq_len=seq_len, output_dim=output_dim, **cfg)
    elif model_name == "patchtst":
        return PatchTSTRegressor(input_dim=input_dim, seq_len=seq_len, output_dim=output_dim, **cfg)
    elif model_name == "grade_tsmixer":
        if num_grades is None:
            with open(PROJECT_ROOT / "data" / "processed" / "grade_mapping.json") as f:
                num_grades = json.load(f)["num_grades"]
        cfg_clean = {k: v for k, v in cfg.items() if k not in ("hidden_dim",)}
        return GradeAwareTSMixer(input_dim=input_dim, num_grades=num_grades,
                                 seq_len=seq_len, hidden_dim=cfg.get("hidden_dim", 128),
                                 output_dim=output_dim, **cfg_clean)
    elif model_name == "grade_patchtst":
        if num_grades is None:
            with open(PROJECT_ROOT / "data" / "processed" / "grade_mapping.json") as f:
                num_grades = json.load(f)["num_grades"]
        return GradeAwarePatchTST(input_dim=input_dim, num_grades=num_grades,
                                  seq_len=seq_len, output_dim=output_dim, **cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")
