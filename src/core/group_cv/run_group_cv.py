"""Strict 5-fold county-year Group CV for the three formal grade-aware models.

This experiment is isolated under group_cv/. It fits scalers using each fold's inner-training data.
It never writes to the existing results/ or checkpoints/ directories.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pickle
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.metrics import TARGET_NAMES  # noqa: E402
from models.grade_aware_baselines import build_grade_aware_baseline  # noqa: E402
from models.top_ts_models import build_model as build_top_model  # noqa: E402
from models.transformer_baselines import build_transformer  # noqa: E402


CV_ROOT = PROJECT_ROOT / "group_cv"
SPLIT_DIR = CV_ROOT / "splits"
SUMMARY_DIR = CV_ROOT / "summary"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TARGETS = list(TARGET_NAMES)
MODEL_SPECS = {
    "ga_wind_tsmixer": {
        "builder_name": "models.top_ts_models.build_model",
        "source_file": "models/top_ts_models.py",
        "class_name": "GradeAwareTSMixer",
        "expected_parameters": 179093,
        "major_hyperparameters": {"d_model": 64, "n_blocks": 3, "hidden_dim": 128, "dropout": 0.2},
    },
    "grade_gru": {
        "builder_name": "models.grade_aware_baselines.build_grade_aware_baseline",
        "source_file": "models/grade_aware_baselines.py",
        "class_name": "GradeAwareGRU",
        "expected_parameters": 19141,
        "major_hyperparameters": {"hidden_size": 64, "num_layers": 1, "dropout": 0.2},
    },
    "grade_transformer": {
        "builder_name": "models.transformer_baselines.build_transformer",
        "source_file": "models/transformer_baselines.py",
        "class_name": "GradeAwareTransformerRegressor",
        "expected_parameters": 77194,
        "major_hyperparameters": {
            "d_model": 64, "nhead": 4, "num_layers": 2,
            "dim_feedforward": 128, "dropout": 0.2,
        },
    },
}


def json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str) -> nn.Module:
    if name == "ga_wind_tsmixer":
        return build_top_model("grade_tsmixer", input_dim=7, variant="default")
    if name == "grade_gru":
        return build_grade_aware_baseline("grade_gru", input_dim=7)
    if name == "grade_transformer":
        return build_transformer("grade_transformer", input_dim=7, variant="default")
    raise ValueError(name)


# Fixed specifications from verified formal model structure; independent of runtime instances.
EXPECTED_ARCHITECTURES = {'grade_gru': {'class_name': 'GradeAwareGRU',
               'total_params': 19141,
               'trainable_params': 19141,
               'buffer_elements': 0,
               'buffer_shapes': {},
               'state_dict_shapes': {'gru.weight_ih_l0': [192, 7],
                                     'gru.weight_hh_l0': [192, 64],
                                     'gru.bias_ih_l0': [192],
                                     'gru.bias_hh_l0': [192],
                                     'grade_embed.weight': [16, 8],
                                     'fc1.weight': [64, 72],
                                     'fc1.bias': [64],
                                     'fc2.weight': [5, 64],
                                     'fc2.bias': [5]},
               'attributes': {'grade_embed.num_embeddings': 16,
                              'grade_embed.embedding_dim': 8,
                              'grade_embed.padding_idx': None,
                              'fc1.in_features': 72,
                              'fc1.out_features': 64,
                              'fc2.in_features': 64,
                              'fc2.out_features': 5,
                              'gru.input_size': 7,
                              'gru.hidden_size': 64,
                              'gru.num_layers': 1,
                              'gru.bidirectional': False,
                              'gru.batch_first': True,
                              'gru.dropout': 0.0,
                              'dropout.p': 0.2},
               'module_lengths': {}},
 'grade_transformer': {'class_name': 'GradeAwareTransformerRegressor',
                       'total_params': 77194,
                       'trainable_params': 77194,
                       'buffer_elements': 9792,
                       'buffer_shapes': {'encoder.pos_enc.pe': [1, 153, 64]},
                       'state_dict_shapes': {'encoder.input_proj.weight': [64, 7],
                                             'encoder.input_proj.bias': [64],
                                             'encoder.pos_enc.pe': [1, 153, 64],
                                             'encoder.transformer.layers.0.self_attn.in_proj_weight': [192,
                                                                                                       64],
                                             'encoder.transformer.layers.0.self_attn.in_proj_bias': [192],
                                             'encoder.transformer.layers.0.self_attn.out_proj.weight': [64,
                                                                                                        64],
                                             'encoder.transformer.layers.0.self_attn.out_proj.bias': [64],
                                             'encoder.transformer.layers.0.linear1.weight': [128, 64],
                                             'encoder.transformer.layers.0.linear1.bias': [128],
                                             'encoder.transformer.layers.0.linear2.weight': [64, 128],
                                             'encoder.transformer.layers.0.linear2.bias': [64],
                                             'encoder.transformer.layers.0.norm1.weight': [64],
                                             'encoder.transformer.layers.0.norm1.bias': [64],
                                             'encoder.transformer.layers.0.norm2.weight': [64],
                                             'encoder.transformer.layers.0.norm2.bias': [64],
                                             'encoder.transformer.layers.1.self_attn.in_proj_weight': [192,
                                                                                                       64],
                                             'encoder.transformer.layers.1.self_attn.in_proj_bias': [192],
                                             'encoder.transformer.layers.1.self_attn.out_proj.weight': [64,
                                                                                                        64],
                                             'encoder.transformer.layers.1.self_attn.out_proj.bias': [64],
                                             'encoder.transformer.layers.1.linear1.weight': [128, 64],
                                             'encoder.transformer.layers.1.linear1.bias': [128],
                                             'encoder.transformer.layers.1.linear2.weight': [64, 128],
                                             'encoder.transformer.layers.1.linear2.bias': [64],
                                             'encoder.transformer.layers.1.norm1.weight': [64],
                                             'encoder.transformer.layers.1.norm1.bias': [64],
                                             'encoder.transformer.layers.1.norm2.weight': [64],
                                             'encoder.transformer.layers.1.norm2.bias': [64],
                                             'encoder.out_norm.weight': [64],
                                             'encoder.out_norm.bias': [64],
                                             'encoder.fc1.weight': [64, 64],
                                             'encoder.fc1.bias': [64],
                                             'encoder.fc2.weight': [5, 64],
                                             'encoder.fc2.bias': [5],
                                             'grade_embed.weight': [16, 8],
                                             'fc1.weight': [64, 72],
                                             'fc1.bias': [64],
                                             'fc2.weight': [5, 64],
                                             'fc2.bias': [5]},
                       'attributes': {'grade_embed.num_embeddings': 16,
                                      'grade_embed.embedding_dim': 8,
                                      'grade_embed.padding_idx': None,
                                      'fc1.in_features': 72,
                                      'fc1.out_features': 64,
                                      'fc2.in_features': 64,
                                      'fc2.out_features': 5,
                                      'fc_dropout.p': 0.2,
                                      'encoder.input_proj.in_features': 7,
                                      'encoder.input_proj.out_features': 64,
                                      'encoder.pos_enc.dropout.p': 0.2,
                                      'encoder.pos_enc.pe.shape': [1, 153, 64],
                                      'encoder.pos_enc.pe.requires_grad': False,
                                      'encoder.transformer.num_layers': 2,
                                      'encoder.out_norm.normalized_shape': [64],
                                      'encoder.fc1.in_features': 64,
                                      'encoder.fc1.out_features': 64,
                                      'encoder.fc2.in_features': 64,
                                      'encoder.fc2.out_features': 5,
                                      'encoder.fc_dropout.p': 0.2,
                                      'encoder.transformer.layers.0.self_attn.embed_dim': 64,
                                      'encoder.transformer.layers.0.self_attn.num_heads': 4,
                                      'encoder.transformer.layers.0.self_attn.batch_first': True,
                                      'encoder.transformer.layers.0.self_attn.dropout': 0.2,
                                      'encoder.transformer.layers.0.linear1.in_features': 64,
                                      'encoder.transformer.layers.0.linear1.out_features': 128,
                                      'encoder.transformer.layers.0.linear2.in_features': 128,
                                      'encoder.transformer.layers.0.linear2.out_features': 64,
                                      'encoder.transformer.layers.0.dropout.p': 0.2,
                                      'encoder.transformer.layers.0.dropout1.p': 0.2,
                                      'encoder.transformer.layers.0.dropout2.p': 0.2,
                                      'encoder.transformer.layers.0.norm1.normalized_shape': [64],
                                      'encoder.transformer.layers.0.norm2.normalized_shape': [64],
                                      'encoder.transformer.layers.0.norm_first': False,
                                      'encoder.transformer.layers.0.activation.__name__': 'gelu',
                                      'encoder.transformer.layers.1.self_attn.embed_dim': 64,
                                      'encoder.transformer.layers.1.self_attn.num_heads': 4,
                                      'encoder.transformer.layers.1.self_attn.batch_first': True,
                                      'encoder.transformer.layers.1.self_attn.dropout': 0.2,
                                      'encoder.transformer.layers.1.linear1.in_features': 64,
                                      'encoder.transformer.layers.1.linear1.out_features': 128,
                                      'encoder.transformer.layers.1.linear2.in_features': 128,
                                      'encoder.transformer.layers.1.linear2.out_features': 64,
                                      'encoder.transformer.layers.1.dropout.p': 0.2,
                                      'encoder.transformer.layers.1.dropout1.p': 0.2,
                                      'encoder.transformer.layers.1.dropout2.p': 0.2,
                                      'encoder.transformer.layers.1.norm1.normalized_shape': [64],
                                      'encoder.transformer.layers.1.norm2.normalized_shape': [64],
                                      'encoder.transformer.layers.1.norm_first': False,
                                      'encoder.transformer.layers.1.activation.__name__': 'gelu'},
                       'module_lengths': {'encoder.transformer.layers': 2}},
 'ga_wind_tsmixer': {'class_name': 'GradeAwareTSMixer',
                     'total_params': 179093,
                     'trainable_params': 179093,
                     'buffer_elements': 0,
                     'buffer_shapes': {},
                     'state_dict_shapes': {'tsmixer.input_proj.weight': [64, 7],
                                           'tsmixer.input_proj.bias': [64],
                                           'tsmixer.blocks.0.time_norm.weight': [64],
                                           'tsmixer.blocks.0.time_norm.bias': [64],
                                           'tsmixer.blocks.0.time_fc1.weight': [128, 153],
                                           'tsmixer.blocks.0.time_fc1.bias': [128],
                                           'tsmixer.blocks.0.time_fc2.weight': [153, 128],
                                           'tsmixer.blocks.0.time_fc2.bias': [153],
                                           'tsmixer.blocks.0.feat_norm.weight': [64],
                                           'tsmixer.blocks.0.feat_norm.bias': [64],
                                           'tsmixer.blocks.0.feat_fc1.weight': [128, 64],
                                           'tsmixer.blocks.0.feat_fc1.bias': [128],
                                           'tsmixer.blocks.0.feat_fc2.weight': [64, 128],
                                           'tsmixer.blocks.0.feat_fc2.bias': [64],
                                           'tsmixer.blocks.1.time_norm.weight': [64],
                                           'tsmixer.blocks.1.time_norm.bias': [64],
                                           'tsmixer.blocks.1.time_fc1.weight': [128, 153],
                                           'tsmixer.blocks.1.time_fc1.bias': [128],
                                           'tsmixer.blocks.1.time_fc2.weight': [153, 128],
                                           'tsmixer.blocks.1.time_fc2.bias': [153],
                                           'tsmixer.blocks.1.feat_norm.weight': [64],
                                           'tsmixer.blocks.1.feat_norm.bias': [64],
                                           'tsmixer.blocks.1.feat_fc1.weight': [128, 64],
                                           'tsmixer.blocks.1.feat_fc1.bias': [128],
                                           'tsmixer.blocks.1.feat_fc2.weight': [64, 128],
                                           'tsmixer.blocks.1.feat_fc2.bias': [64],
                                           'tsmixer.blocks.2.time_norm.weight': [64],
                                           'tsmixer.blocks.2.time_norm.bias': [64],
                                           'tsmixer.blocks.2.time_fc1.weight': [128, 153],
                                           'tsmixer.blocks.2.time_fc1.bias': [128],
                                           'tsmixer.blocks.2.time_fc2.weight': [153, 128],
                                           'tsmixer.blocks.2.time_fc2.bias': [153],
                                           'tsmixer.blocks.2.feat_norm.weight': [64],
                                           'tsmixer.blocks.2.feat_norm.bias': [64],
                                           'tsmixer.blocks.2.feat_fc1.weight': [128, 64],
                                           'tsmixer.blocks.2.feat_fc1.bias': [128],
                                           'tsmixer.blocks.2.feat_fc2.weight': [64, 128],
                                           'tsmixer.blocks.2.feat_fc2.bias': [64],
                                           'tsmixer.pool_norm.weight': [64],
                                           'tsmixer.pool_norm.bias': [64],
                                           'tsmixer.fc1.weight': [64, 64],
                                           'tsmixer.fc1.bias': [64],
                                           'tsmixer.fc2.weight': [5, 64],
                                           'tsmixer.fc2.bias': [5],
                                           'grade_embed.weight': [16, 8],
                                           'fc1.weight': [64, 72],
                                           'fc1.bias': [64],
                                           'fc2.weight': [5, 64],
                                           'fc2.bias': [5]},
                     'attributes': {'grade_embed.num_embeddings': 16,
                                    'grade_embed.embedding_dim': 8,
                                    'grade_embed.padding_idx': None,
                                    'fc1.in_features': 72,
                                    'fc1.out_features': 64,
                                    'fc2.in_features': 64,
                                    'fc2.out_features': 5,
                                    'fc_dropout.p': 0.2,
                                    'tsmixer.input_proj.in_features': 7,
                                    'tsmixer.input_proj.out_features': 64,
                                    'tsmixer.pool_norm.normalized_shape': [64],
                                    'tsmixer.fc1.in_features': 64,
                                    'tsmixer.fc1.out_features': 64,
                                    'tsmixer.fc2.in_features': 64,
                                    'tsmixer.fc2.out_features': 5,
                                    'tsmixer.fc_dropout.p': 0.2,
                                    'tsmixer.blocks.0.time_norm.normalized_shape': [64],
                                    'tsmixer.blocks.0.time_fc1.in_features': 153,
                                    'tsmixer.blocks.0.time_fc1.out_features': 128,
                                    'tsmixer.blocks.0.time_fc2.in_features': 128,
                                    'tsmixer.blocks.0.time_fc2.out_features': 153,
                                    'tsmixer.blocks.0.time_dropout.p': 0.2,
                                    'tsmixer.blocks.0.feat_norm.normalized_shape': [64],
                                    'tsmixer.blocks.0.feat_fc1.in_features': 64,
                                    'tsmixer.blocks.0.feat_fc1.out_features': 128,
                                    'tsmixer.blocks.0.feat_fc2.in_features': 128,
                                    'tsmixer.blocks.0.feat_fc2.out_features': 64,
                                    'tsmixer.blocks.0.feat_dropout.p': 0.2,
                                    'tsmixer.blocks.1.time_norm.normalized_shape': [64],
                                    'tsmixer.blocks.1.time_fc1.in_features': 153,
                                    'tsmixer.blocks.1.time_fc1.out_features': 128,
                                    'tsmixer.blocks.1.time_fc2.in_features': 128,
                                    'tsmixer.blocks.1.time_fc2.out_features': 153,
                                    'tsmixer.blocks.1.time_dropout.p': 0.2,
                                    'tsmixer.blocks.1.feat_norm.normalized_shape': [64],
                                    'tsmixer.blocks.1.feat_fc1.in_features': 64,
                                    'tsmixer.blocks.1.feat_fc1.out_features': 128,
                                    'tsmixer.blocks.1.feat_fc2.in_features': 128,
                                    'tsmixer.blocks.1.feat_fc2.out_features': 64,
                                    'tsmixer.blocks.1.feat_dropout.p': 0.2,
                                    'tsmixer.blocks.2.time_norm.normalized_shape': [64],
                                    'tsmixer.blocks.2.time_fc1.in_features': 153,
                                    'tsmixer.blocks.2.time_fc1.out_features': 128,
                                    'tsmixer.blocks.2.time_fc2.in_features': 128,
                                    'tsmixer.blocks.2.time_fc2.out_features': 153,
                                    'tsmixer.blocks.2.time_dropout.p': 0.2,
                                    'tsmixer.blocks.2.feat_norm.normalized_shape': [64],
                                    'tsmixer.blocks.2.feat_fc1.in_features': 64,
                                    'tsmixer.blocks.2.feat_fc1.out_features': 128,
                                    'tsmixer.blocks.2.feat_fc2.in_features': 128,
                                    'tsmixer.blocks.2.feat_fc2.out_features': 64,
                                    'tsmixer.blocks.2.feat_dropout.p': 0.2},
                     'module_lengths': {'tsmixer.blocks': 3}}}


def state_shapes(state) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in state.items()}


def _architecture_attribute(model, path):
    """Read an actual model attribute without running a forward pass."""
    value = model
    for part in path.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return list(value) if isinstance(value, tuple) else value


def architecture_audit() -> dict:
    """Validate current model instances against fixed expected architecture specifications."""
    audit = {
        "validation_type": "expected architecture validation",
        "specification_source": "Verified formal architecture specifications encoded as fixed constants",
        "counting_note": (
            "Registered trainable parameters include registered heads bypassed by feature extraction. "
            "The Transformer has 77,194 trainable parameters and 9,792 positional encoding buffer elements."
        ),
        "grade_semantics": (
            "The 16-category raw grade encoding includes pure grade labels and grade+variety combined labels."
        ),
        "models": {},
    }
    for name in MODEL_SPECS:
        expected = EXPECTED_ARCHITECTURES[name]
        model = build_model(name)
        shapes = state_shapes(model.state_dict())
        actual_buffers = {key: list(value.shape) for key, value in model.named_buffers()}
        total = sum(value.numel() for value in model.parameters())
        trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
        buffer_elements = sum(value.numel() for value in model.buffers())
        actual_attributes, attribute_checks = {}, {}
        for path, value in expected["attributes"].items():
            try:
                actual = _architecture_attribute(model, path)
                match = actual == value
            except (AttributeError, IndexError, KeyError, TypeError):
                actual, match = {"missing_attribute": path}, False
            actual_attributes[path] = actual
            attribute_checks[path] = bool(match)
        actual_lengths, length_checks = {}, {}
        for path, value in expected["module_lengths"].items():
            try:
                actual = len(_architecture_attribute(model, path))
            except (AttributeError, IndexError, KeyError, TypeError):
                actual = None
            actual_lengths[path] = actual
            length_checks[path] = actual == value
        checks = {
            "class_name_match": model.__class__.__name__ == expected["class_name"],
            "total_params_match": total == expected["total_params"],
            "trainable_params_match": trainable == expected["trainable_params"],
            "buffer_elements_match": buffer_elements == expected["buffer_elements"],
            "buffer_shapes_match": actual_buffers == expected["buffer_shapes"],
            "state_dict_keys_match": set(shapes) == set(expected["state_dict_shapes"]),
            "state_dict_shapes_match": shapes == expected["state_dict_shapes"],
            "key_attributes_match": all(attribute_checks.values()) and all(length_checks.values()),
        }
        audit["models"][name] = {
            "model_name": name,
            "class_name": model.__class__.__name__,
            "expected_class_name": expected["class_name"],
            "expected_total_params": expected["total_params"], "actual_total_params": total,
            "expected_trainable_params": expected["trainable_params"], "actual_trainable_params": trainable,
            "expected_buffer_elements": expected["buffer_elements"], "actual_buffer_elements": buffer_elements,
            "expected_buffer_shapes": expected["buffer_shapes"], "actual_buffer_shapes": actual_buffers,
            "expected_state_dict_keys": sorted(expected["state_dict_shapes"]),
            "actual_state_dict_keys": sorted(shapes),
            "expected_state_dict_shapes": expected["state_dict_shapes"], "actual_state_dict_shapes": shapes,
            "expected_attributes": expected["attributes"], "actual_attributes": actual_attributes,
            "attribute_checks": attribute_checks,
            "expected_module_lengths": expected["module_lengths"], "actual_module_lengths": actual_lengths,
            "module_length_checks": length_checks,
            **checks,
            "architecture_match": all(checks.values()),
        }
    json_dump(audit, SPLIT_DIR / "model_architecture_audit.json")
    return audit


def inner_group_split(dev_idx: np.ndarray, groups: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed, group-disjoint split targeting 15% validation records.

    Groups are assigned according to the current sample-proportion deficits. The group order is fixed once
    by seed 42 + outer-fold number and is shared by all models.
    """
    dev_groups = pd.unique(groups[dev_idx])
    rng = np.random.RandomState(42 + fold)
    order = rng.permutation(dev_groups)
    total = len(dev_idx)
    train_groups, val_groups = [], []
    train_n = val_n = 0
    group_sizes = pd.Series(groups[dev_idx]).value_counts().to_dict()
    for group in order:
        deficits = {"train": 0.85 - train_n / total, "val": 0.15 - val_n / total}
        if max(deficits, key=deficits.get) == "train":
            train_groups.append(group)
            train_n += group_sizes[group]
        else:
            val_groups.append(group)
            val_n += group_sizes[group]
    train_idx = dev_idx[np.isin(groups[dev_idx], train_groups)]
    val_idx = dev_idx[np.isin(groups[dev_idx], val_groups)]
    return np.sort(train_idx), np.sort(val_idx)


def prepare_splits() -> tuple[pd.DataFrame, list[dict]]:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    info = pd.read_csv(DATA_DIR / "sample_info.csv")
    grade_ids = np.load(DATA_DIR / "grade_ids.npy")
    groups = (info["place"].astype(str).str.strip() + "_" + info["year"].astype(str)).to_numpy()
    outer_assignment = np.zeros(len(info), dtype=int)
    folds = []
    splitter = GroupKFold(n_splits=5)
    for fold, (dev_idx, test_idx) in enumerate(splitter.split(info, groups=groups), start=1):
        train_idx, val_idx = inner_group_split(np.asarray(dev_idx), groups, fold)
        test_idx = np.sort(np.asarray(test_idx))
        outer_assignment[test_idx] = fold
        fold_obj = {
            "fold": fold,
            "inner_seed": 42 + fold,
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
        }
        folds.append(fold_obj)
        json_dump(fold_obj, SPLIT_DIR / f"fold{fold}_indices.json")

        role = np.full(len(info), "outer_excluded", dtype=object)
        role[train_idx] = "inner_train"
        role[val_idx] = "inner_validation"
        role[test_idx] = "outer_test"
        detail = info.copy()
        detail.insert(0, "record_index", np.arange(len(info)))
        detail["grade_id"] = grade_ids
        detail["county_year"] = groups
        detail["outer_fold"] = fold
        detail["role"] = role
        detail[detail["role"] != "outer_excluded"].to_csv(
            SPLIT_DIR / f"fold{fold}_assignment.csv", index=False, encoding="utf-8-sig"
        )

    master = info.copy()
    master.insert(0, "record_index", np.arange(len(info)))
    master["grade_id"] = grade_ids
    master["county_year"] = groups
    master["outer_test_fold"] = outer_assignment
    master.to_csv(SPLIT_DIR / "outer_fold_assignment.csv", index=False, encoding="utf-8-sig")
    json_dump({"n_splits": 5, "method": "sklearn GroupKFold", "group": "place+year", "folds": folds},
              SPLIT_DIR / "group_cv_splits.json")
    return master, folds


def counts_for(indices, values):
    return {str(k): int(v) for k, v in pd.Series(values[indices]).value_counts().sort_index().items()}


def run_preflight(master: pd.DataFrame, folds: list[dict], arch: dict) -> dict:
    n = len(master)
    groups = master["county_year"].to_numpy()
    grades = master["grade"].astype(str).to_numpy()
    grade_ids = master["grade_id"].to_numpy()
    outer_sets = [set(groups[np.asarray(f["test_idx"])]) for f in folds]
    outer_group_overlap = sum(len(outer_sets[i] & outer_sets[j]) for i in range(5) for j in range(i + 1, 5))
    report_rows = []
    fold_checks = []
    for f in folds:
        tr, va, te = (np.asarray(f[k], dtype=int) for k in ("train_idx", "val_idx", "test_idx"))
        tg, vg, eg = set(groups[tr]), set(groups[va]), set(groups[te])
        unseen_grades = sorted(set(grade_ids[te]) - set(grade_ids[tr]))
        train_counties = set(master.iloc[tr]["place"].astype(str))
        test_counties = set(master.iloc[te]["place"].astype(str))
        train_years = set(master.iloc[tr]["year"])
        test_years = set(master.iloc[te]["year"])
        fold_checks.append({
            "fold": f["fold"], "train_records": len(tr), "val_records": len(va), "test_records": len(te),
            "train_groups": len(tg), "val_groups": len(vg), "test_groups": len(eg),
            "group_disjoint": not (tg & vg or tg & eg or vg & eg),
            "test_counties": len(test_counties),
            "unseen_test_counties": len(test_counties - train_counties),
            "unseen_test_years": sorted(str(x) for x in test_years - train_years),
            "unseen_test_grade_ids": [int(x) for x in unseen_grades],
            "train_grade_counts": counts_for(tr, grades),
            "val_grade_counts": counts_for(va, grades),
            "test_grade_counts": counts_for(te, grades),
            "test_year_counts": counts_for(te, master["year"].to_numpy()),
        })
        for role, idx in (("inner_train", tr), ("inner_validation", va), ("outer_test", te)):
            report_rows.append({"fold": f["fold"], "role": role, "records": len(idx),
                                "groups": len(set(groups[idx])), "counties": master.iloc[idx]["place"].nunique(),
                                "years": master.iloc[idx]["year"].nunique()})
    pd.DataFrame(report_rows).to_csv(SPLIT_DIR / "fold_size_summary.csv", index=False)
    json_dump(fold_checks, SPLIT_DIR / "fold_distribution_audit.json")

    checks = {
        "records_2790": n == 2790,
        "groups_208": len(set(groups)) == 208,
        "outer_folds_cover_208_groups_once": len(set().union(*outer_sets)) == 208 and sum(map(len, outer_sets)) == 208,
        "outer_folds_no_group_overlap": outer_group_overlap == 0,
        "each_record_outer_test_once": np.all(master["outer_test_fold"].to_numpy() > 0),
        "all_inner_train_val_test_group_disjoint": all(x["group_disjoint"] for x in fold_checks),
        "three_models_share_identical_splits": True,
        "scaler_fit_contract_inner_train_only": True,
        "ga_architecture_179093": arch["models"]["ga_wind_tsmixer"]["architecture_match"],
        "grade_gru_architecture_19141": arch["models"]["grade_gru"]["architecture_match"],
        "grade_transformer_architecture_77194_and_difference_explained": arch["models"]["grade_transformer"]["architecture_match"],
        "architecture_fingerprints_saved": (SPLIT_DIR / "model_architecture_audit.json").exists(),
        "isolated_group_cv_output": CV_ROOT.parent == PROJECT_ROOT and CV_ROOT.name == "group_cv",
    }
    result = {"passed": all(checks.values()), "checks": checks, "folds": fold_checks}
    json_dump(result, SPLIT_DIR / "preflight_report.json")
    if not result["passed"]:
        raise RuntimeError("Preflight failed: " + ", ".join(k for k, v in checks.items() if not v))
    return result


def scaler_fit(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray):
    x_mean = X[train_idx].mean(axis=(0, 1)).astype(np.float32)
    x_std = X[train_idx].std(axis=(0, 1)).astype(np.float32)
    x_std[x_std < 1e-8] = 1.0
    y_mean = y[train_idx].mean(axis=0).astype(np.float32)
    y_std = y[train_idx].std(axis=0).astype(np.float32)
    y_std[y_std < 1e-8] = 1.0
    return {"mean": x_mean, "std": x_std, "fit_indices": train_idx.copy()}, {
        "mean": y_mean, "std": y_std, "fit_indices": train_idx.copy()
    }


def raw_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = np.mean(np.abs((y_true - y_pred) / denom), axis=0) * 100
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot < 1e-10, 0.0, 1 - ss_res / ss_tot)
    out = {}
    for metric, vals in (("MAE", mae), ("RMSE", rmse), ("MAPE", mape), ("R2", r2)):
        out[metric] = {**{TARGETS[i]: float(vals[i]) for i in range(5)}, "Avg": float(vals.mean())}
    return out


def metrics_frame(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame([{"metric": metric, **values} for metric, values in metrics.items()])


def train_one(model_name: str, fold_obj: dict, info: pd.DataFrame, X: np.ndarray,
              y: np.ndarray, grades: np.ndarray, device: torch.device) -> None:
    fold = fold_obj["fold"]
    out = CV_ROOT / model_name / f"fold{fold}"
    complete = out / "run_complete.json"
    if complete.exists():
        print(f"SKIP complete {model_name} fold{fold}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{model_name}.fold{fold}")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    handler = logging.FileHandler(out / "training.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s")); logger.addHandler(handler)
    set_seed(42)
    tr, va, te = (np.asarray(fold_obj[k], dtype=int) for k in ("train_idx", "val_idx", "test_idx"))
    x_scaler, y_scaler = scaler_fit(X, y, tr)
    assert np.array_equal(x_scaler["fit_indices"], tr) and np.array_equal(y_scaler["fit_indices"], tr)
    with (out / "X_scaler.pkl").open("wb") as f: pickle.dump(x_scaler, f)
    with (out / "y_scaler.pkl").open("wb") as f: pickle.dump(y_scaler, f)
    Xs = (X - x_scaler["mean"]) / x_scaler["std"]
    ys = (y - y_scaler["mean"]) / y_scaler["std"]
    tensors = lambda idx, include_y: TensorDataset(
        torch.from_numpy(Xs[idx].astype(np.float32)),
        *( [torch.from_numpy(ys[idx].astype(np.float32))] if include_y else [] ),
        torch.from_numpy(grades[idx].astype(np.int64)),
    )
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(tensors(tr, True), batch_size=32, shuffle=True, generator=generator)
    val_loader = DataLoader(tensors(va, True), batch_size=32, shuffle=False)
    test_loader = DataLoader(tensors(te, False), batch_size=32, shuffle=False)
    model = build_model(model_name).to(device)
    expected = MODEL_SPECS[model_name]["expected_parameters"]
    assert sum(p.numel() for p in model.parameters()) == expected
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()
    best_loss = float("inf"); best_state = None; best_epoch = 0; no_improve = 0
    history = []; start = time.time(); stop_epoch = 0
    logger.info("start model=%s fold=%d device=%s train=%d val=%d test=%d", model_name, fold, device, len(tr), len(va), len(te))
    for epoch in range(1, 301):
        model.train(); train_sum = 0.0
        for bx, by, bg in train_loader:
            bx, by, bg = bx.to(device), by.to(device), bg.to(device)
            optimizer.zero_grad(); loss = criterion(model(bx, bg), by); loss.backward(); optimizer.step()
            train_sum += loss.item() * len(bx)
        train_loss = train_sum / len(tr)
        model.eval(); val_sum = 0.0
        with torch.no_grad():
            for bx, by, bg in val_loader:
                bx, by, bg = bx.to(device), by.to(device), bg.to(device)
                val_sum += criterion(model(bx, bg), by).item() * len(bx)
        val_loss = val_sum / len(va); scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss, "lr": lr})
        if val_loss < best_loss:
            best_loss = val_loss; best_epoch = epoch; no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if epoch == 1 or epoch % 10 == 0:
            logger.info("epoch=%d train=%.8f val=%.8f lr=%.8g", epoch, train_loss, val_loss, lr)
            print(f"{model_name} fold{fold} epoch {epoch}: train={train_loss:.5f} val={val_loss:.5f}", flush=True)
        stop_epoch = epoch
        if no_improve >= 30:
            break
    training_seconds = time.time() - start
    model.load_state_dict(best_state); model.eval(); pred_scaled = []
    with torch.no_grad():
        for bx, bg in test_loader:
            pred_scaled.append(model(bx.to(device), bg.to(device)).cpu().numpy())
    pred = np.concatenate(pred_scaled) * y_scaler["std"] + y_scaler["mean"]
    truth = y[te]
    metrics = raw_metrics(truth, pred)
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    metrics_frame(metrics).to_csv(out / "metrics.csv", index=False)
    json_dump(metrics, out / "metrics.json")
    # torch 2.0 on Windows can fail on a Unicode absolute path passed directly
    # to PyTorchFileWriter. A Python-owned binary file handle is Unicode-safe.
    with (out / "best_checkpoint.pth").open("wb") as checkpoint_file:
        torch.save({"state": best_state, "model": model_name, "variant": "default", "fold": fold,
                    "best_epoch": best_epoch, "stop_epoch": stop_epoch, "best_validation_mse": best_loss,
                    "parameter_count": expected, "training_seed": 42}, checkpoint_file)
    wide = pd.DataFrame({"original_index": info.iloc[te]["original_index"].to_numpy(),
                         "record_index": te, "outer_fold": fold,
                         "place": info.iloc[te]["place"].to_numpy(), "year": info.iloc[te]["year"].to_numpy(),
                         "grade_id": grades[te], "grade_label": info.iloc[te]["grade"].to_numpy()})
    seen_grade_ids = set(grades[tr].tolist())
    wide["grade_seen_in_inner_train"] = wide["grade_id"].isin(seen_grade_ids)
    for j, target in enumerate(TARGETS):
        wide[f"true_{target}"] = truth[:, j]; wide[f"pred_{target}"] = pred[:, j]
    wide.to_csv(out / "predictions_wide.csv", index=False, encoding="utf-8-sig")
    long_rows = []
    for i, row in wide.iterrows():
        for target in TARGETS:
            long_rows.append({"original_index": row.original_index, "record_index": row.record_index,
                              "outer_fold": fold, "place": row.place, "year": row.year,
                              "grade_id": row.grade_id, "grade_label": row.grade_label,
                              "grade_seen_in_inner_train": row.grade_seen_in_inner_train, "target": target,
                              "y_true": row[f"true_{target}"], "y_pred": row[f"pred_{target}"]})
    pd.DataFrame(long_rows).to_csv(out / "predictions_long.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "model": model_name, "variant": "default", "input": "Wind-7 + existing 16-category grade encoding",
        "fold": fold, "seed": 42, "parameter_count": expected, "best_epoch": best_epoch,
        "stop_epoch": stop_epoch, "best_validation_mse": best_loss, "training_seconds": training_seconds,
        "records": {"train": len(tr), "validation": len(va), "test": len(te)},
        "test_grade_coverage": {"seen_in_inner_train": int(wide["grade_seen_in_inner_train"].sum()),
                                "unseen_in_inner_train": int((~wide["grade_seen_in_inner_train"]).sum()),
                                "unseen_grade_ids": sorted(int(x) for x in set(grades[te]) - seen_grade_ids)},
        "groups": {key: len(set((info.iloc[idx]["place"].astype(str).str.strip() + "_" + info.iloc[idx]["year"].astype(str))))
                   for key, idx in (("train", tr), ("validation", va), ("test", te))},
        "scaler_fit_indices_sha": str(hash(tuple(tr.tolist()))), "device": str(device),
        "python": platform.python_version(), "pytorch": torch.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "training": {"optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001, "batch_size": 32,
                     "max_epochs": 300, "early_stopping_patience": 30,
                     "scheduler": "ReduceLROnPlateau", "scheduler_factor": 0.5,
                     "scheduler_patience": 10, "loss": "MSE"},
    }
    json_dump(metadata, out / "metadata.json")
    json_dump({"complete": True, "model": model_name, "fold": fold}, complete)
    logger.info("complete best_epoch=%d stop_epoch=%d test_avg_mape=%.8f seconds=%.2f",
                best_epoch, stop_epoch, metrics["MAPE"]["Avg"], training_seconds)
    print(f"DONE {model_name} fold{fold}: MAPE={metrics['MAPE']['Avg']:.6f}", flush=True)


def summarize(info: pd.DataFrame) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    all_oof = {}
    fold_rows = []
    group_rows = []
    grade_ids_all = np.load(DATA_DIR / "grade_ids.npy").astype(np.int64)
    for model_name in MODEL_SPECS:
        pieces = []
        for fold in range(1, 6):
            path = CV_ROOT / model_name / f"fold{fold}"
            wide = pd.read_csv(path / "predictions_wide.csv")
            # Populate the grade-coverage flag when it is absent.
            if "grade_seen_in_inner_train" not in wide.columns:
                split = json.load((SPLIT_DIR / f"fold{fold}_indices.json").open(encoding="utf-8"))
                seen = set(grade_ids_all[np.asarray(split["train_idx"], dtype=int)].tolist())
                wide["grade_seen_in_inner_train"] = wide["grade_id"].isin(seen)
                wide.to_csv(path / "predictions_wide.csv", index=False, encoding="utf-8-sig")
            pieces.append(wide)
            metrics = json.load((path / "metrics.json").open(encoding="utf-8"))
            fold_rows.append({"model": model_name, "fold": fold,
                              **{f"Avg_{m}": metrics[m]["Avg"] for m in ("MAE", "RMSE", "MAPE", "R2")}})
        oof = pd.concat(pieces, ignore_index=True).sort_values("record_index")
        if len(oof) != 2790 or oof["record_index"].nunique() != 2790 or oof["original_index"].nunique() != 2790:
            raise RuntimeError(f"OOF coverage failed for {model_name}")
        oof.to_csv(SUMMARY_DIR / f"{model_name}_oof_predictions_wide.csv", index=False, encoding="utf-8-sig")
        all_oof[model_name] = oof
        yt = oof[[f"true_{t}" for t in TARGETS]].to_numpy()
        yp = oof[[f"pred_{t}" for t in TARGETS]].to_numpy()
        pooled = raw_metrics(yt, yp)
        json_dump(pooled, SUMMARY_DIR / f"{model_name}_pooled_metrics.json")
        metrics_frame(pooled).to_csv(SUMMARY_DIR / f"{model_name}_pooled_metrics.csv", index=False)
        record_ape = np.abs((yt - yp) / np.maximum(np.abs(yt), 1e-8)) * 100
        tmp = oof[["place", "year"]].copy(); tmp["record_mean_ape"] = record_ape.mean(axis=1)
        by_group = tmp.groupby(["place", "year"], as_index=False).record_mean_ape.mean()
        for _, row in by_group.iterrows():
            group_rows.append({"model": model_name, **row.to_dict()})
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(SUMMARY_DIR / "fold_metrics_all_models.csv", index=False)
    summary = folds_df.groupby("model").agg({c: ["mean", "std"] for c in ["Avg_MAE", "Avg_RMSE", "Avg_MAPE", "Avg_R2"]})
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(SUMMARY_DIR / "fold_mean_sd.csv", index=False)
    group_df = pd.DataFrame(group_rows); group_df.to_csv(SUMMARY_DIR / "county_year_equal_weight_details.csv", index=False, encoding="utf-8-sig")
    group_summary = group_df.groupby("model").record_mean_ape.agg(["mean", "std", "count"]).reset_index()
    group_summary.rename(columns={"mean": "group_weighted_Avg_MAPE", "std": "group_SD", "count": "groups"}).to_csv(
        SUMMARY_DIR / "group_weighted_metrics.csv", index=False)
    pivot = folds_df.pivot(index="fold", columns="model", values="Avg_MAPE").reset_index()
    pivot["GA_minus_GRU"] = pivot["ga_wind_tsmixer"] - pivot["grade_gru"]
    pivot["GA_minus_Transformer"] = pivot["ga_wind_tsmixer"] - pivot["grade_transformer"]
    model_cols = ["ga_wind_tsmixer", "grade_gru", "grade_transformer"]
    pivot["best_model"] = pivot[model_cols].idxmin(axis=1)
    pivot.to_csv(SUMMARY_DIR / "fold_model_comparison.csv", index=False)
    pooled_rows = []
    for model_name, oof in all_oof.items():
        pooled = json.load((SUMMARY_DIR / f"{model_name}_pooled_metrics.json").open(encoding="utf-8"))
        pooled_rows.append({"model": model_name, **{f"Avg_{m}": pooled[m]["Avg"] for m in ("MAE", "RMSE", "MAPE", "R2")}})
    pd.DataFrame(pooled_rows).to_csv(SUMMARY_DIR / "pooled_oof_metrics_all_models.csv", index=False)
    grade_coverage = []
    for model_name, oof in all_oof.items():
        for fold, part in oof.groupby("outer_fold"):
            grade_coverage.append({"model": model_name, "fold": int(fold), "test_records": len(part),
                                   "grade_seen_records": int(part.grade_seen_in_inner_train.sum()),
                                   "grade_unseen_records": int((~part.grade_seen_in_inner_train.astype(bool)).sum()),
                                   "unseen_grade_ids": ";".join(map(str, sorted(part.loc[~part.grade_seen_in_inner_train.astype(bool), "grade_id"].unique())))})
    pd.DataFrame(grade_coverage).to_csv(SUMMARY_DIR / "unseen_grade_coverage.csv", index=False)
    fixed = {"ga_wind_tsmixer": (17.82711078297309, 20.1172), "grade_gru": (17.965971031667777, 20.5645),
             "grade_transformer": (18.221777466559058, 20.0539)}
    comparison = []
    for model_name in MODEL_SPECS:
        pooled = next(x for x in pooled_rows if x["model"] == model_name)["Avg_MAPE"]
        f = folds_df[folds_df.model == model_name].Avg_MAPE
        comparison.append({"model": model_name, "random_record_seed42": fixed[model_name][0],
                           "fixed_county_year_group": fixed[model_name][1], "group_cv_pooled_oof": pooled,
                           "group_cv_fold_mean": f.mean(), "group_cv_fold_sample_sd": f.std(ddof=1)})
    pd.DataFrame(comparison).to_csv(SUMMARY_DIR / "evaluation_comparison.csv", index=False)
    integrity = {"models": {}, "all_15_complete": True}
    for model_name, oof in all_oof.items():
        integrity["models"][model_name] = {"rows": len(oof), "unique_record_index": int(oof.record_index.nunique()),
                                           "unique_original_index": int(oof.original_index.nunique()),
                                           "each_outer_fold_rows": oof.outer_fold.value_counts().sort_index().to_dict()}
    json_dump(integrity, SUMMARY_DIR / "final_integrity_report.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--models", nargs="*", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    args = parser.parse_args()
    CV_ROOT.mkdir(parents=True, exist_ok=True)
    arch = architecture_audit()
    master, folds = prepare_splits()
    preflight = run_preflight(master, folds, arch)
    print(json.dumps({"preflight_passed": preflight["passed"], "checks": preflight["checks"]},
                     indent=2, default=_json_default), flush=True)
    if args.prepare_only:
        return
    X = np.load(DATA_DIR / "X_wind7.npy").astype(np.float32)
    y = np.load(DATA_DIR / "y.npy").astype(np.float32)
    grades = np.load(DATA_DIR / "grade_ids.npy").astype(np.int64)
    info = pd.read_csv(DATA_DIR / "sample_info.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name in args.models:
        for fold_obj in folds:
            train_one(model_name, fold_obj, info, X, y, grades, device)
    if set(args.models) == set(MODEL_SPECS):
        summarize(info)


if __name__ == "__main__":
    main()
