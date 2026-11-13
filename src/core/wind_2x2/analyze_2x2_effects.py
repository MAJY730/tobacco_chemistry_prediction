"""Generate descriptive four-configuration tables and 2x2 effects; no tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.metrics import TARGET_NAMES, compute_all_metrics

SUMMARY_DIR = EXPERIMENT_DIR / "results" / "summary"
EXISTING_TABLES = PROJECT_ROOT / "results" / "tables"
PREDICTIONS = {
    "Original-5": EXPERIMENT_DIR / "results" / "original5" / "seed42" / "tables"
    / "grade_tsmixer_original5_default_seed42_predictions_long.csv",
    "Original-6": EXISTING_TABLES / "stage5_grade_tsmixer_original6_default_seed42_predictions.csv",
    "Wind-6": EXISTING_TABLES / "stage5_grade_tsmixer_wind6_default_seed42_predictions.csv",
    "Wind-7": EXISTING_TABLES / "stage5_grade_tsmixer_wind7_default_seed42_predictions.csv",
}
FEATURES = {
    "Original-5": "Tmax,Tmin,Rain,Sunshine,Pressure",
    "Original-6": "Original-5 + Tavg",
    "Wind-6": "Original-5 + Wind",
    "Wind-7": "Original-5 + Tavg + Wind",
}


def load_metrics(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"sample_index", "target_name", "y_true", "y_pred"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Invalid prediction file: {path}")
    if len(frame) != 2100 or frame["sample_index"].nunique() != 420:
        raise ValueError(f"Unexpected prediction size: {path}")
    true_wide = frame.pivot(index="sample_index", columns="target_name", values="y_true")
    pred_wide = frame.pivot(index="sample_index", columns="target_name", values="y_pred")
    y_true = true_wide.reindex(columns=TARGET_NAMES).to_numpy()
    y_pred = pred_wide.reindex(columns=TARGET_NAMES).to_numpy()
    return compute_all_metrics(y_true, y_pred), y_true


def metric_value(metrics: pd.DataFrame, metric: str, target: str = "Avg") -> float:
    return float(metrics.loc[metrics["metric"] == metric, target].iloc[0])


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in PREDICTIONS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing predictions: {missing}")
    metrics_by_config = {}
    truths = {}
    for configuration, path in PREDICTIONS.items():
        metrics_by_config[configuration], truths[configuration] = load_metrics(path)
    reference = truths["Original-5"]
    alignment = {
        name: {
            "max_abs_y_true_difference": float(np.max(np.abs(values - reference))),
            "aligned_within_csv_tolerance": bool(np.allclose(values, reference, rtol=0, atol=2e-6)),
        }
        for name, values in truths.items()
    }
    if not all(item["aligned_within_csv_tolerance"] for item in alignment.values()):
        raise ValueError(f"Prediction truth alignment failed: {alignment}")

    overall_rows, target_rows = [], []
    for configuration in ["Original-5", "Original-6", "Wind-6", "Wind-7"]:
        metrics = metrics_by_config[configuration]
        overall_rows.append(
            {
                "configuration": configuration, "features": FEATURES[configuration],
                "avg_mae": metric_value(metrics, "MAE"),
                "avg_rmse": metric_value(metrics, "RMSE"),
                "avg_mape": metric_value(metrics, "MAPE"),
                "avg_r2": metric_value(metrics, "R2"),
            }
        )
        for target in TARGET_NAMES:
            target_rows.append(
                {
                    "configuration": configuration, "target": target,
                    "mae": metric_value(metrics, "MAE", target),
                    "rmse": metric_value(metrics, "RMSE", target),
                    "mape": metric_value(metrics, "MAPE", target),
                    "r2": metric_value(metrics, "R2", target),
                }
            )
    overall = pd.DataFrame(overall_rows)
    per_target = pd.DataFrame(target_rows)
    mape = overall.set_index("configuration")["avg_mape"]

    simple_definitions = [
        ("O5_to_O6", "Original-5", "Original-6", "Add Tavg when Wind is absent"),
        ("O5_to_W6", "Original-5", "Wind-6", "Add Wind when Tavg is absent"),
        ("O6_to_W7", "Original-6", "Wind-7", "Add Wind when Tavg is present"),
        ("W6_to_W7", "Wind-6", "Wind-7", "Add Tavg when Wind is present"),
    ]
    simple_rows = []
    for name, baseline, augmented, meaning in simple_definitions:
        delta = float(mape[augmented] - mape[baseline])
        simple_rows.append(
            {"effect": name, "baseline": baseline, "augmented": augmented,
             "meaning": meaning, "delta_mape_augmented_minus_baseline_pp": delta,
             "mape_improvement_baseline_minus_augmented_pp": -delta}
        )
    simple_effects = pd.DataFrame(simple_rows)
    tavg_absent = float(mape["Original-6"] - mape["Original-5"])
    tavg_present_wind = float(mape["Wind-7"] - mape["Wind-6"])
    wind_absent_tavg = float(mape["Wind-6"] - mape["Original-5"])
    wind_present_tavg = float(mape["Wind-7"] - mape["Original-6"])
    descriptive = pd.DataFrame(
        [
            {"effect": "Tavg_average_main_effect",
             "delta_mape_pp": (tavg_absent + tavg_present_wind) / 2,
             "formula": "[(O6-O5)+(W7-W6)]/2"},
            {"effect": "Wind_average_main_effect",
             "delta_mape_pp": (wind_absent_tavg + wind_present_tavg) / 2,
             "formula": "[(W6-O5)+(W7-O6)]/2"},
            {"effect": "Tavg_x_Wind_interaction",
             "delta_mape_pp": tavg_present_wind - tavg_absent,
             "formula": "(W7-W6)-(O6-O5)"},
        ]
    )
    overall.to_csv(SUMMARY_DIR / "wind_2x2_overall_comparison.csv", index=False, encoding="utf-8-sig")
    per_target.to_csv(SUMMARY_DIR / "wind_2x2_per_target_metrics.csv", index=False, encoding="utf-8-sig")
    simple_effects.to_csv(SUMMARY_DIR / "wind_2x2_simple_effects.csv", index=False, encoding="utf-8-sig")
    descriptive.to_csv(
        SUMMARY_DIR / "wind_2x2_main_and_interaction_effects.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "analysis_type": "descriptive 2x2 factor ablation; no statistical tests",
        "delta_convention": "augmented Avg MAPE minus baseline Avg MAPE; negative is improvement",
        "prediction_files": {name: str(path) for name, path in PREDICTIONS.items()},
        "truth_alignment": alignment,
    }
    (SUMMARY_DIR / "wind_2x2_analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("FOUR-CONFIGURATION OVERALL")
    print(overall.to_string(index=False))
    print("\nSIMPLE EFFECTS")
    print(simple_effects.to_string(index=False))
    print("\nDESCRIPTIVE MAIN/INTERACTION EFFECTS")
    print(descriptive.to_string(index=False))
    print(f"\nSUMMARY: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
