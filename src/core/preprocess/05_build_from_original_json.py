"""
Build three input datasets from seven-variable JSON data.
Read less-rules.json (or more-rules.json) and create:
  data/processed/X_original6.npy
  data/processed/X_wind6.npy
  data/processed/X_wind7.npy
  data/processed/y_all_chemical.npy
  data/processed/y.npy
  data/processed/sample_info.csv
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.utils import ensure_dir, save_report_to_excel

# Load configuration
with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

WORKSPACE = PROJECT_ROOT.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
ensure_dir(PROCESSED_DIR)
ensure_dir(RESULTS_DIR)

# Weather-variable index order in the JSON input.
# 0: Tmax 1: Tmin 2: Tavg 3: Rain 4: Pressure 5: Sunshine 6: Wind
WEATHER_INDICES = {
    "Tmax": 0, "Tmin": 1, "Tavg": 2, "Rain": 3,
    "Pressure": 4, "Sunshine": 5, "Wind": 6,
}

# Column indices for the three input configurations
INPUT_COLUMN_INDICES = {
    "original6": [0, 1, 2, 3, 5, 4],  # Tmax, Tmin, Tavg, Rain, Sunshine, Pressure
    "wind6":    [0, 1, 3, 5, 4, 6],  # Tmax, Tmin, Rain, Sunshine, Pressure, Wind
    "wind7":    [0, 1, 2, 3, 5, 4, 6],  # Tmax, Tmin, Tavg, Rain, Sunshine, Pressure, Wind
}

INPUT_COLUMN_NAMES = {
    "original6": ["Tmax", "Tmin", "Tavg", "Rain", "Sunshine", "Pressure"],
    "wind6":    ["Tmax", "Tmin", "Rain", "Sunshine", "Pressure", "Wind"],
    "wind7":    ["Tmax", "Tmin", "Tavg", "Rain", "Sunshine", "Pressure", "Wind"],
}

# Verified chemical-variable index order in the JSON input.
# 0:total_sugar, 1:reducing_sugar, 2:nicotine, 3:chlorine, 4:potassium,
# 5:total_nitrogen, 6:sugar/nic, 7:K/Cl, 8:sugar/N
CHEMICAL_NAMES = [
    "total_sugar", "reducing_sugar", "nicotine", "chlorine",
    "potassium", "total_nitrogen", "sugar_nicotine_ratio",
    "potassium_chlorine_ratio", "sugar_nitrogen_ratio",
]

# Indices of the five prediction targets in chemical.
# Order: total_sugar, reducing_sugar, nicotine, total_nitrogen, potassium
FIVE_TARGET_INDICES = [0, 1, 2, 5, 4]
FIVE_TARGET_NAMES = ["total_sugar", "reducing_sugar", "nicotine", "total_nitrogen", "potassium"]

SEQ_LENGTH = config["data"]["sequence_length"]  # 153
EXPECTED_WEATHER_FEATURES = 7


def find_json_file(json_name="less-rules.json"):
    """Recursively locate a JSON file in the workspace."""
    for root, dirs, files in os.walk(str(WORKSPACE)):
        for f in files:
            if f == json_name and "\u4e03\u4e2a\u56e0\u5b50" in root:
                return Path(root) / f
    # Fall back to an unrestricted filename search.
    for root, dirs, files in os.walk(str(WORKSPACE)):
        for f in files:
            if f == json_name:
                return Path(root) / f
    return None


def check_sample(sample, idx):
    """Check one record and return (is_valid, reason)."""
    required_keys = ["year", "place", "grade", "chemical", "weather"]
    for key in required_keys:
        if key not in sample:
            return False, f"Missing field: {key}"

    weather = sample["weather"]
    if not isinstance(weather, list) or len(weather) != SEQ_LENGTH:
        return False, f"weather length is not {SEQ_LENGTH}: {len(weather) if isinstance(weather, list) else type(weather)}"

    # Check that each weather time step has seven features.
    for t, step in enumerate(weather):
        if not isinstance(step, list) or len(step) != EXPECTED_WEATHER_FEATURES:
            return False, f"weather[{t}] dimension is not {EXPECTED_WEATHER_FEATURES}: {len(step) if isinstance(step, list) else type(step)}"

    # Check weather for NaN and infinity.
    weather_arr = np.array(weather, dtype=np.float64)
    if np.any(np.isnan(weather_arr)):
        nan_count = np.sum(np.isnan(weather_arr))
        return False, f"weather contains {nan_count} NaN values"
    if np.any(np.isinf(weather_arr)):
        inf_count = np.sum(np.isinf(weather_arr))
        return False, f"weather contains {inf_count} infinite values"

    # Check chemical values.
    chemical = sample["chemical"]
    if not isinstance(chemical, list):
        return False, f"chemical is not a list"

    chem_arr = np.array(chemical, dtype=np.float64)
    if np.any(np.isnan(chem_arr)):
        nan_count = np.sum(np.isnan(chem_arr))
        return False, f"chemical contains {nan_count} NaN values"
    if np.any(np.isinf(chem_arr)):
        inf_count = np.sum(np.isinf(chem_arr))
        return False, f"chemical contains {inf_count} infinite values"

    return True, "OK"


def main():
    json_name = "less-rules.json"
    # Allow the input filename to be selected from the command line.
    if len(sys.argv) > 1:
        json_name = sys.argv[1]

    print("=" * 60)
    print("Build three input datasets from JSON (05_build_from_original_json.py)")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"JSON file: {json_name}")
    print("=" * 60)

    # Locate the JSON file.
    json_path = find_json_file(json_name)
    if json_path is None:
        print(f"[ERROR] Not found: {json_name}")
        print("Confirm that the file exists within the search directory.")
        sys.exit(1)

    print(f"\nJSON path: {json_path}")

    # Load JSON.
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Raw record count: {len(data)}")

    # Validate each record.
    valid_samples = []
    bad_samples = []
    chemical_dims = set()

    for i, sample in enumerate(data):
        is_valid, reason = check_sample(sample, i)
        if is_valid:
            valid_samples.append((i, sample))
        else:
            bad_samples.append({
                "original_index": i,
                "reason": reason,
                "year": sample.get("year", ""),
                "place": sample.get("place", ""),
                "grade": sample.get("grade", ""),
            })

        if isinstance(sample.get("chemical"), list):
            chemical_dims.add(len(sample["chemical"]))

    n_total = len(data)
    n_valid = len(valid_samples)
    n_bad = len(bad_samples)

    print(f"Valid record count: {n_valid}")
    print(f"Invalid record count: {n_bad}")
    print(f"chemical dimension distribution: {chemical_dims}")

    # Save the invalid-record report.
    if bad_samples:
        save_report_to_excel(
            {"bad_samples": pd.DataFrame(bad_samples)},
            str(RESULTS_DIR / "bad_json_samples.xlsx")
        )
        print(f"Invalid-record report saved to: {RESULTS_DIR / 'bad_json_samples.xlsx'}")

    if n_valid == 0:
        print("[ERROR] No valid records.")
        sys.exit(1)

    # Extract arrays.
    N = n_valid
    chemical_dim = max(chemical_dims)

    X_original6 = np.zeros((N, SEQ_LENGTH, 6), dtype=np.float32)
    X_wind6 = np.zeros((N, SEQ_LENGTH, 6), dtype=np.float32)
    X_wind7 = np.zeros((N, SEQ_LENGTH, 7), dtype=np.float32)
    y_all = np.zeros((N, chemical_dim), dtype=np.float32)
    sample_info_list = []

    for new_idx, (orig_idx, sample) in enumerate(valid_samples):
        weather_arr = np.array(sample["weather"], dtype=np.float32)
        chem_arr = np.array(sample["chemical"], dtype=np.float32)

        # Three input configurations.
        X_original6[new_idx] = weather_arr[:, INPUT_COLUMN_INDICES["original6"]]
        X_wind6[new_idx] = weather_arr[:, INPUT_COLUMN_INDICES["wind6"]]
        X_wind7[new_idx] = weather_arr[:, INPUT_COLUMN_INDICES["wind7"]]

        # Complete chemical vector.
        pad_len = chemical_dim - len(chem_arr)
        if pad_len > 0:
            chem_arr = np.pad(chem_arr, (0, pad_len), constant_values=np.nan)
        y_all[new_idx, :len(sample["chemical"])] = chem_arr[:len(sample["chemical"])]

        sample_info_list.append({
            "sample_id": new_idx,
            "year": sample.get("year"),
            "place": str(sample.get("place", "")),
            "grade": str(sample.get("grade", "")),
            "json_source": json_name,
            "original_index": orig_idx,
        })

    # Save processed data.
    print("\nSaving processed data...")
    np.save(str(PROCESSED_DIR / "X_original6.npy"), X_original6)
    np.save(str(PROCESSED_DIR / "X_wind6.npy"), X_wind6)
    np.save(str(PROCESSED_DIR / "X_wind7.npy"), X_wind7)
    np.save(str(PROCESSED_DIR / "y_all_chemical.npy"), y_all)

    # Build y.npy from the five targets.
    if chemical_dim >= 6:
        # Confirm that the target indices exist.
        max_target_idx = max(FIVE_TARGET_INDICES)
        if max_target_idx < chemical_dim:
            y = y_all[:, FIVE_TARGET_INDICES]
            np.save(str(PROCESSED_DIR / "y.npy"), y)
            print(f"y.npy created (N={N}, 5 targets: {FIVE_TARGET_NAMES})")
        else:
            print(f"[WARNING] chemical dimension ({chemical_dim}) is insufficient for five targets (max_index={max_target_idx})")
    else:
        print(f"[WARNING] chemical dimension is {chemical_dim}, so five targets cannot be extracted")

    # Save sample metadata.
    df_info = pd.DataFrame(sample_info_list)
    df_info.to_csv(str(PROCESSED_DIR / "sample_info.csv"), index=False, encoding="utf-8-sig")

    print(f"\nData saved to {PROCESSED_DIR}:")
    print(f"  X_original6.npy: {X_original6.shape}")
    print(f"  X_wind6.npy:    {X_wind6.shape}")
    print(f"  X_wind7.npy:    {X_wind7.shape}")
    print(f"  y_all_chemical.npy: {y_all.shape}")
    if (PROCESSED_DIR / "y.npy").exists():
        y_loaded = np.load(str(PROCESSED_DIR / "y.npy"))
        print(f"  y.npy:          {y_loaded.shape}")
    print(f"  sample_info.csv: {len(df_info)} rows")

    # Generate the dataset summary.
    _generate_summary(df_info, X_original6, X_wind6, X_wind7, y_all, y_all[:, FIVE_TARGET_INDICES] if chemical_dim >= 6 else None, json_name)

    print("\n" + "=" * 60)
    print("Dataset construction complete.")
    print("=" * 60)


def _generate_summary(df_info, X_o6, X_w6, X_w7, y_all, y_5, json_name):
    """Generate the dataset summary Excel workbook."""
    N = len(df_info)

    # Year distribution.
    year_dist = df_info["year"].value_counts().sort_index().reset_index()
    year_dist.columns = ["year", "count"]

    # Place distribution.
    place_dist = df_info["place"].value_counts().reset_index()
    place_dist.columns = ["place", "count"]

    # Grade distribution.
    grade_dist = df_info["grade"].value_counts().reset_index()
    grade_dist.columns = ["grade", "count"]

    # Weather-variable statistics.
    weather_stats = []
    for name, indices in INPUT_COLUMN_INDICES.items():
        col_names = INPUT_COLUMN_NAMES[name]
        X = {"original6": X_o6, "wind6": X_w6, "wind7": X_w7}[name]
        for j, cname in enumerate(col_names):
            vals = X[:, :, j].ravel()
            weather_stats.append({
                "input_set": name,
                "feature": cname,
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "has_nan": bool(np.any(np.isnan(vals))),
                "has_inf": bool(np.any(np.isinf(vals))),
            })

    # Chemical statistics.
    chem_stats = []
    for j, cname in enumerate(CHEMICAL_NAMES):
        if j < y_all.shape[1]:
            vals = y_all[:, j]
            valid = vals[~np.isnan(vals)]
            chem_stats.append({
                "feature": cname,
                "index": j,
                "is_target": cname in FIVE_TARGET_NAMES,
                "non_null": int(np.sum(~np.isnan(vals))),
                "missing_rate": round(float(np.mean(np.isnan(vals))), 4),
                "min": round(float(np.min(valid)), 4) if len(valid) > 0 else None,
                "max": round(float(np.max(valid)), 4) if len(valid) > 0 else None,
                "mean": round(float(np.mean(valid)), 4) if len(valid) > 0 else None,
                "std": round(float(np.std(valid)), 4) if len(valid) > 0 else None,
            })

    # Summary information.
    summary_rows = [
        {"Metric": "JSON file", "Value": json_name},
        {"Metric": "Total records", "Value": N},
        {"Metric": "X_original6 shape", "Value": str(X_o6.shape)},
        {"Metric": "X_wind6 shape", "Value": str(X_w6.shape)},
        {"Metric": "X_wind7 shape", "Value": str(X_w7.shape)},
        {"Metric": "y_all_chemical shape", "Value": str(y_all.shape)},
        {"Metric": "y shape", "Value": str(y_5.shape) if y_5 is not None else "not generated"},
        {"Metric": "weather dimensions", "Value": f"{X_w7.shape[1]} days x {X_w7.shape[2]} variables"},
        {"Metric": "chemical dimensions", "Value": y_all.shape[1]},
        {"Metric": "Five-target indices", "Value": str(FIVE_TARGET_INDICES)},
        {"Metric": "Five-target names", "Value": ", ".join(FIVE_TARGET_NAMES)},
        {"Metric": "Contains NaN (X)", "Value": any(np.any(np.isnan(X)) for X in [X_o6, X_w6, X_w7])},
        {"Metric": "Contains inf (X)", "Value": any(np.any(np.isinf(X)) for X in [X_o6, X_w6, X_w7])},
        {"Metric": "Contains NaN (y_all)", "Value": bool(np.any(np.isnan(y_all)))},
        {"Metric": "Contains inf (y_all)", "Value": bool(np.any(np.isinf(y_all)))},
    ]

    save_report_to_excel({
        "Overview": pd.DataFrame(summary_rows),
        "Year distribution": year_dist,
        "Place distribution": place_dist,
        "Grade distribution": grade_dist,
        "Weather-variable statistics": pd.DataFrame(weather_stats),
        "Chemical statistics": pd.DataFrame(chem_stats),
    }, str(RESULTS_DIR / "stage2_json_dataset_summary.xlsx"))

    print(f"\nDataset summary saved: {RESULTS_DIR / 'stage2_json_dataset_summary.xlsx'}")


if __name__ == "__main__":
    main()
