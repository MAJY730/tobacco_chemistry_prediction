"""
Extract grade IDs from sample_info.csv for grade-aware models.
Outputs: data/processed/grade_ids.npy, data/processed/grade_mapping.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from preprocess.utils import ensure_dir, save_report_to_excel

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"


def main():
    print("=" * 60)
    print("Grade Feature Preparation (09_prepare_grade_features.py)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ensure_dir(PROCESSED_DIR)
    ensure_dir(RESULTS_DIR)

    df = pd.read_csv(PROCESSED_DIR / "sample_info.csv")
    print(f"Total samples: {len(df)}")

    grades = df["grade"].fillna("UNKNOWN").astype(str).str.strip()
    value_counts = grades.value_counts()

    # Build mapping by frequency
    grade_to_id = {}
    for i, (g, _) in enumerate(value_counts.items()):
        grade_to_id[g] = i

    grade_ids = np.array([grade_to_id[g] for g in grades], dtype=np.int64)
    np.save(str(PROCESSED_DIR / "grade_ids.npy"), grade_ids)

    mapping = {
        "num_grades": len(grade_to_id),
        "grade_to_id": grade_to_id,
        "id_to_grade": {v: k for k, v in grade_to_id.items()},
    }
    with open(PROCESSED_DIR / "grade_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Grade classes: {len(grade_to_id)}")
    print(f"grade_ids saved: {PROCESSED_DIR / 'grade_ids.npy'} shape={grade_ids.shape}")
    print(f"grade_mapping saved: {PROCESSED_DIR / 'grade_mapping.json'}")

    # Save distribution
    dist_rows = [{"grade": g, "id": grade_to_id[g], "count": int(c)}
                 for g, c in value_counts.items()]
    save_report_to_excel({"grade_distribution": pd.DataFrame(dist_rows)},
                         str(RESULTS_DIR / "stage5_grade_distribution.xlsx"))
    print(f"Distribution saved: {RESULTS_DIR / 'stage5_grade_distribution.xlsx'}")

    for r in dist_rows:
        print(f"  id={r['id']:2d}  {r['grade']:15s}  n={r['count']}")

    print("Done!")


if __name__ == "__main__":
    main()
