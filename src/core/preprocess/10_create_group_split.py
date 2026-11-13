"""
Create county-year group split to prevent data leakage.
Same place+year must be in same split (train/val/test).
"""

import sys, json
from pathlib import Path
import numpy as np, pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from preprocess.utils import ensure_dir, save_report_to_excel

PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results" / "tables"


def main():
    ensure_dir(PROCESSED); ensure_dir(RESULTS)
    print("=" * 60 + "\n10. County-Year Group Split\n" + "=" * 60)

    df = pd.read_csv(PROCESSED / "sample_info.csv")
    print(f"Samples: {len(df)}")

    # Build group key
    df["group_key"] = df["place"].astype(str).str.strip() + "_" + df["year"].astype(str).str.strip()
    groups = df["group_key"].unique()
    print(f"Unique place-year groups: {len(groups)}")

    # Shuffle groups with fixed seed
    rng = np.random.RandomState(42)
    group_order = rng.permutation(groups)

    # Assign groups to train/val/test (target 70/15/15)
    # Greedy: assign groups to the set that brings total samples closest to target
    train_idx, val_idx, test_idx = [], [], []
    train_n, val_n, test_n = 0, 0, 0
    total = len(df)

    for g in group_order:
        g_mask = (df["group_key"] == g).values
        g_count = g_mask.sum()
        # Assign to the set farthest below its target
        deficits = {
            "train": 0.70 - train_n / max(total, 1),
            "val": 0.15 - val_n / max(total, 1),
            "test": 0.15 - test_n / max(total, 1),
        }
        best = max(deficits, key=deficits.get)
        g_indices = list(np.where(g_mask)[0])

        if best == "train":
            train_idx.extend(g_indices); train_n += g_count
        elif best == "val":
            val_idx.extend(g_indices); val_n += g_count
        else:
            test_idx.extend(g_indices); test_n += g_count

    train_idx = sorted([int(x) for x in train_idx])
    val_idx = sorted([int(x) for x in val_idx])
    test_idx = sorted([int(x) for x in test_idx])
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert train_n + val_n + test_n == total

    print(f"Train: {train_n} ({train_n/total*100:.1f}%), groups={len(set(df.iloc[train_idx]['group_key']))}")
    print(f"Val:   {val_n} ({val_n/total*100:.1f}%), groups={len(set(df.iloc[val_idx]['group_key']))}")
    print(f"Test:  {test_n} ({test_n/total*100:.1f}%), groups={len(set(df.iloc[test_idx]['group_key']))}")

    # Leakage check
    train_g = set(df.iloc[train_idx]["group_key"])
    val_g = set(df.iloc[val_idx]["group_key"])
    test_g = set(df.iloc[test_idx]["group_key"])
    leakage = len(train_g & val_g) + len(train_g & test_g) + len(val_g & test_g)
    print(f"Leakage (cross-set groups): {leakage}")

    # Save
    split = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
             "group_key": "place_year", "random_state": 42}
    with open(PROCESSED / "group_split_indices.json", "w") as f:
        json.dump(split, f)

    # Check tables
    df["set"] = ""
    df.loc[train_idx, "set"] = "train"; df.loc[val_idx, "set"] = "val"; df.loc[test_idx, "set"] = "test"

    size_df = pd.DataFrame([{"set":s, "n":(df["set"]==s).sum(), "groups":df[df["set"]==s]["group_key"].nunique()} for s in ["train","val","test"]])
    grade_dist = df.groupby(["set","grade"]).size().unstack(fill_value=0).reindex(["train","val","test"])
    leakage_df = pd.DataFrame([{"check":"no_cross_group_leakage", "result": leakage==0}])

    save_report_to_excel({"split_size": size_df, "grade_distribution": grade_dist,
                          "leakage_check": leakage_df}, str(RESULTS / "stage9_group_split_check.xlsx"))
    print(f"Saved: {RESULTS / 'stage9_group_split_check.xlsx'}")
    print(f"Saved: {PROCESSED / 'group_split_indices.json'}")


if __name__ == "__main__": main()
