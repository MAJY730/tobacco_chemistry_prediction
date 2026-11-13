"""
Split the more-rules dataset and standardize inputs and targets using training-set statistics.
"""

import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from preprocess.utils import ensure_dir, save_report_to_excel

OUTPUT = PROJECT_ROOT / "data" / "processed_more"
STDIR = OUTPUT / "standardized"; SCDIR = OUTPUT / "scalers"
RESULTS = PROJECT_ROOT / "results" / "tables"
ensure_dir(STDIR); ensure_dir(SCDIR)


def std_3d(X_tr, X_va, X_te):
    F = X_tr.shape[2]
    mean = np.array([np.mean(X_tr[:,:,f]) for f in range(F)], dtype=np.float32)
    std = np.array([np.std(X_tr[:,:,f]) for f in range(F)], dtype=np.float32)
    std[std < 1e-10] = 1.0
    def t(X):
        Xs = X.copy()
        for f in range(F): Xs[:,:,f] = (X[:,:,f] - mean[f]) / std[f]
        return Xs
    return t(X_tr), t(X_va), t(X_te), {"mean": mean, "std": std}


def std_2d(y_tr, y_va, y_te):
    F = y_tr.shape[1]
    mean = np.array([np.mean(y_tr[:,f]) for f in range(F)], dtype=np.float32)
    std = np.array([np.std(y_tr[:,f]) for f in range(F)], dtype=np.float32)
    std[std < 1e-10] = 1.0
    def t(y):
        ys = y.copy()
        for f in range(F): ys[:,f] = (y[:,f] - mean[f]) / std[f]
        return ys
    return t(y_tr), t(y_va), t(y_te), {"mean": mean, "std": std}


def main():
    print("=" * 60 + "\n13. Split & Standardize More-Rules\n" + "=" * 60)

    df = pd.read_csv(OUTPUT / "sample_info.csv")
    N = len(df)
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    nt = int(N * 0.70); nv = int(N * 0.15)
    train_idx = sorted(idx[:nt].tolist()); val_idx = sorted(idx[nt:nt+nv].tolist())
    test_idx = sorted(idx[nt+nv:].tolist())

    split = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx, "random_state": 42}
    with open(OUTPUT / "split_indices.json", "w") as f: json.dump(split, f)
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    y = np.load(str(OUTPUT / "y.npy"))
    yt_s, yv_s, yte_s, yp = std_2d(y[train_idx], y[val_idx], y[test_idx])
    y_s = np.zeros_like(y); y_s[train_idx], y_s[val_idx], y_s[test_idx] = yt_s, yv_s, yte_s
    np.save(str(STDIR / "y_scaled.npy"), y_s)
    with open(SCDIR / "y_scaler.pkl", "wb") as f: pickle.dump(yp, f)

    for fname in ["X_original6", "X_wind7"]:
        X = np.load(str(OUTPUT / f"{fname}.npy"))
        Xt_s, Xv_s, Xte_s, xp = std_3d(X[train_idx], X[val_idx], X[test_idx])
        X_s = np.zeros_like(X); X_s[train_idx], X_s[val_idx], X_s[test_idx] = Xt_s, Xv_s, Xte_s
        np.save(str(STDIR / f"{fname}_scaled.npy"), X_s)
        with open(SCDIR / f"{fname}_scaler.pkl", "wb") as f: pickle.dump(xp, f)
        print(f"{fname}: {X_s.shape}")

    # Check table
    grade_dist = df.iloc[train_idx]["grade"].value_counts()
    rows = [{"Metric": "total", "Value": N}, {"Metric": "train", "Value": nt}, {"Metric": "val", "Value": nv},
            {"Metric": "test", "Value": len(test_idx)}, {"Metric": "grades", "Value": df["grade"].nunique()}]
    save_report_to_excel({"split_summary": pd.DataFrame(rows)}, str(RESULTS / "stage9_more_split_scaler_check.xlsx"))
    print("Done.")
if __name__=="__main__": main()
