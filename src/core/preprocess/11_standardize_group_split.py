"""
Standardize data using group split (train only fit). Output group_standardized/ and group_scalers/.
"""

import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from preprocess.utils import ensure_dir, save_report_to_excel

PROCESSED = PROJECT_ROOT / "data" / "processed"
GSTD = PROCESSED / "group_standardized"; GSCL = PROCESSED / "group_scalers"
RESULTS = PROJECT_ROOT / "results" / "tables"
ensure_dir(GSTD); ensure_dir(GSCL)


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
    print("=" * 60 + "\n11. Standardize with Group Split\n" + "=" * 60)
    with open(PROCESSED / "group_split_indices.json") as f: split = json.load(f)
    tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]

    y = np.load(str(PROCESSED / "y.npy"))
    yt_s, yv_s, yte_s, yp = std_2d(y[tr], y[va], y[te])
    y_s = np.zeros_like(y)
    y_s[tr], y_s[va], y_s[te] = yt_s, yv_s, yte_s
    np.save(str(GSTD / "y_group_scaled.npy"), y_s)
    with open(GSCL / "y_group_scaler.pkl", "wb") as f: pickle.dump(yp, f)
    print(f"y: {y_s.shape}")

    check_rows = []
    for name, fname in [("original6","X_original6"), ("wind7","X_wind7")]:
        X = np.load(str(PROCESSED / f"{fname}.npy"))
        Xt_s, Xv_s, Xte_s, xp = std_3d(X[tr], X[va], X[te])
        X_s = np.zeros_like(X); X_s[tr], X_s[va], X_s[te] = Xt_s, Xv_s, Xte_s
        np.save(str(GSTD / f"{fname}_group_scaled.npy"), X_s)
        with open(GSCL / f"{fname}_group_scaler.pkl", "wb") as f: pickle.dump(xp, f)
        print(f"{fname}: {X_s.shape}")
        names = ["Tmax","Tmin","Tavg","Rain","Sunshine","Pressure"] if fname=="X_original6" else ["Tmax","Tmin","Tavg","Rain","Sunshine","Pressure","Wind"]
        for j, fn in enumerate(names):
            check_rows.append({"dataset":name,"feature":fn,
                "train_mean":round(float(xp["mean"][j]),4),"train_std":round(float(xp["std"][j]),4),
                "scaled_train_mean":round(float(np.mean(X_s[tr,:,j])),6)})

    save_report_to_excel({"scaler_check":pd.DataFrame(check_rows)}, str(RESULTS/"stage9_group_scaler_check.xlsx"))
    print(f"Done. {GSTD}/, {GSCL}/")
if __name__=="__main__": main()
