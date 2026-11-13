"""
Build O6 and W7 datasets and five-target arrays from more-rules.json.
"""

import sys, json, os
from pathlib import Path
import numpy as np, pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from preprocess.utils import ensure_dir, save_report_to_excel

WORKSPACE = PROJECT_ROOT.parent
OUTPUT = PROJECT_ROOT / "data" / "processed_more"
RESULTS = PROJECT_ROOT / "results" / "tables"

INPUT_IDX = {"original6": [0,1,2,3,5,4], "wind7": [0,1,2,3,5,4,6]}
FIVE_TARGETS = [0,1,2,5,4]
CHEM_NAMES = ["total_sugar","reducing_sugar","nicotine","chlorine","potassium","total_nitrogen",
              "sugar_nicotine","K_Cl","sugar_N"]


def find_json():
    for root, dirs, files in os.walk(str(WORKSPACE)):
        for f in files:
            if f == "more-rules.json":
                return Path(root) / f
    return None


def main():
    ensure_dir(OUTPUT); ensure_dir(RESULTS)
    print("=" * 60 + "\n12. Build More-Rules Dataset\n" + "=" * 60)

    jp = find_json()
    if jp is None:
        print("[ERROR] more-rules.json not found. Place it in workspace and retry.")
        sys.exit(1)
    print(f"JSON: {jp}")

    with open(jp, "r", encoding="utf-8") as f: data = json.load(f)
    N = len(data)
    print(f"Samples: {N}")

    valid, bad = [], []
    for i, s in enumerate(data):
        w = s.get("weather", [])
        c = s.get("chemical", [])
        ok = True
        if not isinstance(w, list) or len(w) != 153: ok = False
        elif isinstance(w[0], list) and len(w[0]) != 7: ok = False
        wa = np.array(w, dtype=np.float32)
        if np.any(np.isnan(wa)) or np.any(np.isinf(wa)): ok = False
        ca = np.array(c, dtype=np.float32)
        if np.any(np.isnan(ca)) or np.any(np.isinf(ca)): ok = False
        if ok: valid.append((i, s))
        else: bad.append({"idx": i, "reason": "invalid"})

    nv = len(valid); print(f"Valid: {nv}, Bad: {len(bad)}")
    if nv == 0: sys.exit(1)

    X_o6 = np.zeros((nv, 153, 6), dtype=np.float32)
    X_w7 = np.zeros((nv, 153, 7), dtype=np.float32)
    y_arr = np.zeros((nv, 5), dtype=np.float32)
    infos = []

    for ni, (oi, s) in enumerate(valid):
        wa = np.array(s["weather"], dtype=np.float32)
        ca = np.array(s["chemical"], dtype=np.float32)
        X_o6[ni] = wa[:, INPUT_IDX["original6"]]
        X_w7[ni] = wa[:, INPUT_IDX["wind7"]]
        y_arr[ni] = ca[FIVE_TARGETS]
        infos.append({"sample_id": ni, "year": s.get("year"), "place": str(s.get("place","")),
                       "grade": str(s.get("grade","")), "json_source": "more-rules.json",
                       "original_index": oi})

    for fname, arr in [("X_original6", X_o6), ("X_wind7", X_w7), ("y", y_arr)]:
        np.save(str(OUTPUT / f"{fname}.npy"), arr)
    df = pd.DataFrame(infos)
    df.to_csv(str(OUTPUT / "sample_info.csv"), index=False, encoding="utf-8-sig")

    # Grade
    grades = df["grade"].str.strip().value_counts()
    g2id = {g: i for i, (g, _) in enumerate(grades.items())}
    gids = np.array([g2id[g.strip()] for g in df["grade"]], dtype=np.int64)
    np.save(str(OUTPUT / "grade_ids.npy"), gids)
    mapping = {"num_grades": len(g2id), "grade_to_id": g2id, "id_to_grade": {v:k for k,v in g2id.items()}}
    with open(OUTPUT / "grade_mapping.json", "w") as f: json.dump(mapping, f, ensure_ascii=False, indent=2)

    # Summary
    rows = [{"Metric":"samples","Value":nv},{"Metric":"X_original6","Value":str(X_o6.shape)},
            {"Metric":"X_wind7","Value":str(X_w7.shape)},{"Metric":"y","Value":str(y_arr.shape)},
            {"Metric":"grades","Value":len(g2id)},{"Metric":"years","Value":f"{df['year'].min()}-{df['year'].max()}"},
            {"Metric":"places","Value":df['place'].nunique()},
            {"Metric":"has_nan_X","Value":any(np.any(np.isnan(a)) for a in [X_o6, X_w7])},
            {"Metric":"has_nan_y","Value":bool(np.any(np.isnan(y_arr)))}]
    save_report_to_excel({"summary": pd.DataFrame(rows), "grade_dist": grades.reset_index(name="count")},
                         str(RESULTS / "stage9_more_dataset_check.xlsx"))
    print(f"Done. {OUTPUT}/")
if __name__=="__main__": main()
