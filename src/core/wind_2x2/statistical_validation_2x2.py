"""County-year clustered statistical validation for the fixed Wind 2x2 predictions.

This script never trains or loads a model.  Record-level loss is the mean APE
over the five targets, using data/processed/y.npy as the sole truth source.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
OUT = HERE / "results" / "statistical_validation" / "seed42"
TARGETS = ["total_sugar", "reducing_sugar", "nicotine", "total_nitrogen", "potassium"]
PREDICTIONS = {
    "O5": HERE / "results" / "original5" / "seed42" / "tables" /
          "grade_tsmixer_original5_default_seed42_predictions_wide.csv",
    "O6": ROOT / "results" / "tables" /
          "stage5_grade_tsmixer_original6_default_seed42_predictions.csv",
    "W6": ROOT / "results" / "tables" /
          "stage5_grade_tsmixer_wind6_default_seed42_predictions.csv",
    "W7": ROOT / "results" / "tables" /
          "stage5_grade_tsmixer_wind7_default_seed42_predictions.csv",
}
COMPARISONS = [
    ("O6_minus_O5", "O5", "O6", "Add Tavg without Wind"),
    ("W6_minus_O5", "O5", "W6", "Add Wind without Tavg"),
    ("W7_minus_O6", "O6", "W7", "Add Wind with Tavg"),
    ("W7_minus_W6", "W6", "W7", "Add Tavg with Wind"),
    ("W7_minus_O5", "O5", "W7", "Joint Tavg + Wind versus O5"),
]
N_PERM, N_BOOT, RNG_SEED = 100_000, 10_000, 42


def load_predictions(path: Path, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if {f"pred_{t}" for t in TARGETS}.issubset(frame.columns):
        frame = frame.sort_values("sample_index")
        original = frame["original_index"].to_numpy(dtype=int)
        pred = frame[[f"pred_{t}" for t in TARGETS]].to_numpy(dtype=float)
    else:
        wide = frame.pivot(index="sample_index", columns="target_name", values="y_pred")
        pred = wide.reindex(index=range(len(test_idx)), columns=TARGETS).to_numpy(dtype=float)
        original = test_idx.copy()
    if len(pred) != 420 or not np.array_equal(original, test_idx):
        raise ValueError(f"Prediction alignment failed: {path}")
    return original, pred


def holm(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate((len(p) - np.arange(len(p))) * ranked)
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((DATA / "split_indices.json").read_text(encoding="utf-8"))
    test_idx = np.asarray(split["test_idx"], dtype=int)
    if len(test_idx) != 420 or len(np.unique(test_idx)) != 420:
        raise ValueError("Fixed test split is not 420 unique records")
    y_all = np.load(DATA / "y.npy").astype(float)
    y_true = y_all[test_idx]
    info = pd.read_csv(DATA / "sample_info.csv").set_index("original_index").loc[test_idx].reset_index()
    if not np.array_equal(info["original_index"].to_numpy(dtype=int), test_idx):
        raise ValueError("sample_info original_index alignment failed")
    info["cluster"] = info["place"].astype(str) + "__" + info["year"].astype(str)
    codes, cluster_names = pd.factorize(info["cluster"], sort=True)
    n_clusters = len(cluster_names)
    if n_clusters != 149:
        raise ValueError(f"Expected 149 county-year clusters, found {n_clusters}")

    predictions, align = {}, {}
    for name, path in PREDICTIONS.items():
        original, predictions[name] = load_predictions(path, test_idx)
        align[name] = {"path": str(path), "records": len(original),
                       "original_index_exact": bool(np.array_equal(original, test_idx))}
    denom = np.maximum(np.abs(y_true), 1e-8)
    ape = {name: np.mean(np.abs(pred - y_true) / denom * 100.0, axis=1)
           for name, pred in predictions.items()}

    cluster_indices = [np.flatnonzero(codes == c) for c in range(n_clusters)]
    sizes = np.asarray([len(x) for x in cluster_indices], dtype=float)
    rows, group_frames = [], []
    for comparison, before, after, meaning in COMPARISONS:
        diff = ape[after] - ape[before]
        group_mean = np.asarray([diff[ix].mean() for ix in cluster_indices])
        point = float(diff.mean())

        # Flip the entire vector within each cluster; +1 includes the observed draw.
        rng = np.random.default_rng(RNG_SEED)
        extreme = 1
        chunk = 10_000
        for start in range(0, N_PERM, chunk):
            count = min(chunk, N_PERM - start)
            signs = rng.choice(np.array([-1.0, 1.0]), size=(count, n_clusters))
            permuted = (signs @ (group_mean * sizes)) / sizes.sum()
            extreme += int(np.count_nonzero(np.abs(permuted) >= abs(point)))
        permutation_p = extreme / (N_PERM + 1)

        # Pairs-cluster bootstrap: resample county-year clusters with replacement,
        # retaining every record in each selected cluster (record-weighted estimand).
        rng = np.random.default_rng(RNG_SEED)
        draws = rng.integers(0, n_clusters, size=(N_BOOT, n_clusters))
        boot_num = (group_mean * sizes)[draws].sum(axis=1)
        boot_den = sizes[draws].sum(axis=1)
        boot = boot_num / boot_den
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        t = stats.ttest_1samp(group_mean, popmean=0.0)
        try:
            w = stats.wilcoxon(group_mean, alternative="two-sided", zero_method="wilcox")
            wilcoxon_stat, wilcoxon_p = float(w.statistic), float(w.pvalue)
        except ValueError:
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0
        rows.append({
            "comparison": comparison, "before": before, "after": after, "meaning": meaning,
            "delta_mape_pp": point, "n_records": len(diff), "n_county_year_clusters": n_clusters,
            "permutation_p_raw": permutation_p,
            "paired_t_statistic": float(t.statistic), "paired_t_p_raw": float(t.pvalue),
            "wilcoxon_statistic": wilcoxon_stat, "wilcoxon_p_raw": wilcoxon_p,
            "cluster_bootstrap_ci95_low_pp": float(ci_low),
            "cluster_bootstrap_ci95_high_pp": float(ci_high),
            "cluster_bootstrap_ci_crosses_zero": bool(ci_low <= 0 <= ci_high),
        })
        group_frames.append(pd.DataFrame({"comparison": comparison, "cluster": cluster_names,
                                          "n_records": sizes.astype(int), "cluster_mean_delta_mape_pp": group_mean}))

    results = pd.DataFrame(rows)
    for raw, adjusted in [("permutation_p_raw", "permutation_p_holm"),
                          ("paired_t_p_raw", "paired_t_p_holm"),
                          ("wilcoxon_p_raw", "wilcoxon_p_holm")]:
        results[adjusted] = holm(results[raw].tolist())

    interaction_record = (ape["W7"] - ape["W6"]) - (ape["O6"] - ape["O5"])
    interaction_group = np.asarray([interaction_record[ix].mean() for ix in cluster_indices])
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, n_clusters, size=(N_BOOT, n_clusters))
    iboot = ((interaction_group * sizes)[draws].sum(axis=1) / sizes[draws].sum(axis=1))
    ilow, ihigh = np.quantile(iboot, [0.025, 0.975])
    interaction = {
        "definition": "(W7-W6)-(O6-O5)",
        "delta_mape_pp": float(interaction_record.mean()),
        "cluster_bootstrap_ci95_low_pp": float(ilow),
        "cluster_bootstrap_ci95_high_pp": float(ihigh),
        "cluster_bootstrap_ci_crosses_zero": bool(ilow <= 0 <= ihigh),
        "n_county_year_clusters": n_clusters,
    }

    record = info[["original_index", "place", "year", "cluster"]].copy()
    for name in ["O5", "O6", "W6", "W7"]:
        record[f"mean_ape_{name}"] = ape[name]
    for comparison, before, after, _ in COMPARISONS:
        record[f"delta_{comparison}"] = ape[after] - ape[before]
    record["interaction_delta_mape_pp"] = interaction_record

    results.to_csv(OUT / "wind_2x2_cluster_statistical_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(group_frames, ignore_index=True).to_csv(
        OUT / "wind_2x2_cluster_level_differences.csv", index=False, encoding="utf-8-sig")
    record.to_csv(OUT / "wind_2x2_record_level_ape_and_differences.csv", index=False, encoding="utf-8-sig")
    (OUT / "wind_2x2_cluster_statistical_results.json").write_text(json.dumps({
        "analysis": {
            "truth_source": str(DATA / "y.npy"), "split_source": str(DATA / "split_indices.json"),
            "sample_info_source": str(DATA / "sample_info.csv"), "n_test_records": 420,
            "n_county_year_clusters": n_clusters, "ape_definition": "mean over five targets of abs(pred-true)/abs(true)*100",
            "delta_convention": "after minus before; negative means after is better",
            "permutation": {"method": "county-year cluster sign-flip, two-sided", "iterations": N_PERM},
            "bootstrap": {"method": "pairs county-year cluster bootstrap, percentile 95% CI", "iterations": N_BOOT},
            "group_tests": "two-sided one-sample paired t and Wilcoxon signed-rank on 149 cluster means",
            "multiplicity": "Holm separately across five comparisons for each of three p-value families",
            "random_seed": RNG_SEED, "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "prediction_alignment": align,
        },
        "comparisons": results.to_dict(orient="records"), "interaction": interaction,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "wind_2x2_interaction_bootstrap.json").write_text(
        json.dumps(interaction, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results.to_string(index=False))
    print("\nINTERACTION\n" + json.dumps(interaction, ensure_ascii=False, indent=2))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
