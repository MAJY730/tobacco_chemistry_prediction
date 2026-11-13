"""Paired county-year cluster statistics for the completed Group-CV OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "summary"
OUT = SOURCE / "oof_cluster_statistics"
TARGETS = ["total_sugar", "reducing_sugar", "nicotine", "total_nitrogen", "potassium"]
N_PERMUTATIONS = 100_000
N_BOOTSTRAPS = 100_000
RNG_SEED = 42

FILES = {
    "GA": SOURCE / "ga_wind_tsmixer_oof_predictions_wide.csv",
    "Grade-GRU": SOURCE / "grade_gru_oof_predictions_wide.csv",
    "Grade-Transformer": SOURCE / "grade_transformer_oof_predictions_wide.csv",
}


def dump_json(value, path):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def load_and_validate():
    frames = {name: pd.read_csv(path) for name, path in FILES.items()}
    base = frames["GA"].sort_values("original_index").reset_index(drop=True)
    required = {"original_index", "outer_fold", "place", "year", "grade_seen_in_inner_train"}
    checks = {
        "all_files_2790_rows": all(len(frame) == 2790 for frame in frames.values()),
        "all_original_index_unique": all(frame.original_index.nunique() == 2790 for frame in frames.values()),
        "five_outer_folds_present": all(set(frame.outer_fold) == {1, 2, 3, 4, 5} for frame in frames.values()),
    }
    if not required.issubset(base.columns):
        raise RuntimeError(f"Missing columns: {required - set(base.columns)}")
    for name, frame in frames.items():
        aligned = frame.sort_values("original_index").reset_index(drop=True)
        checks[f"{name}_original_index_aligned"] = np.array_equal(base.original_index, aligned.original_index)
        checks[f"{name}_outer_fold_aligned"] = np.array_equal(base.outer_fold, aligned.outer_fold)
        checks[f"{name}_place_year_aligned"] = (
            np.array_equal(base.place.astype(str), aligned.place.astype(str))
            and np.array_equal(base.year, aligned.year)
        )
        checks[f"{name}_grade_seen_flag_aligned"] = np.array_equal(
            base.grade_seen_in_inner_train.astype(bool), aligned.grade_seen_in_inner_train.astype(bool)
        )
        for target in TARGETS:
            checks[f"{name}_{target}_truth_aligned"] = np.allclose(
                base[f"true_{target}"], aligned[f"true_{target}"], rtol=0, atol=1e-7
            )
        frames[name] = aligned
    checks["county_year_clusters_208"] = base.groupby(["place", "year"]).ngroups == 208
    checks["unseen_grade_records_16"] = int((~base.grade_seen_in_inner_train.astype(bool)).sum()) == 16
    if not all(checks.values()):
        raise RuntimeError("Alignment validation failed: " + ", ".join(k for k, v in checks.items() if not v))
    return frames, checks


def mean_ape(frame):
    truth = frame[[f"true_{target}" for target in TARGETS]].to_numpy(dtype=float)
    pred = frame[[f"pred_{target}" for target in TARGETS]].to_numpy(dtype=float)
    return (np.abs((truth - pred) / np.maximum(np.abs(truth), 1e-8)) * 100.0).mean(axis=1)


def sign_flip_pvalue(differences, rng):
    differences = np.asarray(differences, dtype=float)
    observed = abs(differences.mean())
    extreme = 0
    remaining = N_PERMUTATIONS
    chunk = 5_000
    while remaining:
        n = min(chunk, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(differences)))
        permuted = (signs * differences).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted) >= observed - 1e-15))
        remaining -= n
    return (extreme + 1) / (N_PERMUTATIONS + 1)


def cluster_bootstrap(differences, rng):
    differences = np.asarray(differences, dtype=float)
    estimates = np.empty(N_BOOTSTRAPS, dtype=float)
    remaining = N_BOOTSTRAPS
    start = 0
    chunk = 5_000
    while remaining:
        n = min(chunk, remaining)
        sampled = rng.integers(0, len(differences), size=(n, len(differences)))
        estimates[start:start + n] = differences[sampled].mean(axis=1)
        start += n
        remaining -= n
    return estimates, np.percentile(estimates, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, alignment = load_and_validate()
    base = frames["GA"]
    ga_ape = mean_ape(base)
    record_output = base[["original_index", "record_index", "outer_fold", "place", "year",
                          "grade_id", "grade_label", "grade_seen_in_inner_train"]].copy()
    record_output["ga_mean_ape"] = ga_ape
    comparisons = [("GA_minus_Grade_GRU", "Grade-GRU"),
                   ("GA_minus_Grade_Transformer", "Grade-Transformer")]
    group_outputs = []
    results = []
    bootstrap_json = {}
    rng = np.random.default_rng(RNG_SEED)

    for comparison, baseline in comparisons:
        baseline_ape = mean_ape(frames[baseline])
        difference = ga_ape - baseline_ape
        record_output[f"{comparison}_record_difference"] = difference
        working = record_output[["place", "year"]].copy()
        working["difference"] = difference
        grouped = working.groupby(["place", "year"], as_index=False).agg(
            cluster_difference=("difference", "mean"), cluster_records=("difference", "size")
        )
        grouped.insert(0, "comparison", comparison)
        group_outputs.append(grouped)
        d = grouped.cluster_difference.to_numpy()
        permutation_p = sign_flip_pvalue(d, rng)
        bootstrap_values, ci = cluster_bootstrap(d, rng)
        t_result = stats.ttest_1samp(d, popmean=0.0, alternative="two-sided")
        try:
            wilcoxon_result = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
            wilcoxon_p = float(wilcoxon_result.pvalue)
            wilcoxon_stat = float(wilcoxon_result.statistic)
        except ValueError:
            wilcoxon_p = 1.0
            wilcoxon_stat = 0.0
        q1, median, q3 = np.percentile(d, [25, 50, 75])
        seen = base.grade_seen_in_inner_train.astype(bool).to_numpy()
        # Size-influence diagnostics: remove the largest 10% of clusters by record count.
        size_cutoff = grouped.cluster_records.quantile(0.90, interpolation="lower")
        retained_groups = grouped.loc[grouped.cluster_records <= size_cutoff, ["place", "year"]]
        retained_key = set(map(tuple, retained_groups.to_numpy()))
        keep_records = np.array([(place, year) in retained_key for place, year in zip(base.place, base.year)])
        spearman = stats.spearmanr(grouped.cluster_records, grouped.cluster_difference)
        result = {
            "comparison": comparison,
            "baseline": baseline,
            "clusters": len(grouped),
            "records": len(difference),
            "pooled_record_weighted_delta_mape": float(difference.mean()),
            "county_year_equal_weight_delta_mape": float(d.mean()),
            "ga_better_groups": int(np.count_nonzero(d < 0)),
            "ga_better_group_proportion": float(np.mean(d < 0)),
            "ties": int(np.count_nonzero(d == 0)),
            "cluster_difference_median": float(median),
            "cluster_difference_q1": float(q1),
            "cluster_difference_q3": float(q3),
            "cluster_difference_iqr": float(q3 - q1),
            "permutation_p_raw": float(permutation_p),
            "bootstrap_ci_lower": float(ci[0]),
            "bootstrap_ci_upper": float(ci[1]),
            "t_statistic": float(t_result.statistic),
            "t_p_raw": float(t_result.pvalue),
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_p_raw": wilcoxon_p,
            "sensitivity_excluding_16_unseen_grade_records_record_weighted_delta": float(difference[seen].mean()),
            "sensitivity_excluding_16_unseen_grade_records_n": int(seen.sum()),
            "large_group_diagnostic": {
                "largest_10pct_size_cutoff_records": int(size_cutoff),
                "excluded_large_groups": int(len(grouped) - len(retained_groups)),
                "remaining_records": int(keep_records.sum()),
                "record_weighted_delta_after_excluding_largest_10pct_groups": float(difference[keep_records].mean()),
                "cluster_size_difference_spearman_rho": float(spearman.statistic),
                "cluster_size_difference_spearman_p": float(spearman.pvalue),
            },
        }
        results.append(result)
        bootstrap_json[comparison] = {
            "seed": RNG_SEED, "iterations": N_BOOTSTRAPS,
            "statistic": "mean of 208 county-year cluster mean differences",
            "point_estimate": float(d.mean()), "ci_percentile": 95,
            "ci_lower": float(ci[0]), "ci_upper": float(ci[1]),
            "bootstrap_mean": float(bootstrap_values.mean()),
            "bootstrap_sd": float(bootstrap_values.std(ddof=1)),
        }

    for p_field, adjusted_field in (("permutation_p_raw", "permutation_p_holm"),
                                    ("t_p_raw", "t_p_holm"),
                                    ("wilcoxon_p_raw", "wilcoxon_p_holm")):
        adjusted = holm_adjust([result[p_field] for result in results])
        for result, value in zip(results, adjusted):
            result[adjusted_field] = float(value)

    flat_rows = []
    for result in results:
        flat = {key: value for key, value in result.items() if key != "large_group_diagnostic"}
        flat.update({f"large_group_{key}": value for key, value in result["large_group_diagnostic"].items()})
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(OUT / "group_cv_oof_cluster_statistics.csv", index=False)
    pd.concat(group_outputs, ignore_index=True).to_csv(
        OUT / "group_cv_oof_cluster_differences.csv", index=False, encoding="utf-8-sig"
    )
    record_output.to_csv(OUT / "group_cv_oof_record_differences.csv", index=False, encoding="utf-8-sig")
    dump_json(bootstrap_json, OUT / "group_cv_oof_cluster_bootstrap.json")
    summary = {
        "analysis": "5-fold Group-CV OOF paired county-year cluster analysis",
        "difference_definition": "GA mean APE minus baseline mean APE; negative favors GA",
        "formal_unit": "208 place+year county-year clusters",
        "permutations": N_PERMUTATIONS,
        "bootstraps": N_BOOTSTRAPS,
        "alignment_checks": alignment,
        "multiple_testing": "Holm correction separately across two comparisons for permutation, t-test, and Wilcoxon p-values",
        "unseen_grade_policy": "16 records retained in primary analysis; exclusion reported descriptively only",
        "results": results,
    }
    dump_json(summary, OUT / "group_cv_oof_statistical_summary.json")
    print(pd.DataFrame(flat_rows).to_string(index=False))


if __name__ == "__main__":
    main()
