from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_ROOT = (
    REPO_ROOT
    / "src"
    / "core"
    / "group_cv"
    / "summary"
)

EVALUATION_FILE = SUMMARY_ROOT / "evaluation_comparison.csv"
FOLD_FILE = SUMMARY_ROOT / "fold_model_comparison.csv"
POOLED_FILE = SUMMARY_ROOT / "pooled_oof_metrics_all_models.csv"
GROUP_WEIGHTED_FILE = SUMMARY_ROOT / "group_weighted_metrics.csv"
CLUSTER_STATS_FILE = (
    SUMMARY_ROOT
    / "oof_cluster_statistics"
    / "group_cv_oof_cluster_statistics.csv"
)

OUTPUT_DIR = SCRIPT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for required_file in (
    EVALUATION_FILE,
    FOLD_FILE,
    POOLED_FILE,
    GROUP_WEIGHTED_FILE,
    CLUSTER_STATS_FILE,
):
    if not required_file.exists():
        raise FileNotFoundError(f"Required result file not found:\n{required_file}")


# =========================================================
# Style
# =========================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.linewidth": 2.0,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.minor.width": 2.0,
    "ytick.minor.width": 2.0,
    "lines.markeredgewidth": 2.0,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

BLUE = "#5B8CCB"
ORANGE = "#D89A3D"
DARK_GREEN = "#2E6B3A"
GRAY = "#6B7280"
LIGHT_GRAY = "#D9DEE5"
DARK_RED = "#A63D40"

MODEL_ORDER = ["grade_gru", "grade_transformer", "ga_wind_tsmixer"]
MODEL_LABELS = {
    "grade_gru": "Grade-GRU",
    "grade_transformer": "Grade-Transformer",
    "ga_wind_tsmixer": "GA-Wind-TSMixer",
}
MODEL_COLORS = {
    "grade_gru": BLUE,
    "grade_transformer": ORANGE,
    "ga_wind_tsmixer": DARK_GREEN,
}
MODEL_MARKERS = {
    "grade_gru": "o",
    "grade_transformer": "s",
    "ga_wind_tsmixer": "D",
}


def clean_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(length=3, color="#666666")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E7E9EC", linewidth=2.0, zorder=0)
    ax.set_axisbelow(True)


# =========================================================
# Load and validate formal results
# =========================================================
evaluation_df = pd.read_csv(EVALUATION_FILE).set_index("model")
fold_df = pd.read_csv(FOLD_FILE)
pooled_df = pd.read_csv(POOLED_FILE).set_index("model")
group_df = pd.read_csv(GROUP_WEIGHTED_FILE).set_index("model")
cluster_df = pd.read_csv(CLUSTER_STATS_FILE).set_index("comparison")

summary_label_y = {
    "grade_gru": (21.21, 21.64),
    "grade_transformer": (21.23, 21.83),
    "ga_wind_tsmixer": (21.55, 22.02),
}

for model in MODEL_ORDER:
    if model not in evaluation_df.index:
        raise ValueError(f"Missing model in evaluation comparison: {model}")
    if model not in pooled_df.index:
        raise ValueError(f"Missing model in pooled metrics: {model}")
    if model not in group_df.index:
        raise ValueError(f"Missing model in group-weighted metrics: {model}")

if sorted(fold_df["fold"].tolist()) != [1, 2, 3, 4, 5]:
    raise ValueError("Fold comparison must contain folds 1–5 exactly once.")


# =========================================================
# Figure layout
# =========================================================
fig = plt.figure(figsize=(7.1, 7.8), constrained_layout=False)
ax1 = fig.add_axes([0.10, 0.67, 0.42, 0.23])
ax2 = fig.add_axes([0.60, 0.67, 0.35, 0.23])
ax3 = fig.add_axes([0.15, 0.42, 0.78, 0.15])
ax4 = fig.add_axes([0.34, 0.15, 0.41, 0.16])


# =========================================================
# (a) Three evaluation settings
# =========================================================
setting_columns = [
    "random_record_seed42",
    "fixed_county_year_group",
    "group_cv_pooled_oof",
]
setting_labels = [
    "Record-level\nrandom split",
    "Fixed county-year\nsplit",
    "5-fold pooled\nOOF",
]
setting_x = np.arange(3)
offsets = [-0.12, 0.0, 0.12]

for offset, model in zip(offsets, MODEL_ORDER):
    values = evaluation_df.loc[model, setting_columns].to_numpy(dtype=float)
    ax1.scatter(
        setting_x + offset,
        values,
        s=72,
        marker=MODEL_MARKERS[model],
        color=MODEL_COLORS[model],
        edgecolor="white",
        linewidth=2.0,
        label=MODEL_LABELS[model],
        zorder=3,
    )
    for index, value in enumerate(values):
        label_y = value + 0.15
        if index == 2:
            label_y = value + {
                "grade_gru": 0.20,
                "grade_transformer": 0.38,
                "ga_wind_tsmixer": 0.16,
            }[model]
        ax1.text(
            setting_x[index] + [-0.28, 0.0, 0.28][MODEL_ORDER.index(model)],
            label_y,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=MODEL_COLORS[model],
        )

ax1.set_title("(A) Performance across evaluation settings")
ax1.set_ylabel("Average MAPE (%)")
ax1.set_xticks(setting_x, setting_labels)
ax1.set_xlim(-0.42, 2.42)
ax1.set_ylim(17.2, 22.1)
fig.legend(
    *ax1.get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
    frameon=False,
    fontsize=9,
)
clean_axis(ax1, grid_axis="y")


# =========================================================
# (b) Five outer-fold results
# =========================================================
folds = fold_df["fold"].to_numpy(dtype=int)

for model in MODEL_ORDER:
    values = fold_df[model].to_numpy(dtype=float)
    ax2.plot(
        folds,
        values,
        marker=MODEL_MARKERS[model],
        markersize=7,
        linewidth=2.0,
        color=MODEL_COLORS[model],
        label=MODEL_LABELS[model],
        zorder=3,
    )

ax2.set_title("(B) Performance across five outer folds")
ax2.set_ylabel("Average MAPE (%)")
ax2.set_xlabel("Outer fold")
ax2.set_xticks(folds)
ax2.set_xlim(0.75, 5.25)
ax2.set_ylim(18.2, 25.8)
clean_axis(ax2, grid_axis="y")


# =========================================================
# (c) Record-weighted and county-year-equal summaries
# =========================================================
summary_x = np.array([0, 1])
summary_labels = ["Record-weighted\npooled OOF", "County-year\nequal-weight"]

for model in MODEL_ORDER:
    record_weighted = float(pooled_df.loc[model, "Avg_MAPE"])
    equal_weighted = float(group_df.loc[model, "group_weighted_Avg_MAPE"])
    values = np.array([record_weighted, equal_weighted])
    ax3.plot(
        summary_x,
        values,
        marker=MODEL_MARKERS[model],
        markersize=7.5,
        linewidth=2.0,
        color=MODEL_COLORS[model],
        label=MODEL_LABELS[model],
        zorder=3,
    )
    for index, (x, value) in enumerate(zip(summary_x, values)):
        label_x = x + (0.10 if model == "grade_transformer" else 0.0)
        if index == 0 and model == "grade_gru":
            label_x -= 0.10
        ax3.text(
            label_x,
            summary_label_y[model][index],
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=MODEL_COLORS[model],
        )

ax3.set_title("(C) OOF aggregation weighting")
ax3.set_ylabel("Average MAPE (%)")
ax3.set_xticks(summary_x, summary_labels)
ax3.set_xlim(-0.28, 1.28)
ax3.set_ylim(21.15, 22.10)
clean_axis(ax3, grid_axis="y")


# =========================================================
# (d) County-year-equal paired differences
# =========================================================
comparisons = ["GA_minus_Grade_GRU", "GA_minus_Grade_Transformer"]
comparison_labels = [
    "GA-Wind-TSMixer − Grade-GRU",
    "GA-Wind-TSMixer − Grade-Transformer",
]
y_positions = np.array([1, 0])

ax4.axvline(0, color=GRAY, linestyle="--", linewidth=2.0, zorder=0)

for y, comparison, label in zip(y_positions, comparisons, comparison_labels):
    row = cluster_df.loc[comparison]
    mean = float(row["county_year_equal_weight_delta_mape"])
    low = float(row["bootstrap_ci_lower"])
    high = float(row["bootstrap_ci_upper"])
    color = BLUE
    ax4.errorbar(
        mean,
        y,
        xerr=[[mean - low], [high - mean]],
        fmt="D",
        markersize=8,
        color=color,
        ecolor="#303030",
        elinewidth=2.0,
        capsize=4,
        markeredgecolor="white",
        markeredgewidth=2.0,
        zorder=3,
    )
    ax4.text(
        1.05,
        y,
        f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]",
        transform=ax4.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=8.0,
        clip_on=False,
    )

fig.text(
    0.50,
    0.33,
    "(D) County-year-equal paired model differences",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
)
ax4.set_xlabel("GA − baseline mean APE difference (pp)")
ax4.set_yticks(y_positions, comparison_labels)
ax4.set_xlim(-0.60, 0.80)
ax4.set_ylim(-0.65, 1.65)
ax4.grid(axis="x", color="#E7E9EC", linewidth=2.0, zorder=0)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.spines["bottom"].set_color("#666666")
ax4.tick_params(axis="y", length=0)
ax4.tick_params(axis="y", labelsize=8.5)
ax4.tick_params(axis="x", length=3, color="#666666")
ax4.text(
    0.02,
    0.05,
    "← GA lower error",
    transform=ax4.transAxes,
    ha="left",
    va="bottom",
    fontsize=8.5,
    color=DARK_GREEN,
)
ax4.text(
    0.98,
    0.05,
    "GA higher error →",
    transform=ax4.transAxes,
    ha="right",
    va="bottom",
    fontsize=8.5,
    color=DARK_RED,
)


# =========================================================
# Layout and save
# =========================================================
png_file = OUTPUT_DIR / "Figure6_county_year_generalization_composite.png"
pdf_file = OUTPUT_DIR / "Figure6_county_year_generalization_composite.pdf"
svg_file = OUTPUT_DIR / "Figure6_county_year_generalization_composite.svg"

fig.savefig(png_file, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(pdf_file, bbox_inches="tight", facecolor="white")
fig.savefig(svg_file, bbox_inches="tight", facecolor="white")

print("Loaded formal results from:")
for path in (
    EVALUATION_FILE,
    FOLD_FILE,
    POOLED_FILE,
    GROUP_WEIGHTED_FILE,
    CLUSTER_STATS_FILE,
):
    print(path)

print("\nFigure files saved:")
print(png_file)
print(pdf_file)
print(svg_file)

plt.show()
