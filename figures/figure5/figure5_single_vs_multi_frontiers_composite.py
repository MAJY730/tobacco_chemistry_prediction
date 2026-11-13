from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

RESULT_ROOT = (
    REPO_ROOT
    / "src"
    / "core"
    / "single_target"
    / "results"
    / "seed42"
)

TARGET_STATS_FILE = (
    RESULT_ROOT
    / "statistical_validation"
    / "per_target_cluster_statistics.csv"
)

OVERALL_STATS_FILE = (
    RESULT_ROOT
    / "statistical_validation"
    / "overall_cluster_statistics.csv"
)

PARAMETER_FILE = (
    RESULT_ROOT
    / "summary"
    / "parameter_comparison.csv"
)

OUTPUT_DIR = SCRIPT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for required_file in (TARGET_STATS_FILE, OVERALL_STATS_FILE, PARAMETER_FILE):
    if not required_file.exists():
        raise FileNotFoundError(f"Required result file not found:\n{required_file}")


# =========================================================
# Style
# =========================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
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
DARK_BLUE = "#1F4E79"
ORANGE = "#D89A3D"
DARK_ORANGE = "#B87324"
GREEN = "#4F8A5F"
DARK_GREEN = "#2E6B3A"
GRAY = "#6B7280"
LIGHT_GRAY = "#D9DEE5"
VERY_LIGHT_GRAY = "#EEF1F4"


def clean_axis(ax, grid_axis="x"):
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
target_df = pd.read_csv(TARGET_STATS_FILE)
overall_df = pd.read_csv(OVERALL_STATS_FILE)
parameter_df = pd.read_csv(PARAMETER_FILE)

target_columns = [
    "target_name",
    "delta_mape_pp",
    "bootstrap_ci95_low_pp",
    "bootstrap_ci95_high_pp",
    "permutation_p_holm",
]
missing = [column for column in target_columns if column not in target_df.columns]
if missing:
    raise ValueError(f"Missing target-statistics columns: {missing}")

if len(target_df) != 5:
    raise ValueError(f"Expected five target rows, found {len(target_df)}")

if len(overall_df) != 1:
    raise ValueError(f"Expected one overall row, found {len(overall_df)}")

parameter_columns = ["setting", "models_required", "total_parameters"]
missing = [column for column in parameter_columns if column not in parameter_df.columns]
if missing:
    raise ValueError(f"Missing parameter columns: {missing}")


# Fixed scientific order
target_order = [
    "total_sugar",
    "reducing_sugar",
    "nicotine",
    "total_nitrogen",
    "potassium",
]

target_labels = {
    "total_sugar": "Total sugar",
    "reducing_sugar": "Reducing sugar",
    "nicotine": "Nicotine",
    "total_nitrogen": "Total nitrogen",
    "potassium": "Potassium",
}

target_lookup = target_df.set_index("target_name")
missing_targets = [target for target in target_order if target not in target_lookup.index]
if missing_targets:
    raise ValueError(f"Missing target rows: {missing_targets}")


# =========================================================
# Figure layout
# =========================================================
fig = plt.figure(figsize=(7.1, 5.7), constrained_layout=False)
ax1 = fig.add_axes([0.12, 0.56, 0.44, 0.34])
ax2 = fig.add_axes([0.62, 0.56, 0.32, 0.34])
ax3 = fig.add_axes([0.30, 0.20, 0.60, 0.16])


# =========================================================
# (a) Per-target differences and cluster-bootstrap CIs
# =========================================================
means = np.array([
    target_lookup.loc[target, "delta_mape_pp"]
    for target in target_order
])
ci_lows = np.array([
    target_lookup.loc[target, "bootstrap_ci95_low_pp"]
    for target in target_order
])
ci_highs = np.array([
    target_lookup.loc[target, "bootstrap_ci95_high_pp"]
    for target in target_order
])

y_positions = np.arange(len(target_order))[::-1]

ax1.axvline(0, color=GRAY, linestyle="--", linewidth=2.0, zorder=0)

for y, target, mean, low, high in zip(
    y_positions,
    target_order,
    means,
    ci_lows,
    ci_highs,
):
    color = DARK_GREEN if mean > 0 else DARK_ORANGE
    ax1.errorbar(
        mean,
        y,
        xerr=[[mean - low], [high - mean]],
        fmt="o",
        markersize=8,
        color=color,
        ecolor="#303030",
        elinewidth=2.0,
        capsize=4,
        markeredgecolor="white",
        markeredgewidth=2.0,
        zorder=3,
    )

ax1.set_title("(A) Target-specific ST − MT differences", fontsize=11, fontweight="bold")
ax1.set_xlabel("ST − MT MAPE difference (pp)")
ax1.set_yticks(
    y_positions,
    [target_labels[target] for target in target_order],
)
ax1.set_xlim(-0.95, 1.85)
ax1.set_ylim(-0.65, len(target_order) - 0.35)
for tick in ax1.get_xticks():
    if abs(tick) > 1e-10:
        ax1.axvline(tick, color="#E7E9EC", linewidth=2.0, zorder=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["bottom"].set_color("#666666")
ax1.tick_params(axis="y", length=0)
ax1.tick_params(axis="x", length=3, color="#666666")
ax1.text(
    0.02,
    -0.21,
    "← ST lower error",
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    color=DARK_ORANGE,
    clip_on=False,
)
ax1.text(
    0.98,
    -0.21,
    "MT lower error →",
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    color=DARK_GREEN,
    clip_on=False,
)


# =========================================================
# (b) Overall difference and performance context
# =========================================================
overall = overall_df.iloc[0]
overall_delta = float(overall["delta_mape_pp"])
overall_low = float(overall["bootstrap_ci95_low_pp"])
overall_high = float(overall["bootstrap_ci95_high_pp"])
permutation_p = float(overall["permutation_p_raw"])

# Exact overall Avg MAPE values from the formal comparison
mt_avg_mape = 17.8271
st_avg_mape = 18.1482

# Compact dumbbell comparison at the top
ax2.plot(
    [mt_avg_mape, st_avg_mape],
    [1.10, 1.10],
    color=GRAY,
    linewidth=2.0,
    zorder=1,
)
ax2.scatter(mt_avg_mape, 1.10, s=80, color=DARK_GREEN,
            edgecolor="white", linewidth=2.0, zorder=3)
ax2.scatter(st_avg_mape, 1.10, s=80, color=ORANGE,
            edgecolor="white", linewidth=2.0, zorder=3)
ax2.text(mt_avg_mape, 1.24, f"MT\n{mt_avg_mape:.4f}%",
         ha="center", va="bottom", fontsize=9, color=DARK_GREEN)
ax2.text(st_avg_mape, 1.24, f"ST\n{st_avg_mape:.4f}%",
         ha="center", va="bottom", fontsize=9, color=DARK_ORANGE)

# Difference and bootstrap CI on a dedicated lower scale
inset = ax2.inset_axes([0.10, 0.00, 0.82, 0.30])
inset.axvline(0, color=GRAY, linestyle="--", linewidth=2.0, zorder=0)
inset.errorbar(
    overall_delta,
    0,
    xerr=[[overall_delta - overall_low], [overall_high - overall_delta]],
    fmt="D",
    markersize=9,
    color=DARK_GREEN,
    ecolor="#303030",
    elinewidth=2.0,
    capsize=4,
    markeredgecolor="white",
    markeredgewidth=2.0,
    zorder=3,
)
inset.set_xlim(-0.10, 0.72)
inset.set_ylim(-1.0, 0.2)
inset.set_yticks([])
inset.set_xlabel("Overall ST − MT\ndifference (pp)", fontsize=9)
inset.tick_params(axis="x", labelsize=9)
inset.spines["top"].set_visible(False)
inset.spines["right"].set_visible(False)
inset.spines["left"].set_visible(False)
inset.spines["bottom"].set_color("#666666")
for tick in inset.get_xticks():
    if abs(tick) > 1e-10:
        inset.axvline(tick, color="#E7E9EC", linewidth=2.0, zorder=0)

ax2.text(
    0.50,
    0.58,
    f"Δ = +{overall_delta:.4f} pp\n"
    f"95% CI [{overall_low:.4f}, {overall_high:.4f}]",
    transform=ax2.transAxes,
    ha="center",
    va="top",
    fontsize=8.6,
    bbox=dict(
        boxstyle="round,pad=0.16",
        facecolor="white",
        edgecolor=DARK_GREEN,
        linewidth=2.0,
    ),
)

ax2.set_title("(B) Overall ST − MT difference", fontsize=11, fontweight="bold")
ax2.set_xlim(17.70, 18.28)
ax2.set_ylim(0.0, 1.55)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)


# =========================================================
# (c) Total trainable parameters for the five-target task
# =========================================================
parameter_lookup = parameter_df.set_index("setting")
mt_parameters = int(parameter_lookup.loc["Multi-target", "total_parameters"])
st_parameters = int(
    parameter_lookup.loc["Five Single-target models", "total_parameters"]
)
parameter_ratio = 100.0 * mt_parameters / st_parameters
parameter_reduction = 100.0 - parameter_ratio

settings = ["Multi-target\n(1 model)", "Five single-target models\n(5 models)"]
parameter_values = [mt_parameters, st_parameters]
y = np.arange(2)[::-1]

bars = ax3.barh(
    y,
    parameter_values,
    height=0.52,
    color=[DARK_GREEN, ORANGE],
    edgecolor="black",
    linewidth=2.0,
    zorder=3,
)

for bar, value in zip(bars, parameter_values):
    ax3.text(
        value + 18000,
        bar.get_y() + bar.get_height() / 2,
        f"{value:,}",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
    )

ax3.text(
    0.51,
    0.58,
    f"MT = {parameter_ratio:.2f}% of five-ST total\n"
    f"Parameter reduction: {parameter_reduction:.2f}%",
    transform=ax3.transAxes,
    ha="center",
    va="bottom",
    fontsize=8.5,
    color=DARK_GREEN,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor=DARK_GREEN,
        linewidth=2.0,
    ),
)

fig.text(0.08, 0.405, "(C) Parameter sharing efficiency", fontsize=11, fontweight="bold")
ax3.set_xlabel("Total trainable parameters")
ax3.set_yticks(y, settings)
ax3.set_xlim(0, 1_030_000)
ax3.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda value, _: f"{value / 1000:.0f}k")
)
clean_axis(ax3, grid_axis="x")
# =========================================================
# Save
# =========================================================
png_file = OUTPUT_DIR / "Figure5_single_vs_multi_composite.png"
pdf_file = OUTPUT_DIR / "Figure5_single_vs_multi_composite.pdf"
svg_file = OUTPUT_DIR / "Figure5_single_vs_multi_composite.svg"

fig.savefig(png_file, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(pdf_file, bbox_inches="tight", facecolor="white")
fig.savefig(svg_file, bbox_inches="tight", facecolor="white")

print("Loaded formal results from:")
print(TARGET_STATS_FILE)
print(OVERALL_STATS_FILE)
print(PARAMETER_FILE)
print("\nFigure files saved:")
print(png_file)
print(pdf_file)
print(svg_file)

plt.show()
