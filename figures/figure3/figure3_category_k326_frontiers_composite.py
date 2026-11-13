import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# Global style
# =========================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.linewidth": 2.0,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.minor.width": 2.0,
    "ytick.minor.width": 2.0,
    "lines.markeredgewidth": 2.0,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

# Colors consistent with Figure 1 / Figure 2
BLUE = "#5B8CCB"
DARK_BLUE = "#1F4E79"
GREEN = "#7AA35A"
DARK_GREEN = "#2E6B3A"
ORANGE = "#D89A3D"
DARK_ORANGE = "#B87324"
RED = "#C85A54"
DARK_RED = "#A63D40"
GRAY = "#6B7280"
LIGHT_GRAY = "#D9DEE5"

# =========================================================
# Data
# =========================================================

# ---------------------------------------------------------
# Panel (a): information-source ablation and category-prior baseline
# ---------------------------------------------------------
# Climate-only:
#   remove grade and retain county-year climate sequence
#
# Grade-mean:
#   use training-set grade-specific mean chemistry values only
#
# Climate + Grade:
#   full input setting
info_labels = ["Climate-only", "Grade-mean", "Climate + Grade"]
info_values = [22.0887, 20.1113, 17.8271]

# ---------------------------------------------------------
# Panel (b): category-encoding comparison
# ---------------------------------------------------------
encoding_labels = ["No Grade", "One-hot", "Learnable embedding"]
encoding_values = [22.0887, 18.1150, 17.8271]

# ---------------------------------------------------------
# Panel (c): K326 five-seed comparison
# ---------------------------------------------------------
seeds = [42, 123, 456, 789, 2024]

climate_only = [
    20.6813,
    20.6501,
    20.4997,
    20.2872,
    20.5047
]

climate_grade = [
    16.6135,
    16.3906,
    16.8589,
    16.7121,
    16.0874
]

# Difference = Grade-aware - Climate-only
seed_diff = np.array(climate_grade) - np.array(climate_only)
mean_diff = seed_diff.mean()
sd_diff = seed_diff.std(ddof=1)

# =========================================================
# Panel (d) data
# =========================================================
# CSV must contain one column:
# delta_mape_pp
#
# where:
# delta_mape_pp = Grade-aware mean APE - Climate-only mean APE
#
# Negative values favor grade-aware modeling.
#
# File should contain 47 county-year rows.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

cy_file = (
    REPO_ROOT
    / "src"
    / "core"
    / "k326_grade_validation"
    / "summary"
    / "group_cv_oof_cluster_differences.csv"
)

output_dir = SCRIPT_DIR
output_dir.mkdir(parents=True, exist_ok=True)

if not cy_file.exists():
    raise FileNotFoundError(
        f"County-year result file not found:\n{cy_file}"
    )

cy_df = pd.read_csv(cy_file)

required_column = "delta_mape_pp"

if required_column not in cy_df.columns:
    raise ValueError(
        f"Column '{required_column}' was not found.\n"
        f"Available columns: {list(cy_df.columns)}"
    )

cy_delta = cy_df[required_column].to_numpy(dtype=float)

if len(cy_delta) != 47:
    raise ValueError(
        f"Expected 47 county-year groups, but found {len(cy_delta)}."
    )

print(f"Loaded {len(cy_delta)} county-year differences from:")
print(cy_file)

# Confirmed summary statistics for panel (d)
cy_favor_grade = 36
cy_total = 47

cy_median = -2.2659
cy_iqr_low = -5.2264
cy_iqr_high = -0.2282

cy_equal_mean = -2.9527
ci_low = -4.5756
ci_high = -1.4395

# =========================================================
# Composite layout: A/B side by side; C/D full width
# =========================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.linewidth": 2.0,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

fig = plt.figure(figsize=(7.1, 6.85), facecolor="white")
gs = fig.add_gridspec(
    3, 2,
    height_ratios=[1.0, 1.3, 1.5],
    left=0.11, right=0.985, top=0.855, bottom=0.13,
    wspace=0.28, hspace=0.88,
)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])
ax4 = fig.add_subplot(gs[2, :])

def draw_bar_panel(ax, labels, values, title):
    x = np.arange(len(labels))
    bars = ax.bar(
        x, values, width=0.58,
        color=[BLUE, ORANGE, DARK_GREEN],
        edgecolor="black", linewidth=2.0,
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=57)
    ax.set_ylabel("Average MAPE (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 25)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.tick_params(labelsize=9)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35, f"{value:.2f}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

draw_bar_panel(
    ax1, ["Climate-only", "Grade-mean", "Climate +\nGrade"], info_values,
    "(A) Information-source ablation\nand category-prior baseline",
)
draw_bar_panel(
    ax2, ["No Grade", "One-hot", "Learnable\nembedding"], encoding_values,
    "(B) Category encoding\ncomparison",
)

# Pairwise brackets sit above the unchanged 0–25% y-axis.
bracket_transform = ax2.get_xaxis_transform()
for x_left, x_right, y_base, label in [
    (0, 1, 1.06, "−3.97 pp"),
    (1, 2, 1.30, "−0.29 pp"),
]:
    ax2.plot(
        [x_left, x_left, x_right, x_right],
        [y_base, y_base + 0.055, y_base + 0.055, y_base],
        transform=bracket_transform, color=GRAY,
        linewidth=2.0, clip_on=False,
    )
    ax2.text(
        (x_left + x_right) / 2, y_base + 0.075, label,
        transform=bracket_transform, ha="center", va="bottom",
        color=DARK_BLUE, fontsize=9, fontweight="bold",
        clip_on=False,
    )

# Panel C: the same five paired seed results, shown as horizontal dumbbells.
y_seed = np.arange(len(seeds))[::-1]
for yi, climate_value, grade_value in zip(y_seed, climate_only, climate_grade):
    ax3.plot(
        [grade_value, climate_value], [yi, yi],
        color=GRAY, linewidth=2.0, linestyle=":", zorder=1,
    )
ax3.scatter(
    climate_only, y_seed, s=58, color=DARK_BLUE,
    edgecolor="white", linewidth=2.0,
    label="Climate-only", zorder=3,
)
ax3.scatter(
    climate_grade, y_seed, s=58, color=DARK_GREEN,
    edgecolor="white", linewidth=2.0,
    label="Grade-aware", zorder=3,
)
for yi, climate_value, grade_value in zip(y_seed, climate_only, climate_grade):
    ax3.text(
        climate_value + 0.18, yi, f"{climate_value:.2f}",
        ha="left", va="center", color=DARK_BLUE, fontsize=9,
    )
    ax3.text(
        grade_value - 0.18, yi, f"{grade_value:.2f}",
        ha="right", va="center", color=DARK_GREEN, fontsize=9,
    )
ax3.set_title(
    "(C) K326 fixed-split stability across five training seeds",
    loc="left", pad=8, fontsize=11, fontweight="bold",
)
ax3.set_ylabel("Training seed", fontsize=10)
ax3.set_xlabel("Average MAPE (%)", fontsize=10)
ax3.set_yticks(y_seed, seeds)
ax3.set_ylim(-0.55, 6.8)
ax3.set_xlim(14.6, 22.2)
ax3.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
ax3.text(
    0.98, 0.98,
    f"Mean difference = {mean_diff:.2f} ± {sd_diff:.2f} pp\n"
    "Same direction in 5/5 seeds",
    transform=ax3.transAxes, ha="right", va="top", fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor="white",
        edgecolor=DARK_BLUE, linewidth=2.0,
    ),
)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Panel D: the original 47 differences, axis range and summary statistics.
ax4.set_title(
    "(D) County-year paired differences in K326 Group CV",
    loc="left", pad=8, fontsize=11, fontweight="bold",
)
ax4.set_xlabel("Δ mean APE = Grade-aware − Climate-only (pp)", fontsize=10)
ax4.axvline(0, color=GRAY, linestyle="--", linewidth=2.0, zorder=0)
rng = np.random.default_rng(42)
y = rng.uniform(0.33, 0.62, size=len(cy_delta))
point_colors = np.where(cy_delta < 0, DARK_GREEN, ORANGE)
ax4.scatter(
    cy_delta, y, s=34, alpha=0.78, c=point_colors,
    edgecolor="white", linewidth=2.0, zorder=3,
)
summary_y = 0.16
ax4.errorbar(
    cy_equal_mean, summary_y,
    xerr=[[cy_equal_mean - ci_low], [ci_high - cy_equal_mean]],
    fmt="D", markersize=9, color=DARK_GREEN,
    ecolor="black", elinewidth=2.0, capsize=4, capthick=2.0,
    zorder=5,
)
ax4.text(
    0.02, 0.96,
    f"{cy_favor_grade} / {cy_total} county-year groups\n"
    "favored grade-aware modeling",
    transform=ax4.transAxes, ha="left", va="top", fontsize=9, color=DARK_GREEN,
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor="white",
        edgecolor=DARK_GREEN, linewidth=2.0,
    ),
)
ax4.text(
    0.98, 0.96,
    f"Median Δ = {cy_median:.2f} pp\n"
    f"IQR = [{cy_iqr_low:.2f}, {cy_iqr_high:.2f}] pp",
    transform=ax4.transAxes, ha="right", va="top", fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor="white",
        edgecolor=GRAY, linewidth=2.0,
    ),
)
ax4.text(
    0.02, 0.08,
    f"CY-equal mean = {cy_equal_mean:.2f} pp\n"
    f"95% bootstrap CI [{ci_low:.2f}, {ci_high:.2f}] pp",
    transform=ax4.transAxes, ha="left", va="bottom", fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor="white",
        edgecolor=DARK_GREEN, linewidth=2.0,
    ),
)
ax4.text(
    0.03, -0.43, "← Favors grade-aware modeling",
    transform=ax4.transAxes, ha="left", va="top", fontsize=9, color=DARK_GREEN,
)
ax4.text(
    0.97, -0.43, "Favors climate-only modeling →",
    transform=ax4.transAxes, ha="right", va="top", fontsize=9, color=DARK_ORANGE,
)
xmin = min(cy_delta.min(), ci_low) - 1.5
xmax = max(cy_delta.max(), ci_high) + 1.5
ax4.set_xlim(xmin, xmax)
ax4.set_ylim(0, 1)
ax4.set_yticks([])
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["left"].set_visible(False)

png_file = output_dir / "Figure3_category_information_K326_composite.png"
pdf_file = output_dir / "Figure3_category_information_K326_composite.pdf"
svg_file = output_dir / "Figure3_category_information_K326_composite.svg"
fig.savefig(png_file, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(pdf_file, bbox_inches="tight", facecolor="white")
fig.savefig(svg_file, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Figure files saved:")
print(png_file)
print(pdf_file)
print(svg_file)
