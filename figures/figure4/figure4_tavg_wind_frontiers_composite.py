from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

WIND_ROOT = (
    REPO_ROOT
    / "src"
    / "core"
    / "wind_2x2"
)

SEED_FILE = (
    WIND_ROOT
    / "multiseed"
    / "summary"
    / "ten_seed_four_configuration_avg_mape.csv"
)

EFFECT_FILE = (
    WIND_ROOT
    / "multiseed"
    / "summary"
    / "effect_cross_seed_statistics.csv"
)

OUTPUT_DIR = SCRIPT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for required_file in (SEED_FILE, EFFECT_FILE):
    if not required_file.exists():
        raise FileNotFoundError(f"Required result file not found:\n{required_file}")


# =========================================================
# Global style
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
PURPLE = "#7A68A6"
GRAY = "#6B7280"
LIGHT_GRAY = "#D9DEE5"
VERY_LIGHT_GRAY = "#EEF1F4"


def clean_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(length=3, color="#666666")
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            color="#E7E9EC",
            linewidth=2.0,
            zorder=0,
        )
    ax.set_axisbelow(True)


# =========================================================
# Load and validate formal results
# =========================================================
seed_df = pd.read_csv(SEED_FILE)
effect_df = pd.read_csv(EFFECT_FILE)

required_seed_columns = ["seed", "O5", "O6", "W6", "W7"]
missing_seed_columns = [c for c in required_seed_columns if c not in seed_df.columns]
if missing_seed_columns:
    raise ValueError(f"Missing columns in seed file: {missing_seed_columns}")

if len(seed_df) != 10:
    raise ValueError(f"Expected 10 training seeds, found {len(seed_df)}")

required_effect_columns = [
    "name",
    "mean",
    "bootstrap_mean_ci95_low",
    "bootstrap_mean_ci95_high",
]
missing_effect_columns = [c for c in required_effect_columns if c not in effect_df.columns]
if missing_effect_columns:
    raise ValueError(f"Missing columns in effect file: {missing_effect_columns}")

seed42 = seed_df.loc[seed_df["seed"] == 42]
if len(seed42) != 1:
    raise ValueError("The seed-42 row is missing or duplicated.")
seed42 = seed42.iloc[0]


# =========================================================
# Figure layout
# =========================================================
fig = plt.figure(figsize=(7.1, 6.9), constrained_layout=False)
ax1 = fig.add_axes([0.09, 0.62, 0.37, 0.27])
ax2 = fig.add_axes([0.56, 0.62, 0.37, 0.27])
ax3 = fig.add_axes([0.30, 0.145, 0.41, 0.27])


# =========================================================
# (a) Seed-42 2 × 2 interaction plot
# =========================================================
tavg_x = np.array([0, 1])
no_wind = np.array([seed42["O5"], seed42["O6"]])
with_wind = np.array([seed42["W6"], seed42["W7"]])

ax1.plot(
    tavg_x,
    no_wind,
    marker="o",
    markersize=8,
    linewidth=2.0,
    color=BLUE,
    label="Without Wind",
    zorder=3,
)
ax1.plot(
    tavg_x,
    with_wind,
    marker="o",
    markersize=8,
    linewidth=2.0,
    color=ORANGE,
    label="With Wind",
    zorder=3,
)

config_labels = [
    (0, no_wind[0], "O5"),
    (1, no_wind[1], "O6"),
    (0, with_wind[0], "W6"),
    (1, with_wind[1], "W7"),
]
for x_value, y_value, label in config_labels:
    vertical_offset = 0.035 if label in {"O5", "O6"} else -0.055
    va = "bottom" if vertical_offset > 0 else "top"
    ax1.text(
        x_value,
        y_value + vertical_offset,
        f"{label}\n{y_value:.4f}",
        ha="center",
        va=va,
        fontsize=9,
        color=DARK_BLUE if label.startswith("O") else DARK_ORANGE,
    )

interaction_seed42 = (
    (seed42["W7"] - seed42["W6"])
    - (seed42["O6"] - seed42["O5"])
)

ax1.text(
    0.43,
    0.08,
    f"Interaction = {interaction_seed42:+.4f} pp".replace("-", "−"),
    transform=ax1.transAxes,
    ha="center",
    va="bottom",
    fontsize=9.2,
    bbox=dict(
        boxstyle="round,pad=0.38",
        facecolor="white",
        edgecolor=GRAY,
        linewidth=2.0,
    ),
)

ax1.set_title("(A) Seed-42 Tavg × Wind interaction", fontsize=11, fontweight="bold")
ax1.set_ylabel("Average MAPE (%)")
ax1.set_xlabel("Tavg included")
ax1.set_xticks([0, 1], ["No", "Yes"])
ax1.set_xlim(-0.25, 1.25)
ax1.set_ylim(17.65, 18.47)
ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
fig.legend(
    ax1_handles, ax1_labels, loc="center",
    bbox_to_anchor=(0.27, 0.53), ncol=2, frameon=False,
    fontsize=9,
)
clean_axis(ax1, grid_axis="y")


# =========================================================
# (b) Ten-seed paired configuration profiles
# =========================================================
configurations = ["O5", "O6", "W6", "W7"]
config_x = np.arange(len(configurations))
config_colors = [BLUE, ORANGE, PURPLE, DARK_GREEN]

for _, row in seed_df.iterrows():
    values = row[configurations].to_numpy(dtype=float)
    is_seed42 = int(row["seed"]) == 42
    ax2.plot(
        config_x,
        values,
        color=BLUE if is_seed42 else GRAY,
        linewidth=2.0,
        alpha=0.62 if is_seed42 else 0.43,
        zorder=1 if not is_seed42 else 2,
    )

    for xi, yi, color in zip(config_x, values, config_colors):
        ax2.scatter(
            xi,
            yi,
            s=45 if is_seed42 else 28,
            color=color,
            alpha=1.0 if is_seed42 else 0.72,
            edgecolor="white",
            linewidth=2.0,
            zorder=4,
        )

means = seed_df[configurations].mean(axis=0).to_numpy(dtype=float)
for xi, mean_value in zip(config_x, means):
    ax2.scatter(
        xi,
        mean_value,
        marker="D",
        s=72,
        color="black",
        edgecolor="white",
        linewidth=2.0,
        zorder=6,
    )
    ax2.text(
        xi,
        19.20,
        f"{mean_value:.4f}",
        ha="center",
        va="top",
        fontsize=9,
    )

ax2.set_title("(B) Ten-seed configuration stability", fontsize=11, fontweight="bold")
ax2.set_ylabel("Average MAPE (%)")
ax2.set_xlabel("Climate input configuration")
ax2.set_xticks(config_x, configurations)
ax2.set_xlim(-0.35, 3.35)
ax2.set_ylim(17.72, 19.25)

legend_handles = [
    Line2D(
        [0], [0], color=GRAY, lw=2.0, alpha=0.55,
        marker="o", markerfacecolor=LIGHT_GRAY,
        markeredgecolor="white", markersize=6,
        label="Individual seed",
    ),
    Line2D(
        [0], [0], color=DARK_BLUE, lw=2.0,
        marker="o", markerfacecolor=BLUE,
        markeredgecolor="white", markersize=6,
        label="Seed 42",
    ),
    Line2D(
        [0], [0], color="none", marker="D",
        markerfacecolor="black", markeredgecolor="white",
        markersize=7, label="Cross-seed mean",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="center",
    bbox_to_anchor=(0.75, 0.53),
    ncol=2,
    frameon=False,
)
clean_axis(ax2, grid_axis="y")


# =========================================================
# (c) Six cross-seed effects with bootstrap CIs
# =========================================================
effect_order = [
    "O6_minus_O5",
    "W6_minus_O5",
    "W7_minus_O6",
    "W7_minus_W6",
    "W7_minus_O5",
    "Interaction",
]

effect_labels = {
    "O6_minus_O5": "Add Tavg without Wind\nO6 − O5",
    "W6_minus_O5": "Add Wind without Tavg\nW6 − O5",
    "W7_minus_O6": "Add Wind with Tavg\nW7 − O6",
    "W7_minus_W6": "Add Tavg with Wind\nW7 − W6",
    "W7_minus_O5": "Joint difference\nW7 − O5",
    "Interaction": "Interaction\n(W7 − W6) − (O6 − O5)",
}

effect_lookup = effect_df.set_index("name")
missing_effects = [name for name in effect_order if name not in effect_lookup.index]
if missing_effects:
    raise ValueError(f"Missing effects in effect summary: {missing_effects}")

means_effect = np.array([effect_lookup.loc[name, "mean"] for name in effect_order])
ci_lows = np.array([
    effect_lookup.loc[name, "bootstrap_mean_ci95_low"] for name in effect_order
])
ci_highs = np.array([
    effect_lookup.loc[name, "bootstrap_mean_ci95_high"] for name in effect_order
])

y_positions = np.arange(len(effect_order))[::-1]
point_colors = [
    DARK_GREEN if value < 0 else DARK_ORANGE
    for value in means_effect
]

ax3.axvline(0, color=GRAY, linestyle="--", linewidth=2.0, zorder=0)

for y, name, mean_value, low, high, color in zip(
    y_positions,
    effect_order,
    means_effect,
    ci_lows,
    ci_highs,
    point_colors,
):
    ax3.errorbar(
        mean_value,
        y,
        xerr=[[mean_value - low], [high - mean_value]],
        fmt="o" if name != "Interaction" else "D",
        markersize=7.5,
        color=color,
        ecolor="#303030",
        elinewidth=2.0,
        capsize=4,
        markeredgecolor="white",
        markeredgewidth=2.0,
        zorder=3,
    )
    ax3.text(
        1.00,
        y,
        f"{mean_value:+.4f}\n[{low:+.4f}, {high:+.4f}]",
        transform=ax3.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=9,
        color="#303030",
        clip_on=False,
    )

fig.text(0.09, 0.46, "(C) Cross-seed paired effects with 95% bootstrap CIs", fontsize=11, fontweight="bold")
ax3.set_xlabel("Δ Average MAPE (pp)")
ax3.set_yticks(
    y_positions,
    [effect_labels[name] for name in effect_order],
)
ax3.tick_params(axis="y", labelsize=9)
ax3.set_xlim(-0.35, 0.35)
ax3.set_ylim(-0.65, len(effect_order) - 0.35)
for tick in ax3.get_xticks():
    if abs(tick) > 1e-10:
        ax3.axvline(tick, color="#E7E9EC", linewidth=2.0, zorder=0)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.spines["bottom"].set_color("#666666")
ax3.tick_params(axis="y", length=0)
ax3.tick_params(axis="x", length=3, color="#666666")

ax3.text(
    0.02,
    -0.35,
    "← Negative difference",
    transform=ax3.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    color=DARK_GREEN,
    clip_on=False,
)
ax3.text(
    0.98,
    -0.35,
    "Positive difference →",
    transform=ax3.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    color=DARK_ORANGE,
    clip_on=False,
)


# =========================================================
# Save
# =========================================================
png_file = OUTPUT_DIR / "Figure4_Tavg_Wind_2x2_composite.png"
pdf_file = OUTPUT_DIR / "Figure4_Tavg_Wind_2x2_composite.pdf"
svg_file = OUTPUT_DIR / "Figure4_Tavg_Wind_2x2_composite.svg"

fig.savefig(png_file, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(pdf_file, bbox_inches="tight", facecolor="white")
fig.savefig(svg_file, bbox_inches="tight", facecolor="white")

print("Loaded formal results from:")
print(SEED_FILE)
print(EFFECT_FILE)
print("\nFigure files saved:")
print(png_file)
print(pdf_file)
print(svg_file)

plt.show()
