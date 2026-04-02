"""
Quantum vs Classical Model Comparison Chart
============================================
Reads reports/tables/model_comparison.csv and produces a grouped bar chart
at reports/figures/quantum_vs_classical.png showing ROC-AUC for all 7 models.

  • Classical models  → steel blue
  • Quantum models    → burnt orange
  • Dashed line at 0.5 (random baseline)
  • Bar value labels above each bar
  • Clean report-ready style (no chart junk)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all envs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
CSV_PATH = Path("reports/tables/model_comparison.csv")
OUT_PATH = Path("reports/figures/quantum_vs_classical.png")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# Shorten model names for x-axis tick labels
LABEL_MAP = {
    "Logistic Regression"      : "Logistic\nRegression",
    "Random Forest"            : "Random\nForest",
    "Gradient Boosting"        : "Gradient\nBoosting",
    "Classical SVM (RBF)"      : "Classical\nSVM (RBF)",
    "Quantum Kernel SVM (4q)"  : "Q-Kernel\nSVM (4q)",
    "Quantum Kernel SVM (6q)"  : "Q-Kernel\nSVM (6q)",
    "VQC (4q)"                 : "VQC\n(4q)",
}
df["Label"] = df["Model"].map(LABEL_MAP)

# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE   = "#3A7EBF"   # classical
ORANGE = "#E07B39"   # quantum
EDGE   = "#1a1a1a"

def bar_color(model_type: str) -> str:
    return BLUE if model_type == "Classical" else ORANGE

df["Color"] = df["Type"].apply(bar_color)

# ── Figure setup ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "sans-serif",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.color"       : "#e0e0e0",
    "grid.linewidth"   : 0.8,
    "axes.axisbelow"   : True,
})

fig, ax = plt.subplots(figsize=(13, 6.5))

x     = np.arange(len(df))
width = 0.58

# ── Bars ───────────────────────────────────────────────────────────────────────
bars = ax.bar(
    x, df["ROC-AUC"],
    width=width,
    color=df["Color"],
    edgecolor=EDGE,
    linewidth=0.7,
    zorder=3,
)

# ── Value labels above each bar ────────────────────────────────────────────────
for bar, val in zip(bars, df["ROC-AUC"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.013,
        f"{val:.4f}",
        ha="center", va="bottom",
        fontsize=9.5, fontweight="bold",
        color="#222222",
    )

# ── Random-baseline reference line ────────────────────────────────────────────
ax.axhline(
    y=0.5,
    color="#888888",
    linewidth=1.4,
    linestyle="--",
    zorder=2,
    label="Random baseline (AUC = 0.50)",
)

# ── Vertical separator between Classical and Quantum groups ───────────────────
n_classical = (df["Type"] == "Classical").sum()
ax.axvline(
    x=n_classical - 0.5,
    color="#bbbbbb",
    linewidth=1.2,
    linestyle=":",
    zorder=2,
)

# Group labels below the separator
ax.text(
    (n_classical - 1) / 2, -0.115,
    "Classical Models",
    ha="center", va="center",
    fontsize=10, color=BLUE, fontweight="semibold",
    transform=ax.get_xaxis_transform(),
)
ax.text(
    n_classical + (len(df) - n_classical - 1) / 2, -0.115,
    "Quantum Models",
    ha="center", va="center",
    fontsize=10, color=ORANGE, fontweight="semibold",
    transform=ax.get_xaxis_transform(),
)

# ── Axes formatting ────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(df["Label"], fontsize=10.5)
ax.set_xlim(-0.6, len(df) - 0.4)
ax.set_ylim(0, 1.05)
ax.set_ylabel("ROC-AUC", fontsize=12, labelpad=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

# ── Title ──────────────────────────────────────────────────────────────────────
ax.set_title(
    "Quantum vs Classical Wildfire Risk Prediction",
    fontsize=14, fontweight="bold", pad=14,
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=BLUE,   edgecolor=EDGE, linewidth=0.7,
                   label="Classical model"),
    mpatches.Patch(facecolor=ORANGE, edgecolor=EDGE, linewidth=0.7,
                   label="Quantum model"),
    plt.Line2D([0], [0], color="#888888", linewidth=1.4, linestyle="--",
               label="Random baseline (AUC = 0.50)"),
]
ax.legend(
    handles=legend_handles,
    loc="upper right",
    framealpha=0.92,
    edgecolor="#cccccc",
    fontsize=10,
)

# ── Caption-style annotation ───────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    "Val year: 2021  |  Train: 2018-2020  |  "
    "Quantum models: 800 subsampled train samples  |  Classical models: 7,779 train samples",
    ha="center", fontsize=8.5, color="#555555",
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

# ── Save ───────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Chart saved → {OUT_PATH}")
