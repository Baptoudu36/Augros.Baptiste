"""
visualization.py — DEPOXY Project
====================================
All publication-quality figure generation functions.

Every function:
  - returns the matplotlib Figure object
  - saves to results/ at 300 dpi when save=True
  - uses the DEPOXY color scheme (A=red, B=teal, C=orange)

Author: Baptiste AUGROS
"""

from __future__ import annotations
from pathlib import Path
import sys
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Avoid shadowing Python stdlib `statistics` by local `sources/statistics.py`
# when this file is executed directly from the sources directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) in sys.path:
    sys.path.remove(str(_HERE))
if "statistics" in sys.modules:
    mod_file = getattr(sys.modules["statistics"], "__file__", "") or ""
    if mod_file.endswith("/sources/statistics.py"):
        del sys.modules["statistics"]
importlib.import_module("statistics")

import seaborn as sns

# DEPOXY color palette
PALETTE = {"A": "#E63946", "B": "#2A9D8F", "C": "#F4A261"}
PALETTE_LIST = [PALETTE["A"], PALETTE["B"], PALETTE["C"]]

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi":    100,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> None:
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {path}")


def _mark_phases(ax: plt.Axes, timeline: dict, alpha: float = 0.08) -> None:
    """Draw alternating shoe-condition bands on a time-series axis."""
    colors = {"A": "#E63946", "B": "#2A9D8F", "C": "#F4A261"}
    for key, info in timeline.items():
        if key.startswith("run_"):
            code = info.get("shoe_code", "")
            ax.axvspan(info["start"] / 60, info["end"] / 60,
                       alpha=alpha, color=colors.get(code, "gray"))


# ---------------------------------------------------------------------------
# 1. Full time-series overview
# ---------------------------------------------------------------------------

def plot_time_series_overview(k5_data, hr_data, nirs_data, fp_data,
                               timeline: dict,
                               figsize: tuple = (18, 18),
                               save: bool = True) -> plt.Figure:
    """6-row x 2-col overview of all physiological time series."""
    fig, axes = plt.subplots(6, 2, figsize=figsize, sharex=False)
    fig.suptitle("DEPOXY — Complete Physiological Time Series Overview",
                 fontsize=14, fontweight="bold", y=1.01)

    tsi_col  = next((c for c in ["TSI_Quad", "SmO2_Quad"] if c in nirs_data.columns), None)
    tsi_calf = next((c for c in ["TSI_Calf", "SmO2_Calf"] if c in nirs_data.columns), None)
    hhb_col  = next((c for c in ["HHb_Quad", "HHb"] if c in nirs_data.columns), None)

    def _mark(ax):
        _mark_phases(ax, timeline)

    def _plot(ax, df, col, ylabel, title, color):
        if col in df.columns:
            ax.plot(df["time_sec"] / 60,
                    pd.to_numeric(df[col], errors="coerce"),
                    color=color, linewidth=1.2, alpha=0.9)
        _mark(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], k5_data,   "VO2_ml_min",  "VO2 (mL/min)",     "Oxygen Consumption",        "#1565C0")
    _plot(axes[0, 1], hr_data,   "HR_bpm",      "HR (bpm)",          "Heart Rate (FC)",            "#C62828")
    _plot(axes[1, 0], k5_data,   "RER",         "RER",               "Respiratory Exchange Ratio", "#6A1B9A")
    _plot(axes[1, 1], k5_data,   "VE_L_min",    "VE (L/min)",        "Minute Ventilation",         "#E65100")
    _plot(axes[2, 0], nirs_data, tsi_col or "", "TSI / SmO2 (%)",    "TSI — Quadriceps",           "#2E7D32")
    _plot(axes[2, 1], nirs_data, tsi_calf or "","TSI / SmO2 (%)",    "TSI — Calf",                 "#388E3C")
    _plot(axes[3, 0], nirs_data, hhb_col or "", "HHb (uM)",          "HHb — Quadriceps",           "#5D4037")

    o2hb_col = next((c for c in ["O2Hb_Quad", "O2Hb"] if c in nirs_data.columns), None)
    _plot(axes[3, 1], nirs_data, o2hb_col or "", "O2Hb (uM)",        "O2Hb — Quadriceps",          "#00838F")

    hbtot_col = next((c for c in ["HbTot_Quad", "tHb_Quad"] if c in nirs_data.columns), None)
    _plot(axes[4, 0], nirs_data, hbtot_col or "", "HbTot (uM)",      "Total Hemoglobin — Quad",    "#37474F")

    ax_fp = axes[4, 1]
    if "total_force_N" in fp_data.columns:
        step = max(1, len(fp_data) // 5000)
        ax_fp.plot(fp_data["time_sec"][::step] / 60,
                   fp_data["total_force_N"][::step],
                   color="#1A237E", linewidth=0.4, alpha=0.7)
    _mark(ax_fp)
    ax_fp.set_ylabel("Force (N)")
    ax_fp.set_title("Total Vertical Force", fontweight="bold")
    ax_fp.grid(True, alpha=0.3)

    _plot(axes[5, 0], k5_data, "VT_L",        "VT (L)",          "Tidal Volume",        "#4527A0")
    _plot(axes[5, 1], k5_data, "VCO2_ml_min", "VCO2 (mL/min)",   "CO2 Production",      "#AD1457")

    for ax in axes.flat:
        ax.set_xlabel("Time (min)")

    legend_patches = [mpatches.Patch(color=v, label=f"Cond. {k}", alpha=0.5)
                      for k, v in PALETTE.items()]
    fig.legend(handles=legend_patches, loc="upper right",
               bbox_to_anchor=(1.0, 1.0), frameon=True)

    plt.tight_layout()
    if save:
        _save(fig, "Fig1_Time_Series_Overview.png")
    return fig


# ---------------------------------------------------------------------------
# 2. Condition-level boxplots
# ---------------------------------------------------------------------------

def plot_condition_boxplots(block_df: pd.DataFrame,
                             variables: list[str],
                             labels: dict[str, str] | None = None,
                             save: bool = True) -> plt.Figure:
    """Boxplot + strip for each variable by shoe condition (A/B/C)."""
    n = len(variables)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    fig.suptitle("Physiological Responses by Shoe Condition",
                 fontsize=13, fontweight="bold")

    order = ["A", "B", "C"]
    pal   = [PALETTE[c] for c in order]
    labels = labels or {}

    for ax, var in zip(axes, variables):
        if var not in block_df.columns:
            ax.set_visible(False)
            continue
        plot_df = block_df[["condition", var]].copy()
        plot_df[var] = pd.to_numeric(plot_df[var], errors="coerce")
        sns.boxplot(data=plot_df, x="condition", y=var,
                    order=order, palette=pal, ax=ax,
                    width=0.5, fliersize=4, linewidth=1.2)
        sns.stripplot(data=plot_df, x="condition", y=var,
                      order=order, color="black", alpha=0.65,
                      size=7, jitter=True, ax=ax)
        ax.set_title(labels.get(var, var), fontweight="bold")
        ax.set_xlabel("Condition")
        ax.set_ylabel(labels.get(var, var))
        ax.grid(True, alpha=0.25, axis="y")

    for ax in axes[len(variables):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save:
        _save(fig, "Fig2_Condition_Boxplots.png")
    return fig


# ---------------------------------------------------------------------------
# 3. AUC bar chart
# ---------------------------------------------------------------------------

def plot_auc_by_condition(block_df: pd.DataFrame,
                           auc_vars: list[str],
                           labels: dict[str, str] | None = None,
                           save: bool = True) -> plt.Figure:
    """Bar chart of mean AUC +/- SD per condition for each variable."""
    order  = ["A", "B", "C"]
    labels = labels or {}
    n = len(auc_vars)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Area Under Curve (AUC) by Shoe Condition",
                 fontsize=13, fontweight="bold")

    for ax, var in zip(axes, auc_vars):
        if var not in block_df.columns:
            continue
        means, stds = [], []
        for cond in order:
            vals = block_df.loc[block_df["condition"] == cond, var].dropna()
            means.append(vals.mean())
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0)
        ax.bar(order, means, yerr=stds, capsize=6,
               color=[PALETTE[c] for c in order],
               edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.set_title(labels.get(var, var), fontweight="bold")
        ax.set_xlabel("Condition")
        ax.set_ylabel(labels.get(var, var))
        ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    if save:
        _save(fig, "Fig3_AUC_By_Condition.png")
    return fig


# ---------------------------------------------------------------------------
# 4. Slope bar charts (TSI, HHb, FC drift)
# ---------------------------------------------------------------------------

def plot_slopes_by_condition(block_df: pd.DataFrame,
                              slope_vars: list[str],
                              labels: dict[str, str] | None = None,
                              save: bool = True) -> plt.Figure:
    """Bar chart of mean slope +/- SD per condition."""
    order  = ["A", "B", "C"]
    labels = labels or {}
    n = len(slope_vars)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Signal Slopes by Shoe Condition",
                 fontsize=13, fontweight="bold")

    for ax, var in zip(axes, slope_vars):
        if var not in block_df.columns:
            continue
        means, stds = [], []
        for cond in order:
            vals = block_df.loc[block_df["condition"] == cond, var].dropna()
            means.append(vals.mean())
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0)
        ax.bar(order, means, yerr=stds, capsize=6,
               color=[PALETTE[c] for c in order],
               edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(labels.get(var, var), fontweight="bold")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Slope")
        ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    if save:
        _save(fig, "Fig4_Slopes_By_Condition.png")
    return fig


# ---------------------------------------------------------------------------
# 5. Baseline comparison
# ---------------------------------------------------------------------------

def plot_baseline_comparison(baseline_df: pd.DataFrame,
                              save: bool = True) -> plt.Figure:
    """Grouped bar chart: initial vs final baseline per metric."""
    metrics = baseline_df["Metric"].tolist()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Baseline Comparison: Initial vs Final",
                 fontsize=13, fontweight="bold")

    for ax, (_, row) in zip(axes, baseline_df.iterrows()):
        vals = [row["Baseline_Initial"], row["Baseline_Final"]]
        cols = ["#455A64", "#F57F17"]
        ax.bar(["Initial", "Final"], vals, color=cols,
               edgecolor="black", linewidth=0.8, alpha=0.85, width=0.5)
        delta = row["Delta_Final_minus_Initial"]
        ax.set_title(row["Metric"], fontweight="bold")
        ax.set_ylabel(row["Metric"])
        ax.grid(True, alpha=0.25, axis="y")
        if not np.isnan(delta):
            ax.annotate(f"Delta = {delta:+.2f}",
                        xy=(0.5, 0.95), xycoords="axes fraction",
                        ha="center", va="top", fontsize=9,
                        color="green" if delta > 0 else "red")

    plt.tight_layout()
    if save:
        _save(fig, "Fig5_Baseline_Comparison.png")
    return fig


# ---------------------------------------------------------------------------
# 6. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                              title: str = "Correlation Matrix",
                              save: bool = True,
                              filename: str = "Fig6_Correlation_Heatmap.png") -> plt.Figure:
    """Annotated seaborn heatmap of a correlation matrix."""
    fig, ax = plt.subplots(figsize=(max(6, len(corr_matrix)), max(5, len(corr_matrix))))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, square=True, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 7. Force sensor overview
# ---------------------------------------------------------------------------

def plot_force_sensors(fp_data: pd.DataFrame,
                        timeline: dict,
                        save: bool = True) -> plt.Figure:
    """Individual force sensor traces + total force."""
    sensor_n_cols = [c for c in fp_data.columns if c.endswith("_N") and "sensor" in c]
    n_sensors = len(sensor_n_cols)
    nrows = n_sensors + 1
    fig, axes = plt.subplots(nrows, 1, figsize=(16, 2.5 * nrows), sharex=True)
    fig.suptitle("Vertical Force — Individual Sensors & Total",
                 fontsize=13, fontweight="bold")

    colors_sensors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]
    step = max(1, len(fp_data) // 5000)
    t_min = fp_data["time_sec"][::step] / 60

    for i, col in enumerate(sensor_n_cols):
        ax = axes[i]
        ax.plot(t_min, fp_data[col][::step],
                color=colors_sensors[i % len(colors_sensors)],
                linewidth=0.6, alpha=0.8)
        _mark_phases(ax, timeline)
        ax.set_ylabel("Force (N)")
        ax.set_title(f"Sensor {i+1}", fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3)

    ax_total = axes[-1]
    if "total_force_N" in fp_data.columns:
        ax_total.plot(t_min, fp_data["total_force_N"][::step],
                      color="#1A237E", linewidth=0.8, alpha=0.9)
    _mark_phases(ax_total, timeline)
    ax_total.set_ylabel("Force (N)")
    ax_total.set_title("Total Vertical Force", fontweight="bold", fontsize=9)
    ax_total.set_xlabel("Time (min)")
    ax_total.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        _save(fig, "Fig7_Force_Sensors.png")
    return fig


# ---------------------------------------------------------------------------
# 8. Per-block heatmap
# ---------------------------------------------------------------------------

def plot_block_heatmap(block_df: pd.DataFrame,
                        variables: list[str],
                        labels: dict[str, str] | None = None,
                        save: bool = True) -> plt.Figure:
    """Heatmap of z-scored block-level metrics (rows=blocks, cols=variables)."""
    labels = labels or {}
    avail  = [v for v in variables if v in block_df.columns]
    mat    = block_df.set_index("block")[avail].apply(pd.to_numeric, errors="coerce")
    mat_z  = (mat - mat.mean()) / mat.std()

    fig, ax = plt.subplots(figsize=(max(8, len(avail) * 1.2), 4))
    sns.heatmap(mat_z, annot=mat.round(1), fmt=".1f",
                cmap="RdBu_r", center=0, linewidths=0.5,
                xticklabels=[labels.get(v, v) for v in avail],
                yticklabels=[f"Block {b} ({c})"
                             for b, c in zip(block_df["block"], block_df["condition"])],
                ax=ax, cbar_kws={"label": "Z-score"})
    ax.set_title("Block-Level Physiological Metrics (Z-scored)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "Fig8_Block_Heatmap.png")
    return fig


# ---------------------------------------------------------------------------
# Project runner / validation
# ---------------------------------------------------------------------------

def create_all_project_figures(k5_data: pd.DataFrame,
                               hr_data: pd.DataFrame,
                               nirs_data: pd.DataFrame,
                               fp_data: pd.DataFrame,
                               timeline: dict,
                               block_df: pd.DataFrame,
                               baseline_df: pd.DataFrame,
                               corr_matrix: pd.DataFrame,
                               save: bool = True) -> dict[str, plt.Figure]:
    """Generate the full DEPOXY figure set and return figure handles."""
    figs: dict[str, plt.Figure] = {}

    figs["time_series"] = plot_time_series_overview(
        k5_data, hr_data, nirs_data, fp_data,
        timeline=timeline,
        save=save,
    )

    figs["condition_boxplots"] = plot_condition_boxplots(
        block_df,
        variables=["VO2_mean", "HR_mean", "TSI_mean", "HHb_mean", "Force_mean", "step_freq_bpm"],
        labels={
            "VO2_mean": "VO2 mean (mL/min)",
            "HR_mean": "HR mean (bpm)",
            "TSI_mean": "TSI mean (%)",
            "HHb_mean": "HHb mean (uM)",
            "Force_mean": "Force peak mean (N)",
            "step_freq_bpm": "Step frequency (steps/min)",
        },
        save=save,
    )

    figs["auc"] = plot_auc_by_condition(
        block_df,
        auc_vars=["VO2_auc", "HR_auc", "TSI_auc", "HHb_auc"],
        labels={
            "VO2_auc": "VO2 AUC",
            "HR_auc": "HR AUC",
            "TSI_auc": "TSI AUC",
            "HHb_auc": "HHb AUC",
        },
        save=save,
    )

    figs["slopes"] = plot_slopes_by_condition(
        block_df,
        slope_vars=["TSI_slope", "HHb_slope", "FC_drift"],
        labels={
            "TSI_slope": "TSI slope",
            "HHb_slope": "HHb slope",
            "FC_drift": "FC drift (bpm/min)",
        },
        save=save,
    )

    figs["baseline"] = plot_baseline_comparison(baseline_df, save=save)
    figs["corr"] = plot_correlation_heatmap(corr_matrix, save=save)
    figs["force"] = plot_force_sensors(fp_data, timeline=timeline, save=save)
    figs["block_heatmap"] = plot_block_heatmap(
        block_df,
        variables=["VO2_mean", "HR_mean", "TSI_mean", "HHb_mean", "Force_mean", "step_freq_bpm", "FC_drift"],
        save=save,
    )

    return figs


def validate_visualization_outputs() -> tuple[list[str], list[str]]:
    """Run full visualization pipeline on project data and validate saved outputs."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from sources.load_data import load_k5_data, load_hr_data, load_nirs_data, load_force_plate_data
    from sources.synchronization import synchronize_all
    from sources.segmentation import TIMELINE
    from sources.features import build_block_level_summary, compare_baselines
    from sources.statistics import compute_correlation_matrix

    errors: list[str] = []
    warnings: list[str] = []

    data_dir = PROJECT_ROOT / "data"
    k5 = load_k5_data(data_dir / "Données K5 Allan.csv")
    hr = load_hr_data(data_dir / "Données FC Allan.csv")
    nirs = load_nirs_data(data_dir / "Données NIRS Allan.csv")
    fp = load_force_plate_data(data_dir / "Données FP Allan.csv")

    synced = synchronize_all(
        k5, hr, nirs, fp,
        drop_pre_trigger=True,
        use_manual_trigger_inference=False,
    )

    block_df = build_block_level_summary(
        synced["k5"], synced["hr"], synced["nirs"], synced["fp"],
        timeline=TIMELINE,
    )
    baseline_df = compare_baselines(synced["k5"], synced["hr"], synced["nirs"], timeline=TIMELINE)
    corr_matrix = compute_correlation_matrix(
        block_df,
        ["VO2_mean", "HR_mean", "TSI_mean", "HHb_mean", "Force_mean"],
        method="pearson",
    )

    # Keep FP rendering responsive by downsampling already handled in plotting.
    figs = create_all_project_figures(
        synced["k5"], synced["hr"], synced["nirs"], synced["fp"],
        timeline=TIMELINE,
        block_df=block_df,
        baseline_df=baseline_df,
        corr_matrix=corr_matrix,
        save=True,
    )

    expected_files = [
        "Fig1_Time_Series_Overview.png",
        "Fig2_Condition_Boxplots.png",
        "Fig3_AUC_By_Condition.png",
        "Fig4_Slopes_By_Condition.png",
        "Fig5_Baseline_Comparison.png",
        "Fig6_Correlation_Heatmap.png",
        "Fig7_Force_Sensors.png",
        "Fig8_Block_Heatmap.png",
    ]

    for name in expected_files:
        f = RESULTS_DIR / name
        if not f.exists() or f.stat().st_size == 0:
            errors.append(f"Missing or empty figure file: {f}")

    if len(block_df) < 6:
        warnings.append(f"Block table has only {len(block_df)} rows (<6).")
    if baseline_df["Baseline_Final"].isna().all():
        warnings.append("All final baseline metrics are NaN (recording may end before planned final baseline).")

    # Close figures to avoid memory growth in notebooks/reruns.
    for fig in figs.values():
        plt.close(fig)

    return errors, warnings


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VALIDATION: visualization.py")
    print("=" * 70)

    try:
        print("\n[1] Running full visualization pipeline ...")
        errors, warnings = validate_visualization_outputs()

        if warnings:
            print("\n[Warnings]")
            for w in warnings:
                print(f"  ⚠ {w}")

        print("\n" + "=" * 70)
        if errors:
            print("❌ visualization.py validation failed")
            for e in errors:
                print(f"  - {e}")
            raise SystemExit(1)

        print("✅ visualization.py works and is project-adapted")
        print(f"  Output directory: {RESULTS_DIR}")
        print("=" * 70 + "\n")
    except Exception as exc:
        print(f"❌ visualization.py execution failed: {exc}")
        raise SystemExit(1)
