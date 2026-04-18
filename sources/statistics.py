"""
statistics.py — DEPOXY Project
==================================
Statistical analyses:
  - Normality tests (Shapiro-Wilk)
  - Repeated-measures ANOVA (within-subject, shoe condition factor)
  - Friedman test (non-parametric robustness check)
  - Post-hoc pairwise comparisons (Bonferroni correction)
  - Effect sizes (eta-squared, Cohen's d)
  - Correlation matrix + pairwise Pearson correlations

Author: Baptiste AUGROS
"""

from __future__ import annotations
import itertools
import warnings
import numpy as np
import pandas as pd
from scipy import stats

try:
    from segmentation import CONDITION_MAP, PROTOCOL, TIMELINE
except ImportError:
    # Package-style import fallback (e.g. from sources.statistics import ...)
    from .segmentation import CONDITION_MAP, PROTOCOL, TIMELINE

warnings.filterwarnings("ignore")

# Optional libraries — gracefully degraded if absent
try:
    from statsmodels.stats.anova import AnovaRM
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    import pingouin as pg
    _HAS_PINGOUIN = True
except ImportError:
    _HAS_PINGOUIN = False


# ---------------------------------------------------------------------------
# Normality
# ---------------------------------------------------------------------------

def test_normality(block_df: pd.DataFrame,
                   variables: list[str]) -> pd.DataFrame:
    """
    Shapiro-Wilk normality test for each variable across all block observations.

    Returns
    -------
    pd.DataFrame: variable, n, W, p_value, normal (bool)
    """
    rows = []
    for var in variables:
        if var not in block_df.columns:
            continue
        vals = pd.to_numeric(block_df[var], errors="coerce").dropna().values
        if len(vals) < 3:
            rows.append({"variable": var, "n": len(vals),
                          "W": np.nan, "p_value": np.nan, "normal": False})
            continue
        W, p = stats.shapiro(vals)
        rows.append({"variable": var, "n": len(vals),
                      "W": round(W, 4), "p_value": round(p, 4),
                      "normal": bool(p > 0.05)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Repeated-measures ANOVA helpers
# ---------------------------------------------------------------------------

def _build_anova_input(block_df: pd.DataFrame,
                        target: str) -> pd.DataFrame:
    """
    Build a long-format DataFrame for AnovaRM with artificial subject labels.

    For the ABCCBA single-participant design:
      - Block 1 and Block 6 -> condition A -> repeat_1, repeat_2
      - Block 2 and Block 5 -> condition B -> repeat_1, repeat_2
      - Block 3 and Block 4 -> condition C -> repeat_1, repeat_2
    This creates a balanced within-subject design with n=2 repeats.
    """
    rows = []
    for cond, blocks in CONDITION_MAP.items():
        for rep_id, blk in enumerate(blocks, start=1):
            row_data = block_df.loc[block_df["block"] == blk]
            if len(row_data) == 0:
                continue
            rows.append({
                "subject":   f"repeat_{rep_id}",
                "condition": cond,
                target:      float(row_data[target].iloc[0]),
            })
    return pd.DataFrame(rows)


def run_rm_anova(block_df: pd.DataFrame,
                 targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Repeated-measures ANOVA for each target variable.

    Returns
    -------
    anova_df   : ANOVA table with F, df, p, eta-squared per variable
    posthoc_df : Pairwise Bonferroni post-hoc table
    """
    anova_rows   = []
    posthoc_rows = []

    for target in targets:
        if target not in block_df.columns:
            continue

        inp = _build_anova_input(block_df, target).dropna(subset=[target])

        if inp["condition"].nunique() < 2:
            print(f"  ⚠ {target}: fewer than 2 conditions — skipping ANOVA")
            continue

        # --- ANOVA ---
        f_val, p_val, eta_sq = np.nan, np.nan, np.nan
        if _HAS_STATSMODELS and inp["subject"].nunique() >= 2:
            try:
                model = AnovaRM(inp, depvar=target,
                                subject="subject", within=["condition"]).fit()
                tbl   = model.anova_table
                f_val = float(tbl["F Value"].iloc[0])
                df_n  = float(tbl["Num DF"].iloc[0])
                df_d  = float(tbl["Den DF"].iloc[0])
                p_val = float(tbl["Pr > F"].iloc[0])
                eta_sq = (f_val * df_n) / (f_val * df_n + df_d)
            except Exception as e:
                print(f"  ⚠ ANOVA failed for {target}: {e}")

        anova_rows.append({
            "Variable":    target,
            "F":           round(f_val, 3),
            "p_value":     round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "eta_sq":      round(eta_sq, 3) if not np.isnan(eta_sq) else np.nan,
            "significant": bool(p_val < 0.05) if not np.isnan(p_val) else False,
        })

        # --- Post-hoc Bonferroni ---
        wide = inp.pivot(index="subject", columns="condition", values=target)
        conds = sorted(inp["condition"].unique())
        pairs = list(itertools.combinations(conds, 2))
        n_comp = len(pairs)
        raw_ps, pair_stats = [], []

        for a, b in pairs:
            pair_data = wide[[a, b]].dropna()
            if len(pair_data) >= 2:
                t_stat, p_raw = stats.ttest_rel(pair_data[a], pair_data[b])
                diff = pair_data[a] - pair_data[b]
                cohen_d = (diff.mean() / diff.std(ddof=1)
                           if diff.std(ddof=1) > 0 else np.nan)
            else:
                t_stat, p_raw, cohen_d = np.nan, np.nan, np.nan
            raw_ps.append(p_raw)
            pair_stats.append((a, b, t_stat, p_raw, cohen_d, len(pair_data)))

        bonf_ps = [min(p * n_comp, 1.0) if not np.isnan(p) else np.nan for p in raw_ps]
        for (a, b, t, p_r, d, n), p_bonf in zip(pair_stats, bonf_ps):
            posthoc_rows.append({
                "Variable":     target,
                "Comparison":   f"{a} vs {b}",
                "t_stat":       round(t, 3)     if not np.isnan(t)     else np.nan,
                "p_raw":        round(p_r, 4)   if not np.isnan(p_r)   else np.nan,
                "p_bonferroni": round(p_bonf, 4) if not np.isnan(p_bonf) else np.nan,
                "cohens_d":     round(d, 3)     if not np.isnan(d)     else np.nan,
                "n_pairs":      n,
                "significant":  bool(p_bonf < 0.05) if not np.isnan(p_bonf) else False,
            })

    return pd.DataFrame(anova_rows), pd.DataFrame(posthoc_rows)


# ---------------------------------------------------------------------------
# Friedman non-parametric test
# ---------------------------------------------------------------------------

def run_friedman(block_df: pd.DataFrame,
                 targets: list[str]) -> pd.DataFrame:
    """
    Friedman test (non-parametric alternative to rm-ANOVA) for each target.

    Requires >= 2 repeated observations per condition (from ABCCBA: n=2 each).

    Returns
    -------
    pd.DataFrame: Variable, statistic, p_value, significant
    """
    rows = []
    for target in targets:
        if target not in block_df.columns:
            continue
        groups = []
        for cond, blocks in CONDITION_MAP.items():
            vals = block_df.loc[block_df["block"].isin(blocks), target].dropna().values
            if len(vals) > 0:
                groups.append(vals)

        if len(groups) < 2:
            continue
        # Align lengths (take min)
        min_n = min(len(g) for g in groups)
        groups = [g[:min_n] for g in groups]

        try:
            stat, p = stats.friedmanchisquare(*groups)
            rows.append({
                "Variable":    target,
                "statistic":   round(stat, 3),
                "p_value":     round(p, 4),
                "significant": bool(p < 0.05),
            })
        except Exception as e:
            rows.append({
                "Variable":    target,
                "statistic":   np.nan,
                "p_value":     np.nan,
                "significant": False,
            })
            print(f"  ⚠ Friedman failed for {target}: {e}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_pairwise_correlations(block_df: pd.DataFrame,
                                   pairs: list[tuple[str, str]],
                                   method: str = "pearson") -> pd.DataFrame:
    """
    Pairwise Pearson or Spearman correlations between variable pairs.

    Required pairs per spec:
        VO2_mean x TSI_mean
        VO2_mean x HHb_mean
        VO2_mean x HR_mean  (FC)
        HR_mean  x Force_mean

    Returns
    -------
    pd.DataFrame: X, Y, r, p_value, n, method
    """
    rows = []
    for x_col, y_col in pairs:
        if x_col not in block_df.columns or y_col not in block_df.columns:
            rows.append({"X": x_col, "Y": y_col,
                          "r": np.nan, "p_value": np.nan, "n": 0,
                          "method": method})
            continue
        pair_df = (block_df[[x_col, y_col]]
                   .apply(pd.to_numeric, errors="coerce")
                   .dropna())
        n = len(pair_df)
        if n < 3:
            rows.append({"X": x_col, "Y": y_col,
                          "r": np.nan, "p_value": np.nan, "n": n,
                          "method": method})
            continue
        if method == "pearson":
            r, p = stats.pearsonr(pair_df[x_col], pair_df[y_col])
        else:
            r, p = stats.spearmanr(pair_df[x_col], pair_df[y_col])
        rows.append({"X": x_col, "Y": y_col,
                      "r": round(float(r), 3),
                      "p_value": round(float(p), 4),
                      "n": n, "method": method})
    return pd.DataFrame(rows)


def compute_correlation_matrix(block_df: pd.DataFrame,
                                columns: list[str],
                                method: str = "pearson") -> pd.DataFrame:
    """Return a full correlation matrix for the specified columns."""
    available = [c for c in columns if c in block_df.columns]
    numeric   = block_df[available].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method=method)


def run_three_factor_rm_anova(long_df: pd.DataFrame,
                              value_col: str = "value",
                              subject_col: str = "subject",
                              condition_col: str = "condition",
                              task_col: str = "task",
                              repetition_col: str = "repetition") -> pd.DataFrame:
    """
    CERUM-oriented repeated-measures ANOVA with 3 within factors:
      - type de chaussant (condition)
      - type de tâche (task)
      - répétition (repetition)

    The input must be long-format and include multiple subjects.
    With a single subject, this model is not identifiable and a warning row is returned.
    """
    required = [subject_col, condition_col, task_col, repetition_col, value_col]
    missing = [c for c in required if c not in long_df.columns]
    if missing:
        return pd.DataFrame([{
            "status": "error",
            "message": f"Missing columns for 3-factor ANOVA: {missing}",
        }])

    df = long_df[required].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    if df[subject_col].nunique() < 2:
        return pd.DataFrame([{
            "status": "warning",
            "message": "3-factor rm-ANOVA not estimable with <2 subjects; provide group-level long-format data.",
        }])

    if not _HAS_STATSMODELS:
        return pd.DataFrame([{
            "status": "warning",
            "message": "statsmodels unavailable; cannot run 3-factor rm-ANOVA.",
        }])

    try:
        model = AnovaRM(
            df,
            depvar=value_col,
            subject=subject_col,
            within=[condition_col, task_col, repetition_col],
        ).fit()
        out = model.anova_table.reset_index().rename(columns={"index": "Effect"})
        out["status"] = "ok"
        return out
    except Exception as exc:
        return pd.DataFrame([{
            "status": "error",
            "message": f"3-factor rm-ANOVA failed: {exc}",
        }])


def _select_correlation_method(block_df: pd.DataFrame,
                               x_col: str,
                               y_col: str,
                               normality_df: pd.DataFrame | None = None,
                               default: str = "pearson") -> str:
    """Select Pearson if both variables appear normal, else Spearman."""
    if normality_df is None or normality_df.empty:
        return default

    def _is_normal(var: str) -> bool | None:
        row = normality_df[normality_df["variable"] == var]
        if row.empty:
            return None
        v = row["normal"].iloc[0]
        return bool(v) if pd.notna(v) else None

    nx = _is_normal(x_col)
    ny = _is_normal(y_col)
    if nx is True and ny is True:
        return "pearson"
    if nx is False or ny is False:
        return "spearman"
    return default


def compute_pairwise_correlations_auto(block_df: pd.DataFrame,
                                       pairs: list[tuple[str, str]],
                                       normality_df: pd.DataFrame | None = None,
                                       default_method: str = "pearson") -> pd.DataFrame:
    """Pairwise correlations with automatic Pearson/Spearman selection."""
    frames = []
    for x, y in pairs:
        method = _select_correlation_method(block_df, x, y, normality_df, default=default_method)
        one = compute_pairwise_correlations(block_df, [(x, y)], method=method)
        if not one.empty:
            one["method_selected"] = method
        frames.append(one)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Project-level runner and validation
# ---------------------------------------------------------------------------

def run_project_statistics(block_df: pd.DataFrame,
                           targets: list[str] | None = None,
                           correlation_pairs: list[tuple[str, str]] | None = None,
                           correlation_method: str = "auto",
                           long_df_three_factor: pd.DataFrame | None = None,
                           three_factor_value_col: str = "value") -> dict[str, pd.DataFrame]:
    """
    Run full statistical pipeline on block-level summary table.

    Parameters
    ----------
    block_df : output of features.build_block_level_summary
    targets  : variables for normality/ANOVA/Friedman
    correlation_pairs : variable pairs for pairwise correlations

    Returns
    -------
    dict of result DataFrames
    """
    if targets is None:
        targets = ["VO2_mean", "HR_mean", "TSI_mean", "HHb_mean", "Force_mean", "FC_drift"]

    if correlation_pairs is None:
        correlation_pairs = [
            ("VO2_mean", "TSI_mean"),
            ("VO2_mean", "HHb_mean"),
            ("VO2_mean", "HR_mean"),
            ("HR_mean", "Force_mean"),
        ]

    normality_df = test_normality(block_df, targets)
    anova_df, posthoc_df = run_rm_anova(block_df, targets)
    friedman_df = run_friedman(block_df, targets)
    if correlation_method == "auto":
        pair_corr_df = compute_pairwise_correlations_auto(
            block_df,
            correlation_pairs,
            normality_df=normality_df,
            default_method="pearson",
        )
        matrix_method = "pearson"
    else:
        pair_corr_df = compute_pairwise_correlations(block_df, correlation_pairs, method=correlation_method)
        matrix_method = correlation_method

    corr_cols = sorted({x for x, _ in correlation_pairs} | {y for _, y in correlation_pairs})
    corr_matrix_df = compute_correlation_matrix(block_df, corr_cols, method=matrix_method)

    if long_df_three_factor is not None:
        anova_three_factor_df = run_three_factor_rm_anova(
            long_df_three_factor,
            value_col=three_factor_value_col,
        )
    else:
        anova_three_factor_df = pd.DataFrame([{
            "status": "warning",
            "message": "No long-format dataset supplied for CERUM 3-factor rm-ANOVA (condition x task x repetition).",
        }])

    return {
        "normality": normality_df,
        "anova": anova_df,
        "posthoc": posthoc_df,
        "friedman": friedman_df,
        "pairwise_correlations": pair_corr_df,
        "correlation_matrix": corr_matrix_df,
        "anova_three_factor": anova_three_factor_df,
    }


def validate_statistics_adaptation(block_df: pd.DataFrame,
                                   stats_outputs: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    """
    Validate that statistics pipeline is coherent with ABCCBA project constraints.

    Returns
    -------
    (errors, warnings)
    """
    errors: list[str] = []
    warnings: list[str] = []

    if "block" not in block_df.columns:
        errors.append("Missing 'block' column in block summary input.")
        return errors, warnings

    if len(block_df) != PROTOCOL["n_blocks"]:
        errors.append(f"Expected {PROTOCOL['n_blocks']} block rows, got {len(block_df)}.")

    if "condition" not in block_df.columns:
        errors.append("Missing 'condition' column in block summary input.")
    else:
        obs = sorted(block_df["condition"].dropna().unique().tolist())
        if obs != ["A", "B", "C"]:
            errors.append(f"Expected conditions A/B/C, got {obs}.")

    # Repeated-measures shape check: 2 repeats per condition in ABCCBA.
    for cond, blocks in CONDITION_MAP.items():
        n = int(block_df[block_df["block"].isin(blocks)].shape[0])
        if n < 2:
            warnings.append(f"Condition {cond} has {n}/2 expected repeats (blocks={blocks}).")

    anova_df = stats_outputs.get("anova", pd.DataFrame())
    friedman_df = stats_outputs.get("friedman", pd.DataFrame())
    if anova_df.empty and friedman_df.empty:
        errors.append("Both ANOVA and Friedman outputs are empty.")

    corr_df = stats_outputs.get("pairwise_correlations", pd.DataFrame())
    if not corr_df.empty and (corr_df["n"].fillna(0) < 3).all():
        warnings.append("All correlations have n<3; inferential interpretation is weak.")

    anova3 = stats_outputs.get("anova_three_factor", pd.DataFrame())
    if not anova3.empty and "status" in anova3.columns:
        first_status = str(anova3["status"].iloc[0]).lower()
        if first_status == "error":
            errors.append(str(anova3.get("message", pd.Series(["3-factor ANOVA error"])) .iloc[0]))
        elif first_status == "warning":
            warnings.append(str(anova3.get("message", pd.Series(["3-factor ANOVA unavailable"])) .iloc[0]))

    return errors, warnings


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Allow running as script: python sources/statistics.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from sources.load_data import load_k5_data, load_hr_data, load_nirs_data, load_force_plate_data
    from sources.synchronization import synchronize_all
    from sources.features import build_block_level_summary

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"

    print("\n" + "=" * 70)
    print("VALIDATION: statistics.py")
    print("=" * 70)

    try:
        print("\n[1] Loading data ...")
        k5 = load_k5_data(data_dir / "Données K5 Allan.csv")
        hr = load_hr_data(data_dir / "Données FC Allan.csv")
        nirs = load_nirs_data(data_dir / "Données NIRS Allan.csv")
        fp = load_force_plate_data(data_dir / "Données FP Allan.csv")

        print("\n[2] Synchronizing data ...")
        synced = synchronize_all(
            k5, hr, nirs, fp,
            drop_pre_trigger=True,
            use_manual_trigger_inference=False,
        )

        print("\n[3] Building block-level features ...")
        block_df = build_block_level_summary(
            synced["k5"], synced["hr"], synced["nirs"], synced["fp"],
            timeline=TIMELINE,
        )

        print("\n[4] Running statistics ...")
        stats_out = run_project_statistics(block_df)
        errors, warnings = validate_statistics_adaptation(block_df, stats_out)

        print(f"  Block table rows: {len(block_df)}")
        print(f"  Normality rows  : {len(stats_out['normality'])}")
        print(f"  ANOVA rows      : {len(stats_out['anova'])}")
        print(f"  Post-hoc rows   : {len(stats_out['posthoc'])}")
        print(f"  Friedman rows   : {len(stats_out['friedman'])}")
        print(f"  Correlation rows: {len(stats_out['pairwise_correlations'])}")
        print(f"  3-factor ANOVA rows: {len(stats_out['anova_three_factor'])}")

        if warnings:
            print("\n[Warnings]")
            for w in warnings:
                print(f"  ⚠ {w}")

        print("\n" + "=" * 70)
        if errors:
            print("❌ statistics.py validation failed")
            for e in errors:
                print(f"  - {e}")
            raise SystemExit(1)

        print("✅ statistics.py works and is project-adapted")
        print("=" * 70 + "\n")
    except Exception as exc:
        print(f"❌ statistics.py execution failed: {exc}")
        raise SystemExit(1)
