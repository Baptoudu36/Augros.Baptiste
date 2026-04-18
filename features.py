"""
features.py — DEPOXY Project
================================
Physiological feature extraction: AUC, slopes, energy cost, peak detection,
block-level summaries, and condition-level aggregation.

Primary modalities handled:
  VO2, FC (HR), TSI (SmO2), HHb, O2Hb, Force

Author: Baptiste AUGROS
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal, stats

try:
    from segmentation import (
        PROTOCOL, TIMELINE, CONDITION_MAP, SHOE_CONDITIONS,
        extract_steady_state_from_block, extract_window,
    )
except ImportError:
    # Package-style import fallback (e.g. from sources.features import ...)
    from .segmentation import (
        PROTOCOL, TIMELINE, CONDITION_MAP, SHOE_CONDITIONS,
        extract_steady_state_from_block, extract_window,
    )


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------

def compute_window_stats(df: pd.DataFrame,
                         columns: list[str],
                         prefix: str = "") -> dict:
    """
    Compute mean, std, min, max, n for each column in a DataFrame window.

    Returns a flat dict like {"prefix_col_mean": ..., "prefix_col_std": ..., ...}
    """
    stats_dict: dict = {}
    for col in columns:
        if col not in df.columns:
            continue
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        key = f"{prefix}{col}"
        stats_dict[f"{key}_mean"] = float(data.mean()) if len(data) else np.nan
        stats_dict[f"{key}_std"]  = float(data.std())  if len(data) else np.nan
        stats_dict[f"{key}_min"]  = float(data.min())  if len(data) else np.nan
        stats_dict[f"{key}_max"]  = float(data.max())  if len(data) else np.nan
        stats_dict[f"{key}_n"]    = len(data)
    return stats_dict


# ---------------------------------------------------------------------------
# AUC and Slope
# ---------------------------------------------------------------------------

def compute_auc(series: pd.Series | np.ndarray,
                time:   pd.Series | np.ndarray) -> float:
    """Trapezoidal AUC. Returns NaN if < 2 valid points."""
    y = np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)
    t = np.asarray(pd.to_numeric(time,   errors="coerce"), dtype=float)
    valid = np.isfinite(y) & np.isfinite(t)
    if valid.sum() < 2:
        return np.nan
    return float(np.trapz(y[valid], t[valid]))


def compute_slope(series: pd.Series | np.ndarray,
                  time:   pd.Series | np.ndarray) -> float:
    """Linear regression slope (units/s). Returns NaN if < 3 valid points."""
    y = np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)
    t = np.asarray(pd.to_numeric(time,   errors="coerce"), dtype=float)
    valid = np.isfinite(y) & np.isfinite(t)
    if valid.sum() < 3:
        return np.nan
    slope, *_ = stats.linregress(t[valid], y[valid])
    return float(slope)


def compute_fc_drift(hr_series: pd.Series | np.ndarray,
                     time:      pd.Series | np.ndarray) -> float:
    """
    Compute heart rate drift (bpm/min) over a running block.
    Positive slope indicates cardiac drift upward (increasing load).
    """
    slope_per_sec = compute_slope(hr_series, time)
    return slope_per_sec * 60.0 if not np.isnan(slope_per_sec) else np.nan


# ---------------------------------------------------------------------------
# Derived physiological metrics
# ---------------------------------------------------------------------------

def calculate_energy_cost(vo2_exercise_mean: float,
                           vo2_rest_mean:     float,
                           mass_kg:           float = 1.0) -> dict:
    """
    Net energy cost = VO2_exercise - VO2_rest.

    Parameters
    ----------
    vo2_exercise_mean : mean VO2 during exercise (mL/min)
    vo2_rest_mean     : mean VO2 at rest / baseline (mL/min)
    mass_kg           : participant body mass (kg)

    Returns
    -------
    dict with keys: energy_cost_ml_min, energy_cost_ml_kg_min
    """
    net = vo2_exercise_mean - vo2_rest_mean
    return {
        "energy_cost_ml_min":    float(net),
        "energy_cost_ml_kg_min": float(net / mass_kg) if mass_kg > 0 else np.nan,
    }


def calculate_oxygen_pulse(vo2_mean: float, hr_mean: float) -> float:
    """O2 pulse = VO2 / HR (mL/beat)."""
    if hr_mean and hr_mean > 0:
        return float(vo2_mean / hr_mean)
    return np.nan


# ---------------------------------------------------------------------------
# Force peak detection and step frequency
# ---------------------------------------------------------------------------

def _estimate_fs(time_sec: pd.Series | np.ndarray) -> float:
    t = np.asarray(time_sec, dtype=float)
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    return float(1.0 / np.median(diffs)) if len(diffs) else np.nan


def detect_force_peaks(fp_df:            pd.DataFrame,
                       time_col:         str   = "time_sec",
                       force_col:        str   = "total_force_N",
                       height_threshold: float | None = None,
                       distance_samples: int   = 100) -> dict:
    """
    Detect vertical force peaks (footstrikes) and estimate step frequency.

    Parameters
    ----------
    fp_df             : force plate DataFrame
    time_col          : time column name
    force_col         : force column name
    height_threshold  : minimum peak height in N (auto if None)
    distance_samples  : minimum samples between peaks

    Returns
    -------
    dict with keys:
        n_peaks, peak_times, peak_values,
        mean_peak_force, std_peak_force,
        step_frequency_hz, step_frequency_bpm
    """
    if force_col not in fp_df.columns or len(fp_df) < 10:
        return {
            "n_peaks": 0, "peak_times": np.array([]), "peak_values": np.array([]),
            "mean_peak_force": np.nan, "std_peak_force": np.nan,
            "step_frequency_hz": np.nan, "step_frequency_bpm": np.nan,
        }

    force = pd.to_numeric(fp_df[force_col], errors="coerce").fillna(0).values
    time  = pd.to_numeric(fp_df[time_col],  errors="coerce").values

    if height_threshold is None:
        height_threshold = float(np.nanmean(force) + 0.3 * np.nanstd(force))

    peaks, _ = signal.find_peaks(
        force,
        height=height_threshold,
        distance=distance_samples,
    )

    if len(peaks) < 2:
        return {
            "n_peaks":         len(peaks),
            "peak_times":      time[peaks]  if len(peaks) else np.array([]),
            "peak_values":     force[peaks] if len(peaks) else np.array([]),
            "mean_peak_force": float(np.mean(force[peaks])) if len(peaks) else np.nan,
            "std_peak_force":  float(np.std(force[peaks]))  if len(peaks) else np.nan,
            "step_frequency_hz":  np.nan,
            "step_frequency_bpm": np.nan,
        }

    peak_times  = time[peaks]
    peak_values = force[peaks]
    intervals   = np.diff(peak_times)
    step_hz     = 1.0 / float(np.median(intervals))

    print(f"  Peak detection: {len(peaks)} peaks | "
          f"freq = {step_hz:.2f} Hz ({step_hz*60:.1f} steps/min) | "
          f"mean force = {np.mean(peak_values):.1f} N")

    return {
        "n_peaks":            len(peaks),
        "peak_times":         peak_times,
        "peak_values":        peak_values,
        "mean_peak_force":    float(np.mean(peak_values)),
        "std_peak_force":     float(np.std(peak_values)),
        "step_frequency_hz":  step_hz,
        "step_frequency_bpm": step_hz * 60.0,
    }


# ---------------------------------------------------------------------------
# Block-level summary (all modalities including FC)
# ---------------------------------------------------------------------------

def _safe_mean(df: pd.DataFrame | None, col: str) -> float:
    if df is None or col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").dropna().mean())


def build_block_level_summary(k5_df:   pd.DataFrame,
                               hr_df:   pd.DataFrame,
                               nirs_df: pd.DataFrame,
                               fp_df:   pd.DataFrame,
                               timeline: dict | None = None,
                               mass_kg: float = 70.0) -> pd.DataFrame:
    """
    Compute block-level steady-state metrics for all running blocks.

    Includes: VO2, FC (HR), TSI, HHb, O2Hb, AUC (VO2/FC/TSI/HHb),
    slopes (TSI, HHb, FC drift), force mean, step frequency.

    Returns
    -------
    pd.DataFrame with one row per block (6 rows for ABCCBA design)
    """
    tl = timeline or TIMELINE

    # Detect available NIRS columns
    tsi_col  = next((c for c in ["TSI_Quad", "SmO2_Quad", "TSI", "SmO2"] if c in nirs_df.columns), None)
    hhb_col  = next((c for c in ["HHb_Quad", "HHb"] if c in nirs_df.columns), None)
    o2hb_col = next((c for c in ["O2Hb_Quad", "O2Hb"] if c in nirs_df.columns), None)

    rows = []
    for block in range(1, PROTOCOL["n_blocks"] + 1):
        run_key = f"run_{block}"
        rec_key = f"recovery_{block}"
        run_info  = tl[run_key]
        run_start = run_info["start"]
        run_end   = run_info["end"]

        # Full block windows (for AUC and slope computation)
        k5_run   = extract_window(k5_df,   run_start, run_end)
        hr_run   = extract_window(hr_df,   run_start, run_end)
        nirs_run = extract_window(nirs_df, run_start, run_end)
        fp_run   = extract_window(fp_df,   run_start, run_end)

        # Steady-state windows (for condition means)
        k5_ss,   _ = extract_steady_state_from_block(k5_df,   run_start)
        hr_ss,   _ = extract_steady_state_from_block(hr_df,   run_start)
        nirs_ss, _ = extract_steady_state_from_block(nirs_df, run_start)

        t_k5   = k5_run["time_sec"]
        t_hr   = hr_run["time_sec"]
        t_nirs = nirs_run["time_sec"]

        # AUC (full block)
        vo2_auc = compute_auc(k5_run.get("VO2_ml_min", pd.Series(dtype=float)), t_k5)
        hr_auc  = compute_auc(hr_run.get("HR_bpm",     pd.Series(dtype=float)), t_hr)
        tsi_auc = compute_auc(nirs_run[tsi_col] if tsi_col else pd.Series(dtype=float), t_nirs)
        hhb_auc = compute_auc(nirs_run[hhb_col] if hhb_col else pd.Series(dtype=float), t_nirs)

        # Slopes (full block)
        tsi_slope = compute_slope(nirs_run[tsi_col] if tsi_col else pd.Series(dtype=float), t_nirs)
        hhb_slope = compute_slope(nirs_run[hhb_col] if hhb_col else pd.Series(dtype=float), t_nirs)
        fc_drift  = compute_fc_drift(hr_run.get("HR_bpm", pd.Series(dtype=float)), t_hr)

        # Recovery slopes (NIRS)
        tsi_rec_slope, hhb_rec_slope = np.nan, np.nan
        if rec_key in tl:
            rec_info = tl[rec_key]
            nirs_rec = extract_window(nirs_df, rec_info["start"], rec_info["end"])
            t_rec    = nirs_rec["time_sec"]
            tsi_rec_slope = compute_slope(nirs_rec[tsi_col] if tsi_col else pd.Series(dtype=float), t_rec)
            hhb_rec_slope = compute_slope(nirs_rec[hhb_col] if hhb_col else pd.Series(dtype=float), t_rec)

        # Force peaks (steady-state window)
        fp_ss, _ = extract_steady_state_from_block(fp_df, run_start)
        fs_fp    = _estimate_fs(fp_ss["time_sec"]) if len(fp_ss) > 1 else np.nan
        dist_s   = max(20, int(fs_fp * 0.4)) if not np.isnan(fs_fp) else 100
        pk = detect_force_peaks(fp_ss, distance_samples=dist_s)

        rows.append({
            "block":         block,
            "condition":     run_info["shoe_code"],
            "shoe_name":     run_info["shoe_name"],
            # Condition means (steady-state)
            "VO2_mean":      _safe_mean(k5_ss,   "VO2_ml_min"),
            "HR_mean":       _safe_mean(hr_ss,   "HR_bpm"),
            "TSI_mean":      _safe_mean(nirs_ss, tsi_col)  if tsi_col  else np.nan,
            "HHb_mean":      _safe_mean(nirs_ss, hhb_col)  if hhb_col  else np.nan,
            "O2Hb_mean":     _safe_mean(nirs_ss, o2hb_col) if o2hb_col else np.nan,
            "Force_mean":    pk["mean_peak_force"],
            # AUC (full block)
            "VO2_auc":       vo2_auc,
            "HR_auc":        hr_auc,
            "TSI_auc":       tsi_auc,
            "HHb_auc":       hhb_auc,
            # Slopes
            "TSI_slope":     tsi_slope,
            "HHb_slope":     hhb_slope,
            "FC_drift":      fc_drift,
            "TSI_rec_slope": tsi_rec_slope,
            "HHb_rec_slope": hhb_rec_slope,
            # Step frequency
            "step_freq_hz":  pk["step_frequency_hz"],
            "step_freq_bpm": pk["step_frequency_bpm"],
            "n_steps":       pk["n_peaks"],
        })

    return pd.DataFrame(rows).sort_values("block").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Condition-level aggregation (A/B/C)
# ---------------------------------------------------------------------------

def aggregate_by_condition(block_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average block-level metrics across repeated conditions.

    A = mean(Block1, Block6), B = mean(Block2, Block5), C = mean(Block3, Block4)

    Returns
    -------
    pd.DataFrame with one row per condition (A, B, C)
    """
    numeric_cols = block_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "block"]

    rows = []
    for cond, blocks in CONDITION_MAP.items():
        sub = block_df[block_df["block"].isin(blocks)]
        row = {"condition": cond,
               "shoe_name": sub["shoe_name"].iloc[0] if len(sub) else ""}
        for col in numeric_cols:
            if col in sub.columns:
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_std"]  = float(sub[col].std(ddof=1))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Baseline comparison (initial vs final)
# ---------------------------------------------------------------------------

def compare_baselines(k5_df:   pd.DataFrame,
                       hr_df:   pd.DataFrame,
                       nirs_df: pd.DataFrame,
                       timeline: dict | None = None) -> pd.DataFrame:
    """
    Compute mean values at initial and final baseline, and their differences.

    Variables: VO2, FC (HR), TSI, HHb

    Returns
    -------
    pd.DataFrame with columns:
        Metric, Baseline_Initial, Baseline_Final, Delta_Final_minus_Initial
    """
    try:
        from segmentation import extract_baseline
    except ImportError:
        from .segmentation import extract_baseline

    k5_ini,   _ = extract_baseline(k5_df,   phase="initial", timeline=timeline)
    k5_fin,   _ = extract_baseline(k5_df,   phase="final",   timeline=timeline)
    hr_ini,   _ = extract_baseline(hr_df,   phase="initial", timeline=timeline)
    hr_fin,   _ = extract_baseline(hr_df,   phase="final",   timeline=timeline)
    nirs_ini, _ = extract_baseline(nirs_df, phase="initial", timeline=timeline)
    nirs_fin, _ = extract_baseline(nirs_df, phase="final",   timeline=timeline)

    tsi_col = next((c for c in ["TSI_Quad", "SmO2_Quad", "TSI"] if c in nirs_df.columns), None)
    hhb_col = next((c for c in ["HHb_Quad", "HHb"] if c in nirs_df.columns), None)

    result = []
    for m, ini_val, fin_val in [
        ("VO2 (mL/min)",   _safe_mean(k5_ini,   "VO2_ml_min"), _safe_mean(k5_fin,   "VO2_ml_min")),
        ("FC / HR (bpm)",  _safe_mean(hr_ini,   "HR_bpm"),     _safe_mean(hr_fin,   "HR_bpm")),
        ("TSI (%)",        _safe_mean(nirs_ini, tsi_col) if tsi_col else np.nan,
                           _safe_mean(nirs_fin, tsi_col) if tsi_col else np.nan),
        ("HHb (µM)",       _safe_mean(nirs_ini, hhb_col) if hhb_col else np.nan,
                           _safe_mean(nirs_fin, hhb_col) if hhb_col else np.nan),
    ]:
        ini = float(ini_val) if ini_val is not None and not (isinstance(ini_val, float) and np.isnan(ini_val)) else np.nan
        fin = float(fin_val) if fin_val is not None and not (isinstance(fin_val, float) and np.isnan(fin_val)) else np.nan
        delta = (fin - ini) if not (np.isnan(ini) or np.isnan(fin)) else np.nan
        result.append({
            "Metric":                    m,
            "Baseline_Initial":          ini,
            "Baseline_Final":            fin,
            "Delta_Final_minus_Initial": delta,
        })

    return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Validation helper for project pipeline
# ---------------------------------------------------------------------------

def validate_features_pipeline(k5_df: pd.DataFrame,
                               hr_df: pd.DataFrame,
                               nirs_df: pd.DataFrame,
                               fp_df: pd.DataFrame,
                               timeline: dict | None = None) -> dict:
    """
    Run core feature-extraction pipeline and return validation diagnostics.
    """
    tl = timeline or TIMELINE
    errors: list[str] = []
    warnings: list[str] = []

    # Protocol-awareness checks (adapted 75-min design)
    required_keys = ["baseline", "warmup", "pre_block_recovery", "baseline_final"]
    missing_keys = [k for k in required_keys if k not in tl]
    if missing_keys:
        errors.append(f"Missing timeline keys: {missing_keys}")

    expected_run1 = (
        PROTOCOL["baseline_duration"]
        + PROTOCOL["warmup_duration"]
        + PROTOCOL["pre_block_recovery_duration"]
    )
    if "run_1" in tl and not np.isclose(float(tl["run_1"]["start"]), float(expected_run1)):
        warnings.append(
            f"run_1 starts at {tl['run_1']['start']}s, expected {expected_run1}s with warmup+pre-recovery"
        )

    block_df = build_block_level_summary(k5_df, hr_df, nirs_df, fp_df, timeline=tl)
    cond_df = aggregate_by_condition(block_df)
    base_df = compare_baselines(k5_df, hr_df, nirs_df, timeline=tl)

    # Structural checks
    if len(block_df) != PROTOCOL["n_blocks"]:
        errors.append(f"Block summary rows={len(block_df)} (expected {PROTOCOL['n_blocks']}).")

    expected_sequence = [SHOE_CONDITIONS[b]["code"] for b in range(1, PROTOCOL["n_blocks"] + 1)]
    observed_sequence = block_df["condition"].tolist() if "condition" in block_df.columns else []
    if observed_sequence != expected_sequence:
        warnings.append(f"Condition sequence mismatch: observed={observed_sequence}, expected={expected_sequence}")

    if "condition" not in cond_df.columns or sorted(cond_df["condition"].tolist()) != ["A", "B", "C"]:
        errors.append("Condition aggregation must contain A/B/C rows.")

    if len(base_df) < 4:
        warnings.append(f"Baseline table has {len(base_df)} rows (<4 expected metrics).")

    # Data completeness checks on key metrics
    key_cols = ["VO2_mean", "HR_mean", "TSI_mean", "HHb_mean", "VO2_auc", "HR_auc"]
    for col in key_cols:
        if col in block_df.columns:
            frac_nan = float(block_df[col].isna().mean())
            if frac_nan > 0.5:
                warnings.append(f"High NaN fraction in {col}: {frac_nan:.0%}")

    return {
        "block_df": block_df,
        "condition_df": cond_df,
        "baseline_df": base_df,
        "errors": errors,
        "warnings": warnings,
    }


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Allow running as script: python sources/features.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from sources.load_data import load_k5_data, load_hr_data, load_nirs_data, load_force_plate_data
    from sources.synchronization import synchronize_all

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"

    print("\n" + "=" * 70)
    print("VALIDATION: features.py")
    print("=" * 70)

    try:
        print("\n[1] Loading data ...")
        k5 = load_k5_data(data_dir / "Données K5 Allan.csv")
        hr = load_hr_data(data_dir / "Données FC Allan.csv")
        nirs = load_nirs_data(data_dir / "Données NIRS Allan.csv")
        fp = load_force_plate_data(data_dir / "Données FP Allan.csv")

        print("\n[2] Synchronizing data ...")
        # Keep deterministic alignment for feature validation.
        synced = synchronize_all(
            k5, hr, nirs, fp,
            drop_pre_trigger=True,
            use_manual_trigger_inference=False,
        )

        print("\n[3] Building feature tables ...")
        res = validate_features_pipeline(
            synced["k5"], synced["hr"], synced["nirs"], synced["fp"],
            timeline=TIMELINE,
        )

        block_df = res["block_df"]
        cond_df = res["condition_df"]
        base_df = res["baseline_df"]
        errors = res["errors"]
        warnings = res["warnings"]

        print(f"  Blocks table     : {len(block_df)} rows x {len(block_df.columns)} cols")
        print(f"  Conditions table : {len(cond_df)} rows x {len(cond_df.columns)} cols")
        print(f"  Baseline table   : {len(base_df)} rows x {len(base_df.columns)} cols")

        if warnings:
            print("\n[Warnings]")
            for w in warnings:
                print(f"  ⚠ {w}")

        print("\n" + "=" * 70)
        if errors:
            print("❌ features.py validation failed")
            for e in errors:
                print(f"  - {e}")
            raise SystemExit(1)

        print("✅ features.py works and is protocol-adapted")
        print("=" * 70 + "\n")
    except Exception as exc:
        print(f"❌ features.py execution failed: {exc}")
        raise SystemExit(1)