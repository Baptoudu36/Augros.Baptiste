"""
filtering.py — DEPOXY Project
================================
Zero-phase digital filtering for physiological time series.

Filters applied:
  - mNIRS (TSI, HHb, O2Hb, HbTot) : Butterworth low-pass (default 0.1 Hz)
  - Force plate                     : Butterworth low-pass (default 20 Hz)
  - COSMED / HR                     : No filtering (already breath-by-breath
                                      or beat-by-beat; smoothing optional)

Author: Baptiste AUGROS
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal


# ---------------------------------------------------------------------------
# DEPOXY auto-tuning defaults
# ---------------------------------------------------------------------------

# If cutoff/window is None, an automatic selection is applied from candidates.
NIRS_CUTOFF_CANDIDATES_HZ = (0.06, 0.08, 0.10, 0.12)
FP_CUTOFF_CANDIDATES_HZ = (12.0, 15.0, 20.0, 25.0, 30.0)
COSMED_WINDOW_CANDIDATES_SEC = (6.0, 8.0, 10.0, 12.0)


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

def butterworth_lowpass(data: np.ndarray,
                        cutoff_hz: float,
                        fs_hz: float,
                        order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    data     : 1-D array of signal values
    cutoff_hz: filter cutoff frequency in Hz
    fs_hz    : sampling frequency in Hz
    order    : filter order (default 4)

    Returns
    -------
    np.ndarray — filtered signal, same length as input
    """
    if fs_hz <= 0 or cutoff_hz <= 0 or cutoff_hz >= fs_hz / 2:
        raise ValueError(
            f"Invalid filter parameters: cutoff={cutoff_hz} Hz, fs={fs_hz} Hz. "
            f"Cutoff must be < Nyquist ({fs_hz/2:.1f} Hz)."
        )
    nyq = fs_hz / 2.0
    norm_cutoff = cutoff_hz / nyq
    b, a = signal.butter(order, norm_cutoff, btype="low", analog=False)
    # use filtfilt for zero-phase (no time shift)
    filtered = signal.filtfilt(b, a, data, padlen=min(3 * max(len(a), len(b)), len(data) - 1))
    return filtered


def estimate_fs(time_sec: pd.Series | np.ndarray) -> float:
    """Estimate sampling frequency from time vector (median inter-sample interval)."""
    t = np.asarray(time_sec, dtype=float)
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return np.nan
    return float(1.0 / np.median(diffs))


def _composite_score(raw: np.ndarray, filtered: np.ndarray) -> float:
    """
    Score filter quality by balancing noise reduction and shape preservation.

    Higher is better.
    """
    raw = np.asarray(raw, dtype=float)
    filtered = np.asarray(filtered, dtype=float)

    if raw.size < 5 or filtered.size < 5:
        return -np.inf

    # High-frequency proxy: first differences should become less noisy.
    d_raw = np.diff(raw)
    d_flt = np.diff(filtered)
    std_raw = np.nanstd(d_raw)
    std_flt = np.nanstd(d_flt)
    if not np.isfinite(std_raw) or std_raw <= 0 or not np.isfinite(std_flt):
        return -np.inf
    noise_reduction = 1.0 - (std_flt / std_raw)

    # Shape preservation proxy: interquartile amplitude should not collapse.
    iqr_raw = np.nanpercentile(raw, 75) - np.nanpercentile(raw, 25)
    iqr_flt = np.nanpercentile(filtered, 75) - np.nanpercentile(filtered, 25)
    if not np.isfinite(iqr_raw) or iqr_raw <= 0 or not np.isfinite(iqr_flt):
        return -np.inf
    amp_ratio = iqr_flt / iqr_raw

    # Penalize over-smoothing / under-smoothing relative to amplitude retention.
    amplitude_penalty = abs(1.0 - amp_ratio)
    score = noise_reduction - 0.7 * amplitude_penalty
    return float(score)


def _choose_best_cutoff(series: pd.Series,
                        fs_hz: float,
                        candidates_hz: tuple[float, ...],
                        order: int) -> float:
    """Pick cutoff with best composite score on one representative signal."""
    s = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both")
    x = s.values.astype(float)
    if len(x) < 20:
        return candidates_hz[len(candidates_hz) // 2]

    best_cutoff = candidates_hz[len(candidates_hz) // 2]
    best_score = -np.inf

    for cutoff in candidates_hz:
        if cutoff <= 0 or cutoff >= fs_hz / 2:
            continue
        try:
            y = butterworth_lowpass(x, cutoff, fs_hz, order)
            sc = _composite_score(x, y)
            if sc > best_score:
                best_score = sc
                best_cutoff = cutoff
        except Exception:
            continue

    return float(best_cutoff)


def _choose_best_window(series: pd.Series,
                        fs_hz: float,
                        candidates_sec: tuple[float, ...]) -> float:
    """Pick smoothing window with best composite score on one representative signal."""
    s = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both")
    x = s.values.astype(float)
    if len(x) < 20:
        return candidates_sec[len(candidates_sec) // 2]

    best_window = candidates_sec[len(candidates_sec) // 2]
    best_score = -np.inf

    for window_sec in candidates_sec:
        w = max(1, int(round(window_sec * fs_hz)))
        y = pd.Series(x).rolling(window=w, center=True, min_periods=1).median().values
        sc = _composite_score(x, y)
        if sc > best_score:
            best_score = sc
            best_window = window_sec

    return float(best_window)


# ---------------------------------------------------------------------------
# Per-modality filtering wrappers
# ---------------------------------------------------------------------------

def filter_nirs(nirs_df: pd.DataFrame,
                cutoff_hz: float | None = None,
                order: int = 4) -> pd.DataFrame:
    """
    Apply Butterworth low-pass filter to all NIRS channels.

    Target columns (filtered if present):
        TSI_Quad, TSI_Calf, SmO2_Quad, SmO2_Calf,
        HHb_Quad, HHb_Calf,
        O2Hb_Quad, O2Hb_Calf,
        HbTot_Quad, HbTot_Calf

    Parameters
    ----------
    nirs_df   : DataFrame with 'time_sec' column
    cutoff_hz : low-pass cutoff in Hz (None => auto-select for DEPOXY)
    order     : filter order

    Returns
    -------
    pd.DataFrame — filtered copy
    """
    df = nirs_df.copy()
    fs = estimate_fs(df["time_sec"])
    if np.isnan(fs) or fs <= 0:
        print("  ⚠ NIRS: Cannot estimate fs. Skipping filter.")
        return df

    if cutoff_hz is None:
        rep_col = next((c for c in ["TSI_Quad", "SmO2_Quad", "HHb_Quad", "TSI_Calf"] if c in df.columns), None)
        if rep_col is not None:
            cutoff_hz = _choose_best_cutoff(df[rep_col], fs, NIRS_CUTOFF_CANDIDATES_HZ, order)
        else:
            cutoff_hz = 0.10

    target_cols = [c for c in df.columns if c != "time_sec"]
    n_filtered = 0
    for col in target_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        try:
            series_interp = series.interpolate(method="linear", limit_direction="both")
            filtered = butterworth_lowpass(series_interp.values, cutoff_hz, fs, order)
            filtered_full = series.copy().astype(float)
            filtered_full[:] = filtered
            filtered_full[series.isna()] = np.nan
            df[col] = filtered_full
            n_filtered += 1
        except Exception as e:
            print(f"  ⚠ NIRS filter failed for {col}: {e}")

    print(f"✓ NIRS filtered ({n_filtered} channels, cutoff={cutoff_hz} Hz, fs≈{fs:.2f} Hz)")
    return df


def filter_force_plate(fp_df: pd.DataFrame,
                       cutoff_hz: float | None = None,
                       order: int = 4) -> pd.DataFrame:
    """
    Apply Butterworth low-pass filter to force plate channels.

    Force plate is typically sampled at 1000 Hz+; a 20 Hz cutoff removes
    high-frequency noise while preserving ground reaction force waveforms.

    Parameters
    ----------
    fp_df     : DataFrame with 'time_sec' and force columns
    cutoff_hz : low-pass cutoff in Hz (None => auto-select for DEPOXY)
    order     : filter order

    Returns
    -------
    pd.DataFrame — filtered copy
    """
    df = fp_df.copy()
    fs = estimate_fs(df["time_sec"])
    if np.isnan(fs) or fs <= 0:
        print("  ⚠ Force plate: Cannot estimate fs. Skipping filter.")
        return df

    force_cols = [c for c in df.columns if "sensor" in c.lower() or "force" in c.lower()]

    if cutoff_hz is None:
        rep_col = next((c for c in ["total_force_N", "sensor1_N", "sensor1_V"] if c in df.columns), None)
        if rep_col is not None:
            cutoff_hz = _choose_best_cutoff(df[rep_col], fs, FP_CUTOFF_CANDIDATES_HZ, order)
        else:
            cutoff_hz = 20.0

    n_filtered = 0
    for col in force_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        try:
            series_interp = series.interpolate(method="linear", limit_direction="both")
            filtered = butterworth_lowpass(series_interp.values, cutoff_hz, fs, order)
            df[col] = filtered
            n_filtered += 1
        except Exception as e:
            print(f"  ⚠ Force filter failed for {col}: {e}")

    print(f"✓ Force plate filtered ({n_filtered} channels, cutoff={cutoff_hz} Hz, fs≈{fs:.1f} Hz)")
    return df


def smooth_cosmed(k5_df: pd.DataFrame,
                  window_sec: float | None = None) -> pd.DataFrame:
    """
    Optional: apply a rolling-median smoother to COSMED breath-by-breath data.

    Parameters
    ----------
    k5_df      : DataFrame with 'time_sec' and metabolic columns
    window_sec : smoothing window in seconds (None => auto-select for DEPOXY)

    Returns
    -------
    pd.DataFrame — smoothed copy (time column unchanged)
    """
    df = k5_df.copy()
    fs = estimate_fs(df["time_sec"])
    if np.isnan(fs) or fs <= 0:
        print("  ⚠ COSMED: Cannot estimate fs. Skipping smoother.")
        return df

    if window_sec is None:
        rep_col = next((c for c in ["VO2_ml_min", "HR_bpm", "VE_L_min"] if c in df.columns), None)
        if rep_col is not None:
            window_sec = _choose_best_window(df[rep_col], fs, COSMED_WINDOW_CANDIDATES_SEC)
        else:
            window_sec = 10.0

    window_samples = max(1, int(round(window_sec * fs)))
    smooth_cols = ["VO2_ml_min", "VCO2_ml_min", "RER", "VE_L_min", "Freq_Resp", "VT_L", "HR_bpm"]
    for col in smooth_cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .rolling(window=window_samples, center=True, min_periods=1)
                .median()
            )

    print(f"✓ COSMED smoothed (rolling median, window={window_sec}s, ~{window_samples} samples)")
    return df


# ---------------------------------------------------------------------------
# Validation block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Allow running this file directly: python sources/filtering.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from sources.load_data import load_k5_data, load_nirs_data, load_force_plate_data

    print("\n" + "=" * 70)
    print("VALIDATION: filtering.py")
    print("=" * 70)

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"

    try:
        print("\n[1] Loading data ...")
        k5 = load_k5_data(data_dir / "Données K5 Allan.csv")
        nirs = load_nirs_data(data_dir / "Données NIRS Allan.csv")
        fp = load_force_plate_data(data_dir / "Données FP Allan.csv")
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        raise SystemExit(1)

    # Keep force-plate test reasonably fast while preserving frequency content.
    fp_test = fp.head(min(len(fp), 300000)).copy()

    print("\n[2] Applying filters ...")
    nirs_f = filter_nirs(nirs, cutoff_hz=None, order=4)
    fp_f = filter_force_plate(fp_test, cutoff_hz=None, order=4)
    k5_s = smooth_cosmed(k5, window_sec=None)

    print("\n[3] Validating outputs ...")
    checks_ok = True

    # Length / time invariance checks
    if len(nirs_f) != len(nirs) or len(fp_f) != len(fp_test) or len(k5_s) != len(k5):
        print("  ❌ Length mismatch after filtering")
        checks_ok = False
    else:
        print("  ✓ Row counts preserved")

    if not nirs_f["time_sec"].equals(nirs["time_sec"]):
        print("  ❌ NIRS time axis modified")
        checks_ok = False
    else:
        print("  ✓ NIRS time axis preserved")

    if not fp_f["time_sec"].equals(fp_test["time_sec"]):
        print("  ❌ Force-plate time axis modified")
        checks_ok = False
    else:
        print("  ✓ Force-plate time axis preserved")

    # Quantify expected smoothing effect on representative channels
    def _noise_metric(before: pd.Series, after: pd.Series) -> tuple[float, float]:
        b = pd.to_numeric(before, errors="coerce").interpolate(limit_direction="both")
        a = pd.to_numeric(after, errors="coerce").interpolate(limit_direction="both")
        return float(np.nanstd(np.diff(b))), float(np.nanstd(np.diff(a)))

    rep_nirs_col = next((c for c in ["TSI_Quad", "SmO2_Quad", "HHb_Quad"] if c in nirs.columns), None)
    rep_fp_col = next((c for c in ["total_force_N", "sensor1_N", "sensor1_V"] if c in fp_test.columns), None)
    rep_k5_col = next((c for c in ["VO2_ml_min", "HR_bpm"] if c in k5.columns), None)

    if rep_nirs_col is not None:
        n_before, n_after = _noise_metric(nirs[rep_nirs_col], nirs_f[rep_nirs_col])
        print(f"  NIRS {rep_nirs_col}: diff-std {n_before:.4f} -> {n_after:.4f}")
    if rep_fp_col is not None:
        f_before, f_after = _noise_metric(fp_test[rep_fp_col], fp_f[rep_fp_col])
        print(f"  FP {rep_fp_col}: diff-std {f_before:.4f} -> {f_after:.4f}")
    if rep_k5_col is not None:
        k_before, k_after = _noise_metric(k5[rep_k5_col], k5_s[rep_k5_col])
        print(f"  K5 {rep_k5_col}: diff-std {k_before:.4f} -> {k_after:.4f}")

    print("\n" + "=" * 70)
    if checks_ok:
        print("✅ filtering.py works on project data")
    else:
        print("⚠ filtering.py completed with validation issues")
    print("=" * 70 + "\n")