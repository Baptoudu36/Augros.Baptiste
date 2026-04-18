"""
synchronization.py — DEPOXY Project
=====================================
Trigger-based temporal alignment of all physiological signals.

Strategy
--------
The COSMED K5 CSV contains an 'event' column with non-zero values at
protocol trigger points. The first non-zero event is taken as t=0 (protocol
start). Every other modality is then shifted by an offset computed from
their own recording start relative to the COSMED reference.

If a modality has its own trigger column, it is used directly. Otherwise,
the user can supply an explicit `offset_sec` for that modality.

Author: Baptiste AUGROS
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core synchronisation
# ---------------------------------------------------------------------------

def _estimate_fs(time_sec: pd.Series | np.ndarray) -> float:
    t = np.asarray(pd.to_numeric(time_sec, errors="coerce"), dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 3:
        return np.nan
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return np.nan
    return float(1.0 / np.median(diffs))


def _add_manual_event(df: pd.DataFrame,
                      trigger_time: float,
                      time_col: str = "time_sec",
                      event_col: str = "event_manual") -> pd.DataFrame:
    """Return a copy with a synthetic manual trigger event at nearest sample."""
    out = df.copy()
    out[event_col] = 0
    t = pd.to_numeric(out[time_col], errors="coerce")
    if t.notna().sum() == 0:
        return out
    idx = (t - trigger_time).abs().idxmin()
    out.loc[idx, event_col] = 1
    return out


def infer_manual_trigger_time(df: pd.DataFrame,
                              modality: str = "generic",
                              time_col: str = "time_sec") -> float:
    """
    Infer a manual trigger timestamp from signal dynamics.

    The detector looks for the first sustained increase in temporal derivative
    relative to early recording baseline, using modality-specific channels.
    """
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}' for manual trigger inference.")

    t = pd.to_numeric(df[time_col], errors="coerce")
    if t.notna().sum() < 10:
        raise ValueError("Not enough valid time points to infer manual trigger.")

    modality = modality.lower()
    candidate_map = {
        "k5": ["VO2_ml_min", "VCO2_ml_min", "VE_L_min", "HR_bpm", "RER"],
        "nirs": ["HHb_Quad", "HHb_Calf", "TSI_Quad", "TSI_Calf", "SmO2_Quad", "SmO2_Calf"],
        "fp": ["total_force_N", "sensor1_N", "sensor1_V"],
        "hr": ["HR_bpm"],
    }
    candidates = candidate_map.get(modality, []) + [c for c in df.columns if c != time_col]

    fs = _estimate_fs(t)
    if np.isnan(fs) or fs <= 0:
        raise ValueError("Cannot estimate sampling frequency for manual trigger inference.")

    if modality == "fp":
        smooth_sec = 2.0
        sustain_sec = 1.0
        max_search_sec = 600.0
    else:
        smooth_sec = 15.0
        sustain_sec = 12.0
        max_search_sec = 1800.0

    smooth_w = max(3, int(round(smooth_sec * fs)))
    sustain_n = max(3, int(round(sustain_sec * fs)))

    best: tuple[float, float, str] | None = None
    t_num = pd.to_numeric(t, errors="coerce").values

    for col in candidates:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < max(50, sustain_n + 5):
            continue

        y = s.interpolate(limit_direction="both")
        y_s = y.rolling(window=smooth_w, center=True, min_periods=1).median().values
        d = np.abs(np.gradient(y_s, t_num))

        valid = np.isfinite(d) & np.isfinite(t_num)
        if valid.sum() < max(50, sustain_n + 5):
            continue

        t_valid = t_num[valid]
        d_valid = d[valid]

        t0 = np.nanmin(t_valid)
        t1 = np.nanmax(t_valid)
        baseline_limit = min(t0 + max(120.0, 0.15 * (t1 - t0)), t1)
        baseline_mask = t_valid <= baseline_limit
        if baseline_mask.sum() < 20:
            baseline_mask = np.arange(len(t_valid)) < max(20, int(0.2 * len(t_valid)))

        baseline_d = d_valid[baseline_mask]
        med_d = float(np.nanmedian(baseline_d))
        mad_d = float(np.nanmedian(np.abs(baseline_d - med_d)))
        if not np.isfinite(mad_d) or mad_d == 0:
            mad_d = float(np.nanstd(baseline_d))
        if not np.isfinite(mad_d) or mad_d == 0:
            continue

        # Level-change detector complements derivative detector for smoother shifts.
        y_valid = y_s[valid]
        baseline_y = y_valid[baseline_mask]
        med_y = float(np.nanmedian(baseline_y))
        mad_y = float(np.nanmedian(np.abs(baseline_y - med_y)))
        if not np.isfinite(mad_y) or mad_y == 0:
            mad_y = float(np.nanstd(baseline_y))
        if not np.isfinite(mad_y) or mad_y == 0:
            mad_y = 1e-9

        threshold_d = med_d + 4.5 * mad_d
        threshold_level = 4.0 * mad_y

        active_d = d_valid > threshold_d
        active_level = np.abs(y_valid - med_y) > threshold_level
        active = active_d | active_level

        # Ignore earliest seconds to avoid edge effects from filtering.
        active &= (t_valid >= (t0 + 10.0))
        # Prefer early protocol transitions; ignore very late changes.
        active &= (t_valid <= (t0 + min(max_search_sec, 0.6 * (t1 - t0))))

        streak = np.convolve(active.astype(int), np.ones(sustain_n, dtype=int), mode="same")
        hits = np.where(streak >= sustain_n)[0]
        if hits.size == 0:
            continue

        idx = int(hits[0])
        trig_t = float(t_valid[idx])
        score_d = float((d_valid[idx] - threshold_d) / (threshold_d + 1e-9))
        score_l = float((abs(y_valid[idx] - med_y) - threshold_level) / (threshold_level + 1e-9))
        score = max(score_d, score_l)

        if best is None or trig_t < best[0] or (np.isclose(trig_t, best[0]) and score > best[1]):
            best = (trig_t, score, col)

    if best is None:
        raise ValueError(f"Manual trigger inference failed for modality={modality}.")

    trig_t, score, col = best
    print(f"  → Manual trigger inferred at t = {trig_t:.2f} s from {col} (score={score:.2f})")
    return trig_t

def find_trigger_time(df: pd.DataFrame,
                      trigger_col: str = "event",
                      time_col: str = "time_sec") -> float:
    """
    Return the timestamp (in seconds) of the FIRST non-zero trigger event.
    Raises ValueError if no trigger is found.
    """
    if trigger_col not in df.columns:
        raise ValueError(f"Trigger column '{trigger_col}' not found. "
                         f"Available: {list(df.columns)}")
    mask = pd.to_numeric(df[trigger_col], errors="coerce").fillna(0) != 0
    triggers = df.loc[mask, time_col]
    if triggers.empty:
        raise ValueError("No non-zero event found in trigger column.")
    t0 = float(triggers.iloc[0])
    print(f"  → First trigger found at t = {t0:.2f} s ({t0/60:.2f} min)")
    return t0


def shift_time(df: pd.DataFrame,
               offset_sec: float,
               time_col: str = "time_sec") -> pd.DataFrame:
    """
    Subtract offset_sec from the time column so that t=0 corresponds
    to the protocol start trigger.
    """
    df = df.copy()
    df[time_col] = df[time_col] - offset_sec
    return df


def synchronize_all(k5_data: pd.DataFrame,
                    hr_data: pd.DataFrame,
                    nirs_data: pd.DataFrame,
                    fp_data: pd.DataFrame,
                    k5_trigger_col: str = "event",
                    hr_offset_sec: float | None = None,
                    nirs_offset_sec: float | None = None,
                    fp_offset_sec: float | None = None,
                    drop_pre_trigger: bool = True,
                    use_manual_trigger_inference: bool = True) -> dict[str, pd.DataFrame]:
    """
    Synchronise all four modalities to the COSMED K5 trigger.

    Parameters
    ----------
    k5_data : DataFrame
        COSMED K5 data with 'event' and 'time_sec' columns.
    hr_data, nirs_data, fp_data : DataFrame
        Other modalities, each with a 'time_sec' column.
    hr_offset_sec, nirs_offset_sec, fp_offset_sec : float, optional
        If the modality has its own trigger, pass None and the function
        will attempt to auto-detect it. Otherwise supply the recording
        start offset in seconds relative to the K5 trigger.
    drop_pre_trigger : bool
        If True, remove rows with time_sec < 0 (before protocol start).

    Returns
    -------
    dict with keys 'k5', 'hr', 'nirs', 'fp', each a synchronised DataFrame.
    """
    print("=" * 60)
    print("SIGNAL SYNCHRONISATION")
    print("=" * 60)

    # --- K5 reference trigger ---
    print("\n[K5] Detecting reference trigger ...")
    k5_with_events = k5_data.copy()
    try:
        t0_k5 = find_trigger_time(k5_with_events, trigger_col=k5_trigger_col)
    except ValueError as exc:
        print(f"  ⚠ {exc}")
        inferred = False
        if use_manual_trigger_inference:
            try:
                t0_k5 = infer_manual_trigger_time(k5_with_events, modality="k5")
                k5_with_events = _add_manual_event(k5_with_events, t0_k5)
                inferred = True
            except ValueError as exc2:
                print(f"  ⚠ {exc2}")
        if not inferred:
            # Last-resort fallback when no event can be inferred.
            t0_k5 = float(pd.to_numeric(k5_with_events["time_sec"], errors="coerce").dropna().iloc[0])
            print(f"  ⚠ Falling back to first K5 timestamp as t0: {t0_k5:.2f} s")
    k5_sync = shift_time(k5_with_events, offset_sec=t0_k5)

    def _sync_modality(df: pd.DataFrame, name: str,
                       explicit_offset: float | None) -> pd.DataFrame:
        print(f"\n[{name}] Synchronising ...")
        out_df = df.copy()
        if explicit_offset is not None:
            offset = explicit_offset
            print(f"  → Using provided offset: {offset:.2f} s")
        elif "event" in out_df.columns:
            try:
                offset = find_trigger_time(out_df, trigger_col="event")
            except ValueError:
                inferred = False
                if use_manual_trigger_inference:
                    modality = "generic"
                    lname = name.lower()
                    if "nirs" in lname:
                        modality = "nirs"
                    elif "force" in lname:
                        modality = "fp"
                    elif "hr" in lname or "fc" in lname:
                        modality = "hr"
                    try:
                        offset = infer_manual_trigger_time(out_df, modality=modality)
                        out_df = _add_manual_event(out_df, offset)
                        inferred = True
                    except ValueError as exc2:
                        print(f"  ⚠ {exc2}")
                if not inferred:
                    # Assume recording started at the same wall-clock time as K5.
                    offset = t0_k5
                    print(f"  ⚠ No trigger found in {name}; using K5 trigger offset: {offset:.2f} s")
        else:
            inferred = False
            if use_manual_trigger_inference:
                modality = "generic"
                lname = name.lower()
                if "nirs" in lname:
                    modality = "nirs"
                elif "force" in lname:
                    modality = "fp"
                elif "hr" in lname or "fc" in lname:
                    modality = "hr"
                try:
                    offset = infer_manual_trigger_time(out_df, modality=modality)
                    out_df = _add_manual_event(out_df, offset)
                    inferred = True
                except ValueError as exc2:
                    print(f"  ⚠ {exc2}")
            if not inferred:
                # Assume recording started at the same wall-clock time as K5.
                offset = t0_k5
                print(f"  → No trigger column — using K5 trigger offset: {offset:.2f} s")
        return shift_time(out_df, offset_sec=offset)

    hr_sync   = _sync_modality(hr_data,   "HR (FC)",    hr_offset_sec)
    nirs_sync = _sync_modality(nirs_data, "NIRS",       nirs_offset_sec)
    fp_sync   = _sync_modality(fp_data,   "ForcePlate", fp_offset_sec)

    if drop_pre_trigger:
        for label, dfs in [("K5", k5_sync), ("HR", hr_sync),
                            ("NIRS", nirs_sync), ("FP", fp_sync)]:
            n_before = len(dfs)
            dfs.drop(dfs[dfs["time_sec"] < 0].index, inplace=True)
            dfs.reset_index(drop=True, inplace=True)
            if len(dfs) < n_before:
                print(f"  [{label}] Dropped {n_before - len(dfs)} pre-trigger rows")

    print("\n✓ Synchronisation complete")
    return {"k5": k5_sync, "hr": hr_sync, "nirs": nirs_sync, "fp": fp_sync}


# ---------------------------------------------------------------------------
# Resampling utility (optional — brings all signals to a common grid)
# ---------------------------------------------------------------------------

def resample_to_common_grid(signals: dict[str, pd.DataFrame],
                             target_hz: float = 1.0,
                             time_col: str = "time_sec") -> dict[str, pd.DataFrame]:
    """
    Resample all signals to a common time grid using linear interpolation.

    Useful for building a merged DataFrame for correlation analyses.

    Parameters
    ----------
    signals : dict of DataFrames (output of synchronize_all)
    target_hz : float — target sampling rate in Hz
    
    Returns dict of resampled DataFrames, or empty dict if no temporal overlap.
    """
    # common time range = intersection of all signals
    t_min = max(df[time_col].min() for df in signals.values())
    t_max = min(df[time_col].max() for df in signals.values())
    
    # Check for temporal overlap
    if t_min >= t_max:
        print(f"⚠ No temporal overlap between signals. "
              f"Range limits: [{t_min:.1f}–{t_max:.1f}] s")
        return {}
    
    dt = 1.0 / target_hz
    common_t = np.arange(t_min, t_max + dt, dt)

    resampled = {}
    for name, df in signals.items():
        new_df = pd.DataFrame({time_col: common_t})
        for col in df.columns:
            if col == time_col:
                continue
            try:
                new_df[col] = np.interp(common_t, df[time_col].values,
                                        pd.to_numeric(df[col], errors="coerce").values)
            except Exception:
                pass
        resampled[name] = new_df

    print(f"✓ Resampled to {target_hz} Hz — common grid: {t_min:.1f}–{t_max:.1f} s "
          f"({len(common_t)} points)")
    return resampled


# ---------------------------------------------------------------------------
# Test/Validation block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Ajouter le parent directory pour les imports
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    from sources.load_data import (
        load_k5_data, load_hr_data, load_nirs_data, load_force_plate_data
    )

    print("\n" + "="*70)
    print("VALIDATION: synchronization.py")
    print("="*70)

    # --- Load data ---
    print("\n[1] Chargement des données ...")
    try:
        root = Path(__file__).resolve().parent.parent
        data = root / "data"

        k5 = load_k5_data(data / "Données K5 Allan.csv")
        hr = load_hr_data(data / "Données FC Allan.csv")
        nirs = load_nirs_data(data / "Données NIRS Allan.csv")
        fp = load_force_plate_data(data / "Données FP Allan.csv")
    except Exception as e:
        print(f"  ❌ Erreur chargement: {e}")
        exit(1)

    # --- Display raw data ---
    print("\n[2] Données brutes chargées :")
    print(f"  K5        : {len(k5):6d} lignes | Colonnes: {list(k5.columns)}")
    print(f"  HR (FC)   : {len(hr):6d} lignes | Colonnes: {list(hr.columns)}")
    print(f"  NIRS      : {len(nirs):6d} lignes | Colonnes: {list(nirs.columns)}")
    print(f"  ForcePlate: {len(fp):6d} lignes | Colonnes: {list(fp.columns)[:5]}...")

    # --- Verify HR column ---
    if "HR_bpm" not in hr.columns:
        print(f"  ⚠ Attention: colonne HR_bpm manquante dans HR!")
    else:
        hr_vals = hr["HR_bpm"].dropna()
        if len(hr_vals) > 0:
            print(f"  ✓ HR: {len(hr_vals)} valeurs bpm, range [{hr_vals.min():.0f}–{hr_vals.max():.0f}]")

    # --- Synchronize ---
    print("\n[3] Synchronisation ...")
    try:
        sync = synchronize_all(k5, hr, nirs, fp, drop_pre_trigger=True)
    except Exception as e:
        print(f"  ❌ Erreur synchro: {e}")
        exit(1)

    # --- Display synchronized data ---
    print("\n[4] Données synchronisées :")
    for name, df in sync.items():
        if len(df) > 0:
            t_min, t_max = df['time_sec'].min(), df['time_sec'].max()
            print(f"  {name:12}: {len(df):7d} lignes | "
                  f"t: {t_min:7.1f}–{t_max:7.1f} s ({(t_max-t_min)/60:.1f} min)")
        else:
            print(f"  {name:12}: VIDE ⚠")

    # --- Validate ---
    print("\n[5] Validations :")
    all_valid = True

    # K5 should not be empty
    if len(sync["k5"]) > 0:
        print(f"  ✓ K5 synchronisé: {len(sync['k5'])} lignes")
    else:
        print(f"  ❌ K5 vide après synchro")
        all_valid = False

    # NIRS should have quad+calf
    if len(sync["nirs"]) > 0:
        required = ["TSI_Quad", "TSI_Calf", "HHb_Quad", "HHb_Calf"]
        missing = [c for c in required if c not in sync["nirs"].columns]
        if missing:
            print(f"  ⚠ NIRS manque colonnes: {missing}")
            all_valid = False
        else:
            print(f"  ✓ NIRS a quad+calf (TSI, HHb, etc)")
    else:
        print(f"  ⚠ NIRS synchronisé vide")

    # HR should have HR_bpm
    if len(sync["hr"]) > 0 and "HR_bpm" in sync["hr"].columns:
        print(f"  ✓ HR synchronisé avec colonne HR_bpm")
    elif len(sync["hr"]) == 0:
        print(f"  ⚠ HR synchronisé vide (offset timestamp?)")
    else:
        print(f"  ⚠ HR manque colonne HR_bpm")
        all_valid = False

    # No negative times
    neg_vals = {}
    for name, df in sync.items():
        if len(df) > 0:
            n_neg = (df["time_sec"] < 0).sum()
            if n_neg > 0:
                neg_vals[name] = n_neg
    if neg_vals:
        print(f"  ⚠ Temps négatifs trouvés: {neg_vals}")
        all_valid = False
    else:
        print(f"  ✓ Aucun temps négatif")

    # --- Result ---
    print("\n" + "="*70)
    if all_valid:
        print("✅ synchronization.py FONCTIONNE - Tout est OK!")
    else:
        print("⚠ synchronization.py FONCTIONNE (avec warnings)")
    print("="*70 + "\n")
