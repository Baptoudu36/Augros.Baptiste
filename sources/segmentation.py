"""
segmentation.py — DEPOXY Project
===================================
Protocol timeline definition and data windowing.

DEPOXY protocol (ABCCBA design):
    Baseline_initial (5 min)
    Warmup_standardized (5 min @ 10 km/h, participant shoes)
    Recovery_pre_blocks (5 min)
    Block1 A → Recovery1
    Block2 B → Recovery2
    Block3 C → Recovery3
    Block4 C → Recovery4
    Block5 B → Recovery5
    Block6 A
    Baseline_final (5 min, if recorded)

Author: Baptiste AUGROS
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

PROTOCOL = {
    "baseline_duration": 300,   # 5 min
    "warmup_duration":   300,   # 5 min standardized warmup
    "pre_block_recovery_duration": 300,  # 5 min before block sequence
    "run_duration":      300,   # 5 min per running block
    "recovery_duration": 300,   # 5 min per recovery block
    "n_blocks": 6,
}

SHOE_CONDITIONS: dict[int, dict] = {
    1: {"code": "A", "name": "KipRun Foam",   "color": "#E63946"},
    2: {"code": "B", "name": "KipRun Carbon", "color": "#2A9D8F"},
    3: {"code": "C", "name": "Prototype",     "color": "#F4A261"},
    4: {"code": "C", "name": "Prototype",     "color": "#F4A261"},
    5: {"code": "B", "name": "KipRun Carbon", "color": "#2A9D8F"},
    6: {"code": "A", "name": "KipRun Foam",   "color": "#E63946"},
}

CONDITION_MAP: dict[str, list[int]] = {
    "A": [1, 6],
    "B": [2, 5],
    "C": [3, 4],
}


# ---------------------------------------------------------------------------
# Timeline builder
# ---------------------------------------------------------------------------

def create_protocol_timeline() -> dict[str, dict]:
    """
    Build a dictionary of phase timing for the full DEPOXY protocol.

        Keys: 'baseline', 'warmup', 'pre_block_recovery',
            'run_1'...'run_6', 'recovery_1'...'recovery_5',
          'baseline_final'.

    Each value is a dict with: start, end, phase, [block, shoe_code, ...].
    """
    timeline: dict[str, dict] = {}
    t = 0

    # Initial baseline
    timeline["baseline"] = {
        "start": t,
        "end":   t + PROTOCOL["baseline_duration"],
        "phase": "Baseline_initial",
    }
    t += PROTOCOL["baseline_duration"]

    # Standardized warmup (participant shoes)
    timeline["warmup"] = {
        "start": t,
        "end":   t + PROTOCOL["warmup_duration"],
        "phase": "Warmup_standardized",
        "speed_kmh": 10.0,
    }
    t += PROTOCOL["warmup_duration"]

    # Recovery before entering ABCCBA block sequence
    timeline["pre_block_recovery"] = {
        "start": t,
        "end":   t + PROTOCOL["pre_block_recovery_duration"],
        "phase": "Recovery_pre_blocks",
    }
    t += PROTOCOL["pre_block_recovery_duration"]

    for block in range(1, PROTOCOL["n_blocks"] + 1):
        shoe = SHOE_CONDITIONS[block]

        # Running block
        timeline[f"run_{block}"] = {
            "start":      t,
            "end":        t + PROTOCOL["run_duration"],
            "phase":      "Running",
            "block":      block,
            "shoe_code":  shoe["code"],
            "shoe_name":  shoe["name"],
            "color":      shoe["color"],
        }
        t += PROTOCOL["run_duration"]

        # Recovery block (no recovery after the last running block)
        if block < PROTOCOL["n_blocks"]:
            timeline[f"recovery_{block}"] = {
                "start": t,
                "end":   t + PROTOCOL["recovery_duration"],
                "phase": "Recovery",
                "block": block,
            }
            t += PROTOCOL["recovery_duration"]

    # Final baseline (if protocol includes it)
    timeline["baseline_final"] = {
        "start": t,
        "end":   t + PROTOCOL["baseline_duration"],
        "phase": "Baseline_final",
    }

    return timeline


TIMELINE = create_protocol_timeline()


# ---------------------------------------------------------------------------
# Windowing helpers
# ---------------------------------------------------------------------------

def extract_window(df: pd.DataFrame,
                   start_sec: float,
                   end_sec: float,
                   time_col: str = "time_sec") -> pd.DataFrame:
    """Return rows where time_col is in [start_sec, end_sec)."""
    if time_col not in df.columns:
        raise KeyError(f"Missing time column '{time_col}'. Available: {list(df.columns)}")
    if end_sec <= start_sec:
        raise ValueError(f"Invalid window: start={start_sec}, end={end_sec}.")
    mask = (df[time_col] >= start_sec) & (df[time_col] < end_sec)
    return df.loc[mask].copy().reset_index(drop=True)


def extract_steady_state_from_block(df: pd.DataFrame,
                                    block_start_sec: float,
                                    run_duration: float = 300,
                                    ss_start_offset: float = 120,
                                    ss_end_offset:   float = 270,
                                    time_col: str = "time_sec") -> tuple[pd.DataFrame, dict]:
    """
    Extract the steady-state window from a running block.

    Default window: 2:00 → 4:30 after block start (avoids 2 min adaptation
    and last 30 s of anticipatory withdrawal).

    Parameters
    ----------
    df               : full signal DataFrame with time_col
    block_start_sec  : absolute start time of the running block (seconds)
    run_duration     : total block duration (seconds)
    ss_start_offset  : offset from block start to begin SS (default 120 s)
    ss_end_offset    : offset from block start to end SS (default 270 s)
    time_col         : name of the time column

    Returns
    -------
    (steady_df, info_dict)
    """
    ss_start = block_start_sec + ss_start_offset
    ss_end   = block_start_sec + ss_end_offset
    steady_df = extract_window(df, ss_start, ss_end, time_col)
    info = {
        "start_time": ss_start,
        "end_time":   ss_end,
        "duration":   ss_end - ss_start,
        "n_samples":  len(steady_df),
    }
    return steady_df, info


def extract_baseline(df: pd.DataFrame,
                     phase: str = "initial",
                     timeline: dict | None = None,
                     time_col: str = "time_sec") -> tuple[pd.DataFrame, dict]:
    """
    Extract initial or final baseline window.

    Parameters
    ----------
    phase    : 'initial' → baseline / 'final' → baseline_final
    timeline : if None, uses module-level TIMELINE

    Returns
    -------
    (baseline_df, info_dict)
    """
    tl = timeline or TIMELINE
    key = "baseline" if phase == "initial" else "baseline_final"
    info_tl = tl[key]
    start, end = info_tl["start"], info_tl["end"]

    # If final baseline not recorded, fall back to last 5 min of data
    if phase == "final":
        max_t = float(pd.to_numeric(df[time_col], errors="coerce").max())
        if end > max_t:
            start = max(0.0, max_t - PROTOCOL["baseline_duration"])
            end   = max_t
            print(f"  ⚠ Final baseline: using last 5 min of data ({start/60:.1f}–{end/60:.1f} min)")

    baseline_df = extract_window(df, start, end, time_col)
    info = {"start_time": start, "end_time": end,
            "duration": end - start, "n_samples": len(baseline_df)}
    return baseline_df, info


def segment_all_phases(df: pd.DataFrame,
                        timeline: dict | None = None,
                        time_col: str = "time_sec",
                        include_empty: bool = False) -> dict[str, dict]:
    """
    Segment a DataFrame into all protocol phases.

    Returns
    -------
    dict[phase_name -> {"data": DataFrame, "info": dict}]
    """
    tl = timeline or TIMELINE
    segments: dict[str, dict] = {}
    for phase_name, phase_info in tl.items():
        seg_df = extract_window(df, phase_info["start"], phase_info["end"], time_col)
        if len(seg_df) > 0 or include_empty:
            info = dict(phase_info)
            info["n_samples"] = int(len(seg_df))
            segments[phase_name] = {"data": seg_df, "info": info}
    return segments


def validate_segmentation(df: pd.DataFrame,
                          timeline: dict | None = None,
                          time_col: str = "time_sec",
                          label: str = "signal") -> dict:
    """
    Validate phase coverage on one synchronized signal.

    Returns
    -------
    dict with counts and missing phases.
    """
    tl = timeline or TIMELINE
    segments = segment_all_phases(df, tl, time_col=time_col, include_empty=True)

    missing = [k for k, v in segments.items() if len(v["data"]) == 0]
    non_empty = len(segments) - len(missing)

    print(f"\n[{label}] Segmentation summary")
    print(f"  Non-empty phases: {non_empty}/{len(segments)}")
    if missing:
        print(f"  ⚠ Missing phases: {', '.join(missing)}")
    else:
        print("  ✓ All phases contain samples")

    # Focused check on running blocks (core protocol phases)
    run_missing = [f"run_{b}" for b in range(1, PROTOCOL["n_blocks"] + 1)
                   if len(segments.get(f"run_{b}", {}).get("data", [])) == 0]
    if run_missing:
        print(f"  ⚠ Missing running blocks: {', '.join(run_missing)}")
    else:
        print("  ✓ All running blocks present")

    return {
        "label": label,
        "n_total_phases": len(segments),
        "n_non_empty": non_empty,
        "missing_phases": missing,
        "missing_runs": run_missing,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_protocol_summary():
    print("=" * 70)
    print("DEPOXY PROTOCOL — ABCCBA DESIGN")
    print("=" * 70)
    print(f"  Baseline duration : {PROTOCOL['baseline_duration']/60:.0f} min")
    print(f"  Warmup            : {PROTOCOL['warmup_duration']/60:.0f} min (@10 km/h)")
    print(f"  Pre-block recovery: {PROTOCOL['pre_block_recovery_duration']/60:.0f} min")
    print(f"  Running block     : {PROTOCOL['run_duration']/60:.0f} min")
    print(f"  Recovery block    : {PROTOCOL['recovery_duration']/60:.0f} min")
    print(f"  Number of blocks  : {PROTOCOL['n_blocks']}")
    total = TIMELINE["baseline_final"]["end"]
    print(f"  Total duration    : {total/60:.0f} min")
    print()
    print(f"  {'Block':>6}  {'Condition':>10}  {'Shoe':<25}  {'Start':>6}  {'End':>6}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*25}  {'-'*6}  {'-'*6}")
    for b in range(1, PROTOCOL["n_blocks"] + 1):
        ri = TIMELINE[f"run_{b}"]
        print(f"  {b:>6}  {ri['shoe_code']:>10}  {ri['shoe_name']:<25}  "
              f"{ri['start']/60:>5.1f}m  {ri['end']/60:>5.1f}m")
    print("=" * 70)


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Allow running as script: python sources/segmentation.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from sources.load_data import load_k5_data, load_hr_data, load_nirs_data, load_force_plate_data
    from sources.synchronization import synchronize_all

    print_protocol_summary()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"

    print("\n" + "=" * 70)
    print("VALIDATION: segmentation.py")
    print("=" * 70)

    try:
        print("\n[1] Loading data ...")
        k5 = load_k5_data(data_dir / "Données K5 Allan.csv")
        hr = load_hr_data(data_dir / "Données FC Allan.csv")
        nirs = load_nirs_data(data_dir / "Données NIRS Allan.csv")
        fp = load_force_plate_data(data_dir / "Données FP Allan.csv")

        print("\n[2] Synchronizing data ...")
        synced = synchronize_all(k5, hr, nirs, fp, drop_pre_trigger=True)

        print("\n[3] Validating phase segmentation ...")
        results = [
            validate_segmentation(synced["k5"], label="K5"),
            validate_segmentation(synced["hr"], label="HR"),
            validate_segmentation(synced["nirs"], label="NIRS"),
            validate_segmentation(synced["fp"], label="ForcePlate"),
        ]

        has_critical_issue = any(len(r["missing_runs"]) > 0 for r in results if r["label"] in ("K5", "NIRS", "ForcePlate"))

        print("\n" + "=" * 70)
        if has_critical_issue:
            print("⚠ segmentation.py runs but some running blocks are missing")
            raise SystemExit(1)
        print("✅ segmentation.py works on synchronized project data")
        print("=" * 70 + "\n")
    except Exception as exc:
        print(f"❌ Segmentation validation failed: {exc}")
        raise SystemExit(1)
