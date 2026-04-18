"""
load_data.py — DEPOXY Project
==============================
Data loading and parsing for all four physiological modalities.

Canonical column naming:
  - TSI  : muscle oxygen saturation (SmO2 alias)
  - HHb  : deoxyhemoglobin
  - O2Hb : oxyhemoglobin  (= HbTot - HHb, computed here)
  - HbTot: total hemoglobin

Author: Baptiste AUGROS
"""

from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _time_mmss_to_sec(t):
    """Convert MM:SS or HH:MM:SS string to total seconds."""
    try:
        t_str = str(t).strip()
        parts = t_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        pass
    return np.nan


def _european_to_float(series):
    """Replace European comma decimals and parse to float."""
    return (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


# ---------------------------------------------------------------------------
# COSMED K5 loader
# ---------------------------------------------------------------------------

K5_MIN_DURATION_SEC = 75 * 60


def _impute_k5_time_grid(df: pd.DataFrame,
                         min_duration_sec: int = K5_MIN_DURATION_SEC,
                         time_col: str = "time_sec") -> pd.DataFrame:
    """Reindex K5 data on a 1-second grid and extend it to the protocol length."""
    if df.empty:
        return df.copy()

    df = df.copy().sort_values(time_col)

    event_col = "event" if "event" in df.columns else None
    value_cols = [c for c in df.columns if c not in {time_col, event_col}]

    agg_map: dict[str, str] = {col: "mean" for col in value_cols}
    if event_col is not None:
        agg_map[event_col] = "max"

    df = df.groupby(time_col, as_index=False).agg(agg_map)
    df = df.sort_values(time_col).reset_index(drop=True)

    start_time = float(pd.to_numeric(df[time_col], errors="coerce").min())
    observed_end = float(pd.to_numeric(df[time_col], errors="coerce").max())
    target_end = max(observed_end, float(min_duration_sec))

    full_time = np.arange(start_time, target_end + 1.0, 1.0)
    df = df.set_index(time_col).reindex(full_time)
    df.index.name = time_col

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(
            method="index",
            limit_direction="both",
        )

    if event_col is not None:
        df[event_col] = pd.to_numeric(df[event_col], errors="coerce").fillna(0)
    else:
        df["event"] = 0

    numeric_cols = [col for col in value_cols if col in df.columns]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    df = df.reset_index().rename(columns={"index": time_col})
    if event_col is not None:
        df[event_col] = df[event_col].fillna(0)
    return df

def load_k5_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load COSMED K5 metabolic data from CSV.

    The K5 export uses a semi-colon separator with a multi-line header.
    The function locates the 't' (time) column automatically and keeps
    physiologically relevant columns. An 'event' / 'marker' column is
    preserved for trigger-based synchronisation.

    Returns
    -------
    pd.DataFrame with columns (subset present in file):
        time_sec, VO2_ml_min, VCO2_ml_min, RER, VE_L_min, VT_L,
        Freq_Resp, HR_bpm, event
    """
    filepath = Path(filepath)

    # --- detect header row containing 't' ---
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        raw_lines = fh.readlines()

    t_header_row = None
    for i, line in enumerate(raw_lines):
        cols = [c.strip() for c in line.split(";")]
        if "t" in cols:
            t_header_row = i
            headers = cols
            break

    if t_header_row is None:
        raise ValueError(f"Could not find time column 't' in {filepath}")

    t_index = headers.index("t")
    headers = headers[t_index:]

    df = pd.read_csv(
        filepath,
        sep=";",
        skiprows=t_header_row + 1,
        header=None,
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=False,
    )
    df = df.iloc[:, t_index:]
    df.columns = headers[: len(df.columns)]
    df.columns = [str(c).strip() for c in df.columns]

    # --- canonical rename ---
    rename_map = {
        "t": "time_sec",
        "VO2": "VO2_ml_min",
        "VCO2": "VCO2_ml_min",
        "QR": "RER",
        "VE": "VE_L_min",
        "F Resp": "Freq_Resp",
        "VC": "VT_L",
        "HR Echant.": "HR_bpm",
        # common alternative spellings
        "HR": "HR_bpm",
        "Marker": "event",
        "Event": "event",
        "marker": "event",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # --- parse time ---
    df["time_sec"] = df["time_sec"].apply(_time_mmss_to_sec)

    # --- parse numeric columns ---
    numeric_cols = [
        "VO2_ml_min", "VCO2_ml_min", "RER", "VE_L_min",
        "Freq_Resp", "VT_L", "HR_bpm",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _european_to_float(df[col])

    # --- event column: keep as-is, fill NaN with 0 ---
    if "event" in df.columns:
        df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0)
    else:
        df["event"] = 0

    # --- keep only expected columns ---
    keep = ["time_sec"] + [c for c in numeric_cols if c in df.columns] + ["event"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.dropna(subset=["time_sec"]).reset_index(drop=True)

    raw_end = float(pd.to_numeric(df["time_sec"], errors="coerce").max())
    df = _impute_k5_time_grid(df, min_duration_sec=K5_MIN_DURATION_SEC)
    final_end = float(pd.to_numeric(df["time_sec"], errors="coerce").max())

    print(
        f"✓ K5 loaded: {len(df)} rows — raw {raw_end/60:.1f} min, "
        f"imputed {final_end/60:.1f} min (protocol target: 75.0 min)"
    )
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    return df


# ---------------------------------------------------------------------------
# Heart Rate (FC) loader
# ---------------------------------------------------------------------------

def load_hr_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load standalone heart rate (FC) data.

    Expected CSV format (Polar / Garmin export):
        Sample rate,Time,HR (bpm)
        <rate>,HH:MM:SS,<value>

    Returns
    -------
    pd.DataFrame with columns: time_sec, HR_bpm
    """
    filepath = Path(filepath)

    skiprows = 0
    # Prefer the actual sample table header when present.
    strict_header_fragments = ["sample rate", "time", "hr (bpm)"]
    fallback_fragments = ["time", "hr", "bpm", "heart"]
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            lower = line.lower()
            if all(f in lower for f in strict_header_fragments) and "," in line:
                skiprows = i
                break
        else:
            # Fallback for other exports: first likely HR table header.
            fh.seek(0)
            for i, line in enumerate(fh):
                lower = line.lower()
                if any(f in lower for f in fallback_fragments) and "," in line:
                    skiprows = i
                    break

    df = pd.read_csv(filepath, skiprows=skiprows, encoding="utf-8", on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]

    # locate time and HR columns by name fragments
    time_col = next(
        (c for c in df.columns if "time" in c.lower() or "heure" in c.lower()), None
    )
    hr_col = next(
        (c for c in df.columns if "hr" in c.lower() or "bpm" in c.lower() or "fc" in c.lower()), None
    )

    if time_col is None or hr_col is None:
        raise ValueError(
            f"Cannot identify time/HR columns. Found: {list(df.columns)}"
        )

    df = df[[time_col, hr_col]].copy()
    df.columns = ["time_raw", "HR_bpm"]

    df["time_sec"] = df["time_raw"].apply(_time_mmss_to_sec)
    df["HR_bpm"] = _european_to_float(df["HR_bpm"])
    df = df[["time_sec", "HR_bpm"]].dropna(subset=["time_sec"]).reset_index(drop=True)

    print(f"✓ HR (FC) loaded: {len(df)} rows — {df['time_sec'].max()/60:.1f} min")
    return df


# ---------------------------------------------------------------------------
# mNIRS loader (Train.red)
# ---------------------------------------------------------------------------

def load_nirs_data(filepath: str | Path, start_line: int = 61) -> pd.DataFrame:
    """
        Load mNIRS data from Train.red CSV export.

        For the DEPOXY Train.Red export used in this project, metadata/header
        lines are present above the tabular signal. Useful columns for the two
        sensors start at line 61 (1-based):
            - Sensor 1: vastus lateralis (quad)
            - Sensor 2: medial gastrocnemius (calf)

    Canonical output columns (per muscle site):
        TSI_Quad, TSI_Calf     — muscle O2 saturation (SmO2 alias)
        HHb_Quad, HHb_Calf     — deoxyhemoglobin (µM)
        HbTot_Quad, HbTot_Calf — total hemoglobin (µM)
        O2Hb_Quad, O2Hb_Calf  — oxyhemoglobin = HbTot - HHb (computed)

    SmO2_* aliases are preserved alongside TSI_* for traceability.

    Returns
    -------
    pd.DataFrame
    """
    filepath = Path(filepath)
    try:
        # Skip metadata and the sensor-ID row so the next row is the tabular header.
        df = pd.read_csv(
            filepath,
            sep=",",
            skiprows=start_line,
            encoding="utf-8",
            on_bad_lines="skip",
        )

        # Expected Train.Red layout by position:
        #   0 time, 3/9/10 sensor1 SmO2/HHb_u/THb_u, 13/19/20 sensor2 SmO2/HHb_u/THb_u.
        required_max_idx = 20
        if df.shape[1] <= required_max_idx:
            raise ValueError(
                f"Unexpected NIRS column count ({df.shape[1]}). Need > {required_max_idx}."
            )

        out = pd.DataFrame({
            "time_sec": _european_to_float(df.iloc[:, 0]),
            "TSI_Quad": _european_to_float(df.iloc[:, 3]),
            "HHb_Quad": _european_to_float(df.iloc[:, 9]),
            "HbTot_Quad": _european_to_float(df.iloc[:, 10]),
            "TSI_Calf": _european_to_float(df.iloc[:, 13]),
            "HHb_Calf": _european_to_float(df.iloc[:, 19]),
            "HbTot_Calf": _european_to_float(df.iloc[:, 20]),
        })
    except Exception:
        # Generic fallback for other possible exports.
        df = pd.read_csv(filepath, sep=None, engine="python", encoding="utf-8",
                         on_bad_lines="skip")
        df.columns = [str(c).strip() for c in df.columns]

        time_col = next(
            (c for c in df.columns if c.lower() in ("time", "temps", "t", "time_sec")), None
        )
        if time_col is None:
            for c in df.columns:
                test = pd.to_numeric(df[c], errors="coerce")
                if test.dropna().is_monotonic_increasing:
                    time_col = c
                    break
        if time_col is None:
            raise ValueError("Cannot find time column in NIRS file.")

        df["time_sec"] = _european_to_float(df[time_col])
        alias_map = {}
        for c in df.columns:
            cl = c.lower()
            if ("smo2" in cl or "tsi" in cl) and ("quad" in cl or "vastus" in cl or "thigh" in cl or "_1" in cl):
                alias_map["TSI_Quad"] = c
            elif ("smo2" in cl or "tsi" in cl) and ("calf" in cl or "gastro" in cl or "soleus" in cl or "_2" in cl):
                alias_map["TSI_Calf"] = c
            elif ("hhb" in cl or "deoxy" in cl) and ("quad" in cl or "vastus" in cl or "thigh" in cl or "_1" in cl):
                alias_map["HHb_Quad"] = c
            elif ("hhb" in cl or "deoxy" in cl) and ("calf" in cl or "gastro" in cl or "_2" in cl):
                alias_map["HHb_Calf"] = c
            elif ("hbtot" in cl or "thb" in cl or "total" in cl) and ("quad" in cl or "_1" in cl):
                alias_map["HbTot_Quad"] = c
            elif ("hbtot" in cl or "thb" in cl or "total" in cl) and ("calf" in cl or "_2" in cl):
                alias_map["HbTot_Calf"] = c

        out = pd.DataFrame({"time_sec": df["time_sec"]})
        for canonical, src in alias_map.items():
            out[canonical] = _european_to_float(df[src])

    # SmO2 aliases for traceability
    if "TSI_Quad" in out.columns:
        out["SmO2_Quad"] = out["TSI_Quad"]
    if "TSI_Calf" in out.columns:
        out["SmO2_Calf"] = out["TSI_Calf"]

    # --- compute O2Hb = HbTot - HHb ---
    for site in ("Quad", "Calf"):
        hbtot_col = f"HbTot_{site}"
        hhb_col = f"HHb_{site}"
        o2hb_col = f"O2Hb_{site}"
        if hbtot_col in out.columns and hhb_col in out.columns:
            out[o2hb_col] = out[hbtot_col] - out[hhb_col]

    out = out.dropna(subset=["time_sec"]).reset_index(drop=True)
    print(f"✓ NIRS loaded: {len(out)} rows — {out['time_sec'].max()/60:.1f} min")
    print(f"  Columns: {', '.join(out.columns.tolist())}")
    return out


# ---------------------------------------------------------------------------
# Force plate loader
# ---------------------------------------------------------------------------

FORCE_CALIBRATION_FACTOR = 100.0  # N/V  — adjust per calibration sheet


def load_force_plate_data(filepath: str | Path,
                          calibration_factor: float = FORCE_CALIBRATION_FACTOR) -> pd.DataFrame:
    """
    Load force plate CSV (4-sensor configuration).

    Converts voltage to Newtons using calibration_factor (N/V).
    Computes total vertical force as sum of 4 sensors.

    Returns
    -------
    pd.DataFrame with columns:
        time_sec,
        sensor1_V, sensor2_V, sensor3_V, sensor4_V,
        sensor1_N, sensor2_N, sensor3_N, sensor4_N,
        total_force_N
    """
    filepath = Path(filepath)

    # Dedicated parser for the known export format:
    # 4 metadata rows then repeated [time;value] pairs for 4 sensors.
    raw = pd.read_csv(
        filepath,
        sep=";",
        header=None,
        encoding="utf-8",
        on_bad_lines="skip",
    )

    out: pd.DataFrame | None = None
    if raw.shape[1] >= 8 and len(raw) > 5:
        data = raw.iloc[4:].reset_index(drop=True)
        time_series = _european_to_float(data.iloc[:, 0])
        if time_series.notna().sum() > 0:
            out = pd.DataFrame({"time_sec": time_series})
            total_N = pd.Series(np.zeros(len(out)))
            value_indices = [1, 3, 5, 7]
            for i, idx in enumerate(value_indices, start=1):
                if idx >= data.shape[1]:
                    continue
                v_col = f"sensor{i}_V"
                n_col = f"sensor{i}_N"
                out[v_col] = _european_to_float(data.iloc[:, idx])
                out[n_col] = out[v_col] * calibration_factor
                total_N = total_N.add(out[n_col].fillna(0))
            out["total_force_N"] = total_N

    # Fallback for other potential formats.
    if out is None:
        df = pd.read_csv(filepath, sep=None, engine="python", encoding="utf-8",
                         on_bad_lines="skip")
        df.columns = [str(c).strip() for c in df.columns]

        time_col = next(
            (c for c in df.columns if c.lower() in ("time", "temps", "t", "time_sec")), None
        )
        if time_col is None:
            df.insert(0, "time_sec", np.arange(len(df)) / 1000.0)  # generic fallback
            time_col = "time_sec"
        else:
            df["time_sec"] = _european_to_float(df[time_col])

        sensor_cols = [c for c in df.columns if "sensor" in c.lower() or "force" in c.lower() or c.startswith("F")]
        if len(sensor_cols) < 2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            sensor_cols = [c for c in numeric_cols if c != "time_sec"][:4]

        sensor_cols = sensor_cols[:4]
        out = pd.DataFrame({"time_sec": df["time_sec"]})
        total_N = pd.Series(np.zeros(len(df)))
        for i, sc in enumerate(sensor_cols, start=1):
            v_col = f"sensor{i}_V"
            n_col = f"sensor{i}_N"
            out[v_col] = _european_to_float(df[sc])
            out[n_col] = out[v_col] * calibration_factor
            total_N = total_N.add(out[n_col].fillna(0))
        out["total_force_N"] = total_N

    out = out.dropna(subset=["time_sec"]).reset_index(drop=True)
    print(f"✓ Force plate loaded: {len(out)} rows — {out['time_sec'].max()/60:.1f} min")
    print(f"  Calibration: {calibration_factor} N/V")
    return out
if __name__ == "__main__":
    from pathlib import Path

    print("\n=== TEST LOAD DATA MODULE ===")

    # Build data path from this file location so execution works from any CWD.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    try:
        k5 = load_k5_data(DATA_DIR / "Données K5 Allan.csv")
        hr = load_hr_data(DATA_DIR / "Données FC Allan.csv")
        nirs = load_nirs_data(DATA_DIR / "Données NIRS Allan.csv")
        fp = load_force_plate_data(DATA_DIR / "Données FP Allan.csv")

        print("\n✔ ALL DATA LOADED SUCCESSFULLY")
        print(k5.head())

    except Exception as e:
        print("❌ ERROR:", e)
    
