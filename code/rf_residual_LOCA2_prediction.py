#!/usr/bin/env python3
# ==============================================================================
# File: rf_residual_LOCA2_prediction.py
# Author: Ziqi Wei (modified with ChatGPT)
# Date: 2026-02-09
#
# Description
#   Downscale LOCA2 daily max/min temperature (C) to hourly (C) using the
#   county-level RF residual model + scaler stored under:
#     {model_root}/{county}/enforced/
#
#   Calendar robustness (key for your case):
#     1) If a file contains INVALID dates like YYYY-02-29 in a non-leap year,
#        map them to YYYY-02-28 (same time) to avoid NaT.
#     2) If a leap year is MISSING Feb-29 (no-leap calendars), INSERT Feb-29 by
#        copying Feb-28 daily_max/min (consistent with your "copy neighbors"
#        policy when context is missing).
#     3) If fixes create duplicate days, collapse duplicates by:
#        daily_max = max(duplicates), daily_min = min(duplicates).
#
#   Features:
#     - 3-day window daily predictors: prev/today/next daily_max/min
#       (if missing prev/next day -> copy today's values)
#     - Cyclic time features: MonthSin/Cos, DayOfWeekSin/Cos, HourSin/Cos
#     - Baseline: 0.5*(daily_max+daily_min), residual predicted by RF
#     - Optional enforcement of daily min/max via per-day affine transform
#
# Output:
#   /nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/rf_residual_downscaling/{county}/{model}/{SSP}/
# ==============================================================================

from __future__ import annotations

import argparse
import calendar
import glob
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------
def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def set_rf_single_thread(model) -> None:
    """Avoid oversubscription on shared nodes."""
    try:
        if hasattr(model, "set_params"):
            model.set_params(n_jobs=1)
    except Exception:
        pass


# -----------------------------
# Daily min/max hard constraint
# -----------------------------
def enforce_daily_minmax(
    y_pred: pd.Series,
    daily_max: pd.Series,
    daily_min: pd.Series,
    template_peak_hour_utc: float = 15.0,
) -> pd.Series:
    """
    Enforce per-day min/max exactly via an affine map:
      y' = a*y + b such that y'(max_hour)=daily_max and y'(min_hour)=daily_min.

    If degenerate (pred max == pred min), fall back to a simple template
    (baseline with pinned peak/trough hours).
    """
    out = y_pred.copy()
    dates = pd.to_datetime(out.index).normalize().unique()

    for d in dates:
        day_mask = pd.to_datetime(out.index).normalize() == d
        y_day = out.loc[day_mask]
        if y_day.empty:
            continue

        if d not in daily_max.index or d not in daily_min.index:
            continue

        ymax = float(daily_max.loc[d])
        ymin = float(daily_min.loc[d])
        if not np.isfinite(ymax) or not np.isfinite(ymin):
            continue

        i_max = y_day.idxmax()
        i_min = y_day.idxmin()
        y_max_hat = float(y_day.loc[i_max])
        y_min_hat = float(y_day.loc[i_min])

        if np.isfinite(y_max_hat) and np.isfinite(y_min_hat) and (y_max_hat != y_min_hat):
            a = (ymax - ymin) / (y_max_hat - y_min_hat)
            b = ymax - a * y_max_hat
            out.loc[day_mask] = a * y_day + b
        else:
            # Degenerate fallback template
            hours = pd.to_datetime(y_day.index).hour.to_numpy()
            peak_h = int(round(template_peak_hour_utc)) % 24
            trough_h = (peak_h + 12) % 24

            y_new = pd.Series(0.5 * (ymax + ymin), index=y_day.index)

            if (hours == peak_h).any():
                y_new.iloc[np.where(hours == peak_h)[0][0]] = ymax
            else:
                y_new.iloc[0] = ymax

            if (hours == trough_h).any():
                y_new.iloc[np.where(hours == trough_h)[0][0]] = ymin
            else:
                y_new.iloc[-1] = ymin

            out.loc[day_mask] = y_new.values

    return out


# -----------------------------
# Calendar fixes (invalid 2/29 and missing 2/29)
# -----------------------------
def _fix_invalid_feb29_strings(dt_str: pd.Series) -> pd.Series:
    """
    If a file contains invalid 'YYYY-02-29 ...' for non-leap years, map to
    'YYYY-02-28 ...' (same time). This is only applied to rows that failed parsing.
    """
    s = dt_str.astype(str)

    parts = s.str.extract(
        r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\s+(?P<t>\d{2}:\d{2}:\d{2})"
    )
    y = parts["y"]
    m = parts["m"]
    d = parts["d"]
    t = parts["t"].fillna("12:00:00")

    feb29 = (m == "02") & (d == "29") & y.notna()
    fixed = s.copy()
    fixed.loc[feb29] = y.loc[feb29] + "-02-28 " + t.loc[feb29]
    return fixed


def insert_missing_feb29_noleap(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    If a leap year is missing Feb-29 but has Feb-28 (no-leap calendars),
    insert Feb-29 by copying Feb-28 daily_max/min.

    To avoid inserting Feb-29 for partial-year files that end at Feb-28,
    we only insert if that year contains any date strictly after Feb-28.
    """
    df = df_daily.copy().sort_values("date").reset_index(drop=True)
    years = sorted(df["date"].dt.year.unique().tolist())

    inserts = []
    for y in years:
        y = int(y)
        if not calendar.isleap(y):
            continue

        feb28 = pd.Timestamp(f"{y}-02-28")
        feb29 = pd.Timestamp(f"{y}-02-29")

        has_28 = (df["date"] == feb28).any()
        has_29 = (df["date"] == feb29).any()
        has_after_28 = (df["date"].dt.year == y).any() and (df["date"] > feb28).any()

        if has_28 and (not has_29) and has_after_28:
            r28 = df.loc[df["date"] == feb28].iloc[0]
            inserts.append(
                {
                    "date": feb29,
                    "date_time": feb29 + pd.Timedelta(hours=12),
                    "daily_max": float(r28["daily_max"]),
                    "daily_min": float(r28["daily_min"]),
                }
            )

    if not inserts:
        return df

    df2 = pd.concat([df, pd.DataFrame(inserts)], ignore_index=True)

    # Collapse any duplicates safely
    df2 = (
        df2.groupby("date", as_index=False)
        .agg(daily_max=("daily_max", "max"), daily_min=("daily_min", "min"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    df2["date_time"] = df2["date"] + pd.Timedelta(hours=12)
    return df2[["date_time", "date", "daily_max", "daily_min"]]


def read_dailymaxmin_csv(path: str) -> pd.DataFrame:
    """
    Read LOCA2 daily predictors:
      date_time,daily_max,daily_min
      2018-01-01 12:00:00,6.63,1.50

    Steps:
      1) Parse datetime; if invalid Feb-29 causes NaT, map to Feb-28 then reparse.
      2) Normalize to daily 'date' and collapse duplicates.
      3) For leap years missing Feb-29, insert Feb-29 by copying Feb-28.
      4) Collapse duplicates again (safe), sort by date.
    """
    df = pd.read_csv(path)

    if "date_time" not in df.columns:
        raise ValueError(f"Missing 'date_time' in {path}")
    if "daily_max" not in df.columns or "daily_min" not in df.columns:
        raise ValueError(f"Missing 'daily_max'/'daily_min' in {path}")

    df["daily_max"] = pd.to_numeric(df["daily_max"], errors="coerce")
    df["daily_min"] = pd.to_numeric(df["daily_min"], errors="coerce")
    if df["daily_max"].isna().any() or df["daily_min"].isna().any():
        bad = df[df["daily_max"].isna() | df["daily_min"].isna()].head(10)
        raise ValueError(f"Non-numeric daily_max/min in {path}. Examples:\n{bad}")

    dt_raw = df["date_time"].astype(str)
    dt = pd.to_datetime(dt_raw, errors="coerce")

    # Fix invalid Feb-29 strings if any NaT appear
    if dt.isna().any():
        fixed_str = _fix_invalid_feb29_strings(dt_raw)
        dt2 = pd.to_datetime(fixed_str, errors="coerce")

        mask_bad = dt.isna() & dt2.notna()
        dt.loc[mask_bad] = dt2.loc[mask_bad]

    if dt.isna().any():
        bad = df.loc[dt.isna(), ["date_time", "daily_max", "daily_min"]].head(10)
        raise ValueError(
            f"Failed to parse some date_time values in {path}.\nExamples:\n{bad}"
        )

    df["date_time"] = dt
    df["date"] = df["date_time"].dt.normalize()

    # Collapse duplicates by day (max for max, min for min)
    g = (
        df.groupby("date", as_index=False)
        .agg(daily_max=("daily_max", "max"), daily_min=("daily_min", "min"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    g["date_time"] = g["date"] + pd.Timedelta(hours=12)
    g = g[["date_time", "date", "daily_max", "daily_min"]]

    # Insert missing Feb-29 for no-leap calendars (your requested fix)
    g = insert_missing_feb29_noleap(g)

    return g


# -----------------------------
# 3-day window daily features
# -----------------------------
def add_prev_next_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 3-day window predictors (prev/today/next).
    If prev/next day is missing (edge or gap), copy today's values.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    prev_date = df["date"].shift(1)
    next_date = df["date"].shift(-1)

    expected_prev = df["date"] - pd.Timedelta(days=1)
    expected_next = df["date"] + pd.Timedelta(days=1)

    prev_ok = prev_date == expected_prev
    next_ok = next_date == expected_next

    prev_max = df["daily_max"].shift(1)
    prev_min = df["daily_min"].shift(1)
    next_max = df["daily_max"].shift(-1)
    next_min = df["daily_min"].shift(-1)

    df["daily_max_prev"] = np.where(prev_ok, prev_max, df["daily_max"])
    df["daily_min_prev"] = np.where(prev_ok, prev_min, df["daily_min"])
    df["daily_max_next"] = np.where(next_ok, next_max, df["daily_max"])
    df["daily_min_next"] = np.where(next_ok, next_min, df["daily_min"])

    # Aliases (to match possible feature names in inference_config.json)
    df["prev_daily_max"] = df["daily_max_prev"]
    df["prev_daily_min"] = df["daily_min_prev"]
    df["next_daily_max"] = df["daily_max_next"]
    df["next_daily_min"] = df["daily_min_next"]

    df["daily_max_lag1"] = df["daily_max_prev"]
    df["daily_min_lag1"] = df["daily_min_prev"]
    df["daily_max_lead1"] = df["daily_max_next"]
    df["daily_min_lead1"] = df["daily_min_next"]

    return df


# -----------------------------
# Daily -> hourly expansion (no continuity assumption)
# -----------------------------
def expand_daily_to_hourly(df_daily: pd.DataFrame, days_per_chunk: int = 365) -> List[pd.DataFrame]:
    """
    Robust daily->hourly expansion:
      - Do NOT assume daily dates are continuous.
      - Generate exactly 24 rows per existing day in the input file.
    """
    chunks: List[pd.DataFrame] = []
    if len(df_daily) == 0:
        return chunks

    df_daily = df_daily.sort_values("date").reset_index(drop=True)
    daily_cols = [c for c in df_daily.columns if c not in ["date_time", "date"]]

    for start in range(0, len(df_daily), days_per_chunk):
        sub = df_daily.iloc[start:start + days_per_chunk].copy()

        dates = sub["date"].to_numpy()
        rep_dates = np.repeat(dates, 24)
        hours = np.tile(np.arange(24, dtype=int), len(sub))
        date_time = pd.to_datetime(rep_dates) + pd.to_timedelta(hours, unit="h")

        df_hour = pd.DataFrame({"date_time": date_time})
        df_hour["date"] = pd.to_datetime(rep_dates)

        # Broadcast daily predictors by repetition (no join -> no NaNs from gaps)
        for c in daily_cols:
            df_hour[c] = np.repeat(sub[c].to_numpy(), 24)

        chunks.append(df_hour)

    return chunks


def add_time_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic time features for hourly timestamps."""
    t = df["date_time"]
    month = t.dt.month.astype(float)
    dow = t.dt.weekday.astype(float)
    hour = t.dt.hour.astype(float)

    df["MonthSin"] = np.sin(2 * np.pi * month / 12.0)
    df["MonthCos"] = np.cos(2 * np.pi * month / 12.0)
    df["DayOfWeekSin"] = np.sin(2 * np.pi * dow / 7.0)
    df["DayOfWeekCos"] = np.cos(2 * np.pi * dow / 7.0)
    df["HourSin"] = np.sin(2 * np.pi * hour / 24.0)
    df["HourCos"] = np.cos(2 * np.pi * hour / 24.0)

    return df


def compute_baseline(df_hour: pd.DataFrame, baseline_spec: str) -> np.ndarray:
    """Compute baseline hourly temperature from daily inputs."""
    spec = (baseline_spec or "").strip()
    if spec == "" or spec == "0.5*(daily_max+daily_min)":
        return 0.5 * (df_hour["daily_max"].to_numpy() + df_hour["daily_min"].to_numpy())
    raise ValueError(f"Unsupported baseline spec: {baseline_spec}")


# -----------------------------
# Input discovery (LOCA2 synth tree)
# -----------------------------
def list_models_ssps(input_root: str, county: str) -> List[Tuple[str, str]]:
    base = os.path.join(input_root, county, "SYNTH_ENSEMBLES")
    if not os.path.isdir(base):
        return []
    pairs: List[Tuple[str, str]] = []
    for model_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(model_dir):
            continue
        model_name = os.path.basename(model_dir)
        for ssp_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
            if not os.path.isdir(ssp_dir):
                continue
            ssp_name = os.path.basename(ssp_dir)
            pairs.append((model_name, ssp_name))
    return pairs


def list_target_files(input_root: str, county: str, model_name: str, ssp: str) -> List[str]:
    base = os.path.join(
        input_root,
        county,
        "SYNTH_ENSEMBLES",
        model_name,
        ssp,
        "from_CESM2-LENS_SSP370",
    )
    pattern = os.path.join(base, "r*dailymaxmin_synth.csv")
    return sorted(glob.glob(pattern))


def make_output_path(output_root: str, county: str, model_name: str, ssp: str, in_csv: str) -> str:
    out_dir = os.path.join(output_root, county, model_name, ssp)
    safe_makedirs(out_dir)

    base = os.path.basename(in_csv)
    base = base.replace("dailymaxmin", "hourly")
    base = re.sub(r"_synth\.csv$", "_hourly_synth.csv", base)
    return os.path.join(out_dir, base)


# -----------------------------
# Load enforced artifacts
# -----------------------------
def load_county_artifacts(model_root: str, county: str, subdir: str = "enforced"):
    county_dir = os.path.join(model_root, county, subdir)
    if not os.path.isdir(county_dir):
        raise FileNotFoundError(f"Missing county model directory: {county_dir}")

    cfg_path = os.path.join(county_dir, "inference_config.json")
    model_path = os.path.join(county_dir, "rf_resid_model.pkl")
    scaler_path = os.path.join(county_dir, "scaler_residual.pkl")

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    cfg = read_json(cfg_path)
    model = joblib_load(model_path)
    scaler = joblib_load(scaler_path)
    set_rf_single_thread(model)

    return cfg, model, scaler


# -----------------------------
# Core downscaling (chunked)
# -----------------------------
def downscale_one_file_chunked(
    in_csv: str,
    out_csv: str,
    cfg: dict,
    model,
    scaler,
    days_per_chunk: int = 365,
) -> None:
    # If target output already exists, skip the entire prediction workflow.
    if os.path.exists(out_csv):
        return

    df_daily = read_dailymaxmin_csv(in_csv)
    df_daily = add_prev_next_daily(df_daily)

    # For optional enforcement (index by date)
    daily_max_s = df_daily.set_index("date")["daily_max"]
    daily_min_s = df_daily.set_index("date")["daily_min"]

    feature_cols: List[str] = cfg["feature_cols"]
    enforce_flag: bool = bool(cfg.get("enforce_daily_minmax", True))
    template_peak_hour_utc: float = float(cfg.get("template_peak_hour_utc", 15.0))
    baseline_spec: str = str(cfg.get("baseline", "0.5*(daily_max+daily_min)"))

    chunks = expand_daily_to_hourly(df_daily, days_per_chunk=days_per_chunk)

    if os.path.exists(out_csv):
        os.remove(out_csv)
    header_written = False

    for df_hour in chunks:
        df_hour = add_time_cyclic_features(df_hour)

        missing = [c for c in feature_cols if c not in df_hour.columns]
        if missing:
            raise ValueError(
                f"Missing required features: {missing}\n"
                f"Available cols: {list(df_hour.columns)}\n"
                f"File: {in_csv}"
            )

        X = df_hour[feature_cols]
        Xs = scaler.transform(X)

        baseline = compute_baseline(df_hour, baseline_spec)
        r_hat = model.predict(Xs)
        y_hat = baseline + r_hat

        hourly_series = pd.Series(y_hat, index=df_hour["date_time"].values)

        if enforce_flag:
            hourly_series = enforce_daily_minmax(
                hourly_series,
                daily_max=daily_max_s,
                daily_min=daily_min_s,
                template_peak_hour_utc=template_peak_hour_utc,
            )

        out_chunk = pd.DataFrame(
            {
                "date_time": pd.to_datetime(hourly_series.index),
                "temperature_hourly": hourly_series.values,
            }
        )
        out_chunk.to_csv(out_csv, index=False, mode="a", header=not header_written)
        header_written = True

        # Free memory explicitly (helps for long runs)
        del df_hour, X, Xs, baseline, r_hat, y_hat, hourly_series, out_chunk


def process_one_county(
    county: str,
    input_root: str,
    model_root: str,
    output_root: str,
    days_per_chunk: int,
    subdir: str = "enforced",
) -> None:
    cfg, model, scaler = load_county_artifacts(model_root, county, subdir=subdir)

    pairs = list_models_ssps(input_root, county)
    if not pairs:
        print(f"[WARN] No model/SSP folders for county={county}")
        return

    all_files: List[Tuple[str, str, str]] = []
    for model_name, ssp in pairs:
        files = list_target_files(input_root, county, model_name, ssp)
        for f in files:
            all_files.append((model_name, ssp, f))

    if not all_files:
        print(f"[WARN] No daily synth CSVs for county={county}")
        return

    for model_name, ssp, in_csv in tqdm(all_files, desc=f"{county}", leave=True):
        out_csv = make_output_path(output_root, county, model_name, ssp, in_csv)
        try:
            downscale_one_file_chunked(
                in_csv=in_csv,
                out_csv=out_csv,
                cfg=cfg,
                model=model,
                scaler=scaler,
                days_per_chunk=days_per_chunk,
            )
        except Exception as e:
            print(
                f"\n[ERROR] Failed on: {in_csv}\n"
                f"        Output: {out_csv}\n"
                f"        Reason: {repr(e)}\n"
            )


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOCA2 daily->hourly downscaling using county RF residual models (with Feb-29 handling)."
    )
    p.add_argument("--county", required=True, help="County id, e.g., G0100730")
    p.add_argument("--input_root", required=True, help="Root of LOCA2 synth data")
    p.add_argument("--model_root", required=True, help="Root of trained models")
    p.add_argument("--output_root", required=True, help="Output root")
    p.add_argument("--days_per_chunk", type=int, default=365, help="Days per inference chunk (RAM control).")
    p.add_argument(
        "--artifact_subdir",
        type=str,
        default="enforced",
        choices=["enforced", "raw"],
        help="Which subdir to use under each county model folder.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"County: {args.county}")
    print(f"Input root : {args.input_root}")
    print(f"Model root : {args.model_root}")
    print(f"Output root: {args.output_root}")
    print(f"Artifacts  : {args.artifact_subdir}")
    print(f"Chunk days : {args.days_per_chunk}")

    process_one_county(
        county=args.county,
        input_root=args.input_root,
        model_root=args.model_root,
        output_root=args.output_root,
        days_per_chunk=args.days_per_chunk,
        subdir=args.artifact_subdir,
    )


if __name__ == "__main__":
    main()

