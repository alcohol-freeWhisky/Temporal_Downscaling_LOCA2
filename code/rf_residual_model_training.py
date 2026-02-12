#!/usr/bin/env python3
"""
rf_residual_model_training.py
=============================

Purpose
-------
Train a county-level residual Random Forest model that converts daily temperature extremes
(daily_max, daily_min) into an hourly temperature profile.

Assumptions
-----------
- ALL temperatures are already in Celsius (C) for both daily inputs and hourly truth.

Key guarantees
--------------
1) The model uses only:
     - daily_max, daily_min (current day)
     - daily_max/daily_min from the previous day and next day
       (if a neighbor day is missing, it falls back to the current day)
     - cyclic calendar features: month, day-of-week
     - cyclic hourly feature: hour-of-day

2) The evaluation and inference pipeline includes an optional post-processing projection
   that enforces, for each UTC day, that:
        max(hourly_pred) == daily_max
        min(hourly_pred) == daily_min

Test period
-----------
Model evaluation and saved test predictions are restricted to years 2016..2024 (inclusive).
Training uses years <= 2015.

Output separation
-----------------
All artifacts are saved in two fully separated subfolders per county:
  - {OUT_ROOT}/{county}/raw/       : no daily min/max enforcement
  - {OUT_ROOT}/{county}/enforced/  : with daily min/max enforcement

Each variant folder contains its own:
  - rf_resid_model.pkl
  - scaler_residual.pkl
  - inference_config.json
  - evaluation_log.txt
  - plots/
  - test_predictions_2016_2024.csv

Concurrency safety (dedup / no double-write)
--------------------------------------------
If multiple Slurm array tasks accidentally map to the same county_id (duplicate locations,
tolerance collisions, or repeated submissions), they would write into the same output
directory and can create duplicated logs/plots. This script prevents that by acquiring an
atomic lock file under the county output directory. If the lock exists, the task will
skip that county.

Inputs
------
- County mapping CSV (lat/lon -> county id):
    /nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/file/County_selected_revise_3.csv

- Daily max/min predictors per county (Celsius):
    /nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/era5_dailymaxmin_county/era5_dailymaxmin_{county}.csv

- Hourly truth per location (Celsius):
    /nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/era5_county/era5_temperature_{lat}_{lon}.csv

Usage
-----
Train:
  python rf_residual_model_training.py --mode train --location \"{lat}_{lon}\"

Predict (uses enforced variant by default):
  python rf_residual_model_training.py --mode predict --location \"{lat}_{lon}\" \
        --daily_csv /path/to/dailymaxmin.csv --output_csv /path/to/hourly_out.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Paths / constants
# =============================================================================
MAP_CSV = "/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/file/County_selected_revise_3.csv"

DAILY_DIR = "/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/era5_dailymaxmin_county"
HOURLY_DIR = "/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/era5_county"

OUT_ROOT = "/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/rf_residual_model"

TRAIN_CUTOFF_YEAR = 2015
TEST_START_YEAR = 2016
TEST_END_YEAR = 2024

# Robustness: tolerate small numeric formatting mismatch between mapping and filename
LAT_TOL = 5e-4
LON_TOL = 5e-4

# Feature list (single source of truth for train/inference)
FEATURE_COLS = [
    "daily_max",
    "daily_min",
    "daily_max_prev",
    "daily_min_prev",
    "daily_max_next",
    "daily_min_next",
    "MonthSin",
    "MonthCos",
    "DayOfWeekSin",
    "DayOfWeekCos",
    "HourSin",
    "HourCos",
]


# =============================================================================
# Logging
# =============================================================================
def log_and_print(msg: str, log_file: str) -> None:
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


# =============================================================================
# Time handling
# =============================================================================
def to_utc_datetime(series: pd.Series) -> pd.DatetimeIndex:
    """Convert a timestamp-like Series into a UTC DatetimeIndex."""
    t = pd.to_datetime(series, utc=True, errors="coerce")
    if t.isna().any():
        bad = series[t.isna()].head(5).tolist()
        raise ValueError(f"Failed parsing timestamps (example): {bad}")
    return pd.DatetimeIndex(t)


def add_cyclic_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Add cyclic time features based on a UTC timestamp column.

    Adds:
      - _time (DatetimeIndex, UTC)
      - _date (python date)
      - Year, Month, DayOfWeek, Hour
      - MonthSin/Cos, DayOfWeekSin/Cos, HourSin/Cos
    """
    out = df.copy()
    t = to_utc_datetime(out[time_col])
    out["_time"] = t
    out["_date"] = t.date
    out["Year"] = t.year
    out["Month"] = t.month
    out["DayOfWeek"] = t.dayofweek
    out["Hour"] = t.hour

    # Month: 1..12
    out["MonthSin"] = np.sin(2.0 * np.pi * (out["Month"] - 1) / 12.0)
    out["MonthCos"] = np.cos(2.0 * np.pi * (out["Month"] - 1) / 12.0)

    # DayOfWeek: 0..6
    out["DayOfWeekSin"] = np.sin(2.0 * np.pi * out["DayOfWeek"] / 7.0)
    out["DayOfWeekCos"] = np.cos(2.0 * np.pi * out["DayOfWeek"] / 7.0)

    # Hour: 0..23
    out["HourSin"] = np.sin(2.0 * np.pi * out["Hour"] / 24.0)
    out["HourCos"] = np.cos(2.0 * np.pi * out["Hour"] / 24.0)

    return out


# =============================================================================
# County mapping (lat/lon -> county id)
# =============================================================================
def load_mapping_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"in.weather_file_latitude", "in.weather_file_longitude"}
    if not need.issubset(df.columns):
        raise ValueError(f"Mapping file missing required columns {need}: {path}")
    # County id is the 4th column per user (e.g., G0600850), keep it as string
    county_col = df.columns[3]
    df = df.copy()
    df["county_id"] = df[county_col].astype(str)
    df["lat"] = pd.to_numeric(df["in.weather_file_latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["in.weather_file_longitude"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return df


def match_county_id(mapping_df: pd.DataFrame, lat: float, lon: float) -> str:
    """
    Match a (lat,lon) to county_id using tolerance (because filenames may round).
    """
    d = (mapping_df["lat"] - lat).abs() + (mapping_df["lon"] - lon).abs()
    i = int(d.idxmin())
    best = mapping_df.loc[i]
    if abs(best["lat"] - lat) > LAT_TOL or abs(best["lon"] - lon) > LON_TOL:
        raise ValueError(
            f"Could not match location within tolerance. "
            f"Requested ({lat},{lon}), best ({best['lat']},{best['lon']})."
        )
    return str(best["county_id"])


# =============================================================================
# File locking (avoid duplicate writes per county)
# =============================================================================
def acquire_lock(lock_path: str) -> bool:
    """
    Atomically create lock file. Returns True if acquired, False if already exists.
    """
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"time={time.time()}\n")
        return True
    except FileExistsError:
        return False


def release_lock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


# =============================================================================
# Model training helpers
# =============================================================================
def train_or_load_residual_rf(
    X_train: np.ndarray,
    r_train: np.ndarray,
    X_test: np.ndarray,
    baseline_test: np.ndarray,
    model_path: str,
) -> Tuple[RandomForestRegressor, float, np.ndarray]:
    """
    Train (or load) a RandomForestRegressor to predict residuals.
    Returns (model, train_time_seconds, y_pred_raw_hourly_test).
    """
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # Predict anyway for consistent downstream pipeline
        r_hat = model.predict(X_test) if len(X_test) else np.array([])
        y_pred_raw = baseline_test + r_hat if len(baseline_test) else np.array([])
        return model, 0.0, y_pred_raw

    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, r_train)
    train_time = time.time() - t0

    joblib.dump(model, model_path)

    r_hat = model.predict(X_test) if len(X_test) else np.array([])
    y_pred_raw = baseline_test + r_hat if len(baseline_test) else np.array([])
    return model, train_time, y_pred_raw


# =============================================================================
# Enforcement: daily min/max projection
# =============================================================================
def enforce_daily_minmax(
    y_pred: np.ndarray,
    t: pd.DatetimeIndex,
    daily_max: np.ndarray,
    daily_min: np.ndarray,
    log_file: Optional[str] = None,
) -> np.ndarray:
    """
    For each UTC day, linearly rescale the 24 hourly predictions so that:
      - max == daily_max
      - min == daily_min

    If a day is degenerate (pred range ~ 0 or target range ~ 0), it falls back to
    a template diurnal shape centered at template_peak_hour_utc.
    """
    if len(y_pred) == 0:
        return y_pred

    if len(y_pred) != len(t) or len(t) != len(daily_max) or len(t) != len(daily_min):
        raise ValueError("Length mismatch in enforcement inputs.")

    out = np.array(y_pred, dtype=float, copy=True)

    # Group indices by day
    dates = t.date
    uniq_days, day_start_idx = np.unique(dates, return_index=True)

    days = 0
    scaled = 0
    template_used = 0
    skipped = 0

    # Template: fixed sinusoid peaking at 15 UTC
    template_peak_hour_utc = 15.0
    h = np.arange(24, dtype=float)
    template = np.sin(2.0 * np.pi * (h - template_peak_hour_utc) / 24.0)
    template = (template - template.min()) / (template.max() - template.min() + 1e-12)  # 0..1

    for i, day in enumerate(uniq_days):
        days += 1
        start = day_start_idx[i]
        end = day_start_idx[i + 1] if (i + 1) < len(day_start_idx) else len(out)

        # Safety: expect full days in training/eval, but handle partial gracefully
        idx = np.arange(start, end)
        y = out[idx]
        dmax = float(daily_max[start])
        dmin = float(daily_min[start])

        if len(idx) < 2:
            skipped += 1
            continue

        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        y_rng = y_max - y_min
        t_rng = dmax - dmin

        if (not np.isfinite(y_rng)) or (not np.isfinite(t_rng)):
            skipped += 1
            continue

        if abs(y_rng) < 1e-8 or abs(t_rng) < 1e-8:
            # Template fallback: place min/max exactly
            template_used += 1
            if len(idx) == 24:
                out[idx] = dmin + template * (t_rng)
            else:
                # Partial day: just shift/scale current within available hours
                th = np.linspace(0.0, 1.0, len(idx))
                out[idx] = dmin + th * (t_rng)
            continue

        # Linear rescale to match targets
        scaled += 1
        out[idx] = (y - y_min) / y_rng * t_rng + dmin

    if log_file is not None:
        log_and_print(f"[MinMaxEnforce] days={days}, scaled={scaled}, template_used={template_used}, skipped={skipped}", log_file)

    return out


def validate_enforcement(
    y_enforced: np.ndarray,
    t: pd.DatetimeIndex,
    daily_max: np.ndarray,
    daily_min: np.ndarray,
    atol: float = 1e-4,
) -> None:
    """Assert that max/min constraints hold per day (within tolerance)."""
    if len(y_enforced) == 0:
        return

    df = pd.DataFrame({"y": y_enforced, "dmax": daily_max, "dmin": daily_min}, index=t)
    g = df.groupby(df.index.date)
    max_err = (g["y"].max().to_numpy() - g["dmax"].first().to_numpy())
    min_err = (g["y"].min().to_numpy() - g["dmin"].first().to_numpy())
    if np.nanmax(np.abs(max_err)) > atol or np.nanmax(np.abs(min_err)) > atol:
        raise AssertionError(
            f"Daily min/max enforcement check failed. "
            f"max_err_max={np.nanmax(np.abs(max_err))}, min_err_max={np.nanmax(np.abs(min_err))}"
        )


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, title: str, log_file: str) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    log_and_print(f"{title} | MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}", log_file)


def perform_ks_test(y_true: np.ndarray, y_pred: np.ndarray, log_file: str) -> None:
    # Compare distributions of hourly values
    stat, pval = stats.ks_2samp(y_true, y_pred)
    log_and_print(f"KS test | stat={stat:.6f}, p-value={pval:.6g}", log_file)


def evaluate_hourly_by_hour_of_day(
    y_true: np.ndarray, y_pred: np.ndarray, t: pd.DatetimeIndex, plots_dir: str, log_file: str
) -> None:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=t)
    df["hour"] = df.index.hour

    by = df.groupby("hour").mean(numeric_only=True)
    bias = by["y_pred"] - by["y_true"]

    # Bias by hour
    plt.figure()
    plt.plot(bias.index.values, bias.values)
    plt.xlabel("Hour (UTC)")
    plt.ylabel("Mean bias (C)")
    plt.title("Hourly bias by hour (mean pred - mean true)")
    out1 = os.path.join(plots_dir, "hourly_bias_by_hour.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    log_and_print(f"Saved: {out1}", log_file)

    # Mean temperature by hour
    plt.figure()
    plt.plot(by.index.values, by["y_true"].values, label="true")
    plt.plot(by.index.values, by["y_pred"].values, label="pred")
    plt.xlabel("Hour (UTC)")
    plt.ylabel("Temperature (C)")
    plt.title("Hourly mean by hour")
    plt.legend()
    out2 = os.path.join(plots_dir, "hourly_mean_by_hour.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    log_and_print(f"Saved: {out2}", log_file)


def _circ_dist_hours(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular absolute distance on a 24-hour clock."""
    d = np.abs(a - b)
    return np.minimum(d, 24.0 - d)


def evaluate_daily_extreme_hour_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, t: pd.DatetimeIndex, plots_dir: str, log_file: str
) -> None:
    """
    Evaluate daily max/min VALUE errors and the TIMING errors (hour of day) for max/min.
    """
    df = pd.DataFrame({"true": y_true, "pred": y_pred}, index=t)
    g = df.groupby(df.index.date)

    # Daily extreme values
    true_max = g["true"].max()
    pred_max = g["pred"].max()
    true_min = g["true"].min()
    pred_min = g["pred"].min()

    max_mae = float(np.mean(np.abs(pred_max.values - true_max.values)))
    max_rmse = float(np.sqrt(np.mean((pred_max.values - true_max.values) ** 2)))
    min_mae = float(np.mean(np.abs(pred_min.values - true_min.values)))
    min_rmse = float(np.sqrt(np.mean((pred_min.values - true_min.values) ** 2)))

    log_and_print(
        f"Daily extreme VALUES | Max MAE={max_mae:.4f}, Max RMSE={max_rmse:.4f}, "
        f"Min MAE={min_mae:.4f}, Min RMSE={min_rmse:.4f}",
        log_file,
    )

    # Daily extreme timing (argmax/argmin hour)
    def arg_hour(s: pd.Series, is_max: bool) -> float:
        idx = s.idxmax() if is_max else s.idxmin()
        return float(idx.hour)

    true_max_h = g["true"].apply(lambda s: arg_hour(s, True)).to_numpy(dtype=float)
    pred_max_h = g["pred"].apply(lambda s: arg_hour(s, True)).to_numpy(dtype=float)
    true_min_h = g["true"].apply(lambda s: arg_hour(s, False)).to_numpy(dtype=float)
    pred_min_h = g["pred"].apply(lambda s: arg_hour(s, False)).to_numpy(dtype=float)

    max_h_err = _circ_dist_hours(pred_max_h, true_max_h)
    min_h_err = _circ_dist_hours(pred_min_h, true_min_h)

    max_h_mae = float(np.mean(max_h_err))
    min_h_mae = float(np.mean(min_h_err))
    max_acc_1h = float(np.mean(max_h_err <= 1.0))
    min_acc_1h = float(np.mean(min_h_err <= 1.0))

    log_and_print(
        f"Daily extreme TIMING | MaxHour MAE(circ)={max_h_mae:.3f}, MinHour MAE(circ)={min_h_mae:.3f}, "
        f"Acc<=1h (max/min)={max_acc_1h:.3f}/{min_acc_1h:.3f}",
        log_file,
    )

    # Histograms
    plt.figure()
    plt.hist(max_h_err, bins=np.arange(-0.5, 12.6, 1.0))
    plt.xlabel("Circular hour error (hours)")
    plt.ylabel("Count")
    plt.title("Daily max timing error histogram")
    p1 = os.path.join(plots_dir, "daily_max_timing_error_hist.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    log_and_print(f"Saved: {p1}", log_file)

    plt.figure()
    plt.hist(min_h_err, bins=np.arange(-0.5, 12.6, 1.0))
    plt.xlabel("Circular hour error (hours)")
    plt.ylabel("Count")
    plt.title("Daily min timing error histogram")
    p2 = os.path.join(plots_dir, "daily_min_timing_error_hist.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    log_and_print(f"Saved: {p2}", log_file)


def plot_predictions_timeseries(
    y_true: pd.Series, y_pred: pd.Series, plots_dir: str, log_file: str, fig_name: str
) -> None:
    """
    Plot a short time slice for sanity checking.
    """
    if len(y_true) == 0:
        return

    # First ~90 days
    start = y_true.index.min()
    end = start + pd.Timedelta(days=90)
    yt = y_true.loc[(y_true.index >= start) & (y_true.index <= end)]
    yp = y_pred.loc[(y_pred.index >= start) & (y_pred.index <= end)]

    plt.figure(figsize=(12, 4))
    plt.plot(yt.index, yt.values, label="true", linewidth=1)
    plt.plot(yp.index, yp.values, label="pred", linewidth=1)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Temperature (C)")
    plt.title(fig_name)
    plt.legend()
    out = os.path.join(plots_dir, f"{fig_name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log_and_print(f"Saved: {out}", log_file)


# =============================================================================
# Hourly file indexing and reading
# =============================================================================
HOURLY_RE = re.compile(r"^era5_temperature_(?P<lat>-?\d+(?:\.\d+)?)_(?P<lon>-?\d+(?:\.\d+)?)\.csv$")


def build_hourly_index(hourly_dir: str) -> List[Tuple[float, float, str]]:
    out: List[Tuple[float, float, str]] = []
    for fn in os.listdir(hourly_dir):
        m = HOURLY_RE.match(fn)
        if not m:
            continue
        out.append((float(m.group("lat")), float(m.group("lon")), fn))
    if not out:
        raise FileNotFoundError(f"No hourly files found in {hourly_dir} matching era5_temperature_*_*.csv")
    return out


def find_hourly_file(hourly_index: List[Tuple[float, float, str]], lat: float, lon: float) -> str:
    best_fn = None
    best_dist = 1e18
    best_lat = None
    best_lon = None
    for latf, lonf, fn in hourly_index:
        d = abs(latf - lat) + abs(lonf - lon)
        if d < best_dist:
            best_dist = d
            best_fn = fn
            best_lat, best_lon = latf, lonf

    assert best_fn is not None
    if abs(best_lat - lat) > LAT_TOL or abs(best_lon - lon) > LON_TOL:
        raise FileNotFoundError(
            f"Could not find hourly file within tolerance. Requested ({lat},{lon}), nearest ({best_lat},{best_lon})."
        )
    return os.path.join(HOURLY_DIR, best_fn)


# =============================================================================
# Data IO + merge: daily -> hourly
# =============================================================================
def _compute_prev_next_features(daily_unique: pd.DataFrame) -> pd.DataFrame:
    """
    Add prev/next day max/min features.

    Definition:
      - prev/next refer to adjacent CALENDAR days.
      - if the adjacent day is missing, fall back to the current day.

    Input must have one row per day, with columns:
      _date (python date), daily_max, daily_min
    """
    d = daily_unique.copy()
    d["_day"] = pd.to_datetime(d["_date"].astype(str))
    d = d.sort_values("_day")

    base = pd.DataFrame({"daily_max": d["daily_max"].to_numpy(), "daily_min": d["daily_min"].to_numpy()}, index=d["_day"])

    full_idx = pd.date_range(base.index.min(), base.index.max(), freq="D")
    base_full = base.reindex(full_idx)

    smax = base_full["daily_max"]
    smin = base_full["daily_min"]

    prev_max = smax.shift(1)
    prev_min = smin.shift(1)
    next_max = smax.shift(-1)
    next_min = smin.shift(-1)

    base_full["daily_max_prev"] = prev_max.where(~prev_max.isna(), smax)
    base_full["daily_min_prev"] = prev_min.where(~prev_min.isna(), smin)
    base_full["daily_max_next"] = next_max.where(~next_max.isna(), smax)
    base_full["daily_min_next"] = next_min.where(~next_min.isna(), smin)

    # Map back to original days only
    lookup = base_full.loc[base.index]
    d["daily_max_prev"] = lookup["daily_max_prev"].to_numpy()
    d["daily_min_prev"] = lookup["daily_min_prev"].to_numpy()
    d["daily_max_next"] = lookup["daily_max_next"].to_numpy()
    d["daily_min_next"] = lookup["daily_min_next"].to_numpy()

    d = d.drop(columns=["_day"])
    return d


def read_daily(daily_path: str) -> pd.DataFrame:
    df = pd.read_csv(daily_path)
    need = {"date_time", "daily_max", "daily_min"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Daily file {daily_path} missing columns: {miss}")

    df = add_cyclic_time_features(df, "date_time")

    # One row per UTC day. If duplicates exist, keep the first by time.
    df = df.sort_values("_time").drop_duplicates(subset=["_date"], keep="first").reset_index(drop=True)

    df["daily_max"] = df["daily_max"].astype(float)
    df["daily_min"] = df["daily_min"].astype(float)

    df = _compute_prev_next_features(df)

    # Baseline for residual definition (kept simple/deterministic)
    df["baseline"] = 0.5 * (df["daily_max"] + df["daily_min"])
    return df


def read_hourly(hourly_path: str) -> pd.DataFrame:
    df = pd.read_csv(hourly_path)
    need = {"Timestamp", "temperature"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Hourly file {hourly_path} missing columns: {miss}")

    df = add_cyclic_time_features(df, "Timestamp")
    df["temperature"] = df["temperature"].astype(float)
    return df


def build_hourly_table(daily_df: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Merge on UTC date to broadcast daily predictors to each hour."""
    daily_keep = daily_df[
        [
            "_date",
            "daily_max",
            "daily_min",
            "daily_max_prev",
            "daily_min_prev",
            "daily_max_next",
            "daily_min_next",
            "baseline",
        ]
    ].copy()

    merged = hourly_df.merge(daily_keep, on="_date", how="inner")
    if len(merged) == 0:
        raise ValueError("Empty daily->hourly merge. Check overlapping periods and UTC alignment.")

    merged["residual"] = merged["temperature"] - merged["baseline"]

    # Sanity: daily targets should be constant within each day after merge
    chk = merged.groupby("_date")[
        ["daily_max", "daily_min", "daily_max_prev", "daily_min_prev", "daily_max_next", "daily_min_next"]
    ].nunique()
    merged["_daily_target_inconsistent"] = bool(
        (chk.max().max() > 1)
    )

    return merged


# =============================================================================
# Inference helper: generate hourly predictions from daily max/min only
# =============================================================================
def expand_daily_to_hourly_grid(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a daily max/min table, expand each day into 24 hourly rows.

    The input must contain columns:
      - date_time, daily_max, daily_min

    Neighbor features (prev/next day max/min) are computed from calendar adjacency.
    """
    if "date_time" not in daily_df.columns:
        raise ValueError("daily_df must contain 'date_time' column.")

    tmp = daily_df.copy()
    tmp["date_time"] = pd.to_datetime(tmp["date_time"], utc=True, errors="coerce")
    tmp = tmp.dropna(subset=["date_time"]).sort_values("date_time")
    tmp["_time"] = pd.DatetimeIndex(tmp["date_time"])
    tmp["_date"] = tmp["_time"].date
    tmp["Year"] = tmp["_time"].year

    tmp["daily_max"] = tmp["daily_max"].astype(float)
    tmp["daily_min"] = tmp["daily_min"].astype(float)

    tmp = tmp.sort_values("_time").drop_duplicates(subset=["_date"], keep="first").reset_index(drop=True)
    tmp = _compute_prev_next_features(tmp)

    rows = []
    for _, r in tmp.iterrows():
        base = pd.Timestamp(r["date_time"])
        base = pd.Timestamp(year=base.year, month=base.month, day=base.day, tz="UTC")
        for h in range(24):
            rows.append(
                {
                    "Timestamp": base + pd.Timedelta(hours=h),
                    "daily_max": float(r["daily_max"]),
                    "daily_min": float(r["daily_min"]),
                    "daily_max_prev": float(r["daily_max_prev"]),
                    "daily_min_prev": float(r["daily_min_prev"]),
                    "daily_max_next": float(r["daily_max_next"]),
                    "daily_min_next": float(r["daily_min_next"]),
                }
            )

    out = pd.DataFrame(rows)
    out = add_cyclic_time_features(out, "Timestamp")
    out["baseline"] = 0.5 * (out["daily_max"] + out["daily_min"])
    return out


def predict_hourly_from_dailymaxmin(
    daily_df: pd.DataFrame,
    model: RandomForestRegressor,
    scaler: StandardScaler,
    enforce_minmax: bool,
    log_file: Optional[str] = None,
) -> pd.DataFrame:
    """Inference pipeline: daily max/min -> hourly grid -> residual prediction -> optional enforcement."""
    hourly_df = expand_daily_to_hourly_grid(daily_df)

    X_df = hourly_df[FEATURE_COLS].astype(float)
    Xs = scaler.transform(X_df)
    residual_hat = model.predict(Xs)

    y_raw = hourly_df["baseline"].to_numpy(dtype=float) + residual_hat
    t = pd.DatetimeIndex(hourly_df["_time"])

    if enforce_minmax:
        y = enforce_daily_minmax(
            y_pred=y_raw,
            t=t,
            daily_max=hourly_df["daily_max"].to_numpy(dtype=float),
            daily_min=hourly_df["daily_min"].to_numpy(dtype=float),
            log_file=log_file,
        )
        validate_enforcement(
            y_enforced=y,
            t=t,
            daily_max=hourly_df["daily_max"].to_numpy(dtype=float),
            daily_min=hourly_df["daily_min"].to_numpy(dtype=float),
        )
    else:
        y = y_raw

    out = pd.DataFrame(
        {
            "Timestamp": t.astype("datetime64[ns, UTC]"),
            "temperature_pred": y,
            "temperature_pred_raw": y_raw,
            "daily_max": hourly_df["daily_max"].to_numpy(dtype=float),
            "daily_min": hourly_df["daily_min"].to_numpy(dtype=float),
            "daily_max_prev": hourly_df["daily_max_prev"].to_numpy(dtype=float),
            "daily_min_prev": hourly_df["daily_min_prev"].to_numpy(dtype=float),
            "daily_max_next": hourly_df["daily_max_next"].to_numpy(dtype=float),
            "daily_min_next": hourly_df["daily_min_next"].to_numpy(dtype=float),
            "baseline": hourly_df["baseline"].to_numpy(dtype=float),
        }
    )
    return out


# =============================================================================
# Per-county training + evaluation
# =============================================================================
def _write_inference_config(out_dir: str, enforce_flag: bool) -> str:
    cfg = {
        "feature_cols": FEATURE_COLS,
        "train_cutoff_year": TRAIN_CUTOFF_YEAR,
        "test_year_start": TEST_START_YEAR,
        "test_year_end": TEST_END_YEAR,
        "enforce_daily_minmax": bool(enforce_flag),
        "template_peak_hour_utc": 15.0,
        "baseline": "0.5*(daily_max+daily_min)",
        "units": "C",
    }
    config_path = os.path.join(out_dir, "inference_config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return config_path


def run_one_county(
    county_id: str,
    daily_path: str,
    lat: float,
    lon: float,
    hourly_index: List[Tuple[float, float, str]],
) -> None:
    county_root = os.path.join(OUT_ROOT, county_id)
    os.makedirs(county_root, exist_ok=True)

    lock_path = os.path.join(county_root, ".training.lock")
    if not acquire_lock(lock_path):
        print(f"[SKIP] Another process is already training/writing county={county_id}. Lock: {lock_path}")
        return

    raw_dir = os.path.join(county_root, "raw")
    enf_dir = os.path.join(county_root, "enforced")
    raw_plots = os.path.join(raw_dir, "plots")
    enf_plots = os.path.join(enf_dir, "plots")
    os.makedirs(raw_plots, exist_ok=True)
    os.makedirs(enf_plots, exist_ok=True)

    raw_log = os.path.join(raw_dir, "evaluation_log.txt")
    enf_log = os.path.join(enf_dir, "evaluation_log.txt")

    try:
        # Header logs
        for lf in [raw_log, enf_log]:
            with open(lf, "w") as f:
                f.write(f"County: {county_id}\n")
                f.write(f"Daily input: {daily_path}\n")
                f.write(f"Task lat/lon: {lat}, {lon}\n")
                f.write(f"PID: {os.getpid()}\n")
                f.write(f"Start time (unix): {time.time()}\n")
                f.write(f"Units: Celsius (C)\n")

        def log_raw(msg: str) -> None:
            log_and_print(msg, raw_log)

        def log_enf(msg: str) -> None:
            log_and_print(msg, enf_log)

        hourly_path = find_hourly_file(hourly_index, lat, lon)
        log_raw(f"Hourly truth file: {hourly_path}")
        log_enf(f"Hourly truth file: {hourly_path}")

        daily_df = read_daily(daily_path)
        hourly_df = read_hourly(hourly_path)
        df = build_hourly_table(daily_df, hourly_df)

        if bool(df["_daily_target_inconsistent"].any()):
            msg = (
                "[WARN] Detected within-day inconsistencies in daily targets after merge. "
                "This usually means duplicate daily rows with different values. "
                "The model can still run, but results may be unreliable."
            )
            log_raw(msg)
            log_enf(msg)

        # Train: years <= 2015
        df_train = df[df["Year"] <= TRAIN_CUTOFF_YEAR].copy()
        # Test: years 2016..2024
        df_test = df[(df["Year"] >= TEST_START_YEAR) & (df["Year"] <= TEST_END_YEAR)].copy()

        msg = f"Merged hours: {len(df)} | Train: {len(df_train)} | Test(2016-2024): {len(df_test)}"
        log_raw(msg)
        log_enf(msg)

        if len(df_train) == 0:
            log_raw("ERROR: No training data after split. Check TRAIN_CUTOFF_YEAR or input range.")
            log_enf("ERROR: No training data after split. Check TRAIN_CUTOFF_YEAR or input range.")
            return

        X_train_df = df_train[FEATURE_COLS].astype(float)
        r_train = df_train["residual"].to_numpy(dtype=float)

        X_test_df = df_test[FEATURE_COLS].astype(float) if len(df_test) else pd.DataFrame(columns=FEATURE_COLS)
        y_true = df_test["temperature"].to_numpy(dtype=float) if len(df_test) else np.array([])
        baseline_test = df_test["baseline"].to_numpy(dtype=float) if len(df_test) else np.array([])
        t_test = pd.DatetimeIndex(df_test["_time"]) if len(df_test) else pd.DatetimeIndex([])

        daily_max_test = df_test["daily_max"].to_numpy(dtype=float) if len(df_test) else np.array([])
        daily_min_test = df_test["daily_min"].to_numpy(dtype=float) if len(df_test) else np.array([])

        # Fit scaler on DataFrame to retain feature order metadata when supported
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_df)
        X_test_s = scaler.transform(X_test_df) if len(df_test) else np.empty((0, len(FEATURE_COLS)))

        # Canonical artifacts are written under raw/, then duplicated under enforced/
        raw_scaler_path = os.path.join(raw_dir, "scaler_residual.pkl")
        raw_model_path = os.path.join(raw_dir, "rf_resid_model.pkl")
        joblib.dump(scaler, raw_scaler_path)

        model, train_time, y_pred_raw = train_or_load_residual_rf(X_train_s, r_train, X_test_s, baseline_test, raw_model_path)
        log_raw(f"Train time: {train_time:.2f}s")
        log_enf(f"Train time: {train_time:.2f}s")

        # Duplicate artifacts to enforced/ for fully separated outputs
        enf_scaler_path = os.path.join(enf_dir, "scaler_residual.pkl")
        enf_model_path = os.path.join(enf_dir, "rf_resid_model.pkl")
        joblib.dump(scaler, enf_scaler_path)
        joblib.dump(model, enf_model_path)

        # Save inference configs
        raw_cfg_path = _write_inference_config(raw_dir, enforce_flag=False)
        enf_cfg_path = _write_inference_config(enf_dir, enforce_flag=True)
        log_raw(f"Saved: {raw_cfg_path}")
        log_enf(f"Saved: {enf_cfg_path}")

        if len(df_test):
            # Enforced prediction (post-processed)
            y_pred_enf = enforce_daily_minmax(
                y_pred=y_pred_raw,
                t=t_test,
                daily_max=daily_max_test,
                daily_min=daily_min_test,
                log_file=enf_log,
            )
            validate_enforcement(y_pred_enf, t_test, daily_max_test, daily_min_test)

            # Save test (2016-2024) truth + preds (two variants)
            raw_test_csv = os.path.join(raw_dir, "test_predictions_2016_2024.csv")
            pd.DataFrame(
                {
                    "Timestamp": t_test.astype("datetime64[ns, UTC]"),
                    "temperature_true": y_true,
                    "temperature_pred": y_pred_raw,
                    "daily_max": daily_max_test,
                    "daily_min": daily_min_test,
                    "daily_max_prev": df_test["daily_max_prev"].to_numpy(dtype=float),
                    "daily_min_prev": df_test["daily_min_prev"].to_numpy(dtype=float),
                    "daily_max_next": df_test["daily_max_next"].to_numpy(dtype=float),
                    "daily_min_next": df_test["daily_min_next"].to_numpy(dtype=float),
                    "baseline": baseline_test,
                }
            ).to_csv(raw_test_csv, index=False)
            log_raw(f"Saved: {raw_test_csv}")

            enf_test_csv = os.path.join(enf_dir, "test_predictions_2016_2024.csv")
            pd.DataFrame(
                {
                    "Timestamp": t_test.astype("datetime64[ns, UTC]"),
                    "temperature_true": y_true,
                    "temperature_pred": y_pred_enf,
                    "daily_max": daily_max_test,
                    "daily_min": daily_min_test,
                    "daily_max_prev": df_test["daily_max_prev"].to_numpy(dtype=float),
                    "daily_min_prev": df_test["daily_min_prev"].to_numpy(dtype=float),
                    "daily_max_next": df_test["daily_max_next"].to_numpy(dtype=float),
                    "daily_min_next": df_test["daily_min_next"].to_numpy(dtype=float),
                    "baseline": baseline_test,
                }
            ).to_csv(enf_test_csv, index=False)
            log_enf(f"Saved: {enf_test_csv}")

            # Evaluation
            log_raw("=== Metrics on RAW predictions (no daily min/max enforcement) ===")
            evaluate_model(y_true, y_pred_raw, "Overall hourly (raw)", raw_log)
            perform_ks_test(y_true, y_pred_raw, raw_log)
            evaluate_hourly_by_hour_of_day(y_true, y_pred_raw, t_test, raw_plots, raw_log)
            evaluate_daily_extreme_hour_accuracy(y_true, y_pred_raw, t_test, raw_plots, raw_log)

            log_enf("=== Metrics on ENFORCED predictions (daily min/max guaranteed) ===")
            evaluate_model(y_true, y_pred_enf, "Overall hourly (enforced)", enf_log)
            perform_ks_test(y_true, y_pred_enf, enf_log)
            evaluate_hourly_by_hour_of_day(y_true, y_pred_enf, t_test, enf_plots, enf_log)
            evaluate_daily_extreme_hour_accuracy(y_true, y_pred_enf, t_test, enf_plots, enf_log)

            plot_predictions_timeseries(
                pd.Series(y_true, index=t_test),
                pd.Series(y_pred_raw, index=t_test),
                raw_plots,
                raw_log,
                "Pred_vs_True_first_90days_raw",
            )
            plot_predictions_timeseries(
                pd.Series(y_true, index=t_test),
                pd.Series(y_pred_enf, index=t_test),
                enf_plots,
                enf_log,
                "Pred_vs_True_first_90days_enforced",
            )

        else:
            log_raw("No test data to evaluate (2016-2024).")
            log_enf("No test data to evaluate (2016-2024).")

        gc.collect()
        log_raw("DONE.\n")
        log_enf("DONE.\n")

    finally:
        release_lock(lock_path)


# =============================================================================
# CLI / main
# =============================================================================
def parse_location(location: str) -> Tuple[float, float]:
    """location format: "{lat}_{lon}", e.g., "37.37_-121.93"""
    parts = location.strip().split("_")
    if len(parts) != 2:
        raise ValueError(f"Bad --location format: {location}. Expected '{{lat}}_{{lon}}'.")
    return float(parts[0]), float(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True, help="Location string: {lat}_{lon}")
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="train: fit/evaluate model; predict: run inference using saved model/scaler",
    )
    parser.add_argument(
        "--daily_csv",
        default=None,
        help="(predict mode) Daily max/min CSV to downscale. Must have date_time,daily_max,daily_min.",
    )
    parser.add_argument("--output_csv", default=None, help="(predict mode) Output hourly CSV path.")
    args = parser.parse_args()

    lat, lon = parse_location(args.location)

    mapping_df = load_mapping_df(MAP_CSV)
    county_id = match_county_id(mapping_df, lat, lon)

    daily_path = os.path.join(DAILY_DIR, f"era5_dailymaxmin_{county_id}.csv")

    if args.mode == "train":
        if not os.path.exists(daily_path):
            raise FileNotFoundError(f"Daily file not found for county_id={county_id}: {daily_path}")

        hourly_index = build_hourly_index(HOURLY_DIR)

        print(f"Task location: {args.location}")
        print(f"Matched county_id: {county_id}")
        print(f"Daily predictors: {daily_path}")

        run_one_county(county_id, daily_path, lat, lon, hourly_index)
        return

    # Predict mode: load enforced variant by default (fallback to legacy layout if needed)
    county_root = os.path.join(OUT_ROOT, county_id)
    enf_dir = os.path.join(county_root, "enforced")
    model_path = os.path.join(enf_dir, "rf_resid_model.pkl")
    scaler_path = os.path.join(enf_dir, "scaler_residual.pkl")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        # Legacy: artifacts written directly under county_root
        model_path = os.path.join(county_root, "rf_resid_model.pkl")
        scaler_path = os.path.join(county_root, "scaler_residual.pkl")
        enf_dir = county_root

    if args.daily_csv is None or args.output_csv is None:
        raise ValueError("predict mode requires --daily_csv and --output_csv.")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Missing trained artifacts in {enf_dir}. Need rf_resid_model.pkl and scaler_residual.pkl."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    daily_df = pd.read_csv(args.daily_csv)
    need = {"date_time", "daily_max", "daily_min"}
    miss = need - set(daily_df.columns)
    if miss:
        raise ValueError(f"daily_csv missing columns: {miss}")

    pred_log = args.output_csv + ".log"
    with open(pred_log, "w") as f:
        f.write(f"Predict mode for county_id={county_id}\n")
        f.write(f"location={args.location}\n")
        f.write(f"daily_csv={args.daily_csv}\n")
        f.write(f"output_csv={args.output_csv}\n")
        f.write(f"pid={os.getpid()}\n")
        f.write("Units: Celsius (C)\n")

    hourly_pred = predict_hourly_from_dailymaxmin(
        daily_df=daily_df,
        model=model,
        scaler=scaler,
        enforce_minmax=True,
        log_file=pred_log,
    )
    hourly_pred.to_csv(args.output_csv, index=False)
    print(f"Saved hourly downscaled output: {args.output_csv}")
    print(f"Saved inference log: {pred_log}")


if __name__ == "__main__":
    main()

