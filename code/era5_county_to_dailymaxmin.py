#!/usr/bin/env python3
"""
ERA5 County CSV → Daily Max/Min (°C) Aggregator

This script converts each hourly ERA5 county-point file
    era5_temperature_{lat}_{lon}.csv
into a daily summary with three columns:
    date_time, daily_max, daily_min

Key requirements:
- date_time is date-only (datetime at 00:00:00).
- daily_max/min are in Celsius (ERA5 input is already Celsius).
- output file name is:
    era5_dailymaxmin_{county_id}.csv
  where county_id is the 4th column (1-based) in County_selected_revise_3.csv,
  e.g., "G0600850".
- mapping from county rows to ERA5 files is done by matching the *string-form*
  lat/lon in the county CSV to the {lat}_{lon} embedded in the ERA5 filename.

Inputs:
- Hourly ERA5 point CSVs:
    /nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/era5_county/era5_temperature_{lat}_{lon}.csv
- County meta CSV:
    /nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/file/County_selected_revise_3.csv

Outputs:
- Daily max/min CSVs:
    /nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/era5_dailymaxmin_county/era5_dailymaxmin_{county_id}.csv
"""

from __future__ import annotations

import os
import pandas as pd
from tqdm import tqdm


# -----------------------------
# Paths / columns
# -----------------------------
COUNTY_FILE = "/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/file/County_selected_revise_3.csv"

IN_DIR = "/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/era5_county"
OUT_DIR = "/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/era5_dailymaxmin_county"

LAT_COL = "in.weather_file_latitude"
LON_COL = "in.weather_file_longitude"

# County ID is the 4th column in the CSV (1-based) => iloc[:, 3] in pandas (0-based)
COUNTY_ID_COL_INDEX_0BASED = 3


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Read county meta with lat/lon preserved as strings (filename fidelity)
    df = pd.read_csv(COUNTY_FILE, dtype={LAT_COL: "string", LON_COL: "string"})
    if LAT_COL not in df.columns or LON_COL not in df.columns:
        raise KeyError(f"County CSV must contain columns: {LAT_COL}, {LON_COL}")

    if df.shape[1] <= COUNTY_ID_COL_INDEX_0BASED:
        raise ValueError(
            f"County CSV has only {df.shape[1]} columns; cannot access 4th column "
            f"(0-based index {COUNTY_ID_COL_INDEX_0BASED})."
        )

    county_id = df.iloc[:, COUNTY_ID_COL_INDEX_0BASED].astype("string").str.strip()
    lat_str = df[LAT_COL].astype("string").str.strip()
    lon_str = df[LON_COL].astype("string").str.strip()

    # Basic validation
    if county_id.isna().any():
        bad = df.index[county_id.isna()].tolist()[:10]
        raise ValueError(f"Found missing county_id values (4th column). Example indices: {bad}")
    if lat_str.isna().any() or lon_str.isna().any():
        bad = df.index[lat_str.isna() | lon_str.isna()].tolist()[:10]
        raise ValueError(f"Found missing lat/lon values. Example indices: {bad}")

    n_ok = 0
    n_missing = 0

    iterable = zip(county_id.tolist(), lat_str.tolist(), lon_str.tolist())
    for cid, la, lo in tqdm(iterable, total=len(df), desc="Processing counties", unit="county"):
        in_file = os.path.join(IN_DIR, f"era5_temperature_{la}_{lo}.csv")
        if not os.path.exists(in_file):
            tqdm.write(f"[Skip] Missing input: {in_file}")
            n_missing += 1
            continue

        # Read hourly data
        d = pd.read_csv(in_file)
        if "Timestamp" not in d.columns or "temperature" not in d.columns:
            raise KeyError(f"Input file {in_file} must contain columns: Timestamp, temperature")

        ts = pd.to_datetime(d["Timestamp"], errors="coerce")
        if ts.isna().any():
            bad_n = int(ts.isna().sum())
            raise ValueError(f"Failed to parse {bad_n} Timestamp values in {in_file}")

        # Temperature is already in Celsius (°C); NO Kelvin->Celsius conversion here.
        temp_c = pd.to_numeric(d["temperature"], errors="coerce")
        if temp_c.isna().any():
            bad_n = int(temp_c.isna().sum())
            raise ValueError(f"Found {bad_n} non-numeric temperature values in {in_file}")

        # Daily aggregation (date-only at midnight)
        date_time = ts.dt.normalize()
        out = (
            pd.DataFrame({"date_time": date_time, "temp_c": temp_c})
            .groupby("date_time", as_index=False)
            .agg(daily_max=("temp_c", "max"), daily_min=("temp_c", "min"))
            .sort_values("date_time")
        )

        out_file = os.path.join(OUT_DIR, f"era5_dailymaxmin_{cid}.csv")
        out.to_csv(out_file, index=False)
        n_ok += 1

    tqdm.write(f"Done. Wrote {n_ok} files. Missing inputs: {n_missing}. Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()

