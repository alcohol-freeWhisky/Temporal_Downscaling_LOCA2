#!/usr/bin/env python3
"""
ERA5 Temperature → County-point CSV Extractor (Output in °C)

Purpose
-------
Extract hourly ERA5 isobaric temperature ("t", originally in Kelvin) at county-specific
lat/lon points (from a metadata CSV). Each point is snapped to the nearest ERA5 grid cell,
then exported as a single CSV time series.

IMPORTANT CHANGE
---------------
Before saving, temperature is converted from Kelvin to Celsius:
    temp_C = temp_K - 273.15

Design
------
The core workflow intentionally matches the original script:
1) Pre-compute nearest-grid indices (iy, ix) using a reference-year ERA5 file.
2) Loop over yearly ERA5 files and extract time series at those indices.
3) Concatenate across years and write one CSV per point.

Inputs
------
- ERA5 NetCDF:  /nfs/turbo/seas-mtcraig-climate/Ziqi/ERA5/temperature_download/era5_temperature_{year}.nc
- County meta:  /nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/file/County_selected_revise_3.csv
  Uses columns: in.weather_file_latitude, in.weather_file_longitude

Outputs
-------
- /nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/era5_county/era5_temperature_{lat}_{lon}.csv
  Columns: Timestamp, temperature
  NOTE: "temperature" is in Celsius (°C).
  IMPORTANT: {lat} and {lon} are taken from the original CSV strings (no rounding).
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm


# -----------------------------
# User-defined paths
# -----------------------------
ERA5_DIR = "/nfs/turbo/seas-mtcraig-climate/Ziqi/ERA5/temperature_download"
COUNTY_FILE = "/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/file/County_selected_revise_3.csv"
OUT_DIR = "/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/era5_county"

LAT_COL = "in.weather_file_latitude"
LON_COL = "in.weather_file_longitude"

os.makedirs(OUT_DIR, exist_ok=True)

# Kelvin -> Celsius conversion constant
KELVIN_TO_C_OFFSET = 273.15


# -----------------------------
# Read county CSV
# Keep lat/lon as strings for filename fidelity,
# and also create numeric versions for computations.
# -----------------------------
df_county = pd.read_csv(COUNTY_FILE, dtype={LAT_COL: "string", LON_COL: "string"})

if LAT_COL not in df_county.columns or LON_COL not in df_county.columns:
    raise KeyError(f"County CSV must contain columns: {LAT_COL}, {LON_COL}")

df_county[LAT_COL] = df_county[LAT_COL].astype("string").str.strip()
df_county[LON_COL] = df_county[LON_COL].astype("string").str.strip()

lat_num = pd.to_numeric(df_county[LAT_COL], errors="coerce")
lon_num = pd.to_numeric(df_county[LON_COL], errors="coerce")

bad = lat_num.isna() | lon_num.isna()
if bad.any():
    bad_rows = df_county.loc[bad, [LAT_COL, LON_COL]]
    raise ValueError(
        "Found non-numeric lat/lon values in county CSV. "
        f"Example rows:\n{bad_rows.head(10)}"
    )


# -----------------------------
# Step 1: build nearest-grid index map using a reference year
# Original intent: ref_year = 1980
# Robustness: fall back to first available file if 1980 is missing.
# -----------------------------
ref_year = 1980
ref_nc_file = os.path.join(ERA5_DIR, f"era5_temperature_{ref_year}.nc")

if not os.path.exists(ref_nc_file):
    # Fall back: find the first existing file within the loop range
    fallback = None
    for y in range(1980, 2025):
        cand = os.path.join(ERA5_DIR, f"era5_temperature_{y}.nc")
        if os.path.exists(cand):
            fallback = (y, cand)
            break
    if fallback is None:
        raise FileNotFoundError(
            f"Reference file not found: {ref_nc_file}. "
            f"No ERA5 files found in {ERA5_DIR} for years 1980–2024."
        )
    ref_year, ref_nc_file = fallback
    print(f"[Info] 1980 reference file missing. Using fallback reference year: {ref_year}")

print(f"[Step 1] Reference file: {ref_nc_file}")
ds_ref = xr.open_dataset(ref_nc_file)

# ERA5 lat/lon are expected to be 1D arrays
lats = ds_ref["latitude"].values
lons = ds_ref["longitude"].values
if lats.ndim != 1 or lons.ndim != 1:
    ds_ref.close()
    raise ValueError("This script assumes ERA5 latitude/longitude are 1D arrays.")

county_to_index = {}
for idx in df_county.index:
    iy = int(np.abs(lats - float(lat_num.loc[idx])).argmin())
    ix = int(np.abs(lons - float(lon_num.loc[idx])).argmin())
    county_to_index[idx] = (iy, ix)

ds_ref.close()


# -----------------------------
# Step 2: loop over years and extract time series
# Keep the original loop range(1980, 2025), but skip missing files.
# Also handle valid_time units robustly.
#
# IMPORTANT CHANGE:
#   Convert Kelvin -> Celsius at extraction time so everything downstream is in °C.
# -----------------------------
county_time_series = {idx: [] for idx in df_county.index}

for year in tqdm(range(1980, 2025), desc="[Step 2] Processing years", unit="yr"):
    nc_file = os.path.join(ERA5_DIR, f"era5_temperature_{year}.nc")
    if not os.path.exists(nc_file):
        tqdm.write(f"[Skip] Missing file: {nc_file}")
        continue

    ds = xr.open_dataset(nc_file)

    vt = ds["valid_time"].values
    # valid_time units are typically "seconds since 1970-01-01"
    if np.issubdtype(vt.dtype, np.datetime64):
        time_values = pd.to_datetime(vt, utc=True)
    else:
        units = ds["valid_time"].attrs.get("units", "")
        if "seconds since" in units:
            time_values = pd.to_datetime(vt, unit="s", utc=True)
        elif "hours since" in units:
            time_values = pd.to_datetime(vt, unit="h", utc=True)
        else:
            # Conservative fallback: still try interpreting as seconds
            time_values = pd.to_datetime(vt, unit="s", utc=True)

    # Extract each point
    for idx, (iy, ix) in county_to_index.items():
        # Variable t dims: (valid_time, pressure_level, latitude, longitude)
        t_series = ds["t"].isel(pressure_level=0, latitude=iy, longitude=ix)

        # Convert to Celsius (°C)
        temp_c = t_series.values - KELVIN_TO_C_OFFSET

        df_year = pd.DataFrame(
            {
                "Timestamp": time_values,
                "temperature": temp_c,  # NOTE: in °C
            }
        )
        county_time_series[idx].append(df_year)

    ds.close()


# -----------------------------
# Step 3: concatenate all years and write CSV per point
# Filename must preserve the original lat/lon string formatting from CSV.
# Output temperature is in Celsius (°C).
# -----------------------------
print(f"[Step 3] Writing outputs to: {OUT_DIR}")

for idx, df_list in county_time_series.items():
    if len(df_list) == 0:
        print(f"[Warn] No data extracted for idx={idx}. Output skipped.")
        continue

    df_all = pd.concat(df_list, ignore_index=True)
    df_all.sort_values("Timestamp", inplace=True)

    # Ensure deterministic dtypes
    df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"], utc=True)
    df_all["temperature"] = pd.to_numeric(df_all["temperature"], errors="coerce")

    orig_lat_str = str(df_county.loc[idx, LAT_COL]).strip()
    orig_lon_str = str(df_county.loc[idx, LON_COL]).strip()

    file_name = os.path.join(OUT_DIR, f"era5_temperature_{orig_lat_str}_{orig_lon_str}.csv")
    df_all.to_csv(file_name, index=False)
    print(f"Saved idx={idx}: {file_name}")

print("All processing completed.")

