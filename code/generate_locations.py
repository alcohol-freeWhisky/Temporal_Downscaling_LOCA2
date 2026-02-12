"""
Generate locations.txt from a county metadata CSV.

This script reads the county meta CSV, extracts the latitude/longitude columns
(as raw strings to preserve the original decimal precision), and writes a plain
text file where each line is:
    {lat}_{lon}

Output file:
- /nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/file/locations.txt
"""

from __future__ import annotations

import os
import pandas as pd


# -----------------------------
# User-defined paths
# -----------------------------
COUNTY_FILE = "/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/file/County_selected_revise_3.csv"
OUT_FILE = "/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/file/locations.txt"

LAT_COL = "in.weather_file_latitude"
LON_COL = "in.weather_file_longitude"


def main() -> None:
    # Read lat/lon as strings to preserve original formatting/precision
    df = pd.read_csv(COUNTY_FILE, dtype={LAT_COL: "string", LON_COL: "string"})

    if LAT_COL not in df.columns or LON_COL not in df.columns:
        raise KeyError(f"Input CSV must contain columns: {LAT_COL}, {LON_COL}")

    lat_str = df[LAT_COL].astype("string").str.strip()
    lon_str = df[LON_COL].astype("string").str.strip()

    # Basic validation: ensure no missing values
    if lat_str.isna().any() or lon_str.isna().any():
        bad_idx = df.index[lat_str.isna() | lon_str.isna()].tolist()[:10]
        raise ValueError(f"Found missing lat/lon values. Example row indices: {bad_idx}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    # Write one location per line: "{lat}_{lon}" (no headers, no indices)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for la, lo in zip(lat_str.tolist(), lon_str.tolist()):
            f.write(f"{la}_{lo}\n")

    print(f"Saved: {OUT_FILE} (n={len(df)})")


if __name__ == "__main__":
    main()

