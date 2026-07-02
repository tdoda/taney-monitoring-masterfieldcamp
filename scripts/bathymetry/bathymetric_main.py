# -*- coding: utf-8 -*-
"""
Main processing pipeline for Deeper Smart Sonar Pro+ 2 bathymetric data.
Calls functions from bathymetric_functions.py to process L0 → L1 → L2.

@author: Quentin Noël
"""

import os
import json
import glob
import pandas as pd
from bathymetric_function import (
    create_folder,
    read_L0,
    process_L1,
    process_L2,
    export_L1_netcdf,
    export_L2_netcdf,
)

# Run script where it is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
#%% CONFIGURATION — only date_campaign needs editing for a new transect
# =============================================================================

date_campaign = "20260603"

campaign_folder = os.path.join("..", "..", "data", "bathymetry", date_campaign)
level0_dir = os.path.join(campaign_folder, "Level 0")

# Find the .meta file by content, not by exact name (export tools sometimes
# prefix it with an ID, e.g. "1783023678222_sonar_deeper_20260604.meta").
meta_candidates = glob.glob(os.path.join(level0_dir, "*.meta"))
if len(meta_candidates) == 0:
    raise FileNotFoundError(f"No .meta file found in {level0_dir}")
if len(meta_candidates) > 1:
    raise ValueError(f"Multiple .meta files found in {level0_dir}, expected one: {meta_candidates}")

with open(meta_candidates[0]) as f:
    meta = json.load(f)

input_file = os.path.join(level0_dir, meta["filenames"]["level0"])

# Transect endpoints (start = hydropower outflow, end = cliff CTD)
LAT_START, LON_START = meta["transect"]["lat_start"], meta["transect"]["lon_start"]
LAT_END,   LON_END   = meta["transect"]["lat_end"],   meta["transect"]["lon_end"]

# Time crop
TIME_END = pd.Timestamp(meta["processing"]["time_crop"], tz=meta["processing"]["timezone"])

# Smoothing resolution
BIN_SIZE_M = meta["transect"]["bin_size_m"]

# Output filenames (without extension; export_*_netcdf appends ".nc")
FILENAME_L1 = meta["filenames"]["level1"].rsplit(".", 1)[0]
FILENAME_L2 = meta["filenames"]["level2"].rsplit(".", 1)[0]

# =============================================================================
#%% MAIN PROCESSING PIPELINE
# =============================================================================

if __name__ == "__main__":

    # Create output folders
    folder_L1 = create_folder(campaign_folder, "Level1")
    folder_L2 = create_folder(campaign_folder, "Level2")

    # L0 → read raw data
    df_L0 = read_L0(input_file)

    # L0 → L1 : clean
    df_L1 = process_L1(df_L0, TIME_END)
    export_L1_netcdf(df_L1, folder_L1, FILENAME_L1)

    # L1 → L2 : smooth
    df_L2, df_L1_full, t_len, lateral_max = process_L2(
        df_L1, LAT_START, LON_START, LAT_END, LON_END, BIN_SIZE_M
    )
    export_L2_netcdf(df_L2, df_L1_full, t_len, lateral_max, folder_L2, FILENAME_L2)

    print("\n=== Processing complete ===")
    print(f"  L1 → {folder_L1}")
    print(f"  L2 → {folder_L2}")