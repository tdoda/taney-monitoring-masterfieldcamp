# -*- coding: utf-8 -*-
"""
Main processing pipeline for Deeper Smart Sonar Pro+ 2 bathymetric data.
Calls functions from bathymetric_functions.py to process L0 → L1 → L2.

@author: Quentin Noël
"""

import os
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
#%% CONFIGURATION — modify here only
# =============================================================================

date_campaign = "20260604"

base_folder  = r"C:\Users\Quentin\Desktop\Geoscience\PREMIERE MASTER\projet_camp_062026"
input_file   = os.path.join(base_folder, "sonar_deeper_bathymetric_level0_2026.csv")

# Transect endpoints (start = hydropower outflow, end = cliff CTD)
LAT_START, LON_START = 46.34453798644245, 6.841974053531885
LAT_END,   LON_END   = 46.346515654586256, 6.841558394953608

# Time crop
TIME_END = pd.Timestamp("2026-06-04 12:30:00", tz="Europe/Zurich")

# Smoothing resolution
BIN_SIZE_M = 1.0

# =============================================================================
#%% MAIN PROCESSING PIPELINE
# =============================================================================

if __name__ == "__main__":

    # Create output folders
    folder_L1 = create_folder(base_folder, "Level1")
    folder_L2 = create_folder(base_folder, "Level2")

    # L0 → read raw data
    df_L0 = read_L0(input_file)

    # L0 → L1 : clean
    df_L1 = process_L1(df_L0, TIME_END)
    export_L1_netcdf(df_L1, folder_L1,
                     f"sonar_deeper_bathymetric_level1_{date_campaign}")

    # L1 → L2 : smooth
    df_L2, df_L1_full, t_len, lateral_max = process_L2(
        df_L1, LAT_START, LON_START, LAT_END, LON_END, BIN_SIZE_M
    )
    export_L2_netcdf(df_L2, df_L1_full, t_len, lateral_max, folder_L2,
                     f"sonar_deeper_bathymetric_level2_{date_campaign}")

    print("\n=== Processing complete ===")
    print(f"  L1 → {folder_L1}")
    print(f"  L2 → {folder_L2}")