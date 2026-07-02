# -*- coding: utf-8 -*-
"""
Read the Deeper Smart Sonar Pro+ 2 data and export it to netCDF files.
Processing levels:
    L0 : Raw CSV data from the Deeper device
    L1 : Cleaned data (GPS filled, timestamp converted, time crop)
    L2 : Smoothed data (orthogonal projection on transect + 1m binning)

@author: Quentin Noël
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timezone

# NOTE: date_campaign, base_folder, transect endpoints, TIME_END and
# BIN_SIZE_M are configured in bathymetric_main.py, which is the only
# entry point that should be run. They used to be duplicated here as dead
# code (never read by any function below); removed to avoid the two files
# silently drifting apart if only one copy gets edited.

# CTD deployment times (local time UTC+2)
CTD_STATIONS = {
    "T2_P1":  "2026-06-04 11:30:00",
    "T2_P2":  "2026-06-04 11:35:00",
    "T2_P3":  "2026-06-04 11:40:00",
    "T2_P4":  "2026-06-04 11:46:00",
    "T2_P5":  "2026-06-04 11:50:00",
    "T2_P6":  "2026-06-04 11:56:00",
    "T2_P7":  "2026-06-04 12:03:00",
    "T2_P8":  "2026-06-04 12:07:00",
    "T2_P9":  "2026-06-04 12:14:00",
    "T2_P10": "2026-06-04 12:16:00",
    "T2_P11": "2026-06-04 12:18:00",
    "T2_P12": "2026-06-04 12:22:00",
    "T2_P13": "2026-06-04 12:24:00",
    "T2_P14": "2026-06-04 12:26:00",
    "T2_P15": "2026-06-04 12:28:00",
}

# =============================================================================
#%% FUNCTIONS
# =============================================================================

def create_folder(base, level):
    """Create output folder if it doesn't exist."""
    path = os.path.join(base, level)
    os.makedirs(path, exist_ok=True)
    return path


def read_L0(filepath):
    """
    Read raw CSV from Deeper Smart Sonar Pro+ 2.
    Returns raw DataFrame (Level 0).
    """
    df = pd.read_csv(filepath, sep=",")
    df.columns = ["latitude", "longitude", "depth", "temperature", "time_ms"]
    print(f"[L0] Loaded {len(df)} raw measurements from {os.path.basename(filepath)}")
    return df


def process_L1(df, time_end):
    """
    Clean raw data → Level 1:
      - Forward/backward fill GPS coordinates
      - Convert Unix timestamp (ms) to UTC+2 datetime
      - Crop to time_end
    """
    df = df.copy()

    # Fill GPS gaps (recorded every ~10-20s)
    df["latitude"]    = df["latitude"].ffill().bfill()
    df["longitude"]   = df["longitude"].ffill().bfill()
    df["temperature"] = df["temperature"].ffill().bfill()

    # Convert timestamp
    df["datetime"] = (pd.to_datetime(df["time_ms"], unit="ms")
                      .dt.tz_localize("UTC")
                      .dt.tz_convert("Europe/Zurich"))

    # Time crop
    df = df[df["datetime"] < time_end].copy()
    df = df.reset_index(drop=True)

    print(f"[L1] {len(df)} measurements after cleaning and time crop")
    print(f"     Start : {df['datetime'].min()}")
    print(f"     End   : {df['datetime'].max()}")
    return df


def process_L2(df, lat_start, lon_start, lat_end, lon_end, bin_size=1.0):
    """
    Smooth data → Level 2:
      - Convert lat/lon to local metric coordinates (m)
      - Project each point orthogonally onto the straight transect axis
      - Average depth, temperature, position within bin_size metre bins
    Returns smoothed DataFrame and transect length.
    """
    df = df.copy()
    R = 6371000  # Earth radius (m)

    # Local metric origin
    lat0 = df["latitude"].iloc[0]
    lon0 = df["longitude"].iloc[0]

    df["x_m"] = np.radians(df["longitude"] - lon0) * np.cos(np.radians(lat0)) * R
    df["y_m"] = np.radians(df["latitude"]  - lat0) * R

    # Transect vector
    x_start = np.radians(lon_start - lon0) * np.cos(np.radians(lat0)) * R
    y_start = np.radians(lat_start - lat0) * R
    x_end   = np.radians(lon_end   - lon0) * np.cos(np.radians(lat0)) * R
    y_end   = np.radians(lat_end   - lat0) * R

    tx = x_end - x_start
    ty = y_end - y_start
    t_len = np.sqrt(tx**2 + ty**2)
    tx_n, ty_n = tx / t_len, ty / t_len

    # Orthogonal projection
    dx = df["x_m"] - x_start
    dy = df["y_m"] - y_start
    df["dist_transect"] = dx * tx_n + dy * ty_n
    df["dist_lateral"]  = -dx * ty_n + dy * tx_n

    # Binning
    bins = np.arange(df["dist_transect"].min(),
                     df["dist_transect"].max() + bin_size, bin_size)
    df["bin"] = pd.cut(df["dist_transect"], bins=bins, labels=bins[:-1]).astype(float)

    df_smooth = (df.groupby("bin", as_index=False)
                   .agg(
                       latitude      = ("latitude",      "mean"),
                       longitude     = ("longitude",     "mean"),
                       depth         = ("depth",          "mean"),
                       temperature   = ("temperature",    "mean"),
                       datetime      = ("datetime",       "first"),
                       dist_lateral  = ("dist_lateral",   "mean"),
                   ))
    df_smooth = df_smooth.dropna().reset_index(drop=True)
    df_smooth["dist_transect"] = df_smooth["bin"]

    lateral_max = df["dist_lateral"].abs().max()
    print(f"[L2] {len(df_smooth)} smoothed points at {bin_size}m resolution")
    print(f"     Transect length  : {t_len:.1f} m")
    print(f"     Max lateral drift: {lateral_max:.1f} m")
    print(f"     Max depth        : {df_smooth['depth'].max():.2f} m")

    return df_smooth, df, t_len, lateral_max


def export_L1_netcdf(df, output_folder, filename):
    """Export Level 1 data to NetCDF."""
    time_vals = df["datetime"].apply(lambda x: x.timestamp()).values

    ds = xr.Dataset(
        {
            "depth":       (["index"], df["depth"].values,
                            {"units": "m",
                             "long_name": "Water depth from surface to lake bottom"}),
            "temperature": (["index"], df["temperature"].values,
                            {"units": "degrees_Celsius",
                             "long_name": "Surface water temperature",
                             "comment": "Potentially biased due to solar heating of black sensor casing"}),
            "latitude":    (["index"], df["latitude"].values,
                            {"units": "degrees_north",
                             "long_name": "Latitude (GPS, forward-filled)"}),
            "longitude":   (["index"], df["longitude"].values,
                            {"units": "degrees_east",
                             "long_name": "Longitude (GPS, forward-filled)"}),
            "time":        (["index"], time_vals,
                            {"units": "seconds since 1970-01-01 00:00:00 UTC",
                             "long_name": "Timestamp (UTC)"}),
        },
        attrs={
            "title":            "Sonar bathymetric transect T2 — Level 1",
            "processing_level": "L1",
            "instrument":       "Deeper Smart Sonar Pro+ 2 (A2AD)",
            "source_file":      "scan_data_2026-06-04_112703.csv",
            "processing":       "GPS forward/backward filled; timestamp converted UTC+2; time crop at 12:30",
            "date_created":     datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "author":           "Quentin Noël",
            "institution":      "Sampling and sensing aquatic ecosystems - Master of Science in Environmental Sciences - Aquatic science",
            "conventions":      "CF-1.8",
        }
    )

    outpath = os.path.join(output_folder, filename + ".nc")
    ds.to_netcdf(outpath, engine="netcdf4")
    print(f"[L1] Exported → {outpath}")


def export_L2_netcdf(df_smooth, df_raw, t_len, lateral_max, output_folder, filename):
    """Export Level 2 smoothed data to NetCDF."""
    time_vals = df_smooth["datetime"].apply(lambda x: x.timestamp()).values

    ds = xr.Dataset(
        {
            "depth":          (["distance"], df_smooth["depth"].values,
                               {"units": "m",
                                "long_name": "Mean water depth per 1m bin"}),
            "temperature":    (["distance"], df_smooth["temperature"].values,
                               {"units": "degrees_Celsius",
                                "long_name": "Mean surface water temperature per 1m bin",
                                "comment": "Potentially biased due to solar heating of black sensor casing"}),
            "latitude":       (["distance"], df_smooth["latitude"].values,
                               {"units": "degrees_north",
                                "long_name": "Mean latitude per 1m bin"}),
            "longitude":      (["distance"], df_smooth["longitude"].values,
                               {"units": "degrees_east",
                                "long_name": "Mean longitude per 1m bin"}),
            "dist_transect":  (["distance"], df_smooth["dist_transect"].values,
                               {"units": "m",
                                "long_name": "Projected distance along straight transect axis"}),
            "time":           (["distance"], time_vals,
                               {"units": "seconds since 1970-01-01 00:00:00 UTC",
                                "long_name": "Timestamp of first measurement in bin (UTC)"}),
        },
        coords={
            "distance": (["distance"], df_smooth["dist_transect"].values,
                         {"units": "m",
                          "long_name": "Distance along orthogonal transect (1m bins)"})
        },
        attrs={
            "title":              "Sonar bathymetric transect T2 — Level 2",
            "processing_level":   "L2",
            "instrument":         "Deeper Smart Sonar Pro+ 2 (A2AD)",
            "source_file":        "scan_data_2026-06-04_112703.csv",
            "processing":         "Orthogonal projection onto straight transect axis + 1m bin averaging",
            "transect":           "T2 transversal",
            "transect_length_m":  str(round(t_len, 1)),
            "lateral_drift_max_m": str(round(lateral_max, 1)),
            "bin_size_m":         "1.0",
            "n_raw_points":       str(len(df_raw)),
            "n_smoothed_points":  str(len(df_smooth)),
            "time_coverage_start": "2026-06-04T09:27:05Z",
            "time_coverage_end":   "2026-06-04T10:29:59Z",
            "time_zone":           "Europe/Zurich (UTC+2, CEST)",
            "geospatial_lat_min":  str(df_smooth["latitude"].min()),
            "geospatial_lat_max":  str(df_smooth["latitude"].max()),
            "geospatial_lon_min":  str(df_smooth["longitude"].min()),
            "geospatial_lon_max":  str(df_smooth["longitude"].max()),
            "depth_max_m":         str(df_smooth["depth"].max()),
            "date_created":        datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "author":              "Quentin",
            "institution":         "Sampling and sensing aquatic ecosystems - Master of Science in Environmental Sciences - Aquatic science",
            "conventions":         "CF-1.8",
        }
    )

    outpath = os.path.join(output_folder, filename + ".nc")
    ds.to_netcdf(outpath, engine="netcdf4")
    print(f"[L2] Exported → {outpath}")

