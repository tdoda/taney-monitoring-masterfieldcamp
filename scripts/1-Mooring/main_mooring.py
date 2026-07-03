# -*- coding: utf-8 -*-
"""
Single entry point for the Lake Taney mooring temperature pipeline.

To process any campaign, only two things must change:
    1. ``date_campaign``  -> the YYYYMMDD campaign folder name
    2. ``meta_filename``  -> the .meta file inside that folder's Level0/

The .meta file alone determines file formats, validity, depths and the mooring
configuration. Nothing in this script hardcodes a depth.

Mooring configurations (read from meta["mooring_config"]):
    "single_chain" : old campaign (.xlsx). Depths taken directly from the .meta.
                     One Level 3 grid.
    "option1"      : new campaign, single unified depth axis across both buoys.
                     One Level 3 grid (L3_mooring_*.nc).
    "option2"      : new campaign, two independent chains (A and B) sharing the
                     same absolute depth reference but kept separate.
                     Two Level 3 grids (L3_mooring_chainA_*.nc, ...chainB_*.nc).
    "option3"      : new campaign, pressure-derived depths. Analysis window
                     truncated to pressure-sensor availability (Jun 2025 – Jan 2026).
                     Produces four L3 files:
                       - L3_mooring_option3A_chainA_*.nc  (dynamic depth, chain A)
                       - L3_mooring_option3A_chainB_*.nc  (dynamic depth, chain B)
                       - L3_mooring_option3B_*.nc         (fixed depth, all sensors fused)

Data layout (SwitchDrive + Git):
    data/Mooring/HOBO_T/<campaign>/Level0/   <- raw thermistor CSV/XLSX + .meta
    data/Mooring/HOBO_T/<campaign>/Level1/   <- produced here (L1 netCDF)
    data/Mooring/HOBO_T/<campaign>/Level2/   <- produced here (L2 netCDF, masked)
    data/Mooring/HOBO_T/<campaign>/Level3/   <- produced here (L3 netCDF grids)
    data/Mooring/HOBO_P/<campaign>/Level3/   <- water_level_variation.csv (option3)
                                                produced by the HOBO_P pressure pipeline

@author: T. Doda (original) — extended for multi-format / multi-config campaigns
"""
import os
import json
import numpy as np
from datetime import datetime, timezone

from thermistor import thermistor_series, thermistor_grid
from functions_mooring import (
    read_data, export, create_temp_grid, create_folder,
    compute_depths_from_meta, filter_meta_by_chain,
    read_pressure_hobo, read_water_level,
    gmt_to_local_wallclock,
    compute_nominal_depths_option3,
    create_temp_grid_option3A, create_temp_grid_option3B,
    export_option3A, export_option3B,
)

# Run the script from its own location.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %% Specify field campaign here
# Old campaigns (single_chain):
# date_campaign = '20240527'
# date_campaign = '20250604'
# New campaign (2025-2026):
date_campaign = '20260603'

# Choose which .meta to run.
# meta_filename = "thermistors_{}.meta".format(date_campaign)            # option1 / single_chain
# meta_filename = "thermistors_{}_option2.meta".format(date_campaign)  # option2
meta_filename = "thermistors_{}_option3.meta".format(date_campaign)  # option3

# %% Setup paths
mooring_data_folder = os.path.join("..", "..", "data", "Mooring", "HOBO_T")
input_folder = os.path.join(mooring_data_folder, date_campaign)
meta_path = os.path.join(input_folder, "Level0", meta_filename)

create_folder(input_folder, "Level1")
create_folder(input_folder, "Level2")
create_folder(input_folder, "Level3")

# %% Load mooring metadata
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
else:
    raise FileNotFoundError("Metadata file not found: {}".format(meta_path))

config = meta.get("mooring_config", "single_chain")
print("Mooring configuration: {}".format(config))

# %% Resolve depths from the metadata (raises if total_depth is missing)
# Overwrite the working "Depth (m)" so every L1/L2 file and the L3 grid use the
# config-consistent depth, even if the stored values were stale.
resolved_depths = compute_depths_from_meta(meta)
if "Depth (m)" in meta:
    stored = np.array(meta["Depth (m)"], dtype=float)
    if len(stored) == len(resolved_depths) and not np.allclose(stored, resolved_depths, atol=0.01):
        print("WARNING: stored 'Depth (m)' differs from values recomputed from "
              "total_depth; using the recomputed values.")
meta["Depth (m)"] = resolved_depths

# %% Read the data files (only the valid ones)
valid_mask = np.array(meta["valid"], dtype=bool)
files = np.array(meta["filenames"])[valid_mask]
file_types = np.array(meta["filetypes"])[valid_mask]

for k, file in enumerate(files):
    file_path = os.path.join(input_folder, "Level0", file)
    if not os.path.exists(file_path):
        print("WARNING: file {} not found, skipping".format(file))
        continue
    try:
        data_temp = read_data(file_path, file_types[k])
    except Exception as e:
        print(e)
        print("Failed to process {}".format(file))
        continue

    temp_series = thermistor_series()
    if temp_series.read_timeseries(data_temp, meta):
        temp_series.quality_assurance()
        file_name = file.rsplit('.', 1)[0]
        print("Export to L1 netCDF files")
        export(temp_series, os.path.join(input_folder, "Level1"),
               "L1_mooring_{}_{}".format(file_types[k], file_name), overwrite=True)
        temp_series.mask_data()  # flagged values -> NaN
        print("Export to L2 netCDF files")
        export(temp_series, os.path.join(input_folder, "Level2"),
               "L2_mooring_{}_{}".format(file_types[k], file_name), overwrite=True)

# %% Build the common time axis for the L3 grid
tnum_start = datetime.strptime(meta["campaign"]["Time of deployment"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
tnum_end = datetime.strptime(meta["campaign"]["Time of retrieval"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
tnum_interp = np.arange(tnum_start, tnum_end, thermistor_grid().dt_sec)

# List all Level 2 files produced above.
L2_dir = os.path.join(input_folder, "Level2")
files_L2 = [f for f in os.listdir(L2_dir) if f.endswith(".nc")]


def _l2_chain_lookup(meta_full):
    """Map each L2 filename to its buoy chain using the campaign metadata.

    L2 files are named 'L2_mooring_<filetype>_<rawname_without_ext>_<stamp>.nc'.
    The prefix 'L2_mooring_<filetype>_<rawname_without_ext>' is reconstructed
    per sensor and matched against the actual filenames.
    """
    prefixes = {}
    for i, ok in enumerate(meta_full["valid"]):
        if not ok:
            continue
        no_ext = meta_full["filenames"][i].rsplit('.', 1)[0]
        prefix = "L2_mooring_{}_{}".format(meta_full["filetypes"][i], no_ext)
        prefixes[prefix] = meta_full["chain"][i]

    def chain_of(fname):
        for prefix, ch in prefixes.items():
            if fname.startswith(prefix + "_") and fname.endswith(".nc"):
                return ch
        return None
    return chain_of


# %% Create Level 3 grid(s) depending on configuration
if config in ("single_chain", "option1"):
    temp_grid = thermistor_grid()
    data_grid = create_temp_grid(L2_dir, files_L2, tnum_interp)
    temp_grid.add_grid(data_grid, meta)
    print("Export to L3 netCDF file (single grid)")
    export(temp_grid, os.path.join(input_folder, "Level3"), "L3_mooring", overwrite=True)

elif config == "option2":
    chain_of = _l2_chain_lookup(meta)
    for ch in ("A", "B"):
        ch_files = [f for f in files_L2 if chain_of(f) == ch]
        if not ch_files:
            print("WARNING: no Level 2 files for chain {}, skipping its grid".format(ch))
            continue
        meta_chain = filter_meta_by_chain(meta, ch)
        temp_grid = thermistor_grid()
        data_grid = create_temp_grid(L2_dir, ch_files, tnum_interp)
        temp_grid.add_grid(data_grid, meta_chain)
        print("Export to L3 netCDF file (chain {})".format(ch))
        export(temp_grid, os.path.join(input_folder, "Level3"),
               "L3_mooring_chain{}".format(ch), overwrite=True)

elif config == "option3":
    # ------------------------------------------------------------------
    # option3: pressure-derived dynamic depths (3A) and fixed depth axis
    # for cross-chain comparison (3B). Analysis window is truncated to
    # the pressure-sensor period (June 2025 – January 2026).
    # ------------------------------------------------------------------

    # --- 1. Load pressure files and water-level variation ---------------
    pres_meta   = meta["pressure_sensors"]
    pres_consts = meta["pressure_constants"]
    tw          = meta["time_window"]

    level0_dir = os.path.join(input_folder, "Level0")

    # Water-level variation — canonical location: HOBO_P/<campaign>/Level3/
    # This file is produced by the HOBO_P pressure pipeline and must be
    # present on SwitchDrive at data/Mooring/HOBO_P/<campaign>/Level3/.
    hobo_p_l2_dir = os.path.join(
        mooring_data_folder.replace("HOBO_T", "HOBO_P"),
        date_campaign,
        "Level2"
    )
    wl_path = os.path.join(hobo_p_l2_dir, "water_level_variation.csv")
    if not os.path.exists(wl_path):
        raise FileNotFoundError(
            "water_level_variation.csv not found: {}\n"
            "  -> Copy from SwitchDrive: data/Mooring/HOBO_P/{}/Level2/water_level_variation.csv".format(
                wl_path, date_campaign
            )
        )

    df_wl = read_water_level(wl_path)
    # Convert GMT timestamps to local naive wall-clock for alignment with thermistors.
    df_wl["time_local"] = gmt_to_local_wallclock(df_wl["time"])

    # Pressure sensors (raw files remain in HOBO_T Level0 alongside thermistors)
    pres_data = {}
    for ps in pres_meta:
        p_path = os.path.join(level0_dir, ps["filename"])
        if not os.path.exists(p_path):
            print("WARNING: pressure file {} not found, skipping".format(ps["filename"]))
            continue
        dp = read_pressure_hobo(p_path)
        dp["data"]["time_local"] = gmt_to_local_wallclock(dp["data"]["time"])
        pres_data[ps["chain"]] = dp["data"]
        print("Loaded pressure sensor chain {}: {} rows, {} to {}".format(
            ps["chain"],
            len(dp["data"]),
            dp["data"]["time_local"].iloc[0],
            dp["data"]["time_local"].iloc[-1],
        ))

    # --- 2. Build the option3 time axis (local wall-clock, truncated) ----
    import pandas as pd
    tw_start_local = pd.Timestamp(tw["start_local"])
    tw_end_local   = pd.Timestamp(tw["end_local"])

    tw_start_unix = tw_start_local.timestamp()
    tw_end_unix   = tw_end_local.timestamp()

    tnum_option3 = tnum_interp[
        (tnum_interp >= tw_start_unix) & (tnum_interp <= tw_end_unix)
    ]
    print("option3 time axis: {} steps ({} to {})".format(
        len(tnum_option3),
        datetime.fromtimestamp(tnum_option3[0], timezone.utc).strftime("%Y-%m-%d %H:%M"),
        datetime.fromtimestamp(tnum_option3[-1], timezone.utc).strftime("%Y-%m-%d %H:%M"),
    ))

    # --- 3. Interpolate delta_wl onto the option3 10-min time axis -------
    wl_tnum  = df_wl["time_local"].astype("int64").values / 1e9
    wl_delta = df_wl["delta_wl"].values.astype(float)

    delta_wl_interp = np.interp(
        tnum_option3,
        wl_tnum,
        wl_delta,
        left=np.nan,
        right=np.nan,
    )

    # --- 4. Compute fixed depth axis for option3B ------------------------
    fixed_depth_axis = compute_nominal_depths_option3(meta, wl_delta)
    print("option3B fixed depth axis (m):", np.round(fixed_depth_axis, 3).tolist())

    # --- 5. Build per-chain L2 file lists --------------------------------
    chain_of = _l2_chain_lookup(meta)
    l2_files_by_chain = {}
    for ch in ("A", "B"):
        l2_files_by_chain[ch] = [f for f in files_L2 if chain_of(f) == ch]
        if not l2_files_by_chain[ch]:
            print("WARNING: no L2 files for chain {}, skipping".format(ch))

    L3_dir = os.path.join(input_folder, "Level3")

    # --- 6. option3A: dynamic depths, one file per chain -----------------
    for ch in ("A", "B"):
        ch_files = l2_files_by_chain.get(ch, [])
        if not ch_files:
            continue
        meta_chain = filter_meta_by_chain(meta, ch)
        print("Building option3A grid for chain {} ({} sensors)".format(ch, len(ch_files)))
        grid_3A = create_temp_grid_option3A(
            L2_dir, ch_files, tnum_option3, delta_wl_interp, chain=ch
        )
        export_option3A(
            grid_3A, meta_chain, L3_dir,
            "L3_mooring_option3A_chain{}".format(ch),
            overwrite=True,
        )

    # --- 7. option3B: fixed depth axis, all sensors fused ----------------
    all_l2_files = l2_files_by_chain.get("A", []) + l2_files_by_chain.get("B", [])
    if all_l2_files:
        print("Building option3B grid (all chains, {} sensors)".format(len(all_l2_files)))
        grid_3B = create_temp_grid_option3B(
            L2_dir, all_l2_files, tnum_option3, fixed_depth_axis
        )
        export_option3B(
            grid_3B, meta, L3_dir,
            "L3_mooring_option3B",
            overwrite=True,
        )
    else:
        print("WARNING: no L2 files found for option3B, skipping.")

else:
    raise ValueError("Unknown mooring_config: {}".format(config))

print("Done.")
