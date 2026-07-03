# -*- coding: utf-8 -*-
"""
Single entry point for the Lake Taney miniDOT oxygen logger pipeline.

Loops over every sensor listed in the campaign's oxygen .meta file and runs
oxygen_logger.process() (L0 -> L1 -> L2) for each valid one.

To process any campaign, only ``date_campaign`` needs to change. Everything
else comes from
data/Mooring/miniDOT/<date_campaign>/Level0/oxygen_loggers_<date_campaign>.meta

Serial number handling
-----------------------
oxygen_logger expects the raw file at
    Level0/7450-<serial_id>/Cat.txt
The serial_id is not stored in the .meta, so this script auto-discovers it by
scanning Level0/ for a "7450-*" folder rather than hardcoding it. If several
oxygen loggers are deployed in the same campaign in the future, this
one-folder assumption will need revisiting (raise a clear error rather than
silently guessing wrong).

@author: Estelle
"""
import os
import json
import glob
from oxygen_logger import oxygen_logger

# Run the script from its own location.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %% Specify field campaign here
date_campaign = '20260603'

# %% Load oxygen metadata
meta_path = f'../../data/Mooring/miniDOT/{date_campaign}/Level0/oxygen_loggers_{date_campaign}.meta'
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Metadata file not found: {meta_path}")

with open(meta_path) as f:
    meta = json.load(f)

level0_dir = f'../../data/Mooring/miniDOT/{date_campaign}/Level0'

# %% Make sure Level1/Level2 exist (oxygen_logger.write_to_nc does not
# create them itself; a missing folder surfaces as a misleading
# "Permission denied" from netCDF4 rather than a clear "not found").
campaign_dir = f'../../data/Mooring/miniDOT/{date_campaign}'
os.makedirs(os.path.join(campaign_dir, "Level1"), exist_ok=True)
os.makedirs(os.path.join(campaign_dir, "Level2"), exist_ok=True)

# %% Discover the miniDOT serial folder(s) (format "7450-<serial_id>")
serial_folders = sorted(
    os.path.basename(p) for p in glob.glob(os.path.join(level0_dir, "7450-*"))
    if os.path.isdir(p)
)

if len(serial_folders) == 0:
    raise FileNotFoundError(
        f"No '7450-<serial_id>' folder found under {level0_dir}. "
        "Check that the raw miniDOT export was copied there."
    )
if len(serial_folders) != sum(meta["valid"]):
    print(f"WARNING: found {len(serial_folders)} serial folder(s) but "
          f"{sum(meta['valid'])} valid sensor(s) in the .meta. Matching "
          "them in order — verify this is correct if you have several loggers.")

serial_ids = [f.split("7450-")[1] for f in serial_folders]

# %% Process every valid sensor
serial_iter = iter(serial_ids)
for idx, valid in enumerate(meta["valid"]):
    filename = meta["filenames"][idx]
    if not valid:
        print(f"Sensor {idx} ({filename}) marked invalid, skipping.")
        continue

    try:
        serial_id = next(serial_iter)
    except StopIteration:
        print(f"No serial folder left to match sensor {idx} ({filename}), skipping.")
        continue

    t_offset = meta.get("t_offset", [None] * len(meta["valid"]))[idx]

    print(f"Processing oxygen logger {serial_id} ({filename})...")
    try:
        ol = oxygen_logger(date_campaign, idx, serial_id, t_offset=t_offset)
        ol.process()
    except Exception as e:
        print(f"FAILED to process {filename}: {e}")
        continue

print("Done.")
