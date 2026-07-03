# -*- coding: utf-8 -*-
"""
Single entry point for the Lake Taney HOBO_P pressure sensor pipeline.

Loops over every sensor listed in the campaign's pressure .meta file and runs
pressure_sensor.process() (L0 -> L1 -> L2, i.e. raw CSV -> netCDF, then
quality-flagged and masked) for each valid one.

To process any campaign, only ``date_campaign`` needs to change. Everything
else (filenames, depths, validity) comes from
data/Mooring/HOBO_P/<date_campaign>/Level0/pressure_sensors_<date_campaign>.meta

NOTE: this script produces the per-sensor L1/L2 pressure files. It does NOT
compute water_level_variation.csv (that is done separately by the pressure
analysis notebook, which combines both sensors' pressure series). Run this
script BEFORE that notebook, and run the notebook BEFORE main_mooring.py
when mooring_config == "option3".

@author: Estelle
"""
import os
import json
from pressure_sensor import pressure_sensor

# Run the script from its own location.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %% Specify field campaign here
date_campaign = '20260603'

# %% Load pressure metadata
meta_path = f'../../data/Mooring/HOBO_P/{date_campaign}/Level0/pressure_sensors_{date_campaign}.meta'
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Metadata file not found: {meta_path}")

with open(meta_path) as f:
    meta = json.load(f)

# %% Make sure Level1/Level2 exist (pressure_sensor.write_to_nc does not
# create them itself; a missing folder surfaces as a misleading
# "Permission denied" from netCDF4 rather than a clear "not found").
campaign_dir = f'../../data/Mooring/HOBO_P/{date_campaign}'
os.makedirs(os.path.join(campaign_dir, "Level1"), exist_ok=True)
os.makedirs(os.path.join(campaign_dir, "Level2"), exist_ok=True)

# %% Process every valid sensor
for idx, valid in enumerate(meta["valid"]):
    filename = meta["filenames"][idx]
    if not valid:
        print(f"Sensor {idx} ({filename}) marked invalid, skipping.")
        continue

    # Serial number is not stored as its own field in the .meta; it is the
    # prefix of the raw filename (e.g. "20598412_B_15.csv" -> "20598412"),
    # confirmed against the "reference" field documented in the thermistor
    # option3 .meta for the same campaign.
    serial_id = filename.split("_")[0]

    t_offset = meta.get("t_offset", [None] * len(meta["valid"]))[idx]

    print(f"Processing pressure sensor {serial_id} ({filename})...")
    try:
        ps = pressure_sensor(date_campaign, idx, serial_id, t_offset=t_offset)
        ps.process()
    except Exception as e:
        print(f"FAILED to process {filename}: {e}")
        continue

print("Done. Next step: run the pressure analysis notebook to produce "
      "water_level_variation.csv, then main_mooring.py.")
