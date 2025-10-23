# -*- coding: utf-8 -*-

"""
Read the meteo data and export it to netCDF files (Level 1 only).

@author: T. Doda / Alejandro Perez
"""
import os
import sys
import json
import numpy as np
from datetime import datetime, timezone

from Meteo import meteo_series 
from functions_meteo import read_data, export, create_folder

#%% Setup paths

meteo_data_folder = r'..\..\data\Meteo\Model\20250606'
input_folder = os.path.join(meteo_data_folder, "Level0")
meta_path = os.path.join(input_folder, "meteo_20250606.meta")


create_folder(meteo_data_folder, "Level1")

#%% Load metadata

if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
else:
    raise Exception("Metadata file not found!")

#%% Read the data files
if isinstance(meta["filename"], list):
    files = np.array(meta["filename"])[meta["valid"]]
else:
    files = [meta["filename"]] if meta["valid"] else []


for k, file in enumerate(files):
    try:
        data_temp = read_data(os.path.join(input_folder, file))
    except Exception as e:
        print(e)
        print(f"Failed to process {file}")
        continue
    
    met_series = meteo_series()
    if met_series.read_timeseries(data_temp, meta):
        file_name = file.rsplit('.', 1)[0]
        print(f"Export to L1 netCDF file: L1_meteo_{file_name}.nc")
        export(met_series, os.path.join(meteo_data_folder, "Level1"), f"L1_meteo_{file_name}", overwrite=True)

print("Meteorological data exported to Level 1.")
  

    