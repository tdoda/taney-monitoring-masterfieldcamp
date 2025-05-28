# -*- coding: utf-8 -*-
"""
Read the thermistors data and export it to netCDF files.

@author: T. Doda
"""
import os
import sys
import json
import numpy as np
#sys.path.append(os.path.join(os.path.dirname(__file__), r'..\..\functions\1-Mooring'))
from thermistor import thermistor_series,thermistor_grid
from datetime import datetime, timezone
from functions_mooring import read_data, export, create_temp_grid, create_folder

#%% Specify field campaign here:

# date_campaign='20240527'
# date_campaign='20240910'
date_campaign='20241126'

#%% Setup paths

mooring_data_folder='..\..\data\Mooring\HOBO_T'
input_folder=os.path.join(mooring_data_folder,date_campaign)
meta_path=os.path.join(input_folder,"Level0","thermistors_"+date_campaign+".meta")

create_folder(input_folder, "Level1")
create_folder(input_folder, "Level2")
create_folder(input_folder, "Level3")

#%% Load metadata of the mooring
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
else:
    raise Exception("Metadata file not found!")

#%% Read the data files
files = np.array(meta["filenames"])[meta["valid"]]
file_types=np.array(meta["filetypes"])[meta["valid"]]


for k,file in enumerate(files):
    metadata = True
    try:
        data_temp = read_data(os.path.join(input_folder,"Level0",file),file_types[k])
    except Exception as e:
        print(e)
        print("Failed to process {}".format(file))
        continue
    temp_series = thermistor_series()
    if temp_series.read_timeseries(data_temp,meta):
        temp_series.quality_assurance()
        file_name = file.rsplit('.', 1)[0]
        print("Export to L1 netCDF files")
        export(temp_series,os.path.join(input_folder, "Level1"), "L1_mooring_{}_{}".format(file_types[k], file_name),overwrite=True)
        temp_series.mask_data() # Replace flagged data by nan 
        print("Export to L2 netCDF files")
        export(temp_series,os.path.join(input_folder, "Level2"), "L2_mooring_{}_{}".format(file_types[k], file_name), overwrite=True) # Create Level 2 file      
 
#%% Load L2 files and interpolate to grid
temp_grid = thermistor_grid()
files_L2=os.listdir(os.path.join(input_folder, "Level2"))
# tnum_files=[None]*len(files_L2)
# for k,f in enumerate(files_L2):
#     ind_dash=[i for i, c in enumerate(f) if c == "_"]
#     tnum_files[k]=datetime.strptime(f[ind_dash[-2]+1:f.find(".nc")],"%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()
tnum_start=datetime.strptime(meta["campaign"]["Time of deployment"],"%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
tnum_end=datetime.strptime(meta["campaign"]["Time of retrieval"],"%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
tnum_interp=np.arange(tnum_start,tnum_end,temp_grid.dt_sec)
data_grid=create_temp_grid(os.path.join(input_folder, "Level2"),files_L2,tnum_interp)
temp_grid.add_grid(data_grid,meta)
print("Export to L3 netCDF file")
export(temp_grid,os.path.join(input_folder, "Level3"), "L3_mooring", overwrite=True) # Create Level 3 file      

#%% Interpolated Visualization
depth_interp=np.arange(data_grid["depth"][0],data_grid["depth"][-1],0.1)
temp_interp=np.full((len(depth_interp),len(data_grid["time"])),np.nan)
for kt in range(len(data_grid["time"])):
    temp_interp[:,kt]=np.interp(depth_interp,data_grid["depth"],data_grid["temp"][:,kt])

import matplotlib.pyplot as plt
plt.figure()
plt.pcolormesh(data_grid["time"].astype("datetime64[s]").astype(datetime),depth_interp,temp_interp)
plt.gca().invert_yaxis()
    