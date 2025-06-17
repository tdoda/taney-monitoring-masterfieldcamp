# -*- coding: utf-8 -*-
import os
import sys
import json
import shutil
import numpy as np
from datetime import datetime, timezone
from ctd import CTD
from functions_ctd import create_file_list, copy_files, read_data, process_profiles, create_folder

#%% Specify field campaign here:

date_campaign='20250605'

# For RBR profiles:
ctd_data_folder='..\..\data\Profiles\RBR_237207'
extensions = [".rsk"]
DO_umol=True # Do data is in umol/l and needs to be converted to mg/l

# For EXO profiles:
# ctd_data_folder='..\..\data\Profiles\EXO'
# extensions = [".csv"]

#%% Other parameters

input_folder=os.path.join(ctd_data_folder,date_campaign)

create_folder(input_folder, "Level1")
create_folder(input_folder, "Level2")

files = create_file_list(os.path.join(input_folder, "Level0"))
metadata_required=[]
#%% Read and export CTD data
for file in files:
    print("Processing file {}".format(file["path"]))
    try:
        profiles = read_data(file["path"], file["type"],DO_umol)
    except Exception as e:
        print(e)
        print("Failed to process {}".format(file["path"]))
        continue
    
    list_metafiles=[f for f in os.listdir(os.path.join(os.path.dirname(file["path"]))) if f.endswith(".meta") and profiles[0]["name"] in f]
    search_meta=False
    if len(list_metafiles)>len(profiles):
        print("**** WARNING: more meta files than detected profiles! ****") 
        search_meta=True
    elif len(list_metafiles)<len(profiles):
        print("**** WARNING: not all detected profiles avec metadata! ****")
        search_meta=True
        
    if search_meta: # List profiling time from all metadata files
        time_prof=np.full(len(list_metafiles),np.nan)
        for k,meta_file in enumerate(list_metafiles):
            meta_path=os.path.join(os.path.dirname(file["path"]),meta_file)
            with open(meta_path) as f:
                meta = json.load(f)
            time_prof_str=meta["campaign"]["Date of measurement"]+" "+meta["profile"]["Time of measurement (local)"]
            time_prof[k]=datetime.strptime(time_prof_str,"%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()
            time_prof[k]=time_prof[k]+(meta["campaign"]["Time Zone device (UTC+)"]-meta["campaign"]["Time Zone local (UTC+)"])*3600 # Device time
        metafiles_found=[""]*len(profiles)
        if len(list_metafiles)>len(profiles): # Find the metadata file for each profile
            for kp,profile in enumerate(profiles):
                indmin=np.argmin(np.abs(time_prof-profile["data"]["time"].iloc[0]))
                metafiles_found[kp]=list_metafiles[indmin]
        else: # Find the profile for each metadatafile (leave empty otherwise)
            time_start_profiles=[profile["data"]["time"].iloc[0] for profile in profiles]
            for km in range(len(list_metafiles)):
                indmin=np.argmin(np.abs(time_start_profiles-time_prof[km]))
                metafiles_found[indmin]=list_metafiles[km]
                         
    for kp,profile in enumerate(profiles):
        if search_meta: # Search for corresponding metadata files
            profilemeta=metafiles_found[kp]   
        else: 
            profilemeta=profile["name"] + ".meta"            
        if profilemeta and os.path.isfile(os.path.join(os.path.dirname(file["path"]), profilemeta)):
            print("Processing profile {}".format(profile["name"]))
            ctd = CTD()
            if ctd.read_profile(profile):
                ctd.quality_assurance('quality_assurance_ctd.json')
                file_name = os.path.basename(file["path"]).rsplit('.', 1)[0]
                ctd.export(os.path.join(input_folder, "Level1"), "L1_CTD_{}_{}".format(file["type"], file_name),overwrite=True)
                ctd.mask_data() # Replace flagged data by nan 
                ctd.derive_variables() # Compute additional variables to add to Level 2
                ctd.export(os.path.join(input_folder, "Level2"), "L2_CTD_{}_{}".format(file["type"], file_name), overwrite=True) # Create Level 2 file      
                
        else:
            print("No metadata for profile {}".format(profile["name"]))
            metadata_required.append(profile)


