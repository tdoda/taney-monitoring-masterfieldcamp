# -*- coding: utf-8 -*-
import os
import sys
import json
import shutil
from ctd import CTD
from functions_ctd import create_file_list, copy_files, read_data, process_profiles, create_folder

#%% Specify field campaign here:

date_campaign='20250605'

# For RBR profiles:
ctd_data_folder='..\..\data\Profiles\RBR_66131'
extensions = [".rsk"]

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
        profiles = read_data(file["path"], file["type"])
    except Exception as e:
        print(e)
        print("Failed to process {}".format(file["path"]))
        continue
    for profile in profiles:
        if os.path.isfile(os.path.join(os.path.dirname(file["path"]), profile["name"] + ".meta")):
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


