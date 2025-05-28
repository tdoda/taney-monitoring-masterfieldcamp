# -*- coding: utf-8 -*-
import os
import json
import sys
import netCDF4
import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import functions_mooring as func


class thermistor_series:
    def __init__(self):
        self.general_attributes = {
            "institution": "Unil",
            "source": "",
            "references": "Aquatic Science Master field camp",
            "history": "See history on Renku",
            "conventions": "CF 1.7",
            "comment": "Monitoring data in Lake Taney performed by Aquatic Science Master students",
            "title": "Mooring Lake Taney"
        }

        self.dimensions = {
            'time': {'dim_name': 'time', 'dim_size': None}
        }

        self.variables = {
            'time': {'var_name': 'time', 'dim': ('time',), 'unit': 'seconds since 1970-01-01 00:00:00', 'long_name': 'time'},
            'Temp': {'var_name': 'Temp', 'dim': ('time',), 'unit': 'degC', 'long_name': 'temperature'},
        }


        self.start_time = False
        self.end_time = False
        self.latitude = False
        self.altitude = False
        self.depth = False
        self.data = {}
        self.filename = False

    def read_timeseries(self, data_temp,meta_mooring):
        self.filename = data_temp["file"]
        file_noext = self.filename.rsplit('.', 1)[0]
        df = data_temp["data"]
        for variable in self.variables:
            if variable in df.columns:
                self.data[variable] = np.array(df[variable].values).astype("float")
            else:
                self.data[variable] = np.array([np.nan] * len(df))

        meta_path = os.path.join(data_temp["folder"], file_noext + ".meta")
         
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            print("{} not found, use mooring metadata instead".format(meta_path))
            indfile=np.where(np.array(meta_mooring["filenames"])==self.filename)[0][0]
            meta={"valid":np.array(meta_mooring["valid"])[indfile],"Depth (m)":np.array(meta_mooring["Depth (m)"])[indfile],"campaign":meta_mooring["campaign"]}
            
        if "valid" in meta and not meta["valid"]:
            print("File {} marked invalid, not processing.".format(data_temp["file"]))
            return False
        for key in meta["campaign"]:
            if isinstance(meta["campaign"][key], bool):
                self.general_attributes[key] = str(meta["campaign"][key])
            else:
                self.general_attributes[key] = meta["campaign"][key]
        self.general_attributes["Depth (m)"] = meta["Depth (m)"]
        if ("X Coordinate (CH1903)" in self.general_attributes and
                self.general_attributes["X Coordinate (CH1903)"] != ""):
            latitude, longitude = func.ch1903_to_latlng(int(self.general_attributes["X Coordinate (CH1903)"]),
                                                        int(self.general_attributes["Y Coordinate (CH1903)"]))
            self.latitude = latitude
            self.general_attributes["latitude"] = latitude
            self.general_attributes["longitude"] = longitude
        
        elif "Latitude" in self.general_attributes and self.general_attributes["Latitude"] != "":
            self.latitude = self.general_attributes["Latitude"]
        if "Altitude (m)" in self.general_attributes and self.general_attributes["Altitude (m)"] != "":
            self.altitude = float(self.general_attributes["Altitude (m)"])
        if "Depth (m)" in self.general_attributes and self.general_attributes["Depth (m)"] != "":
            self.depth = float(self.general_attributes["Depth (m)"])
        if "Time of deployment" in self.general_attributes and self.general_attributes["Time of deployment"] != "":
            self.start_time=datetime.strptime(self.general_attributes["Time of deployment"],"%Y-%m-%d %H:%M:%S")
        if "Time of retrieval" in self.general_attributes and self.general_attributes["Time of retrieval"] != "":
            self.end_time=datetime.strptime(self.general_attributes["Time of retrieval"],"%Y-%m-%d %H:%M:%S")
            
        return True
    
    def quality_assurance(self):  
        for key, values in self.variables.copy().items():
            if "_qual" not in key: 
                if key != "time": # Only add quality assurance on non temporal data (i.e., remove data before deployment and after retrieval)
                    name = key + "_qual"
                    self.variables[name] = {'var_name': name, 'dim': values["dim"],
                                            'unit': '0 = nothing to report, 1 = more investigation',
                                            'long_name': name, }
                    self.data[name]=np.zeros(self.data[key].shape)
                    if self.start_time:
                        self.data[name][self.data["time"]<self.start_time.replace(tzinfo=timezone.utc).timestamp()] = 1
                    if self.end_time:
                        self.data[name][self.data["time"]>self.end_time.replace(tzinfo=timezone.utc).timestamp()] = 1


    def mask_data(self):
        for var in self.variables:
            if var + "_qual" in self.data:
                idx = self.data[var + "_qual"][:] > 0
                self.data[var][idx] = np.nan



class thermistor_grid:
    def __init__(self):
        self.general_attributes = {
            "institution": "Unil",
            "source": "",
            "references": "Aquatic Science Master field camp",
            "history": "See history on Renku",
            "conventions": "CF 1.7",
            "comment": "Monitoring data in Lake Taney performed by Aquatic Science Master students",
            "title": "Mooring Lake Taney"
        }

        self.dimensions = {
            'time': {'dim_name': 'time', 'dim_size': None},
            "depth": {'dim_name': "depth", 'dim_size': None}
        }

        self.variables = {
            'time': {'var_name': 'time', 'dim': ('time',), 'unit': 'seconds since 1970-01-01 00:00:00', 'long_name': 'Time'},
            'depth': {'var_name': 'depth', 'dim': ('depth',), 'unit': 'm', 'long_name': "Depth"},
            'temp': {'var_name': 'temp', 'dim': ('depth', 'time'), 'unit': 'degC', 'long_name': 'Temperature'}
        }

        self.data = {}
        self.dt_sec=10*60 # [s]
        
    def add_grid(self, data_grid,meta):
        for variable in self.variables:
            if variable in data_grid.keys():
                self.data[variable] = data_grid[variable]
            else:
                dim_names=self.variables[variable]["dim"]
                self.data[variable] = np.full(tuple([len(self.data[d]) for d in dim_names]),np.nan)
  
        if "valid" in meta:
            ind_sensors=np.where(meta["valid"])[0]
        else:
            ind_sensors=np.arange(len(data_grid["depth"]))

        for key in meta["campaign"]:
            if isinstance(meta["campaign"][key], bool):
                self.general_attributes[key] = str(meta["campaign"][key])
            else:
                self.general_attributes[key] = meta["campaign"][key]
        self.general_attributes["Depth (m)"] = np.array(meta["Depth (m)"])[ind_sensors]
        self.general_attributes["Filenames"] = np.array(meta["filenames"])[ind_sensors]
        self.general_attributes["Filetypes"] = np.array(meta["filetypes"])[ind_sensors]
        if ("X Coordinate (CH1903)" in self.general_attributes and
                self.general_attributes["X Coordinate (CH1903)"] != ""):
            latitude, longitude = func.ch1903_to_latlng(int(self.general_attributes["X Coordinate (CH1903)"]),
                                                        int(self.general_attributes["Y Coordinate (CH1903)"]))
            self.general_attributes["latitude"] = latitude
            self.general_attributes["longitude"] = longitude
        
        elif "Latitude" in self.general_attributes and self.general_attributes["Latitude"] != "":
            self.latitude = self.general_attributes["Latitude"]
