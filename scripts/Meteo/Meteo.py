# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from datetime import datetime
import pandas as pd



# FUuction to transform ch1093 to lat/ton
def ch1903_to_latlng(x, y):
    x_aux = (x - 600000) / 1000000
    y_aux = (y - 200000) / 1000000
    lat = 16.9023892 + 3.238272 * y_aux - 0.270978 * x_aux ** 2 - 0.002528 * y_aux ** 2 - 0.0447 * x_aux ** 2 * y_aux - 0.014 * y_aux ** 3
    lng = 2.6779094 + 4.728982 * x_aux + 0.791484 * x_aux * y_aux + 0.1306 * x_aux * y_aux ** 2 - 0.0436 * x_aux ** 3
    lat = (lat * 100) / 36
    lng = (lng * 100) / 36
    return lat, lng




class meteo_series:
    def __init__(self):
        self.general_attributes = {
            "institution": "Unil",
            "source": "",
            "references": "Aquatic Science Master field camp",
            "history": "See history on Github",
            "conventions": "CF 1.7",
            "comment": "Monitoring data in Lake Taney performed by Aquatic Science Master students",
            "title": "Mooring Lake Taney"
        }

        self.dimensions = {
            'time': {'dim_name': 'time', 'dim_size': None}
        }

        self.variables = {
            'time': {'var_name': 'time', 'dim': ('time',), 'unit': 'seconds since 1970-01-01 00:00:00', 'long_name': 'time'},
            'T_2M [°C]': {'var_name': 'T2M_C', 'dim': ('time',), 'unit': 'degC', 'long_name': 'air_temperature_2m_C'},
            'T_2M [K]': {'var_name': 'T2M_K', 'dim': ('time',), 'unit': 'K', 'long_name': 'air_temperature_2m_K'},
            'U [m/s]': {'var_name': 'U10', 'dim': ('time',), 'unit': 'm/s', 'long_name': 'zonal_wind_speed_10m'},
            'V [m/s]': {'var_name': 'V10', 'dim': ('time',), 'unit': 'm/s', 'long_name': 'meridional_wind_speed_10m'},
            'GLOB [W/m2]': {'var_name': 'GLOB', 'dim': ('time',), 'unit': 'W/m2', 'long_name': 'global_radiation_surface'},
            'RELHUM_2M [%]': {'var_name': 'RH2M', 'dim': ('time',), 'unit': '%', 'long_name': 'relative_humidity_2m'},
            'PMSL [Pa]': {'var_name': 'PMSL', 'dim': ('time',), 'unit': 'Pa', 'long_name': 'pressure_mean_sea_level'},
            'CLCT [%]': {'var_name': 'CLCT', 'dim': ('time',), 'unit': '%', 'long_name': 'total_cloud_cover'}
        }

        self.start_time = False
        self.end_time = False
        self.latitude = False
        self.altitude = False
        self.data = {}
        self.filename = False

    def read_timeseries(self, data_meteo, meta_meteo):
        self.filename = data_meteo["file"]
        file_noext = self.filename.rsplit('.', 1)[0]
        df = data_meteo["data"]

        # COnvert timeseries to timestamp in seconds
        if "UTC_time" in df.columns:
            df["time"] = df["UTC_time"].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
            self.data["time"] = np.array(df["time"].values)
        else:
            raise ValueError("Missing UTC_time column in input data.")


        # Load data columns
        for variable in self.variables:
            if variable in df.columns:
                self.data[variable] = np.array(df[variable].values).astype("float")
            else:
                self.data[variable] = np.full(len(df), np.nan)

        # Try to read metadata
        meta_path = os.path.join(data_meteo["folder"], file_noext + ".meta")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            print(f"{meta_path} not found, using meteo metadata instead.")
            meta = meta_meteo

        # Check validity
        if "valid" in meta and not meta["valid"]:
            print(f"File {self.filename} marked invalid, not processing.")
            return False

        # Copy dataset-level attributes
        if "dataset" in meta:
            for key, val in meta["dataset"].items():
                self.general_attributes[key] = str(val) if isinstance(val, bool) else val

        # Coordinate conversion
        if ("X Coordinate (CH1903)" in self.general_attributes and
                self.general_attributes["X Coordinate (CH1903)"] != ""):
            latitude, longitude = ch1903_to_latlng(
                int(self.general_attributes["X Coordinate (CH1903)"]),
                int(self.general_attributes["Y Coordinate (CH1903)"])
            )
            self.latitude = latitude
            self.general_attributes["latitude"] = latitude
            self.general_attributes["longitude"] = longitude
        elif "Latitude" in self.general_attributes and self.general_attributes["Latitude"] != "":
            self.latitude = self.general_attributes["Latitude"]

        # Altitude
        if "Altitude (m)" in self.general_attributes and self.general_attributes["Altitude (m)"] != "":
            self.altitude = float(self.general_attributes["Altitude (m)"])

        # Start and end time
        if "Starting time (UTC)" in self.general_attributes:
            self.start_time = datetime.strptime(self.general_attributes["Starting time (UTC)"], "%Y-%m-%d %H:%M:%S")
        if "End time (UTC)" in self.general_attributes:
            self.end_time = datetime.strptime(self.general_attributes["End time (UTC)"], "%Y-%m-%d %H:%M:%S")

        return True


class meteo_grid:
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
            'T_2M [°C]': {'var_name': 'T2M_C', 'dim': ('time',), 'unit': 'degC', 'long_name': 'air_temperature_2m_C'},
            'T_2M [K]': {'var_name': 'T2M_K', 'dim': ('time',), 'unit': 'K', 'long_name': 'air_temperature_2m_K'},
            'U [m/s]': {'var_name': 'U10', 'dim': ('time',), 'unit': 'm/s', 'long_name': 'zonal_wind_speed_10m'},
            'V [m/s]': {'var_name': 'V10', 'dim': ('time',), 'unit': 'm/s', 'long_name': 'meridional_wind_speed_10m'},
            'GLOB [W/m2]': {'var_name': 'GLOB', 'dim': ('time',), 'unit': 'W/m2', 'long_name': 'global_radiation_surface'},
            'RELHUM_2M [%]': {'var_name': 'RH2M', 'dim': ('time',), 'unit': '%', 'long_name': 'relative_humidity_2m'},
            'PMSL [Pa]': {'var_name': 'PMSL', 'dim': ('time',), 'unit': 'Pa', 'long_name': 'pressure_mean_sea_level'},
            'CLCT [%]': {'var_name': 'CLCT', 'dim': ('time',), 'unit': '%', 'long_name': 'total_cloud_cover'}
        }

        self.data = {}
        self.dt_sec = 3600  # [s] — hourly data
        self.latitude = False

    def add_grid(self, data_grid, meta):
        # Load available variables
        for variable in self.variables:
            if variable in data_grid.keys():
                self.data[variable] = data_grid[variable]
            else:
                dim_names = self.variables[variable]["dim"]
                self.data[variable] = np.full(tuple([len(self.data[d]) for d in dim_names]), np.nan)

        # Copy metadata
        if "dataset" in meta:
            for key, val in meta["dataset"].items():
                self.general_attributes[key] = str(val) if isinstance(val, bool) else val

        # Coordinates
        if ("X Coordinate (CH1903)" in self.general_attributes and
                self.general_attributes["X Coordinate (CH1903)"] != ""):
            latitude, longitude = ch1903_to_latlng(
                int(self.general_attributes["X Coordinate (CH1903)"]),
                int(self.general_attributes["Y Coordinate (CH1903)"])
            )
            self.general_attributes["latitude"] = latitude
            self.general_attributes["longitude"] = longitude
        elif "Latitude" in self.general_attributes and self.general_attributes["Latitude"] != "":
            self.latitude = self.general_attributes["Latitude"]
