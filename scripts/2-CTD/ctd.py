# -*- coding: utf-8 -*-
import os
import json
import sys
import netCDF4
import numpy as np
import seawater as sw
from copy import deepcopy
from envass import qualityassurance
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import functions_ctd as func


class CTD:
    def __init__(self):
        self.general_attributes = {
            "institution": "Eawag",
            "source": "SURF CTD",
            "references": "EAWAG SURF CTD profiles",
            "history": "See history on Renku",
            "conventions": "CF 1.7",
            "comment": "Data from CTD profiles collected by the SURF Department at Eawag",
            "title": "Surf CTD"
        }

        self.dimensions = {
            'time': {'dim_name': 'time', 'dim_size': None}
        }

        self.variables = {
            'time': {'var_name': 'time', 'dim': ('time',), 'unit': 'seconds since 1970-01-01 00:00:00', 'long_name': 'time'},
            'Press': {'var_name': 'Press', 'dim': ('time',), 'unit': 'dbar', 'long_name': 'pressure'},
            'Temp': {'var_name': 'Temp', 'dim': ('time',), 'unit': 'degC', 'long_name': 'temperature'},
            'Cond': {'var_name': 'Cond', 'dim': ('time',), 'unit': 'mS/cm', 'long_name': 'conductivity'},
            'Chl_A': {'var_name': 'Chl_A', 'dim': ('time',), 'unit': 'µg/l', 'long_name': 'chlorophyll A'},
            'Turb': {'var_name': 'Turb', 'dim': ('time',), 'unit': 'FTU', 'long_name': 'Turbidity'},
            'pH': {'var_name': 'pH', 'dim': ('time',), 'unit': '_', 'long_name': 'pH'},
            'sat': {'var_name': 'sat', 'dim': ('time',), 'unit': '%', 'long_name': 'oxygen saturation'},
            'DO_mg': {'var_name': 'DO_mg', 'dim': ('time',), 'unit': 'mg/l', 'long_name': 'oxygen concentration'},
            'Flur': {'var_name': 'Flur', 'dim': ('time',), 'unit': 'mg/m3', 'long_name': 'Fluorescence'},
        }

        self.derived_variables = {
            "rho": {'var_name': "rho", 'dim': ('time',), 'unit': 'kg/m3', 'long_name': "Density", },
            "depth": {'var_name': "depth", 'dim': ('time',), 'unit': 'm', 'long_name': "Depth", },
            "SALIN": {'var_name': 'SALIN', 'dim': ('time',), 'unit': ['PSU', 'ppt'], 'long_name': 'salinity', }
        }

        self.start_profile_index = False
        self.bottom_profile_index = False
        self.end_profile_index = False
        self.start_pressure = False
        self.end_pressure = False
        self.air_pressure = False
        self.latitude = False
        self.altitude = False
        self.data = {}
        self.filename = False

    def read_profile(self, profile,profilemeta=None):
        if not profilemeta:
            profilemeta=profile["name"] + ".meta"
        meta_path = os.path.join(profile["folder"],profilemeta)
        self.filename = profile["file"]
        df = profile["data"]
        for variable in self.variables:
            if variable in df.columns:
                self.data[variable] = np.array(df[variable].values).astype("float")
            else:
                self.data[variable] = np.array([np.nan] * len(df))
        if "bottom_index" in profile:
            self.bottom_profile_index = profile["bottom_index"]
        if "air_pressure" in profile:
            self.air_pressure = profile["air_pressure"]
  
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if "valid" in meta and not meta["valid"]:
                self.logger.warning("Profile {} marked invalid, not processing.".format(profile["name"]))
                return False
            for key in meta["campaign"]:
                if isinstance(meta["campaign"][key], bool):
                    self.general_attributes[key] = str(meta["campaign"][key])
                else:
                    self.general_attributes[key] = meta["campaign"][key]
            for key in meta["profile"]:
                if isinstance(meta["profile"][key], bool):
                    self.general_attributes[key] = str(meta["profile"][key])
                else:
                    self.general_attributes[key] = meta["profile"][key]
            if ("X Coordinate (CH1903)" in self.general_attributes and
                    self.general_attributes["X Coordinate (CH1903)"] != ""):
                latitude, longitude = func.ch1903_to_latlng(int(self.general_attributes["X Coordinate (CH1903)"]),
                                                            int(self.general_attributes["Y Coordinate (CH1903)"]))
                self.latitude = latitude
                self.general_attributes["latitude"] = latitude
                self.general_attributes["longitude"] = longitude
            elif "Latitude" in self.general_attributes and self.general_attributes["Latitude"] != "":
                self.latitude = self.general_attributes["Latitude"]
            if "Start pressure (dbar)" in self.general_attributes and self.general_attributes["Start pressure (dbar)"] != "":
                self.start_pressure = float(self.general_attributes["Start pressure (dbar)"])
            if "End pressure (dbar)" in self.general_attributes and self.general_attributes["End pressure (dbar)"] != "":
                self.start_pressure = float(self.general_attributes["End pressure (dbar)"])
            if "Altitude (m)" in self.general_attributes and self.general_attributes["Altitude (m)"] != "":
                self.altitude = float(self.general_attributes["Altitude (m)"])
            return True
        else:
            self.logger.warning("{} not found.".format(meta_path))
            return False

    def quality_assurance(self, file_path, simple=True):
        with open(file_path) as f:
            quality_assurance_dict = func.json_converter(json.load(f))
        for key, values in self.variables.copy().items():
            if "_qual" not in key:
                if (quality_assurance_dict[key]["advanced"]) or (quality_assurance_dict[key]["simple"]):
                    name = key + "_qual"
                    self.variables[name] = {'var_name': name, 'dim': values["dim"],
                                            'unit': '0 = nothing to report, 1 = more investigation',
                                            'long_name': name, }
                    if simple:
                        self.data[name] = qualityassurance(np.array(self.data[key]), np.array(self.data["time"]), **quality_assurance_dict[key]["simple"])
                    else:
                        quality_assurance_all = dict(quality_assurance_dict[key]["simple"], **quality_assurance_dict[key]["advanced"])
                        self.data[name] = qualityassurance(np.array(self.data[key]), np.array(self.data["time"]), **quality_assurance_all)
                    if key != "time":
                        if self.bottom_profile_index:
                            self.data[name][self.bottom_profile_index:] = 1
                        if self.start_profile_index:
                            self.data[name][:self.start_profile_index] = 1
                        if self.start_pressure:
                            self.data[name][self.data["Press"] > self.start_pressure] = 1
                        if self.end_pressure:
                            self.data[name][self.data["Press"] < self.end_pressure] = 1

    def export(self, folder, title, output_period="file", time_label="time", profile_to_grid=False, overwrite=False):
        if profile_to_grid:
            variables = self.grid_variables
            dimensions = self.grid_dimensions
            data = self.grid
        else:
            variables = self.variables
            dimensions = self.dimensions
            data = self.data

        time = data[time_label]
        time_min = datetime.utcfromtimestamp(np.nanmin(time)).replace(tzinfo=timezone.utc)
        time_max = datetime.utcfromtimestamp(np.nanmax(time)).replace(tzinfo=timezone.utc)

        if output_period == "file":
            file_start = time_min
            file_period = time_max - time_min
        elif output_period == "daily":
            file_start = time_min.replace(hour=0, minute=0, second=0, microsecond=0)
            file_period = timedelta(days=1)
        elif output_period == "weekly":
            file_start = (time_min - timedelta(days=time_min.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            file_period = timedelta(weeks=1)
        elif output_period == "monthly":
            file_start = time_min.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            file_period = relativedelta(months=+1)
        elif output_period == "yearly":
            file_start = time_min.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            file_period = relativedelta(year=+1)
        else:
            self.logger.warning('Output period "{}" not recognised.'.format(output_period))
            return

        if not os.path.exists(folder):
            os.makedirs(folder)

        output_files = []
        while file_start < time_max:
            file_end = file_start + file_period
            filename = "{}_{}.nc".format(title, file_start.strftime('%Y%m%d_%H%M%S'))
            out_file = os.path.join(folder, filename)
            output_files.append(out_file)
            valid_time = (time >= datetime.timestamp(file_start)) & (time <= datetime.timestamp(file_end))

            if not os.path.isfile(out_file):
                with netCDF4.Dataset(out_file, mode='w', format='NETCDF4') as nc:
                    # nc=create_netCDF(out_file, mode_name='w', format_name='NETCDF4')
                    for key in self.general_attributes:
                        setattr(nc, key, self.general_attributes[key])
                    for key, values in dimensions.items():
                        nc.createDimension(values['dim_name'], values['dim_size'])
                    for key, values in variables.items():
                        var = nc.createVariable(values["var_name"], np.float64, values["dim"], fill_value=np.nan)
                        var.units = values["unit"]
                        var.long_name = values["long_name"]
                        if profile_to_grid and key == time_label:
                            var[0] = time[0]
                        elif profile_to_grid and len(values["dim"]) == 2:
                            if values["dim"][0] == time_label:
                                var[0, :] = data[key]
                            elif values["dim"][1] == time_label:
                                var[:, 0] = data[key]
                            else:
                                raise ValueError("Failed to write variable {} with dimensions: {} to file".format(key, ", ".join(values["dim"])))
                        else:
                            if len(values["dim"]) == 1:
                                if values["dim"][0] == time_label:
                                    var[:] = data[key][valid_time]
                                else:
                                    var[:] = data[key]
                            elif len(values["dim"]) == 2:
                                if values["dim"][0] == time_label:
                                    var[:] = data[key][valid_time, :]
                                elif values["dim"][1] == time_label:
                                    var[:] = data[key][:, valid_time]
                            else:
                                raise ValueError("Failed to write variable {} with dimensions: {} to file".format(key, ", ".join(values["dim"])))
                    #close_netCDF(nc,out_file)
            
            else:
                with netCDF4.Dataset(out_file, mode='a', format='NETCDF4') as nc:
                    #nc=create_netCDF(out_file, mode_name='a', format_name='NETCDF4')
                    nc_time = np.array(nc.variables[time_label][:])
                    if profile_to_grid:
                        if time[0] in nc_time:
                            if overwrite:
                                idx = np.where(nc_time == time[0])[0][0]
                                for key, values in variables.items():
                                    if key not in dimensions:
                                        if len(values["dim"]) == 1:
                                            if hasattr(data[key], "__len__"):
                                                nc.variables[key][idx] = data[key][0]
                                            else:
                                                nc.variables[key][idx] = data[key]
                                        elif len(values["dim"]) == 2 and values["dim"][1] == time_label:
                                            nc.variables[key][:, idx] = data[key]
                                        else:
                                            self.logger.warning("Unable to write {} with {} dimensions.".format(key, len(
                                                values["dim"])))
    
                            else:
                                self.logger.warning("Grid data already exists in NetCDF, skipping.")
                        else:
                            idx = func.position_in_array(nc_time, time[0])
                            nc.variables[time_label][:] = np.insert(nc_time, idx, time[0])
                            for key, values in variables.items():
                                if key not in dimensions:
                                    var = nc.variables[key]
                                    if len(values["dim"]) == 1:
                                        if hasattr(data[key], "__len__"):
                                            var[idx] = data[key][0]
                                        else:
                                            var[idx] = data[key]
                                    elif len(values["dim"]) == 2 and values["dim"][1] == time_label:
                                        end = len(var[:][0]) - 1
                                        if idx != end:
                                            var[:, end] = data[key]
                                            var[:] = var[:, np.insert(np.arange(end), idx, end)]
                                        else:
                                            var[:, idx] = data[key]
                                    else:
                                        self.logger.warning(
                                            "Unable to write {} with {} dimensions.".format(key, len(values["dim"])))
                    else:
                        if np.all(np.isin(time, nc_time)) and not overwrite:
                            self.logger.warning("Data already exists in NetCDF, skipping.")
                        else:
                            non_duplicates = ~np.isin(time, nc_time)
                            valid = np.logical_and(valid_time, non_duplicates)
                            combined_time = np.append(nc_time, time[valid])
                            order = np.argsort(combined_time)
                            nc_copy = func.copy_variables(nc.variables)
                            for key, values in self.variables.items():
                                if time_label in values["dim"]:
                                    if len(values["dim"]) == 1:
                                        combined = np.append(nc_copy[key][:], np.array(data[key])[valid])
                                        if overwrite:
                                            combined[np.isin(combined_time, time)] = np.array(data[key])[
                                                np.isin(time, combined_time)]
                                        out = combined[order]
                                    elif len(values["dim"]) == 2 and values["dim"][1] == time_label:
                                        combined = np.concatenate((np.array(nc_copy[key][:]), np.array(data[key])[:, valid]), axis=1)
                                        if overwrite:
                                            combined[:, np.isin(combined_time, time)] = np.array(data[key])[:, np.isin(time, combined_time)]
                                        out = combined[:, order]
                                    else:
                                        raise ValueError(
                                            "Failed to write variable {} with dimensions: {} to file"
                                            .format(key, ", ".join(values["dim"])))
                                    nc.variables[key][:] = out
                    #close_netCDF(nc,out_file)
            file_start = file_start + file_period
        return output_files

    def mask_data(self):
        for var in self.variables:
            if var + "_qual" in self.data:
                idx = self.data[var + "_qual"][:] > 0
                self.data[var][idx] = np.nan

    def derive_variables(self, y_cond=0.874e-3, beta=0.807e-3):
        if self.altitude == False or self.latitude == False:
            raise ValueError("Altitude and latitude must be provided in metadata to calculate additional parameters")
        data = deepcopy(self.data)
        for var in self.variables:
            if "_qual" not in var:
                idx = data[var + "_qual"] > 0
                data[var][idx] = np.nan
        data["adj_press"] = data["Press"] - self.air_pressure # Atmospheric pressure is computed from measurements in the air in function extract_single_profile
        threshold = data["Temp"].shape[0] * 0.9
        if sum(np.isnan(data["Temp"])) > threshold or sum(np.isnan(data["Cond"])) > threshold or \
                sum(np.isnan(data["adj_press"])) > threshold:
            raise ValueError("Not enough valid parameters to calculate additional parameters.")
        else:
            self.variables.update(self.derived_variables)

        self.data["SALIN"] = func.salinity(data["Temp"], data["Cond"], y_cond, temperature_func=func.default_salinity_temperature)
        self.data["rho"] = np.asarray([1000] * len(data["Press"]))
        self.data["rho"] = func.density(data["Temp"], self.data["SALIN"])
        self.data["depth"] = 1e4 * data["adj_press"] / self.data["rho"] / sw.g(self.latitude)

        try:
            self.data["pt"] = func.potential_temperature_sw(data["Temp"], self.data["SALIN"], data["adj_press"], 0)
        except Exception:
            self.data["pt"] = np.asarray([np.nan] * len(data["time"]))
            self.logger.warning("Failed to calculate potential temperature")

        try:
            self.data["prho"] = func.density(self.data["pt"], self.data["SALIN"])
        except Exception:
            self.data["prho"] = np.asarray([np.nan] * len(data["time"]))
            self.logger.warning("Failed to calculate potential density")

        try:
            theoretical_saturation = func.oxygen_saturation(self.data["pt"], self.data["SALIN"], self.altitude, self.latitude)
            self.data["sat"] = (self.data["DO_mg"] / theoretical_saturation) * 100
        except Exception:
            self.logger.warning("Failed to replace oxygen saturation")

        # try:
        #     sorted_pt = np.argsort(self.data["pt"])[::-1]
        #     self.data["thorpe"] = -(self.data["depth"] - self.data["depth"][sorted_pt])
        # except Exception:
        #     self.data["thorpe"] = np.asarray([np.nan] * len(data["time"]))
        #     self.logger.warning("Failed to calculate Thorpe Displacements")

    def get_lake(self):
        return self.general_attributes["Lake"].replace(" ", "").lower()

