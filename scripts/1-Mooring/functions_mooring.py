import os
import json
import shutil
import numpy as np
import pandas as pd
import netCDF4
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta



def read_data(file_path, file_type):
    if file_type == "hobo_T":
        temp = read_temp_hobo(file_path)
    else:
        raise ValueError("File type not recognised: {}".format(file_type))
    return temp

def read_temp_hobo(file_path):
    ind_name=file_path.rfind("\\")
    if ind_name==-1: # The character was not found
        ind_name=file_path.rfind("/")
        if ind_name==-1:
            ind_name=np.nan
            
    if ~np.isnan(ind_name): 
        filename=file_path[ind_name+1:]
        filepath=file_path[:ind_name]   
    else: # only the filename is in the path
        filename=file_path
        filepath=''
        
    df = pd.read_excel(file_path, header=None, sheet_name='Data',skiprows=1,usecols=[1,2], names=["time","Temp"],index_col=False)
    df["time"]=df["time"].astype("datetime64[s]")
    data_temp={"folder":filepath,"file":filename,"data":df}
    return data_temp


def ch1903_to_latlng(x, y):
    x_aux = (x - 600000) / 1000000
    y_aux = (y - 200000) / 1000000
    lat = 16.9023892 + 3.238272 * y_aux - 0.270978 * x_aux ** 2 - 0.002528 * y_aux ** 2 - 0.0447 * x_aux ** 2 * y_aux - 0.014 * y_aux ** 3
    lng = 2.6779094 + 4.728982 * x_aux + 0.791484 * x_aux * y_aux + 0.1306 * x_aux * y_aux ** 2 - 0.0436 * x_aux ** 3
    lat = (lat * 100) / 36
    lng = (lng * 100) / 36
    return lat, lng

def copy_variables(variables_dict):
    var_dict = dict()
    for var in variables_dict:
        var_dict[var] = variables_dict[var][:]
    nc_copy = deepcopy(var_dict)
    return nc_copy

def export(obj, folder, title, output_period="file", time_label="time", profile_to_grid=False, overwrite=False):
    # If profile_to_grid=True, variable has been interpolated to a grid (e.g., profle with fixed depths)
    if profile_to_grid:
        variables = obj.grid_variables
        dimensions = obj.grid_dimensions
        data = obj.grid
    else:
        variables = obj.variables
        dimensions = obj.dimensions
        data = obj.data

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
        print('Output period "{}" not recognised.'.format(output_period))
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
                for key in obj.general_attributes:
                    setattr(nc, key, obj.general_attributes[key])
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
                                        obj.logger.warning("Unable to write {} with {} dimensions.".format(key, len(
                                            values["dim"])))

                        else:
                            obj.logger.warning("Grid data already exists in NetCDF, skipping.")
                    else:
                        idx = position_in_array(nc_time, time[0])
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
                                    obj.logger.warning(
                                        "Unable to write {} with {} dimensions.".format(key, len(values["dim"])))
                else:
                    if np.all(np.isin(time, nc_time)) and not overwrite:
                        obj.logger.warning("Data already exists in NetCDF, skipping.")
                    else:
                        non_duplicates = ~np.isin(time, nc_time)
                        valid = np.logical_and(valid_time, non_duplicates)
                        combined_time = np.append(nc_time, time[valid])
                        order = np.argsort(combined_time)
                        nc_copy = copy_variables(nc.variables)
                        for key, values in obj.variables.items():
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

def position_in_array(arr, value):
    for i in range(len(arr)):
        if value < arr[i]:
            return i
    return len(arr)

def create_temp_grid(path,files,tnum_interp):
    data_grid=dict()
    data_grid["time"]=tnum_interp
    data_grid["depth"]=np.full(len(files),np.nan)
    data_grid["temp"]=np.full((len(files),len(tnum_interp)),np.nan)
    
    
    for k,file in enumerate(files):
        L2_data=netCDF4.Dataset(os.path.join(path,file), mode='r', format='NETCDF4_CLASSIC')
        data_grid["depth"][k]=float(getattr(L2_data,"Depth (m)"))
        data_grid["temp"][k,:]=np.interp(tnum_interp,L2_data.variables["time"][:].data,L2_data.variables["Temp"][:].data,left=np.nan,right=np.nan)
    
    indsort=np.argsort(data_grid["depth"])
    data_grid["temp"]=data_grid["temp"][indsort,:]
    data_grid["depth"]=data_grid["depth"][indsort]
    
    
    return data_grid

def create_folder(input_folder,output_folder):
    if os.path.exists(os.path.join(input_folder, output_folder)):
        print("Folder {} already exists: delete it".format(output_folder))
        shutil.rmtree(os.path.join(input_folder, output_folder))
    os.makedirs(os.path.join(input_folder, output_folder))
    



        