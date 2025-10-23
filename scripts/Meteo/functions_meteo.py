import os
import json
import shutil
import numpy as np
import pandas as pd
import netCDF4
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta


def read_data(file_path):
    """
    Read a meteorological CSV file from MeteoSwiss model output.
    Expected to contain a column 'UTC_time' and multiple meteorological variables.
    """
    filename = os.path.basename(file_path)
    filepath = os.path.dirname(file_path)

    try:
        df = pd.read_csv(file_path)

        # Convert timestamps
        if "UTC_time" in df.columns:
            df["UTC_time"] = pd.to_datetime(df["UTC_time"], format="%d/%m/%Y %H:%M", errors="coerce")
        elif "time" in df.columns:
            df["UTC_time"] = pd.to_datetime(df["time"], errors="coerce")
        else:
            raise ValueError("Missing time column ('UTC_time').")

        df = df.dropna(subset=["UTC_time"])

    except Exception as e:
        raise ValueError(f"Error reading meteorological file {filename}: {e}")

    return {"folder": filepath, "file": filename, "data": df}


def ch1903_to_latlng(x, y):
    """Convert Swiss CH1903 coordinates to latitude and longitude (WGS84)."""
    x_aux = (x - 600000) / 1000000
    y_aux = (y - 200000) / 1000000
    lat = 16.9023892 + 3.238272 * y_aux - 0.270978 * x_aux**2 - 0.002528 * y_aux**2 - 0.0447 * x_aux**2 * y_aux - 0.014 * y_aux**3
    lng = 2.6779094 + 4.728982 * x_aux + 0.791484 * x_aux * y_aux + 0.1306 * x_aux * y_aux**2 - 0.0436 * x_aux**3
    lat = (lat * 100) / 36
    lng = (lng * 100) / 36
    return lat, lng


def copy_variables(variables_dict):
    """Deep copy NetCDF-like variable dictionary."""
    return deepcopy({var: variables_dict[var][:] for var in variables_dict})


def export(obj, folder, title, output_period="file", time_label="time", overwrite=False):
    """
    Export meteorological data (time series) to NetCDF files.
    """
    variables = obj.variables
    dimensions = obj.dimensions
    data = obj.data

    time = data[time_label]
    time_min = datetime.utcfromtimestamp(np.nanmin(time)).replace(tzinfo=timezone.utc)
    time_max = datetime.utcfromtimestamp(np.nanmax(time)).replace(tzinfo=timezone.utc)

    # Determine export period
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
    else:
        raise ValueError(f'Output period "{output_period}" not recognised.')

    # Create output folder
    os.makedirs(folder, exist_ok=True)
    output_files = []

    while file_start < time_max:
        file_end = file_start + file_period
        filename = f"{title}_{file_start.strftime('%Y%m%d_%H%M%S')}.nc"
        out_file = os.path.join(folder, filename)
        output_files.append(out_file)

        valid_time = (time >= datetime.timestamp(file_start)) & (time <= datetime.timestamp(file_end))

        with netCDF4.Dataset(out_file, mode='w', format='NETCDF4') as nc:
            # Global attributes
            for key in obj.general_attributes:
                setattr(nc, key, obj.general_attributes[key])

            # Dimensions
            for key, values in dimensions.items():
                size = len(data[time_label][valid_time]) if values['dim_name'] == time_label else values['dim_size']
                nc.createDimension(values['dim_name'], size)

            # Variables
            for key, values in variables.items():
                var = nc.createVariable(values["var_name"], np.float64, values["dim"], fill_value=np.nan)
                var.units = values["unit"]
                var.long_name = values["long_name"]

                if len(values["dim"]) == 1 and values["dim"][0] == time_label:
                    var[:] = data[key][valid_time]
                elif key == time_label:
                    var[:] = data[time_label][valid_time]

        file_start += file_period

    return output_files


def position_in_array(arr, value):
    """Find insertion index of a value in a sorted array."""
    for i in range(len(arr)):
        if value < arr[i]:
            return i
    return len(arr)


def create_folder(input_folder, output_folder):
    """Create (or recreate) a clean output folder."""
    full_path = os.path.join(input_folder, output_folder)
    if os.path.exists(full_path):
        print(f"Folder {output_folder} already exists: deleting it.")
        shutil.rmtree(full_path)
    os.makedirs(full_path)
