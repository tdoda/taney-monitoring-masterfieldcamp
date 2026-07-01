# -*- coding: utf-8 -*-
"""
Reading and export functions for the Lake Taney mooring thermistor pipeline.
 
This module is format-agnostic: raw sensor files are routed to the correct
reader by file extension (.xlsx -> HOBO/RBR Excel export, .csv -> new-campaign
RBR CSV export). All depth information comes EXCLUSIVELY from the .meta file;
nothing here hardcodes a depth.
 
Time convention (IMPORTANT)
---------------------------
Both readers return the raw timestamps as a naive ``datetime64[s]`` wall-clock.
``thermistor_series.read_timeseries`` later turns that into a unix timestamp by
dividing the int64 nanoseconds by 1e9, i.e. the local wall-clock is treated as
the time axis WITHOUT applying an explicit UTC offset. This is intentional and
must stay consistent across:
  - HOBO .xlsx files (old campaign, all CEST),
  - new .csv files (local civil time WITH daylight saving),
  - the "Time of deployment"/"Time of retrieval" strings in the .meta.
Applying a real timezone conversion in one reader only would shift it relative
to the others and corrupt the QAQC window and the L3 grid alignment.
 
@author: T. Doda (original) — extended for the 2025-2026 CSV campaign
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import netCDF4
from copy import deepcopy
from datetime import datetime, time, timezone, timedelta, UTC
from dateutil.relativedelta import relativedelta
 
 
# ---------------------------------------------------------------------------
# File routing and readers
# ---------------------------------------------------------------------------
def read_data(file_path, file_type):
    """Route a raw sensor file to the correct reader based on its extension.
 
    Routing is done on the file EXTENSION rather than the ``file_type`` label,
    so a wrong/legacy label in the .meta cannot send a file to the wrong reader.
    ``file_type`` is still passed through the pipeline for documentation and is
    written into the netCDF filenames.
 
    Parameters
    ----------
    file_path : str
        Path to the raw Level 0 file.
    file_type : str
        Logical type from the .meta ("hobo_T" for .xlsx, "csv_T" for .csv).
        Currently informational only.
 
    Returns
    -------
    dict
        {"folder": str, "file": str, "data": pandas.DataFrame} with the
        DataFrame holding columns ["time", "Temp"]. See the individual readers.
 
    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".xlsx":
        return read_temp_hobo(file_path)
    elif ext == ".csv":
        return read_temp_csv(file_path)
    else:
        raise ValueError("Unsupported file format: {} for file {}".format(ext, file_path))
 
 
def _split_path(file_path):
    """Return (folder, filename) for a path using either separator.
 
    Returns ("", filename) when the path has no directory component.
    """
    ind = file_path.rfind("\\")
    if ind == -1:
        ind = file_path.rfind("/")
    if ind == -1:
        return "", file_path
    return file_path[:ind], file_path[ind + 1:]
 
 
def read_temp_hobo(file_path):
    """Read an old-campaign (2024-2025) HOBO/RBR thermistor .xlsx file.
 
    The Excel export stores data on a sheet named 'Data'; the first data row is
    a header, the timestamp is in column index 1 and the temperature in column
    index 2. Timestamps are naive CEST wall-clock (see module docstring).
 
    Parameters
    ----------
    file_path : str
        Path to the .xlsx file.
 
    Returns
    -------
    dict
        {"folder": str, "file": str, "data": pandas.DataFrame} with DataFrame
        columns ["time" (datetime64[s]), "Temp" (float, degC)].
    """
    filepath, filename = _split_path(file_path)
    df = pd.read_excel(
        file_path, header=None, sheet_name="Data", skiprows=1,
        usecols=[1, 2], names=["time", "Temp"], index_col=False,
    )
    # Force nanosecond resolution so that read_timeseries' "(...)/1e9" yields
    # epoch seconds under ANY pandas version. Under pandas <2.0 datetime64[s]
    # was silently upcast to [ns]; under pandas >=2.0 it is preserved, which
    # would make the /1e9 conversion off by 1e9 and silently produce an
    # all-NaN Level 3 grid. Pinning [ns] here makes the pipeline version-robust.
    df["time"] = df["time"].astype("datetime64[ns]")
    return {"folder": filepath, "file": filename, "data": df}
 
 
def read_temp_csv(file_path):
    """Read a new-campaign (2025-2026) RBR thermistor .csv file.
 
    File layout (columns):
        # | Date-Time (CEST) | Temperature   (°C) | Started | Host Connected | End of File
 
    Key facts verified against the raw data, NOT assumed from the brief:
      * Timestamp format is **MM/DD/YYYY HH:MM:SS** (month-first). The project
        brief stated DD/MM/YYYY, which is wrong: '10/26/2025' exists in the
        files while '26/10/2025' does not, and the series ends on the retrieval
        day '06/03/2026' = 3 June 2026. Parsing with the wrong order would
        silently corrupt every date whose day <= 12.
      * Timestamps are local civil time WITH daylight saving applied by the
        logger. This produces duplicate timestamps at the autumn fall-back
        (26/10/2025) and a ~1 h gap at the spring forward (29/03/2026). These
        artefacts are deliberately LEFT in the returned series and are detected
        and flagged in ``thermistor_series.quality_assurance``.
      * Logger-event rows (Started / Host Connected / End of File markers) have
        an empty temperature and may carry off-cadence timestamps; they are
        dropped here.
 
    Time is returned as a naive ``datetime64[s]`` wall-clock, identical to
    ``read_temp_hobo`` (see module docstring). The columns Started,
    Host Connected and End of File are ignored.
 
    Parameters
    ----------
    file_path : str
        Path to the .csv file.
 
    Returns
    -------
    dict
        {"folder": str, "file": str, "data": pandas.DataFrame} with DataFrame
        columns ["time" (datetime64[s]), "Temp" (float, degC)]. Identical
        structure to ``read_temp_hobo``.
    """
    filepath, filename = _split_path(file_path)
 
    # utf-8-sig strips the BOM present at the start of the header line.
    df_raw = pd.read_csv(file_path, encoding="utf-8-sig")
 
    # Identify columns by content: the header contains a degree sign and
    # multiple spaces that make exact-name matching fragile.
    time_col = next(c for c in df_raw.columns if "Date-Time" in c)
    temp_col = next(c for c in df_raw.columns if "Temperature" in c)
 
    time = pd.to_datetime(df_raw[time_col], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    temp = pd.to_numeric(df_raw[temp_col], errors="coerce")
 
    df = pd.DataFrame({"time": time, "Temp": temp})
    # Drop logger-event rows (no temperature) and any unparseable timestamps.
    # Duplicates (DST) are NOT dropped here — that is QAQC's responsibility.
    df = df[df["time"].notna() & df["Temp"].notna()].reset_index(drop=True)
    # Nanosecond resolution -> version-robust epoch-seconds conversion in
    # read_timeseries (see read_temp_hobo for the detailed rationale).
    df["time"] = df["time"].astype("datetime64[ns]")
 
    return {"folder": filepath, "file": filename, "data": df}
 
 
# ---------------------------------------------------------------------------
# Depth handling (multi-configuration) — all depths come from the .meta
# ---------------------------------------------------------------------------
def compute_depths_from_meta(meta):
    """Return depth-from-surface (m) for every sensor listed in ``meta``.
 
    The rule depends on ``meta["mooring_config"]``:
      * "single_chain" (old campaign): depths are taken directly from
        ``meta["Depth (m)"]`` — no conversion.
      * "option1" / "option2" (new campaign): for each sensor, if its
        ``raw_depth_reference`` is "height_from_bottom" the depth is
        ``total_depth - raw_depth_value``; if it is "depth_from_surface" the
        raw value is already a depth and is used directly.
 
    Parameters
    ----------
    meta : dict
        Parsed .meta content.
 
    Returns
    -------
    list of float
        Depth-from-surface (m), one per entry of ``meta["filenames"]``.
 
    Raises
    ------
    ValueError
        If ``total_depth`` is null/missing for option1/option2 (never silently
        use a wrong depth), or if a config / depth reference is unknown.
    """
    config = meta.get("mooring_config", "single_chain")
 
    if config == "single_chain":
        return [float(d) for d in meta["Depth (m)"]]
 
    if config in ("option1", "option2", "option3"):
        total_depth = meta.get("total_depth", None)
        if total_depth is None:
            raise ValueError(
                "total_depth is null in .meta — run Option 3 pressure calibration "
                "first and update the .meta file before proceeding."
            )
        depths = []
        for ref, raw in zip(meta["raw_depth_reference"], meta["raw_depth_values"]):
            if ref == "height_from_bottom":
                depths.append(float(total_depth) - float(raw))
            elif ref == "depth_from_surface":
                depths.append(float(raw))
            else:
                raise ValueError("Unknown raw_depth_reference: {}".format(ref))
        return depths
 
    raise ValueError("Unknown mooring_config: {}".format(config))
 
 
def filter_meta_by_chain(meta, chain):
    """Return a shallow copy of ``meta`` restricted to one buoy chain.
 
    Used by option2 so that each per-chain L3 grid carries metadata describing
    only the sensors on that chain. Per-sensor list fields are filtered; scalar
    fields and the ``campaign`` block are copied unchanged.
 
    Parameters
    ----------
    meta : dict
        Full parsed .meta content (must contain a "chain" list).
    chain : str
        "A" or "B".
 
    Returns
    -------
    dict
        Filtered metadata.
    """
    idx = [i for i, c in enumerate(meta["chain"]) if c == chain]
    list_fields = [
        "filenames", "filetypes", "valid", "sensor_numbers", "sensor_references",
        "chain", "raw_depth_reference", "raw_depth_values", "Depth (m)",
    ]
    out = dict(meta)  # keep scalars (mooring_config, total_depth, campaign, ...)
    for f in list_fields:
        if f in meta:
            out[f] = [meta[f][i] for i in idx]
    return out
 
 
# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------
def ch1903_to_latlng(x, y):
    """Convert Swiss CH1903 (LV03) coordinates to WGS84 latitude/longitude.
 
    Parameters
    ----------
    x, y : float
        CH1903 easting (x) and northing (y) in metres.
 
    Returns
    -------
    (float, float)
        (latitude, longitude) in decimal degrees.
    """
    x_aux = (x - 600000) / 1000000
    y_aux = (y - 200000) / 1000000
    lat = 16.9023892 + 3.238272 * y_aux - 0.270978 * x_aux ** 2 - 0.002528 * y_aux ** 2 - 0.0447 * x_aux ** 2 * y_aux - 0.014 * y_aux ** 3
    lng = 2.6779094 + 4.728982 * x_aux + 0.791484 * x_aux * y_aux + 0.1306 * x_aux * y_aux ** 2 - 0.0436 * x_aux ** 3
    lat = (lat * 100) / 36
    lng = (lng * 100) / 36
    return lat, lng
 
 
# ---------------------------------------------------------------------------
# netCDF export and gridding (unchanged behaviour from the original pipeline)
# ---------------------------------------------------------------------------
def copy_variables(variables_dict):
    """Deep-copy the numeric content of a netCDF variables mapping."""
    var_dict = dict()
    for var in variables_dict:
        var_dict[var] = variables_dict[var][:]
    nc_copy = deepcopy(var_dict)
    return nc_copy
 
 
def export(obj, folder, title, output_period="file", time_label="time", profile_to_grid=False, overwrite=False):
    """Write a thermistor object (series or grid) to netCDF file(s).
 
    Splits the output by ``output_period`` ("file", "daily", "weekly",
    "monthly", "yearly"). If a target file already exists, new (non-duplicate)
    time steps are appended; existing steps are overwritten only if
    ``overwrite=True``.
 
    Parameters
    ----------
    obj : thermistor_series or thermistor_grid
        Object exposing ``general_attributes``, ``dimensions``, ``variables``
        and ``data`` (and ``grid_*`` when ``profile_to_grid=True``).
    folder : str
        Output directory (created if missing).
    title : str
        Filename prefix; the time stamp of the file start is appended.
    output_period : str, optional
        Temporal chunking of output files. Default "file" (one file).
    time_label : str, optional
        Name of the time variable. Default "time".
    profile_to_grid : bool, optional
        True when writing an interpolated grid object. Default False.
    overwrite : bool, optional
        Overwrite already-present time steps. Default False.
 
    Returns
    -------
    list of str
        Paths of the netCDF files written/updated.
    """
    if profile_to_grid:
        variables = obj.grid_variables
        dimensions = obj.grid_dimensions
        data = obj.grid
    else:
        variables = obj.variables
        dimensions = obj.dimensions
        data = obj.data
 
    time = data[time_label]
    time_min = datetime.fromtimestamp(np.nanmin(time), UTC)
    time_max = datetime.fromtimestamp(np.nanmax(time), UTC)
 
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
        file_start = file_start + file_period
    return output_files
 
 
def position_in_array(arr, value):
    """Return the insertion index keeping ``arr`` sorted ascending."""
    for i in range(len(arr)):
        if value < arr[i]:
            return i
    return len(arr)
 
 
def create_temp_grid(path, files, tnum_interp):
    """Interpolate a set of Level 2 sensor files onto a common time axis.
 
    Each L2 netCDF carries a single sensor's masked temperature series and a
    "Depth (m)" global attribute. Temperatures are linearly interpolated onto
    ``tnum_interp`` (no extrapolation: values outside each sensor's own time
    span become NaN). The resulting field is sorted by increasing depth.
 
    Parameters
    ----------
    path : str
        Directory containing the L2 .nc files.
    files : list of str
        L2 filenames to assemble into the grid.
    tnum_interp : numpy.ndarray
        Target time axis (unix seconds), regularly spaced.
 
    Returns
    -------
    dict
        {"time": tnum_interp, "depth": (n_sensors,), "temp": (n_sensors, n_time)}
        with depth and temp sorted by increasing depth.
    """
    data_grid = dict()
    data_grid["time"] = tnum_interp
    data_grid["depth"] = np.full(len(files), np.nan)
    data_grid["temp"] = np.full((len(files), len(tnum_interp)), np.nan)
 
    for k, file in enumerate(files):
        L2_data = netCDF4.Dataset(os.path.join(path, file), mode='r', format='NETCDF4_CLASSIC')
        data_grid["depth"][k] = float(getattr(L2_data, "Depth (m)"))
        data_grid["temp"][k, :] = np.interp(
            tnum_interp,
            L2_data.variables["time"][:].data,
            L2_data.variables["Temp"][:].data,
            left=np.nan, right=np.nan,
        )
 
    indsort = np.argsort(data_grid["depth"])
    data_grid["temp"] = data_grid["temp"][indsort, :]
    data_grid["depth"] = data_grid["depth"][indsort]
    return data_grid
 
 
def create_folder(input_folder, output_folder):
    """(Re)create ``input_folder/output_folder`` empty, deleting it if present."""
    if os.path.exists(os.path.join(input_folder, output_folder)):
        print("Folder {} already exists: delete it".format(output_folder))
        shutil.rmtree(os.path.join(input_folder, output_folder))
    os.makedirs(os.path.join(input_folder, output_folder))
# This is the block to append to functions_mooring.py
# Content starts after the existing create_folder function
 
def read_pressure_hobo(file_path):
    """Read a HOBO pressure logger CSV file (option3 pressure sensors).
 
    File layout (columns):
        # | Date Heure, GMT+00:00 | Pres. abs., Pa | Temp., °C | Coupleur détaché | Fin de fichier
 
    Key facts:
      * Timestamp format is MM.DD.YY HH:MM:SS AM/PM (month-first, two-digit year).
        Verified empirically: '06.13.25' exists (June 13) while '13.06.25' does not.
      * Timestamps are GMT+00:00 as stated in the column header.
      * The pipeline converts GMT to local naive wall-clock in
        ``gmt_to_local_wallclock`` before merging with thermistor data.
      * Logger-event rows (Coupleur détaché / Fin de fichier markers) have
        non-numeric pressure; they are dropped here.
 
    Parameters
    ----------
    file_path : str
        Path to the HOBO pressure .csv file.
 
    Returns
    -------
    dict
        {"folder": str, "file": str, "data": pandas.DataFrame} with DataFrame
        columns:
          - "time"  : datetime64[ns], naive GMT wall-clock (NOT yet converted
                      to local time — caller must apply ``gmt_to_local_wallclock``).
          - "Pres"  : float, absolute pressure in Pa.
          - "Temp"  : float, temperature in °C (co-located with pressure sensor).
    """
    filepath, filename = _split_path(file_path)
 
    # utf-8-sig strips the BOM present at the start of the header line.
    df_raw = pd.read_csv(file_path, encoding="utf-8-sig", skiprows=1)
 
    # Identify columns by keyword to handle variable serial-number suffixes.
    time_col = next(c for c in df_raw.columns if "Date" in c or "Heure" in c)
    pres_col = next(c for c in df_raw.columns if "Pres" in c)
    temp_col = next(c for c in df_raw.columns if "Temp" in c)
 
    time_parsed = pd.to_datetime(df_raw[time_col], format="%m.%d.%y %I:%M:%S %p", errors="coerce")
    pres = pd.to_numeric(df_raw[pres_col], errors="coerce")
    temp = pd.to_numeric(df_raw[temp_col], errors="coerce")
 
    df = pd.DataFrame({"time": time_parsed, "Pres": pres, "Temp": temp})
    df = df[df["time"].notna() & df["Pres"].notna()].reset_index(drop=True)
    df["time"] = df["time"].astype("datetime64[ns]")
 
    return {"folder": filepath, "file": filename, "data": df}
 
 
def read_water_level(file_path):
    """Read the water-level-variation CSV file.
 
    File layout (columns): date | time | delta_wl
 
    Key facts:
      * Timestamps are GMT+00:00 (aligned with pressure sensor files).
      * delta_wl is in **centimetres**, zero-mean variation around a fixed
        reference level over the pressure-sensor window (June 2025 – Jan 2026).
      * Positive delta_wl -> lake surface is above the reference -> sensors are
        slightly deeper than nominal. Negative -> sensors slightly shallower.
 
    Parameters
    ----------
    file_path : str
        Path to water_level_variation.csv.
 
    Returns
    -------
    pandas.DataFrame
        Columns:
          - "time"     : datetime64[ns], naive GMT wall-clock.
          - "delta_wl" : float, water-level variation in cm.
    """
    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["date"] + " " + df["time"]).astype("datetime64[ns]")
    df["delta_wl"] = pd.to_numeric(df["delta_wl"], errors="coerce")
    df = df[["time", "delta_wl"]].dropna().reset_index(drop=True)
    return df
 
 
def gmt_to_local_wallclock(dt_series):
    """Convert a naive GMT datetime64 series to local naive wall-clock.
 
    The pipeline convention stores all timestamps as naive local civil time
    (Europe/Zurich: UTC+2 CEST in summer, UTC+1 CET in winter). Pressure and
    water-level files are GMT; this function applies the correct offset so they
    can be merged with thermistor data on the same naive time axis.
 
    DST transition (Europe/Zurich 2025-2026):
      - Before 2025-10-26 01:00 UTC : CEST, offset = +2 h
      - From   2025-10-26 01:00 UTC : CET,  offset = +1 h
      - Spring-forward 2026-03-29 01:00 UTC is beyond the pressure-data window.
 
    Parameters
    ----------
    dt_series : pandas.Series of datetime64[ns]
        Naive GMT timestamps.
 
    Returns
    -------
    pandas.Series of datetime64[ns]
        Naive local civil-time timestamps.
    """
    dst_end_utc = pd.Timestamp("2025-10-26 01:00:00")  # clock falls back at this UTC moment
    offset_summer = pd.Timedelta("2h")   # CEST = UTC+2
    offset_winter = pd.Timedelta("1h")   # CET  = UTC+1
 
    offset = pd.Series(
        np.where(dt_series < dst_end_utc, offset_summer, offset_winter),
        index=dt_series.index,
    )
    local = dt_series + offset
    return local.astype("datetime64[ns]")
 
 
def compute_dynamic_depths(nominal_depths_m, delta_wl_cm, chain="B"):
    """Compute time-varying sensor depths from nominal depths and water-level variation.
 
    The correction depends on which buoy chain the sensors belong to:
 
    - Chain B (anchored at the bottom): when the lake level rises by delta_wl,
      the sensors move DOWN relative to the surface by the same amount.
      Correction: depth_actual(t) = depth_nominal + delta_wl(t) / 100
 
    - Chain A (anchored at the surface): the rope hangs from a surface buoy.
      When the lake level rises, the anchor rises with it and the sensors stay
      at the same depth-from-surface. No correction is applied.
      Correction: depth_actual(t) = depth_nominal  (constant)
 
    Parameters
    ----------
    nominal_depths_m : array-like of float, shape (n_sensors,)
        Nominal depth-from-surface in metres for each sensor.
    delta_wl_cm : array-like of float, shape (n_time,)
        Water-level variation in centimetres at each time step (positive = lake
        level above reference).
    chain : str, optional
        "B" (default) applies the water-level correction.
        "A" returns a constant depth matrix (no correction).
 
    Returns
    -------
    numpy.ndarray, shape (n_sensors, n_time)
        Dynamic depth-from-surface (m) for each sensor at each time step.
    """
    nd     = np.asarray(nominal_depths_m, dtype=float)        # (n_sensors,)
    n_time = len(delta_wl_cm)
 
    if chain == "B":
        dw = np.asarray(delta_wl_cm, dtype=float) / 100.0     # (n_time,) in metres
        return nd[:, np.newaxis] + dw[np.newaxis, :]           # (n_sensors, n_time)
    elif chain == "A":
        # Surface-anchored: depth does not change with lake level.
        return np.tile(nd[:, np.newaxis], (1, n_time))         # (n_sensors, n_time)
    else:
        raise ValueError(
            "compute_dynamic_depths: chain must be 'A' or 'B', got '{}'".format(chain)
        )
 
 
def compute_nominal_depths_option3(meta, delta_wl_cm_series):
    """Return the fixed depth axis for option3B.
 
    The fixed axis is the set of per-sensor nominal depths (depth_from_surface),
    corrected by the **mean** water-level variation over the analysis window.
    Because delta_wl is zero-mean by construction, the mean correction is
    negligible (< 1 mm), but it is applied explicitly for physical correctness.
 
    Parameters
    ----------
    meta : dict
        Parsed option3 .meta content. Must contain "Depth (m)" already resolved
        by ``compute_depths_from_meta``.
    delta_wl_cm_series : array-like of float
        Full delta_wl time series in cm over the analysis window.
 
    Returns
    -------
    numpy.ndarray of float
        Fixed depth-from-surface axis (m), sorted ascending, one value per
        thermistor sensor listed in meta["filenames"].
    """
    nominal = np.array(meta["Depth (m)"], dtype=float)
    mean_wl_m = float(np.nanmean(delta_wl_cm_series)) / 100.0
    fixed_depths = np.sort(nominal + mean_wl_m)
    return fixed_depths
 
 
def create_temp_grid_option3A(path, files_L2, tnum_interp, delta_wl_cm_interp, chain="B"):
    """Build the option3A grid: temperature with a time-varying depth axis.
 
    The depth correction depends on the buoy chain:
    - Chain B (bottom-anchored): depth_actual(t) = depth_nominal + delta_wl(t)/100
    - Chain A (surface-anchored): depth is constant; delta_wl has no effect on
      depth-from-surface because the anchor rises with the lake level.
 
    Parameters
    ----------
    path : str
        Directory containing the L2 .nc files.
    files_L2 : list of str
        L2 filenames for the sensors in this chain.
    tnum_interp : numpy.ndarray, shape (n_time,)
        Target time axis (unix seconds, regularly spaced, already truncated to
        the pressure-sensor window).
    delta_wl_cm_interp : numpy.ndarray, shape (n_time,)
        Water-level variation (cm) interpolated onto ``tnum_interp``.
    chain : str, optional
        "B" (default) applies the water-level correction to sensor depths.
        "A" keeps sensor depths constant (surface-anchored rope).
 
    Returns
    -------
    dict
        {
          "time"  : tnum_interp (n_time,),
          "depth" : (n_sensors, n_time)  depth matrix (constant rows for A,
                                          time-varying rows for B),
          "temp"  : (n_sensors, n_time)  temperature matrix,
        }
        Rows are sorted by the mean depth of each sensor (ascending).
    """
    n_sensors = len(files_L2)
    n_time    = len(tnum_interp)
 
    nominal_depths = np.full(n_sensors, np.nan)
    temp_matrix    = np.full((n_sensors, n_time), np.nan)
 
    for k, fname in enumerate(files_L2):
        ds = netCDF4.Dataset(os.path.join(path, fname), mode="r", format="NETCDF4_CLASSIC")
        nominal_depths[k] = float(getattr(ds, "Depth (m)"))
        temp_matrix[k, :] = np.interp(
            tnum_interp,
            ds.variables["time"][:].data,
            ds.variables["Temp"][:].data,
            left=np.nan, right=np.nan,
        )
        ds.close()
 
    # Dynamic depth matrix: chain A is constant, chain B follows delta_wl
    depth_matrix = compute_dynamic_depths(nominal_depths, delta_wl_cm_interp, chain=chain)
 
    # Sort rows by mean sensor depth (ascending)
    mean_depths = depth_matrix.mean(axis=1)
    sort_idx    = np.argsort(mean_depths)
    return {
        "time":  tnum_interp,
        "depth": depth_matrix[sort_idx, :],
        "temp":  temp_matrix[sort_idx, :],
    }
 
 
def create_temp_grid_option3B(path, files_L2, tnum_interp, fixed_depth_axis):
    """Build the option3B grid: temperature interpolated onto a fixed depth axis.
 
    For each time step, the available (sensor_index, temperature) pairs are
    linearly interpolated onto ``fixed_depth_axis``. This produces a regular
    (depth × time) grid suitable for heatmaps and cross-chain comparison.
 
    Extrapolation is disabled: time steps where the target depth falls outside
    the range of available sensors produce NaN.
 
    Parameters
    ----------
    path : str
        Directory containing the L2 .nc files.
    files_L2 : list of str
        L2 filenames for ALL sensors that contribute to this grid (may span
        both chains for a fused product).
    tnum_interp : numpy.ndarray, shape (n_time,)
        Target time axis (unix seconds).
    fixed_depth_axis : numpy.ndarray, shape (n_nodes,)
        Fixed depth-from-surface nodes (m), sorted ascending.
 
    Returns
    -------
    dict
        {
          "time"  : tnum_interp (n_time,),
          "depth" : fixed_depth_axis (n_nodes,),
          "temp"  : (n_nodes, n_time) temperature interpolated onto fixed axis,
        }
    """
    n_sensors  = len(files_L2)
    n_time     = len(tnum_interp)
    n_nodes    = len(fixed_depth_axis)
 
    sensor_depths = np.full(n_sensors, np.nan)
    sensor_temps  = np.full((n_sensors, n_time), np.nan)
 
    for k, fname in enumerate(files_L2):
        ds = netCDF4.Dataset(os.path.join(path, fname), mode="r", format="NETCDF4_CLASSIC")
        sensor_depths[k] = float(getattr(ds, "Depth (m)"))
        sensor_temps[k, :] = np.interp(
            tnum_interp,
            ds.variables["time"][:].data,
            ds.variables["Temp"][:].data,
            left=np.nan, right=np.nan,
        )
        ds.close()
 
    # Sort sensors by depth
    sort_idx      = np.argsort(sensor_depths)
    sensor_depths = sensor_depths[sort_idx]
    sensor_temps  = sensor_temps[sort_idx, :]
 
    # Interpolate each time step onto fixed_depth_axis
    temp_grid = np.full((n_nodes, n_time), np.nan)
    for t in range(n_time):
        col = sensor_temps[:, t]
        valid = np.isfinite(col)
        if valid.sum() < 2:
            continue
        temp_grid[:, t] = np.interp(
            fixed_depth_axis,
            sensor_depths[valid],
            col[valid],
            left=np.nan, right=np.nan,
        )
 
    return {
        "time":  tnum_interp,
        "depth": fixed_depth_axis,
        "temp":  temp_grid,
    }
 
 
def export_option3A(data_grid, meta, folder, title, overwrite=True):
    """Write an option3A grid (dynamic depth) to a single netCDF file.
 
    Unlike the standard ``export`` function which expects a scalar "Depth (m)"
    attribute per sensor, option3A stores a 2-D depth matrix (n_sensors x n_time).
    This requires a dedicated export that mirrors the standard L3 layout but
    writes depth as a full (depth_index, time) variable.
 
    Dimensions:  depth_index (sensor count), time
    Variables:
      - time        : (time,)             unix seconds
      - depth       : (depth_index, time) dynamic depth in m
      - temp        : (depth_index, time) temperature in °C
 
    Parameters
    ----------
    data_grid : dict
        Output of ``create_temp_grid_option3A``:
        {"time": (n_time,), "depth": (n_sensors, n_time), "temp": (n_sensors, n_time)}.
    meta : dict
        Campaign metadata (already filtered to one chain if per-chain export).
    folder : str
        Output directory (created if missing).
    title : str
        Filename prefix; the time stamp of the first record is appended.
    overwrite : bool
        If True and file exists, delete and recreate. Default True.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
 
    time_arr  = data_grid["time"]
    depth_mat = data_grid["depth"]   # (n_sensors, n_time)
    temp_mat  = data_grid["temp"]    # (n_sensors, n_time)
    n_sensors, n_time = depth_mat.shape
 
    from datetime import datetime, timezone as _tz
    t0 = datetime.fromtimestamp(float(np.nanmin(time_arr)), _tz.utc)
    stamp = t0.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(folder, "{}_{}.nc".format(title, stamp))
 
    if os.path.exists(out_path):
        if overwrite:
            os.remove(out_path)
        else:
            print("File {} already exists and overwrite=False, skipping.".format(out_path))
            return out_path
 
    with netCDF4.Dataset(out_path, mode="w", format="NETCDF4") as nc:
        # Global attributes from campaign metadata
        for key, val in meta["campaign"].items():
            if isinstance(val, dict):
                setattr(nc, key, json.dumps(val, ensure_ascii=False))
            elif isinstance(val, bool):
                setattr(nc, key, str(val))
            else:
                setattr(nc, key, val)
        setattr(nc, "mooring_config", "option3A")
        setattr(nc, "depth_note",
                "depth is a 2-D variable (depth_index x time): each sensor has a "
                "time-varying depth corrected by water-level variation (delta_wl/100 m). "
                "Rows sorted by mean sensor depth ascending.")
        if "X Coordinate (CH1903)" in meta["campaign"] and meta["campaign"]["X Coordinate (CH1903)"] != "":
            lat, lon = ch1903_to_latlng(
                int(meta["campaign"]["X Coordinate (CH1903)"]),
                int(meta["campaign"]["Y Coordinate (CH1903)"]),
            )
            setattr(nc, "latitude",  lat)
            setattr(nc, "longitude", lon)
 
        # Dimensions
        nc.createDimension("depth_index", n_sensors)
        nc.createDimension("time",        n_time)
 
        # Variables
        v_time = nc.createVariable("time", np.float64, ("time",), fill_value=np.nan)
        v_time.units     = "seconds since 1970-01-01 00:00:00"
        v_time.long_name = "time (local naive wall-clock treated as UTC for storage)"
        v_time[:]        = time_arr
 
        v_depth = nc.createVariable("depth", np.float64, ("depth_index", "time"), fill_value=np.nan)
        v_depth.units     = "m"
        v_depth.long_name = "depth from surface (dynamic, corrected by water-level variation)"
        v_depth[:]        = depth_mat
 
        v_temp = nc.createVariable("temp", np.float64, ("depth_index", "time"), fill_value=np.nan)
        v_temp.units     = "degC"
        v_temp.long_name = "water temperature"
        v_temp[:]        = temp_mat
 
    print("option3A written: {}".format(out_path))
    return out_path
 
 
def export_option3B(data_grid, meta, folder, title, overwrite=True):
    """Write an option3B grid (fixed depth axis) to a single netCDF file.
 
    Layout is identical to the standard L3 grid (option1/option2) so the same
    visualisation code can be used for cross-option comparison.
 
    Dimensions:  depth (n_nodes), time
    Variables:
      - time  : (time,)         unix seconds
      - depth : (depth,)        fixed depth axis in m
      - temp  : (depth, time)   temperature in °C
 
    Parameters
    ----------
    data_grid : dict
        Output of ``create_temp_grid_option3B``:
        {"time": (n_time,), "depth": (n_nodes,), "temp": (n_nodes, n_time)}.
    meta : dict
        Campaign metadata.
    folder : str
        Output directory (created if missing).
    title : str
        Filename prefix.
    overwrite : bool
        Default True.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
 
    time_arr  = data_grid["time"]
    depth_vec = data_grid["depth"]
    temp_mat  = data_grid["temp"]
 
    from datetime import datetime, timezone as _tz
    t0 = datetime.fromtimestamp(float(np.nanmin(time_arr)), _tz.utc)
    stamp = t0.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(folder, "{}_{}.nc".format(title, stamp))
 
    if os.path.exists(out_path):
        if overwrite:
            os.remove(out_path)
        else:
            print("File {} already exists and overwrite=False, skipping.".format(out_path))
            return out_path
 
    with netCDF4.Dataset(out_path, mode="w", format="NETCDF4") as nc:
        for key, val in meta["campaign"].items():
            if isinstance(val, dict):
                setattr(nc, key, json.dumps(val, ensure_ascii=False))
            elif isinstance(val, bool):
                setattr(nc, key, str(val))
            else:
                setattr(nc, key, val)
        setattr(nc, "mooring_config", "option3B")
        setattr(nc, "depth_note",
                "Fixed depth axis: nominal sensor depths corrected by mean water-level "
                "variation. All sensors (both chains) interpolated onto this common axis.")
        if "X Coordinate (CH1903)" in meta["campaign"] and meta["campaign"]["X Coordinate (CH1903)"] != "":
            lat, lon = ch1903_to_latlng(
                int(meta["campaign"]["X Coordinate (CH1903)"]),
                int(meta["campaign"]["Y Coordinate (CH1903)"]),
            )
            setattr(nc, "latitude",  lat)
            setattr(nc, "longitude", lon)
 
        nc.createDimension("depth", len(depth_vec))
        nc.createDimension("time",  len(time_arr))
 
        v_time = nc.createVariable("time", np.float64, ("time",), fill_value=np.nan)
        v_time.units     = "seconds since 1970-01-01 00:00:00"
        v_time.long_name = "time (local naive wall-clock treated as UTC for storage)"
        v_time[:]        = time_arr
 
        v_depth = nc.createVariable("depth", np.float64, ("depth",), fill_value=np.nan)
        v_depth.units     = "m"
        v_depth.long_name = "fixed depth from surface (nominal sensor positions)"
        v_depth[:]        = depth_vec
 
        v_temp = nc.createVariable("temp", np.float64, ("depth", "time"), fill_value=np.nan)
        v_temp.units     = "degC"
        v_temp.long_name = "water temperature interpolated onto fixed depth axis"
        v_temp[:]        = temp_mat
 
    print("option3B written: {}".format(out_path))
    return out_path