# -*- coding: utf-8 -*-
"""
Thermistor data objects for the Lake Taney mooring pipeline.

``thermistor_series`` holds one sensor's time series, runs quality assurance
(deployment-window flagging, physical-range check, DST duplicate flagging and
DST/time-gap detection) and masks flagged values for Level 2.
``thermistor_grid`` assembles the interpolated multi-sensor Level 3 field.

@author: T. Doda (original) — QAQC extended for the 2025-2026 CSV campaign
"""
import os
import json
import sys
import logging
import netCDF4
import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import functions_mooring as func


# Module-level logger. Configured once; messages are emitted as "WARNING: ...".
logger = logging.getLogger("taney.mooring")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


def _fmt_ts(unix_seconds):
    """Format a (naive wall-clock) unix timestamp as 'YYYY-MM-DD HH:MM:SS'."""
    return datetime.fromtimestamp(float(unix_seconds), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


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
        self.logger = logger

    def read_timeseries(self, data_temp, meta_mooring):
        """Load one sensor's data and attach its metadata.

        Reads a per-file ``<name>.meta`` if present in the Level 0 folder,
        otherwise falls back to the campaign-level metadata. Returns False (and
        skips processing) when the sensor is flagged ``valid: false``.

        Parameters
        ----------
        data_temp : dict
            Output of ``functions_mooring.read_data`` ({"folder","file","data"}).
        meta_mooring : dict
            Campaign-level parsed .meta (already depth-resolved by the caller).

        Returns
        -------
        bool
            True if the series was loaded, False if the sensor is invalid.
        """
        self.filename = data_temp["file"]
        file_noext = self.filename.rsplit('.', 1)[0]
        df = data_temp["data"]
        for variable in self.variables:
            if variable in df.columns:
                if variable == "time":
                    self.data[variable] = np.array(df[variable].values).astype("float") / 10 ** 9
                else:
                    self.data[variable] = np.array(df[variable].values).astype("float")
            else:
                self.data[variable] = np.array([np.nan] * len(df))

        meta_path = os.path.join(data_temp["folder"], file_noext + ".meta")

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            print("{} not found, use mooring metadata instead".format(meta_path))
            indfile = np.where(np.array(meta_mooring["filenames"]) == self.filename)[0][0]
            meta = {"valid": np.array(meta_mooring["valid"])[indfile],
                    "Depth (m)": np.array(meta_mooring["Depth (m)"])[indfile],
                    "campaign": meta_mooring["campaign"]}

        if "valid" in meta and not meta["valid"]:
            print("File {} marked invalid, not processing.".format(data_temp["file"]))
            return False
        for key in meta["campaign"]:
            if isinstance(meta["campaign"][key], bool):
                self.general_attributes[key] = str(meta["campaign"][key])
            elif isinstance(meta["campaign"][key], dict):
                # e.g. failed_sensors block: store as a JSON string attribute.
                self.general_attributes[key] = json.dumps(meta["campaign"][key], ensure_ascii=False)
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
            self.start_time = datetime.strptime(self.general_attributes["Time of deployment"], "%Y-%m-%d %H:%M:%S")
        if "Time of retrieval" in self.general_attributes and self.general_attributes["Time of retrieval"] != "":
            self.end_time = datetime.strptime(self.general_attributes["Time of retrieval"], "%Y-%m-%d %H:%M:%S")

        return True

    def quality_assurance(self):
        """Flag suspect data with a quality variable (0 = ok, 1 = investigate).

        Adds a ``<var>_qual`` companion for each non-time variable and applies,
        for the "Temp" variable:
          1. deployment-window flagging (data before deployment / after
             retrieval),
          2. physical-range flagging (outside [0, 30] degC),
          3. DST duplicate flagging (autumn fall-back: keep first occurrence),
          4. DST / time-gap detection (spring forward: logged, never filled).

        Flags are additive: a value can be flagged by several checks. No data
        is removed here; masking to NaN happens in ``mask_data``.
        """
        for key, values in self.variables.copy().items():
            if "_qual" not in key:
                if key != "time":  # quality only on non-temporal variables
                    name = key + "_qual"
                    self.variables[name] = {'var_name': name, 'dim': values["dim"],
                                            'unit': '0 = nothing to report, 1 = more investigation',
                                            'long_name': name}
                    self.data[name] = np.zeros(self.data[key].shape)
                    if self.start_time:
                        self.data[name][self.data["time"] < self.start_time.replace(tzinfo=timezone.utc).timestamp()] = 1
                    if self.end_time:
                        self.data[name][self.data["time"] > self.end_time.replace(tzinfo=timezone.utc).timestamp()] = 1

        # --- Additional checks (new campaign; harmless on the old campaign) ---
        if "Temp" in self.data and "Temp_qual" in self.data:
            self._flag_physical_range(var="Temp", vmin=0.0, vmax=30.0)
            self._flag_dst_duplicates(time_label="time", var="Temp")
            self._detect_time_gaps(time_label="time", expected_dt_sec=600, gap_factor=1.5)

    def _flag_physical_range(self, var="Temp", vmin=0.0, vmax=30.0):
        """Flag finite values of ``var`` outside [vmin, vmax] and log each one."""
        v = self.data[var]
        qual = var + "_qual"
        mask = np.isfinite(v) & ((v < vmin) | (v > vmax))
        for i in np.where(mask)[0]:
            self.data[qual][i] = 1
            self.logger.warning(
                "temperature %.3f at %s outside physical range [%g, %g degC]",
                v[i], _fmt_ts(self.data["time"][i]), vmin, vmax,
            )

    def _flag_dst_duplicates(self, time_label="time", var="Temp"):
        """Flag duplicate timestamps (DST autumn fall-back), keeping the first.

        The first occurrence of each timestamp (lowest original index) keeps
        quality 0; every later occurrence is flagged 1 so it is masked to NaN in
        Level 2. ``np.unique(return_index=True)`` returns the first-occurrence
        indices in the original array order.
        """
        t = self.data[time_label]
        qual = var + "_qual"
        _, first_idx = np.unique(t, return_index=True)
        keep = np.zeros(t.shape, dtype=bool)
        keep[first_idx] = True
        dup_mask = ~keep
        n_dup = int(dup_mask.sum())
        if n_dup:
            for i in np.where(dup_mask)[0]:
                self.data[qual][i] = 1
            ts_examples = sorted({_fmt_ts(t[i]) for i in np.where(dup_mask)[0]})
            self.logger.warning(
                "%d duplicate timestamp(s) detected (DST fall-back), first occurrence kept. "
                "Affected times: %s", n_dup, ", ".join(ts_examples),
            )

    def _detect_time_gaps(self, time_label="time", expected_dt_sec=600, gap_factor=1.5):
        """Log time gaps larger than ``gap_factor`` x ``expected_dt_sec``.

        Detection only — gaps are never filled. The spring-forward DST jump
        (29/03/2026) shows up here as a single ~70 min gap.
        """
        t = np.sort(np.unique(np.asarray(self.data[time_label], dtype=float)))
        if t.size < 2:
            return
        dt = np.diff(t)
        threshold = expected_dt_sec * gap_factor
        for i in np.where(dt > threshold)[0]:
            self.logger.warning(
                "time gap of %.0f min detected at %s (no filling applied; "
                "expected sampling %.0f min). Likely the 29/03 DST spring-forward gap.",
                dt[i] / 60.0, _fmt_ts(t[i]), expected_dt_sec / 60.0,
            )

    def mask_data(self):
        """Replace every flagged value (quality > 0) by NaN for Level 2."""
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
        self.dt_sec = 10 * 60  # [s]
        self.logger = logger

    def add_grid(self, data_grid, meta):
        """Attach an interpolated grid and the campaign metadata as attributes.

        Parameters
        ----------
        data_grid : dict
            Output of ``functions_mooring.create_temp_grid``.
        meta : dict
            Metadata describing the sensors present in this grid. For option2
            this should already be filtered to a single chain.
        """
        for variable in self.variables:
            if variable in data_grid.keys():
                self.data[variable] = data_grid[variable]
            else:
                dim_names = self.variables[variable]["dim"]
                self.data[variable] = np.full(tuple([len(self.data[d]) for d in dim_names]), np.nan)

        if "valid" in meta:
            ind_sensors = np.where(meta["valid"])[0]
        else:
            ind_sensors = np.arange(len(data_grid["depth"]))

        for key in meta["campaign"]:
            if isinstance(meta["campaign"][key], bool):
                self.general_attributes[key] = str(meta["campaign"][key])
            elif isinstance(meta["campaign"][key], dict):
                self.general_attributes[key] = json.dumps(meta["campaign"][key], ensure_ascii=False)
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
