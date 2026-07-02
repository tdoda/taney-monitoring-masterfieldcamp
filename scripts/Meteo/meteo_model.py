# imports
import json
import os
import numpy as np
import pandas as pd
import xarray as xr
import warnings


class meteo_model:
    GENERAL_ATTRS = {
        "institution": "Unil",
        "source": "ICON weather model",
        "references": "Aquatic Science Master field camp",
        "history": "See history on Renku",
        "conventions": "CF 1.7",
        "comment": "Meteorological model data for Lake Taney",
        "title": "Meteo model Lake Taney"
    }

    BASE_PATH = "data/Meteo"
    MODEL_PATH = os.path.join(BASE_PATH, "Model", "{date}")

    MD_PATH = os.path.join(MODEL_PATH, "Level0", "meteo_model_{date}.meta")
    DPATH = MODEL_PATH

    VARS_MAP_ICON = {
        "T_2M": "Temp_K",
        "RELHUM_2M": "RH",
        "PMSL": "pres_Pa",
        "GLOB": "global_radiation",
        "CLCT": "cloud_cover"
    }

    QA_VARS = [
        "Temp_K",
        "Temp",
        "RH",
        "pres_Pa",
        "pres",
        "U",
        "V",
        "wind_speed",
        "wind_dir",
        "global_radiation",
        "cloud_cover"
    ]

    VAR_ATTRS = {
        "time": {"long_name": "Coordinated Universal Time (UTC)"},
        "Temp_K": {"units": "K", "long_name": "Air temperature at 2 m"},
        "Temp": {"units": "degC", "long_name": "Air temperature at 2 m"},
        "RH": {"units": "%", "long_name": "Relative humidity at 2 m"},
        "pres_Pa": {"units": "Pa", "long_name": "Pressure reduced to mean sea level"},
        "pres": {"units": "hPa", "long_name": "Pressure reduced to mean sea level"},
        "U": {"units": "m s-1", "long_name": "Eastward wind component"},
        "V": {"units": "m s-1", "long_name": "Northward wind component"},
        "wind_speed": {"units": "m s-1", "long_name": "Wind speed"},
        "wind_dir": {"units": "degree", "long_name": "Wind direction from north"},
        "global_radiation": {"units": "W m-2", "long_name": "Global radiation"},
        "cloud_cover": {"units": "%", "long_name": "Total cloud cover"}
    }

    def __init__(self, date, idx=0, model_id="icon_taney", t_offset=None):
        self.date = date
        self.idx = idx
        self.model_id = model_id
        self.t_offset = t_offset

        self.md_file = self.locate_md_file()
        self.sensor = self.get_sensor_type()
        self.dpath_L0, self.dpath_L1, self.dpath_L2, self.dpath_L3 = self.locate_data_dirs()

    def locate_md_file(self):
        return self.MD_PATH.format(date=self.date)

    def open_md_file(self):
        with open(self.md_file, "r") as f:
            md = json.load(f)
        return md

    def get_sensor_type(self):
        md = self.open_md_file()
        return md["filetypes"][self.idx]

    def locate_data_dirs(self):
        dpath = self.DPATH.format(date=self.date)

        dpath_L0 = os.path.join(dpath, "Level0")
        dpath_L1 = os.path.join(dpath, "Level1")
        dpath_L2 = os.path.join(dpath, "Level2")
        dpath_L3 = os.path.join(dpath, "Level3")

        for path in [dpath_L0, dpath_L1, dpath_L2, dpath_L3]:
            os.makedirs(path, exist_ok=True)

        return dpath_L0, dpath_L1, dpath_L2, dpath_L3

    def locate_file_L0(self):
        md = self.open_md_file()
        filename = md.get("filename", f"icon_{self.date}.csv")
        return os.path.join(self.dpath_L0, filename)

    def parse_icon_L0(self, fpath_L0):
        df = pd.read_csv(fpath_L0)

        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["time"] = df["time"].dt.tz_localize(None)

        numeric_cols = ["T_2M", "U", "V", "GLOB", "RELHUM_2M", "PMSL", "CLCT"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        df = df.dropna(subset=["time"])
        df = df.sort_values("time").drop_duplicates("time")

        return df

    def parse_L0(self):
        fpath_L0 = self.locate_file_L0()

        if self.sensor == "icon":
            data = self.parse_icon_L0(fpath_L0)
        else:
            raise NotImplementedError("Only ICON model data are handled.")

        data = data.set_index("time")
        ds = xr.Dataset.from_dataframe(data)

        return ds

    def organize_data_vars(self, ds):
        if self.sensor == "icon":
            vars_map = {k: v for k, v in self.VARS_MAP_ICON.items() if k in ds.data_vars}
            ds = ds.rename_vars(vars_map)
        else:
            raise NotImplementedError("Only ICON model data are handled.")

        return ds

    def derive_model_vars(self, ds):
        if "Temp_K" in ds:
            ds["Temp"] = ds["Temp_K"] - 273.15

        if "pres_Pa" in ds:
            ds["pres"] = ds["pres_Pa"] / 100.0

        if "U" in ds and "V" in ds:
            ds["wind_speed"] = np.sqrt(ds["U"] ** 2 + ds["V"] ** 2)

            # Meteorological wind direction: direction from which the wind comes
            wind_dir = (270 - np.degrees(np.arctan2(ds["V"], ds["U"]))) % 360
            ds["wind_dir"] = wind_dir

        return ds

    def get_campaign_time(self, md, kind):
        campaign = md["campaign"]

        if kind == "start":
            value = campaign.get("Time of deployment") or campaign.get("Starting time (UTC)")
        elif kind == "end":
            value = campaign.get("Time of retrieval") or campaign.get("End time (UTC)")
        else:
            raise ValueError("kind must be 'start' or 'end'")

        if value is None:
            raise KeyError(
                f"Could not find {kind} time in metadata. "
                "Expected either 'Time of deployment'/'Time of retrieval' "
                "or 'Starting time (UTC)'/'End time (UTC)'."
            )

        return value

    def assign_attributes(self, ds):
        for var, attrs in self.VAR_ATTRS.items():
            if var in ds:
                ds[var].attrs.update(attrs)

        md = self.open_md_file()

        md_xr = {
            "lon": md["campaign"].get("Longitude", ""),
            "lat": md["campaign"].get("Latitude", ""),
            "altitude": md["campaign"].get("Altitude (m)", ""),
            "deployment": self.get_campaign_time(md, "start"),
            "retrieval": self.get_campaign_time(md, "end"),
            "sensor": self.sensor,
            "model_id": str(self.model_id),
            "t_offset": str(self.t_offset),
            "filename": md.get("filename", ""),
            "dataset_name": md["campaign"].get("Dataset Name", ""),
            "source_dataset": md["campaign"].get("Source of dataset", "")
        }

        md_xr.update(self.GENERAL_ATTRS)
        ds = ds.assign_attrs(md_xr)

        return ds

    def derive_vars(self, ds):
        ds = self.organize_data_vars(ds)
        ds = self.derive_model_vars(ds)
        ds = self.assign_attributes(ds)
        return ds

    def correct_clock_offset(self, ds):
        ds["time"] = ds["time"] + pd.to_timedelta(self.t_offset)
        return ds

    def quality_assurance(self, ds):
        if self.t_offset:
            ds = self.correct_clock_offset(ds)

        md = self.open_md_file()

        deploy = pd.to_datetime(self.get_campaign_time(md, "start"))
        retrieve = pd.to_datetime(self.get_campaign_time(md, "end"))

        flag = (ds["time"] < deploy) | (ds["time"] > retrieve)

        for var in self.QA_VARS:
            if var in ds:
                ds[f"{var}_qual"] = flag.astype(int)

        return ds

    def mask_data(self, ds):
        for var in self.QA_VARS:
            if var in ds and f"{var}_qual" in ds:
                ds[var] = ds[var].where(ds[f"{var}_qual"] == 0)

        return ds

    def write_to_nc(self, ds, level, overwrite=True):
        if level == "L1":
            fpath = os.path.join(self.dpath_L1, f"L1_meteo_{self.sensor}_{self.model_id}.nc")
        elif level == "L2":
            fpath = os.path.join(self.dpath_L2, f"L2_meteo_{self.sensor}_{self.model_id}.nc")
        elif level == "L3":
            fpath = os.path.join(self.dpath_L3, f"L3_meteo_{self.sensor}_{self.model_id}.nc")
        else:
            raise ValueError("Writing level must be L1, L2, or L3.")

        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        if os.path.exists(fpath) and not overwrite:
            warnings.warn(f"{fpath} already exists and overwrite = False.")
        else:
            ds.to_netcdf(fpath)

        return fpath

    def process(self):
        ds = self.parse_L0()
        ds = self.derive_vars(ds)
        ds = self.quality_assurance(ds)
        fpath_L1 = self.write_to_nc(ds, "L1")

        ds = self.mask_data(ds)
        fpath_L2 = self.write_to_nc(ds, "L2")

        return fpath_L1, fpath_L2


if __name__ == "__main__":
    mm = meteo_model(date="20260604", idx=0, model_id="icon_taney", t_offset=None)

    fpath_L1, fpath_L2 = mm.process()

    print("L1 written to:", fpath_L1)
    print("L2 written to:", fpath_L2)
    print("Level0:", mm.dpath_L0)
    print("Level1:", mm.dpath_L1)
    print("Level2:", mm.dpath_L2)
    print("Level3:", mm.dpath_L3)