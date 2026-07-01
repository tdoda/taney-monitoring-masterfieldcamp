### Class for processing weather station data

# imports
import json
import os
import pandas as pd
import xarray as xr
import warnings


class weather_station:
    GENERAL_ATTRS = {
        "institution": "Unil",
        "source": "",
        "references": "Aquatic Science Master field camp",
        "history": "See history on Renku",
        "conventions": "CF 1.7",
        "comment": "Monitoring data in Lake Taney performed by Aquatic Science Master students",
        "title": "Weather station Lake Taney"
    }

    BASE_PATH = 'data/Meteo'
    STATION_PATH = os.path.join(BASE_PATH, 'Station', '{date}')

    MD_PATH = os.path.join(STATION_PATH, 'Level0', 'meteo_station_{date}.meta')
    DPATH = STATION_PATH

    # Netatmo CSV is ";"-separated, has a BOM, and the real header is row index 3.
    # The meaningful columns are taken by POSITION (names repeat / are localized):
    #   0 date | 1 RH | 2 CO2 | 3 Pressure(hPa) | 4 wind dir | 5 wind speed |
    #   6 gust speed | 7 gust dir | 8 Temperature | 9 Rain   (rest = Min/Max/Mean block)
    COLS_NETATMO = {
        0: 'time', 1: 'RH', 2: 'CO2', 3: 'pres', 4: 'wind_dir',
        5: 'wind_speed', 6: 'gust_speed', 7: 'gust_dir', 8: 'Temp', 9: 'rain',
    }
    DT_FMT_NETATMO = '%d/%m/%Y %H:%M'   # local time in the export
    TZ_LOCAL = 'Europe/Zurich'          # timestamps are Europe/Zurich -> converted to UTC
    COLS_FLOAT_NETATMO = ['RH', 'CO2', 'pres', 'wind_dir', 'wind_speed',
                          'gust_speed', 'gust_dir', 'Temp', 'rain']
    VARS_DROP_NETATMO = []              # nothing to drop: we already selected the columns
    VARS_MAP_NETATMO = {}               # already named in COLS_NETATMO
    QA_VARS = ['Temp', 'RH', 'pres', 'CO2', 'wind_speed', 'wind_dir',
               'gust_speed', 'gust_dir', 'rain']
    VAR_ATTRS = {
        'time': {'long_name': 'Coordinated Universal Time (UTC)'},
        'Temp': {'units': 'degC', 'long_name': 'Air temperature'},
        'RH': {'units': '%', 'long_name': 'Relative humidity'},
        'pres': {'units': 'hPa', 'long_name': 'Atmospheric pressure'},
        'CO2': {'units': 'ppm', 'long_name': 'CO2 concentration'},
        'wind_speed': {'units': 'km/h', 'long_name': 'Wind speed'},
        'wind_dir': {'units': 'degree', 'long_name': 'Wind direction'},
        'gust_speed': {'units': 'km/h', 'long_name': 'Gust speed'},
        'gust_dir': {'units': 'degree', 'long_name': 'Gust direction'},
        'rain': {'units': 'mm', 'long_name': 'Precipitation'},
        'serial_id': {'long_name': 'Serial ID'}
    }

    def __init__(self, date, idx=0, serial_id=None, t_offset=None):
        """
        Initialize weather_station object.

        Parameters
        ----------
        date : str
            Date (YYYYMMDD) of weather data retrieval/export.
        idx : int
            Index of the station in the metadata file.
        serial_id : str
            Identifier of the weather station.
        t_offset : str
            Time offset from station clock to correct time (e.g., +/-HH:MM:SS).
        """
        self.date = date
        self.idx = idx
        self.serial_id = serial_id
        self.t_offset = t_offset

        self.md_file = self.locate_md_file()
        self.sensor = self.get_sensor_type()
        self.dpath_L0, self.dpath_L1, self.dpath_L2, self.dpath_L3 = self.locate_data_dirs()

    # ---------- Metadata ----------

    def locate_md_file(self):
        """Locate metadata file."""
        return self.MD_PATH.format(date=self.date)

    def open_md_file(self):
        """Open metadata file."""
        with open(self.md_file, 'r') as f:
            md = json.load(f)
        return md

    def get_sensor_type(self):
        """Parse metadata file for sensor type."""
        md = self.open_md_file()
        return md['filetypes'][self.idx]

    # ---------- Navigation ----------

    def locate_data_dirs(self):
        """Locate and create data directories for L0, L1, L2, and L3 data."""
        dpath = self.DPATH.format(date=self.date)

        dpath_L0 = os.path.join(dpath, 'Level0')
        dpath_L1 = os.path.join(dpath, 'Level1')
        dpath_L2 = os.path.join(dpath, 'Level2')
        dpath_L3 = os.path.join(dpath, 'Level3')

        for path in [dpath_L0, dpath_L1, dpath_L2, dpath_L3]:
            os.makedirs(path, exist_ok=True)

        return dpath_L0, dpath_L1, dpath_L2, dpath_L3

    def locate_file_L0(self):
        """Locate file with raw (L0) weather station data."""
        if self.sensor == 'netatmo':
            fpath_L0 = f'{self.dpath_L0}/Weather_data_{self.date}.csv'
        else:
            raise NotImplementedError("Only netatmo stations are handled.")
        return fpath_L0

    # ---------- L0 to L1 ----------

    def parse_netatmo_L0(self, fpath_L0):
        """Parse raw (L0) data from a Netatmo weather station export."""
        # ";"-separated, BOM, header on row index 3
        df = pd.read_csv(fpath_L0, sep=';', header=3, encoding='utf-8-sig')

        # keep the meaningful columns by position, then name them
        df = df.iloc[:, list(self.COLS_NETATMO.keys())]
        df.columns = list(self.COLS_NETATMO.values())

        # drop the Min/Max/Mean summary rows (no date)
        df = df[df['time'].notna()].copy()

        # local Europe/Zurich time -> UTC -> naive-UTC (for NetCDF)
        local = pd.to_datetime(df['time'], format=self.DT_FMT_NETATMO)
        df['time'] = (local
                      .dt.tz_localize(self.TZ_LOCAL, nonexistent='shift_forward', ambiguous='NaT')
                      .dt.tz_convert('UTC')
                      .dt.tz_localize(None))

        # cast to proper datatypes
        df[self.COLS_FLOAT_NETATMO] = df[self.COLS_FLOAT_NETATMO].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=['time']).sort_values('time').drop_duplicates('time')

        return df

    def parse_L0(self):
        """Load raw (L0) weather station data into xarray Dataset."""
        fpath_L0 = self.locate_file_L0()

        if self.sensor == 'netatmo':
            data = self.parse_netatmo_L0(fpath_L0)
        else:
            raise NotImplementedError("Only netatmo stations are handled.")

        data = data.set_index('time')
        ds = xr.Dataset.from_dataframe(data)

        return ds

    def organize_data_vars(self, ds):
        """Drop and rename data variables."""
        if self.sensor == 'netatmo':
            ds = ds.drop_vars(self.VARS_DROP_NETATMO, errors='ignore')
            vars_map = {k: v for k, v in self.VARS_MAP_NETATMO.items() if k in ds.data_vars}
        else:
            raise NotImplementedError('Only netatmo stations are handled.')

        return ds.rename_vars(vars_map)

    def get_campaign_time(self, md, kind):
        """
        Get start/end time from metadata.
        Accepts both old and new metadata key names.
        """
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
        """Assign attributes to data variables and dataset; add serial id coordinate."""
        ds = ds.assign_coords(serial_id=self.serial_id)

        for var, attrs in self.VAR_ATTRS.items():
            if var in ds:
                ds[var].attrs.update(attrs)

        md = self.open_md_file()

        md_xr = {
            'xsc': md['campaign'].get('X Coordinate (CH1903)', ''),
            'ysc': md['campaign'].get('Y Coordinate (CH1903)', ''),
            'altitude': md['campaign'].get('Altitude (m)', ''),
            'deployment': self.get_campaign_time(md, 'start'),
            'retrieval': self.get_campaign_time(md, 'end'),
            'sensor': self.sensor,
            'serial_id': str(self.serial_id),
            't_offset': str(self.t_offset),
            'filename': md.get('filename', ''),
            'dataset_name': md['campaign'].get('Dataset Name', ''),
            'source_dataset': md['campaign'].get('Source of dataset', '')
        }

        md_xr.update(self.GENERAL_ATTRS)
        ds = ds.assign_attrs(md_xr)

        return ds

    def derive_vars(self, ds):
        """Organize variables and assign attributes."""
        ds = self.organize_data_vars(ds)
        ds = self.assign_attributes(ds)
        return ds

    def correct_clock_offset(self, ds):
        """Apply correction to station clock offset."""
        ds['time'] = ds['time'] + pd.to_timedelta(self.t_offset)
        return ds

    def quality_assurance(self, ds):
        """Run quality assurance: clock offset + flag outside deployment window."""
        if self.t_offset:
            ds = self.correct_clock_offset(ds)

        md = self.open_md_file()

        deploy = pd.to_datetime(self.get_campaign_time(md, 'start'))
        retrieve = pd.to_datetime(self.get_campaign_time(md, 'end'))

        flag = (ds['time'] < deploy) | (ds['time'] > retrieve)

        for var in self.QA_VARS:
            if var in ds:
                ds[f'{var}_qual'] = flag.astype(int)

        return ds

    def mask_data(self, ds):
        """Apply QA flags to mask data."""
        for var in self.QA_VARS:
            if var in ds and f'{var}_qual' in ds:
                ds[var] = ds[var].where(ds[f'{var}_qual'] == 0)

        return ds

    def write_to_nc(self, ds, level, overwrite=True):
        """Write xarray Dataset to .nc file."""
        if level == 'L1':
            fpath = os.path.join(self.dpath_L1, f'L1_meteo_{self.sensor}_{self.serial_id}.nc')
        elif level == 'L2':
            fpath = os.path.join(self.dpath_L2, f'L2_meteo_{self.sensor}_{self.serial_id}.nc')
        elif level == 'L3':
            fpath = os.path.join(self.dpath_L3, f'L3_meteo_{self.sensor}_{self.serial_id}.nc')
        else:
            raise ValueError('Writing level must be L1, L2, or L3.')

        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        if os.path.exists(fpath) and not overwrite:
            warnings.warn(f'{fpath} already exists and overwrite = False.')
        else:
            ds.to_netcdf(fpath)

        return fpath

    def process(self):
        """
        Process raw L0 weather station data.
        Create L1 and L2 NetCDF files.
        """
        ds = self.parse_L0()
        ds = self.derive_vars(ds)
        ds = self.quality_assurance(ds)
        fpath_L1 = self.write_to_nc(ds, 'L1')

        ds = self.mask_data(ds)
        fpath_L2 = self.write_to_nc(ds, 'L2')

        return fpath_L1, fpath_L2
    
ws = weather_station(date='20260521', idx=0, serial_id='netatmo_taney', t_offset=None)

fpath_L1, fpath_L2 = ws.process()

print("L1 written to:", fpath_L1)
print("L2 written to:", fpath_L2)
print(ws.dpath_L0)
print(ws.dpath_L1)
print(ws.dpath_L2)
print(ws.dpath_L3)