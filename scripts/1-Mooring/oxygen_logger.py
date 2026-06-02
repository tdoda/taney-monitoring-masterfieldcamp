### Class for processing oxygen logger data

# imports
import json
import os
from glob import glob
import pandas as pd
import pyrsktools as rsk
import xarray as xr
import warnings


class oxygen_logger:
    GENERAL_ATTRS = {
        "institution": "Unil",
        "source": "",
        "references": "Aquatic Science Master field camp",
        "history": "See history on Renku",
        "conventions": "CF 1.7",
        "comment": "Monitoring data in Lake Taney performed by Aquatic Science Master students",
        "title": "Mooring Lake Taney"
    }
    MD_PATH = '../../data/Mooring/miniDOT/{date}/Level0/oxygen_loggers_{date}.meta'
    DPATH = '../../data/Mooring/miniDOT/{date}/'
    COLS_DT_MINIDOT = ['UTC_Date_&_Time', 'Coordinated Universal Time']
    COLS_INT_MINIDOT = ['Unix Timestamp']
    COLS_FLOAT_MINIDOT = ['Battery', 'Temperature', 'Dissolved Oxygen', 'Dissolved Oxygen Saturation', 'Q']
    COLS_MAP_MINIDOT = {'UTC_Date_&_Time': 'time'}
    VARS_DROP_MINIDOT = ['Unix Timestamp', 'Coordinated Universal Time', 'Battery', 'Q']
    VARS_MAP_MINIDOT = {
        'Temperature': 'temp', 
        'Dissolved Oxygen': 'do2_conc', 
        'Dissolved Oxygen Saturation': 'do2_sat'
    }
    VAR_ATTRS = {
        'time': {'long_name': 'Coordinated Universal Time (UTC)'},
        'do2_conc': {'units': 'mg/l', 'long_name': 'Dissolved Oxygen Concentration'},
        'do2_sat': {'units': '%', 'long_name': 'Dissolved Oxygen Saturation'},
        'temp': {'units': '°C', 'long_name': 'Temperature'},
        'depth': {'units': 'm', 'long_name': 'Depth'},
        'serial_id': {'long_name': 'Serial ID'}
    }


    def __init__(self, date, idx, serial_id, t_offset=None):
        """
        Initialize O2Processor object.

        Parameters
        ----------
        date : str
            Date (YYYYMMDD) of oxygen logger retrieval.
        idx : int
            Index of oxygen logger in metadata file.
        serial_id : str
            Serial number of oxygen logger.
        t_offset : str
            Time offset from sensor clock to correct time (e.g., +/-HH:MM:SS).
        """
        self.date = date
        self.idx = idx
        self.serial_id = serial_id
        self.t_offset = t_offset

        self.md_file = self.locate_md_file()
        self.sensor = self.get_sensor_type()
        self.depth = self.get_depth()
        self.dpath_L0, self.dpath_L1, self.dpath_L2, self.dpath_L3 = self.locate_data_dirs()    

    
    # ---------- Metadata ----------
    
    def locate_md_file(self):
        """
        Locate metadata file.

        Returns
        -------
        md_path : str
            File path to metadata JSON file.
        """
        return self.MD_PATH.format(date=self.date)
    

    def open_md_file(self):
        """
        Open metadata file.

        Returns
        -------
        md : dict
            Mooring metadata.
        """
        with open(self.md_file, 'r') as f:
            md = json.load(f)

        return md
    
    
    def get_sensor_type(self):
        """
        Parse metadata file for sensor type.

        Returns
        -------
        sensor : str
            Type of sensor.
        """
        md = self.open_md_file()

        return md['filetypes'][self.idx]
    

    def get_depth(self):
        """
        Parse metadata file for instrument depth.

        Returns
        -------
        depth : float
            Depth [m] of sensor.
        """
        md = self.open_md_file()
        
        return md['Depth (m)'][self.idx]
    

    # ---------- Navigation ----------

    def locate_data_dirs(self):
        """
        Locate data directories for L0, L1, L2, and L3 data.

        Returns
        -------
        dpath_L0 : str
            Path to L0 data directory.
        dpath_L1 : str
            Path to L1 data directory.
        dpath_L2 : str
            Path to L2 data directory.
        dpath_L3 : str
            Path to L3 data directory.
        """
        dpath = self.DPATH.format(lake=self.lake, location=self.location, year=self.year, date=self.date)

        return os.path.join(dpath, 'Level0'), os.path.join(dpath, 'Level1'), os.path.join(dpath, 'Level2'), os.path.join(dpath, 'Level3')


    def locate_file_L0(self):
        """
        Locate file with raw (L0) oxygen logger data.

        Returns
        -------
        fpath_L0 : str
            Path to L0 data file.
        """
        if self.sensor == 'minidot':
            fpath_L0 = f'{self.dpath_L0}/7450-{self.serial_id}/Cat.txt'
        else:
            raise NotImplementedError("Only minidot sensors are handled.")
        
        return fpath_L0
    

    # ---------- L0 to L1 ----------
    
    def parse_minidot_L0(self, fpath_L0):
        """
        Parse raw (L0) data from Minidot oxygen logger.

        Parameters
        ----------
        fpath_L0 : str
            File path to raw (L0) Minidot oxygen logger data.

        Returns
        -------
        data : pd.DataFrame
            Data from Minidot oxygen logger.
        """
        with open(fpath_L0, 'r') as f:
            lines = [x[:-1] for x in f if len(x.split(',')) > 1]

        # extract column names
        cols = [x.lstrip(' ') for x in lines[0].split(',')]

        data = []
        for line in lines[2:]:
            data.append([x.lstrip(' ') for x in line.split(',')])
        data = pd.DataFrame(data, columns=cols)

        # cast to proper datatypes, map to time dimension
        data[self.COLS_DT_MINIDOT] = data[self.COLS_DT_MINIDOT].apply(pd.to_datetime)
        data[self.COLS_INT_MINIDOT] = data[self.COLS_INT_MINIDOT].astype(int)
        data[self.COLS_FLOAT_MINIDOT] = data[self.COLS_FLOAT_MINIDOT].astype(float)
        data = data.rename(columns=self.COLS_MAP_MINIDOT)

        return data
    

    def parse_L0(self):
        """
        Load raw (L0) oxygen logger data into xarray Dataset.

        Returns
        -------
        ds : xr.Dataset
            Dataset of data recorded by oxygen logger.
        """
        fpath_L0 = self.locate_file_L0()

        if self.sensor == 'minidot':
            data = self.parse_minidot_L0(fpath_L0)
        else:
            raise NotImplementedError("Only minidot sensors are handled.")
        
        data = data.set_index('time')
        ds = xr.Dataset.from_dataframe(data)

        return ds


    def organize_data_vars(self, ds):
        """
        Drop and rename data variables.

        Parameters
        ----------
        ds : xr.Dataset
            Oxygen logger data.
        
        Returns
        -------
        ds : xr.Dataset
            Oxygen logger data with desired data variables.
        """
        if self.sensor == 'minidot':
            ds = ds.drop_vars(self.VARS_DROP_MINIDOT)
            vars_map = {k: v for k, v in self.VARS_MAP_MINIDOT.items() if k in ds.data_vars}
        else:
            raise NotImplementedError('Only minidot sensors are handled.')

        return ds.rename_vars(vars_map)
    
    
    def assign_attributes(self, ds):
        """
        Assign attributes to data variables and to dataset.
        Add depth and serial id coordinates.

        Parameters
        ----------
        ds : xr.Dataset
            Oxygen logger data.

        Returns
        -------
        ds : xr.Dataset
            Oxygen logger data with attributes.
        """
        # add depth and serial id coordinates
        ds = ds.assign_coords(depth=self.depth, serial_id=self.serial_id)
        
        # data variables
        for var, attrs in self.VAR_ATTRS.items():
            if var in ds:
                ds[var].attrs.update(attrs)

        # dataset
        md = self.open_md_file()
        md_xr = {
            'xsc': md['campaign']['X Coordinate (CH1903)'],
            'ysc': md['campaign']['Y Coordinate (CH1903)'],
            'deployment': md['campaign']['Time of deployment'],
            'retrieval': md['campaign']['Time of retrieval'],
            'sensor': self.sensor,
            'serial_id': self.serial_id,
            'depth': self.depth,
            't_offset': str(self.t_offset)
        }
        ds = ds.assign_attrs(md_xr)

        return ds
    
    
    def derive_vars(self, ds):
        """
        Process oxygen logger data to organize variables and assign attributes.
        
        Parameters
        ----------
        ds : xr.Dataset
            Oxygen logger data.

        Returns
        -------
        ds : xr.Dataset
            Processed oxygen logger data.
        """
        ds = self.organize_data_vars(ds)
        ds = self.assign_attributes(ds)

        return ds
    

    def correct_clock_offset(self, ds):
        """
        Apply correction to sensor clock offset.

        Parameters
        ----------
        ds : xr.Dataset
            Oxygen logger data.

        Returns
        -------
        ds : xr.Dataset
            Oxygen logger data with corrected time dimension.
        
        """
        ds['time'] = ds['time'] + pd.to_timedelta(self.t_offset)

        return ds
    

    def quality_assurance(self, ds):
        """
        Run quality assurance on oxygen logger data.

        Parameters
        ----------
        ds : xr.Dataset
            Oxygen logger data.

        Returns
        -------
        ds : xr.Dataset
            Quality assured oxygen logger data.
        """
        # correct clock offset
        if self.t_offset:
            ds = self.correct_clock_offset(ds)

        # flag prior to deployment and after retrieval
        md = self.open_md_file()
        deploy = pd.to_datetime(md['campaign']['Time of deployment'])
        retrieve = pd.to_datetime(md['campaign']['Time of retrieval'])
        flag = (ds['time'] < deploy) | (ds['time'] > retrieve)
        ds['DO_mg_qual'] = flag.astype(int)
        ds['sat_qual'] = flag.astype(int)

        return ds
    

    def mask_data(self, ds):
        """
        Apply QA flags to mask data.

        Parameters
        ----------
        ds : xr.Dataset
            Oxygen logger data with QA flags.

        Returns
        -------
        ds : xr.Dataset
            Oxygen logger data with masked values.
        """
        ds['DO_mg'] = ds['DO_mg'].where(ds['DO_mg_qual'] == 0)
        ds['sat'] = ds['sat'].where(ds['sat_qual'] == 0)

        return ds
    

    # ---------- Writing ----------

    def write_to_nc(self, ds, level, overwrite=True):
        """
        Write xarray Dataset to .nc file.

        Parameters
        ----------
        ds : xr.Dataset
            Oxygen logger data.
        level : str
            L1, L2, or L3.
        overwrite : bool
            If True, overwrite existing L1 data.

        Returns
        -------
        fpath : str
            File path to written data.
        """
        if level == 'L1':
            fpath = os.path.join(self.dpath_L1, f'L1_mooring_{self.sensor}_{self.serial_id}.nc')
        elif level == 'L2':
            fpath = os.path.join(self.dpath_L2, f'L2_mooring_{self.sensor}_{self.serial_id}.nc')
        elif level == 'L3':
            fpath = os.path.join(self.dpath_L3, f'L3_mooring_{self.sensor}_{self.serial_id}.nc')
        else:
            raise ValueError('Writing level must be L1, L2, or L3.')

        if os.path.exists(fpath) and not overwrite:
            warnings.warn(f'{fpath} already exists and overwrite = False.')
        else:
            ds.to_netcdf(fpath)

        return fpath
    

    # ---------- Pipeline ----------

    def process(self):
        """
        Process raw (L0) oxygen logger data.  Convert to xarray and write to .nc (L1).
        Run quality assurance and write to .nc (L2).
        """
        # L0 to L1
        ds = self.parse_L0()
        ds = self.derive_vars(ds)
        ds = self.quality_assurance(ds)
        fpath_L1 = self.write_to_nc(ds, 'L1')

        # L1 to L2
        ds = self.mask_data(ds)
        fpath_L2 = self.write_to_nc(ds, 'L2')