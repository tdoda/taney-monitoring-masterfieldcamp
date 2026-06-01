### Class for processing oxygen logger data

# imports
import pandas as pd
import xarray as xr


class oxygen_logger:
    general_attributes = {
        "institution": "Unil",
        "source": "",
        "references": "Aquatic Science Master field camp",
        "history": "See history on Renku",
        "conventions": "CF 1.7",
        "comment": "Monitoring data in Lake Taney performed by Aquatic Science Master students",
        "title": "Mooring Lake Taney"
    }
    dimesions = {'time': {'dim_name': 'time', 'dim_size': None}}
    variables = {
        'time': {'long_name': 'Coordinated Universal Time (UTC)'},
        'DO_mg': {'units': 'mg/l', 'long_name': 'Dissolved Oxygen Concentration'},
        'sat': {'units': '%', 'long_name': 'Dissolved Oxygen Saturation'},
        'Temp': {'units': '°C', 'long_name': 'Temperature'},
        'depth': {'units': 'm', 'long_name': 'Depth'},
        'serial_id': {'long_name': 'Serial ID'}
    }
    cols_dt_minidot = ['UTC_Date_&_Time', 'Coordinated Universal Time']
    cols_int_minidot = ['Unix Timestamp']
    cols_float_minidot = ['Battery', 'Temperature', 'Dissolved Oxygen', 'Dissolved Oxygen Saturation', 'Q']
    cols_map_minidot = {'UTC_Date_&_Time': 'time'}
    vars_drop_minidot = ['Unix Timestamp', 'Coordinated Universal Time', 'Battery', 'Q']
    vars_map_minidot = {
        'Temperature': 'Temp', 
        'Dissolved Oxygen': 'DO_mg', 
        'Dissolved Oxygen Saturation': 'sat'
    }


    def __init__(self):
        """
        Initialize oxygen_logger object.
        """
        raise NotImplementedError
    

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
        ds : xr.Dataset
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
        data[self.cols_dt_minidot] = data[self.cols_dt_minidot].apply(pd.to_datetime)
        data[self.cols_int_minidot] = data[self.cols_int_minidot].astype(int)
        data[self.cols_float_minidot] = data[self.cols_float_minidot].astype(float)
        data = data.rename(columns=self.cols_map_minidot)

        # convert to xarray Dataset
        data = data.set_index('time')
        ds = xr.Dataset.from_dataframe(data)

        # drop and rename data variables
        ds = ds.drop_vars(self.vars_drop_minidot)
        vars_map = {k: v for k, v in self.vars_map_minidot.items() if k in ds.data_vars}
        ds = ds.rename_vars(vars_map)

        return ds
    

    
    # ---------- TO DO ----------
    
    def assign_attributes(self, ds):
        """
        Add depth and serial id coordinates.
        Assign attributes (i.e., descriptions) to dimensions/data variables.
        Include metadata in dataset attributes.

        Parameters
        ----------
        ds : xr.Dataset
            Oxygen logger data.

        Returns
        -------
        ds : xr.Dataset
            Oxygen logger data with attributes.
        """
        raise NotImplementedError
    

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
        raise NotImplementedError