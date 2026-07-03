#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from datetime import datetime
import cmocean

data_folder = 'path_to_file'
date_campaign = '20250605'
dz_top = 0.5  # [m] exclusion zone en surface
dz_bot = 0.5  # [m] exclusion zone au fond
varnames = ["Temp", "sat", "DO_mg"]

folder_path = os.path.join(data_folder, date_campaign, 'Level2')
if not os.path.exists(folder_path):
    folder_path = data_folder

filenames = [f for f in os.listdir(folder_path) if f.endswith('.nc')]


ctd_data = []
for file in filenames:
    full_path = os.path.join(folder_path, file)
    try:
        nc = netCDF4.Dataset(full_path)
    except Exception as e:
        print(f"⚠️ Erreur lecture {file} : {e}")
        continue

    # Coordonnées globales
    try:
        xcoord = float(getattr(nc, 'X Coordinate (CH1903)', np.nan))
        ycoord = float(getattr(nc, 'Y Coordinate (CH1903)', np.nan))
    except:
        continue

    # Profondeur et temps
    depth = nc.variables["depth"][:].data

    dict_ctd = {"filename": file, "xcoord": xcoord, "ycoord": ycoord, "depth": depth}

    # Lecture des variables
    for var in varnames:
        if var not in nc.variables:
            dict_ctd[var] = np.full_like(depth, np.nan)
            continue
        vardata = nc.variables[var][:].data.astype(float)
        # Filtrage par qualité si dispo
        if var + "_qual" in nc.variables:
            vardata[nc.variables[var + "_qual"][:] > 0] = np.nan
        # Exclusion surface/fond
        vardata[depth < dz_top] = np.nan
        vardata[depth > (np.nanmax(depth) - dz_bot)] = np.nan
        dict_ctd[var] = vardata

    ctd_data.append(dict_ctd)
    nc.close()

df = pd.DataFrame(ctd_data)
df_exploded = df.explode(['depth'] + varnames).reset_index(drop=True)

# Conversion
for col in ['xcoord', 'ycoord', 'depth'] + varnames:
    df_exploded[col] = pd.to_numeric(df_exploded[col], errors='coerce')

# Distance horizontale
x0, y0 = df_exploded.loc[0, 'xcoord'], df_exploded.loc[0, 'ycoord']
df_exploded['dist'] = np.sqrt((df_exploded['xcoord'] - x0)**2 +
                              (df_exploded['ycoord'] - y0)**2)

def plot_variable(df, var, title, cmap, unit):
    df_var = df.dropna(subset=[var])
    if df_var.empty:
        print(f"No valid data available for {var}")
        return

    # Regular grid
    xi = np.linspace(df_var['dist'].min(), df_var['dist'].max(), 200)
    yi = np.linspace(df_var['depth'].min(), df_var['depth'].max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    # Linear interpolation
    Zi = griddata(
        points=(df_var['dist'], df_var['depth']),
        values=df_var[var],
        xi=(Xi, Yi),
        method='linear'
    )

    # Figure
    plt.figure(figsize=(10, 6))
    c = plt.pcolormesh(Xi, Yi, Zi, shading='auto', cmap=cmap)
    plt.gca().invert_yaxis()
    plt.xlabel('Distance along profile [m]')
    plt.ylabel('Depth [m]')
    plt.title(title)
    plt.colorbar(c, label=unit)
    plt.tight_layout()
    plt.show()

plot_variable(df_exploded, 'Temp', 'Interpolated temperature along transect 3', 'cmo.thermal', 'Temperature [°C]')
plot_variable(df_exploded, 'DO_mg', 'Interpolated dissolved oxygen along transect 3', 'cmo.oxy', 'Dissolved O₂ [mg/L]')
plot_variable(df_exploded, 'sat', 'Interpolated oxygen saturation along transect 3', 'cmo.matter', 'O₂ Saturation [%]')
