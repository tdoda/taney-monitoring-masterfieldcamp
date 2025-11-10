#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 10:54:24 2025

@author: romaindubois
"""

import os
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from datetime import datetime
import cmocean


# === PARAMÈTRES ===
data_folder = '/Users/romaindubois/Downloads/Romain/taney-monitoring-masterfieldcamp-master/data/Profiles/RBR_66131'
date_campaign = '20250605'
varnames = ["Temp", "Cond", "sat"]
dz_top = 0.5  # [m] zone de surface à exclure
dz_bot = 0.5  # [m] zone de fond à exclure

# === LISTER LES FICHIERS ===
folder_path = os.path.join(data_folder, date_campaign, 'Level2')
if not os.path.exists(folder_path):
    folder_path = data_folder  # si les fichiers sont directement dans donnees_transect

filenames = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
if not filenames:
    raise FileNotFoundError(f"Aucun fichier .nc trouvé dans {folder_path}")

# === LECTURE DES DONNÉES ===
ctd_data = []
for file in filenames:
    full_path = os.path.join(folder_path, file)
    try:
        nc = netCDF4.Dataset(full_path)
    except Exception as e:
        print(f"⚠️ Erreur lecture {file} : {e}")
        continue

    # Extraction des métadonnées
    try:
        xcoord = float(getattr(nc, 'X Coordinate (CH1903)', np.nan))
        ycoord = float(getattr(nc, 'Y Coordinate (CH1903)', np.nan))
    except:
        print(f"Coordonnées manquantes dans {file}")
        continue

    # Temps
    time_num = nc.variables["time"][:].data
    time_date = np.array(time_num, dtype="datetime64[s]").astype(datetime)

    # Structure de données pour ce profil
    dict_ctd = {
        "filename": file,
        "xcoord": xcoord,
        "ycoord": ycoord,
        "time_start": time_date[0],
        "depth": nc.variables["depth"][:].data,
    }

    # Lecture des variables d’intérêt
    for var in varnames:
        vardata = nc.variables[var][:].data.astype(float)
        # Masquage via qualité si disponible
        if var + "_qual" in nc.variables:
            vardata[nc.variables[var + "_qual"][:] > 0] = np.nan
        # Supprimer zones trop proches surface/fond
        depth = dict_ctd["depth"]
        vardata[depth < dz_top] = np.nan
        vardata[depth > (np.nanmax(depth) - dz_bot)] = np.nan
        dict_ctd[var] = vardata

    ctd_data.append(dict_ctd)
    nc.close()

# === CRÉATION DU DATAFRAME ===
df = pd.DataFrame(ctd_data)
df_exploded = df.explode(['depth', 'Temp', 'Cond', 'sat']).reset_index(drop=True)

# Conversion en float
for col in ['xcoord', 'ycoord', 'depth', 'Temp']:
    df_exploded[col] = pd.to_numeric(df_exploded[col], errors='coerce')

# Calcul de la distance le long du transect
x0, y0 = df_exploded.loc[0, 'xcoord'], df_exploded.loc[0, 'ycoord']
df_exploded['dist'] = np.sqrt((df_exploded['xcoord'] - x0)**2 +
                              (df_exploded['ycoord'] - y0)**2)

# Filtrage des NaN
df_plot = df_exploded.dropna(subset=['Temp'])

if df_plot.empty:
    print("Aucune donnée valide à interpoler.")
else:
    # Grille régulière
    xi = np.linspace(df_plot['dist'].min(), df_plot['dist'].max(), 200)
    yi = np.linspace(df_plot['depth'].min(), df_plot['depth'].max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolation
    Zi = griddata(
        points=(df_plot['dist'], df_plot['depth']),
        values=df_plot['Temp'],
        xi=(Xi, Yi),
        method='linear'
    )

    # === VISUALISATION ===
    plt.figure(figsize=(10, 6))
    c = plt.pcolormesh(Xi, Yi, Zi, shading='auto', cmap='cmo.thermal')
    plt.gca().invert_yaxis()
    plt.xlabel('Distance [m]')
    plt.ylabel('Profondeur [m]')
    plt.title("Température interpolée (°C)")
    plt.colorbar(c, label='Température [°C]')
    plt.tight_layout()
    plt.show()
varnames = ["Temp", "Cond", "sat"]
dz_top = 0.5  # [m] zone de surface à exclure
dz_bot = 0.5  # [m] zone de fond à exclure

# === LISTER LES FICHIERS ===
folder_path = os.path.join(data_folder, date_campaign, 'Level2')
if not os.path.exists(folder_path):
    folder_path = data_folder  # si les fichiers sont directement dans donnees_transect

filenames = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
if not filenames:
    raise FileNotFoundError(f"Aucun fichier .nc trouvé dans {folder_path}")

# === LECTURE DES DONNÉES ===
ctd_data = []
for file in filenames:
    full_path = os.path.join(folder_path, file)
    try:
        nc = netCDF4.Dataset(full_path)
    except Exception as e:
        print(f"⚠️ Erreur lecture {file} : {e}")
        continue

    # Extraction des métadonnées
    try:
        xcoord = float(getattr(nc, 'X Coordinate (CH1903)', np.nan))
        ycoord = float(getattr(nc, 'Y Coordinate (CH1903)', np.nan))
    except:
        print(f"Coordonnées manquantes dans {file}")
        continue

    # Temps
    time_num = nc.variables["time"][:].data
    time_date = np.array(time_num, dtype="datetime64[s]").astype(datetime)

    # Structure de données pour ce profil
    dict_ctd = {
        "filename": file,
        "xcoord": xcoord,
        "ycoord": ycoord,
        "time_start": time_date[0],
        "depth": nc.variables["depth"][:].data,
    }

    # Lecture des variables d’intérêt
    for var in varnames:
        vardata = nc.variables[var][:].data.astype(float)
        # Masquage via qualité si disponible
        if var + "_qual" in nc.variables:
            vardata[nc.variables[var + "_qual"][:] > 0] = np.nan
        # Supprimer zones trop proches surface/fond
        depth = dict_ctd["depth"]
        vardata[depth < dz_top] = np.nan
        vardata[depth > (np.nanmax(depth) - dz_bot)] = np.nan
        dict_ctd[var] = vardata

    ctd_data.append(dict_ctd)
    nc.close()

# === CRÉATION DU DATAFRAME ===
df = pd.DataFrame(ctd_data)
df_exploded = df.explode(['depth', 'Temp', 'Cond', 'sat']).reset_index(drop=True)

# Conversion en float
for col in ['xcoord', 'ycoord', 'depth', 'Temp']:
    df_exploded[col] = pd.to_numeric(df_exploded[col], errors='coerce')

# Calcul de la distance le long du transect
x0, y0 = df_exploded.loc[0, 'xcoord'], df_exploded.loc[0, 'ycoord']
df_exploded['dist'] = np.sqrt((df_exploded['xcoord'] - x0)**2 +
                              (df_exploded['ycoord'] - y0)**2)

# Filtrage des NaN
df_plot = df_exploded.dropna(subset=['Temp'])

# === DÉFINITION DES GROUPES DE PROFILS ===
# (on ne drop rien, on choisit juste les bons index)

profil1_idx = [5, 6, 1, 2, 11, 12, 7]          # profil vertical 1
profil2_idx = [4, 16, 13, 0, 3, 15, 14]        # profil vertical 2

# Extraction directe dans df
df_p1 = df.loc[profil1_idx]
df_p2 = df.loc[profil2_idx]

# === FONCTION D’AFFICHAGE ===
def plot_vertical_profile(df_subset, title):
    df_exploded = df_subset.explode(['depth', 'Temp']).reset_index(drop=True)
    df_exploded['depth'] = pd.to_numeric(df_exploded['depth'], errors='coerce')
    df_exploded['Temp'] = pd.to_numeric(df_exploded['Temp'], errors='coerce')

    # Calcul d’une distance relative à partir du premier point
    x0, y0 = df_exploded.iloc[0]['xcoord'], df_exploded.iloc[0]['ycoord']
    df_exploded['dist'] = np.sqrt(
        (df_exploded['xcoord'] - x0)**2 + (df_exploded['ycoord'] - y0)**2
    )

    # Nettoyage
    df_plot = df_exploded.dropna(subset=['Temp'])
    if df_plot.empty:
        print(f"Aucune donnée valide pour {title}")
        return

    # Grille régulière
    xi = np.linspace(df_plot['dist'].min(), df_plot['dist'].max(), 200)
    yi = np.linspace(df_plot['depth'].min(), df_plot['depth'].max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolation
    Zi = griddata(
        points=(df_plot['dist'], df_plot['depth']),
        values=df_plot['Temp'],
        xi=(Xi, Yi),
        method='linear'
    )

    # === VISUALISATION ===
    plt.figure(figsize=(9, 5))
    c = plt.pcolormesh(Xi, Yi, Zi, shading='auto', cmap='cmo.thermal')
    plt.gca().invert_yaxis()
    plt.xlabel('Distance along profile [m]')
    plt.ylabel('Depth [m]')
    plt.title(title)
    plt.colorbar(c, label='Temperature [°C]')
    plt.tight_layout()
    plt.show()


# === TRACE DES DEUX PROFILS ===
plot_vertical_profile(df_p1, "Interpolated temperature along transect 1 ")
plot_vertical_profile(df_p2, "Interpolated temperature along transect 2")

    
    
 
  
    
    