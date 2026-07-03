#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# === Fichier à lire ===
file_path = "path_to_file"

# === Lecture du fichier ===
nc = netCDF4.Dataset(file_path)

time = nc.variables["time"][:]
depth = nc.variables["depth"][:]
temp = nc.variables["temp"][:]  # (depth, time)

# Conversion du temps
origin = datetime(1970, 1, 1)
time_dt = np.array([origin + timedelta(seconds=float(t)) for t in time])

nc.close()

plt.figure(figsize=(11, 6))
T = np.ma.masked_invalid(temp)
time_grid, depth_grid = np.meshgrid(time_dt, depth)

plt.pcolormesh(time_grid, depth_grid, T, shading='auto', cmap='plasma')
plt.gca().invert_yaxis()
plt.colorbar(label="Temperature (°C)")
plt.title("Temperature evolution in Lake Taney", fontsize=13)
plt.xlabel("Date")
plt.ylabel("Depth (m)")

ax = plt.gca()
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.show()

plt.figure(figsize=(11, 6))
for i, z in enumerate(depth):
    plt.plot(time_dt, temp[i, :], label=f"{z:.1f} m", linewidth=1.8)

plt.title("Temperature over time at different depths (mooring data)", fontsize=13)
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend(
    title="Depth (m)",
    loc='upper left',             # coin supérieur gauche
    bbox_to_anchor=(0.05, 0.95),  # (x, y) relatif à la figure
    fontsize=9,
    frameon=True)
plt.grid(True, linestyle=':', alpha=0.5)

ax = plt.gca()
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

plt.tight_layout()
plt.show()
