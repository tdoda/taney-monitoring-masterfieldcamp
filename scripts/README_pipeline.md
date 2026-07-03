# Lake Taney mooring — pipeline execution order

Three sensor types, three independent entry points. Run in this order for a
new campaign (only needed if `mooring_config == "option3"` for the
thermistors; for `single_chain`/`option1`/`option2`, only step 4 is needed):

1. **`main_pressure.py`** — processes the two HOBO pressure loggers
   (`data/Mooring/HOBO_P/<date>/Level0/pressure_sensors_<date>.meta`).
   Produces per-sensor L1/L2 netCDF files.

2. **Pressure analysis notebook** — combines the two pressure sensors'
   L2 output into `water_level_variation.csv`, written to
   `data/Mooring/HOBO_P/<date>/Level2/water_level_variation.csv`.
   This file is a hard dependency for step 4 (option3).

3. **`main_oxygen.py`** — processes the miniDOT oxygen logger(s)
   (`data/Mooring/miniDOT/<date>/Level0/oxygen_loggers_<date>.meta`).
   Independent of steps 1-2; can run any time.

4. **`main_mooring.py`** — processes the thermistor chain(s) and builds the
   L3 grid(s). For `mooring_config == "option3"`, requires
   `water_level_variation.csv` from step 2 to already exist.

Each script only requires editing `date_campaign` at the top to process a
new campaign; all sensor names, depths, and validity flags come from the
corresponding `.meta` file — nothing is hardcoded.

## Meteo

Three independent pipelines, covering two different data sources and two
campaign vintages. None of them depends on the mooring pipeline above.

- **`main_meteo.py`** (+ `Meteo.py`, `functions_meteo.py`) — old campaign
  (2024-03-01 to 2025-06-06) ICON/COSMO model reanalysis, read from
  `data/Meteo/Model/20250606/Level0/meteo_20250606.meta`. Produces **L1
  only** (no L2 masking in this version).

- **`meteo_model.py`** — new campaign (2025-06-01 to 2026-06-03) ICON
  model data, read from
  `data/Meteo/Model/<date>/Level0/meteo_model_<date>.meta`. This file is
  its own entry point (`if __name__ == "__main__":` block at the bottom) —
  run directly with `python meteo_model.py`, no separate `main_*.py` needed.

- **`weather_station.py`**  — real Netatmo
  weather station measurements (not a model), read from
  `data/Meteo/Station/<date>/Level0/meteo_station_<date>.meta`. **This
  `.meta` file does not exist yet** — field names assumed in
  `main_weather_station.py` (`valid`, `serial_id`, `t_offset`) are inferred
  from the pressure/oxygen `.meta` convention and need to be confirmed once
  the real file is created.

## Bathymetry

- **`bathymetric_main.py`** (+ `bathymetric_function.py`, authored by
  Quentin Noël) — processes Deeper Smart Sonar Pro+ 2 transect data into
  L1 (cleaned GPS/time) and L2 (orthogonal projection onto the transect
  axis + 1 m binning).

  Reads its config from a `.meta` file, found by pattern
  (`glob("*.meta")`, not by exact name, since export tools sometimes prefix
  it with an ID) in
  `data/Bathymetry/<date_campaign>/Level0/`. Only `date_campaign` at the
  top of `bathymetric_main.py` needs editing for a new transect; transect
  endpoints, time crop, bin size, and output filenames all come from the
  `.meta`, consistent with the rest of the repo's "nothing hardcoded"
  convention.

## Full execution order for a new campaign

```
main_pressure.py            (HOBO_P pressure loggers)
   -> pressure analysis notebook   (produces water_level_variation.csv)
main_oxygen.py               (miniDOT oxygen loggers, no dependency)
main_mooring.py               (thermistors + L3 grid, needs water_level_variation.csv if option3)
meteo_model.py                (ICON model, new campaign, no dependency)
main_weather_station.py       (Netatmo station, no dependency, .meta not created yet)
bathymetric_main.py           (sonar transect, no dependency, config at top of file)
```
