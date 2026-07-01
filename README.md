# Taney Monitoring — Master Field Camp

Monitoring of Lake Taney by the Aquatic Science Master students of the University of Lausanne.

**Repository:** https://github.com/tdoda/taney-monitoring-masterfieldcamp.git

> A guide on how to use Git collaboratively is provided in `Notes/Instructions_Git`.

---

## Introduction

This project studies the thermal structure of Lake Taney (Valais, Switzerland) and its fluctuations in relation to significant water level variability. The data were collected as part of the Master's Field Camp in Aquatic Science at the University of Lausanne and are used to analyse the physical and biogeochemical dynamics of the lake. This student-led monitoring program started in June 2025.

---
## Installation

### 1. Python installation

Python 3 is required to run the scripts. Three installation are possible:
- Recommended option: download [Miniforge](https://github.com/conda-forge/miniforge). 
- User-friendly option: download the [Anaconda distribution](https://www.anaconda.com/products/individual).
- Classic option: download Python from the [official website](https://www.python.org/downloads/).

### 2. Repository installation

- If using GIT, clone the repository to your local machine using the command in Git Bash: 

    ``` 
    git clone  https://github.com/tdoda/taney-monitoring-masterfieldcamp.git 
    ```
 
    Or use the "clone" options in VS Code or Git Desktop.
    Note that the repository will be copied to your current working directory.
- Without GIT, just download the entire ZIP folder from https://github.com/tdoda/taney-monitoring-masterfieldcamp.git ("Code" > "Download ZIP") and extract it.

### 3. Packages installation

1. Open the terminal (e.g., Anaconda Prompt), and move to the `taney-monitoring-masterfieldcamp` repository.
2. Create a new environment *taney-monitoring* and install the packages as follows:
    - If using conda (Anaconda or Miniforge installation):
        ```
        conda env create -f environment.yml
        conda activate taney-monitoring 
        ```
        It is also possible to install the packages from `requirements.txt` with pip instead:
        ```
        conda create -n taney-monitoring python=3.11
        conda activate taney-monitoring
        pip install -r requirements.txt
        ```
    - If using mamba (Anaconda or Miniforge installation):
        ```
        mamba env create -f environment.yml
        mamba activate taney-monitoring 
        ```
    - If using pip (classic Python installation):
        ```
        python -m venv taney-monitoring  
        # For Linux/macOS:     
        source taney-monitoring/bin/activate 
        # For Windows:
        taney-monitoring\Scripts\activate   
        pip install -r requirements.txt
        ```

---

## Repository Structure

The repository separates raw data, processing scripts, and analysis notebooks.

### Data

Contains datasets collected in the field, organised by type:

- **`Meteo/`** — meteorological data, including local weather station observations, ICON model data, and cloud-cover data used for atmospheric forcing calculations.
- **`Mooring/`** — data from sensors installed on the lake mooring, including thermistors, MiniDOT sensors, and HOBO pressure loggers.
- **`Profiles/`** — vertical profile measurements, including CTD profiles of temperature, oxygen, conductivity, and related variables.

Each data type folder contains subfolders with the following organization: `data_type/sensor_type/campaign_date/processing_level/`

**Important notes:**
- Disregard `EXO_...` probes — use RBR probes only
- The `HOBO_P` file (pressure sensor) is empty for 2024-2025 data — disregard
- For the 2025–2026 campaign, HOBO pressure data were used to derive water-level variation. The pressure-derived correction is available until 18 January 2026.
- Meteorological data come from different sources and should not be treated as interchangeable. Local station data represent observed conditions near Lac de Taney, while ICON data provide hourly modelled atmospheric forcing for the lake area.

**Campaign folders:** data files are stored in different subfolders for different campaigns with the date of the campaign as a folder name: format `YYYYMMDD` (e.g. `20250603`).

**Processing levels:**

The data files are organized in processing levels:

| Level | Description |
|-------|-------------|
| **Level 0** | Raw data — files exported directly from the logger (sensor-specific file format) |
| **Level 1** | Cleaned data converted to NetCDF (`.nc`) format |
| **Level 2** | Corrected and merged data ready for analysis (metadata, time alignment, hourly/daily averages) |
| **Level 3** *(optional)* | Interpreted or aggregated data for final analyses |

---

### Notebooks

Jupyter notebooks used for visualisation and analysis:

- `plot_ctd.ipynb` — CTD profile visualisation
- `clean_mooring_minidot_topbottom_rtrm.ipynb` — mooring data visualisation
- `plot_weather.ipynb`- weathter data visualisation
- `oxygen notebook.ipynb` - oxygen data visualisation
- `Notebook_Temperature.ipynb`- Temperature data visualisation
- `bathymetric_analysis_visu.ipynb`- bathymetric data visualisation
- `plot_mooring.ipynb`- this notebook visualise the data of the mooring for the 2025 campaign
- **`Figures/`** (*optional*) — figures automatically generated by the notebooks

Each notebook is self-contained and documented for easy reuse by students and researchers.

---

### Scripts

Reusable Python scripts and functions are organised in subfolders according to data type, including CTD profiles, mooring sensors, pressure data, and meteorological forcing.
The scripts are used to:
- read raw data files from the different sensors or external sources;
- convert data into consistent formats;
- attach deployment metadata;
- generate Level 1 and Level 2 products;
- compute derived variables such as water-level variation, wind speed, or thermal structure indicators.

> Modify the CTD script to analyse only profiles after a specific time if needed.

---

### Notes

Manual notes, mooring diagrams, logbook, etc.

---

## Data Storage & Workflow

Due to file size limitations on GitHub, raw data files (`.rsk`, `.xlsx`) are not pushed to git (see `.gitignore` for details). All data are stored on [**SwitchDrive**]( https://drive.switch.ch/index.php/s/yyV253N9xIICeSs).

### Uploading Data from the Field

#### CTD
1. Upload RBR CTD data using Ruskin (smartphone or laptop)
2. Upload the `.rsk` file to the `Data` folder on SwitchDrive following the structure: `YYYYMMDD/Level0/`
3. Create a `.meta` file in the new folder (follow an example from past measurements — required for post-processing)
4. Proceed to **Data Use** below

#### HOBO T
1. Upload HOBO sensor data using the dedicated device and field laptop
2. Save as `.xlsx` — filename should match the deployment depth (e.g. `5m.xlsx` for a logger at 5 m)
3. Add a `.meta` file following the example from past measurements (required for post-processing)
4. Upload to SwitchDrive
5. Proceed to **Data Use** below

#### HOBO P 

1. Export the two HOBO pressure logger files from the logger software in CSV format.
2. Name the files using the sensor serial number, mooring position and deployment depth, for example `20598413_A_14.csv` and `20598412_B_15.csv`.
3. Add or update the corresponding `.meta` file
4. Upload to SwitchDrive.
5. Proceed to **Data Use** below

#### DO Minidot
1. Export the MiniDOT oxygen logger file from the logger software in `txt` format. The expected raw file is `Cat.txt`.
2. Save the file in the appropriate MiniDOT Level 0 folder on SwitchDrive.
3. Add or update the corresponding `.meta` file.
4. Upload to SwitchDrive.
5. Proceed to **Data Use** below

#### Meteorological data
1. Store the local weather station file, the ICON model file in the appropriate Meteo folder. 
2. Add or update the corresponding `.meta` files.
3. Upload to SwitchDrive
4. Proceed to **Data Use** below

### Using the Data

*See also Sect.3 in `notes/Ìnstructions_Git/Instructions_Git.pdf` for more details.*

1. Pull the most recent version of the git repository locally
2. Copy the most recent `Data` folder from SwitchDrive to your local folder
3. Run or modify the scripts as needed
4. If updated L1/L2/L3 products are produced, upload them to SwitchDrive
5. Push any code changes (scripts, notebooks, figures, reports) to git

---

## Open Questions

The following can be addressed by analysing the data and writing new scripts:

- Yearly minimum and maximum surface temperature
- Duration of the stratified period
- Number of mixing events
- Duration of the ice-covered period
- Dynamics of oxygen content
- Link with meteorological forcing
- Heat fluxes

---

## Authors

**Tomy Doda, Damien Bouffard**

**2025:** Noémie Bagnoud, Huey Bickerstaffe, Romain Du Bois, Pierre Herold, Rachel Jacot-Descombes, Justin Knight, Alejandro Perez Pardo, Margaux Python, Célestin Pythoud, Marco Zaninetti

**2026:** Bauermeister Lynn, Bulotti Nicola, Criado Gabriela, Fringeli Coralie, Grillon James, Korkmaz Nihal, Messerli Estelle, Noël Quentin, Quillet Térence, Randrianasolo Iriana, Rottigni Nathan , Rouyard Cyprile, Sanwald Giulia,Saravia Guevara Nicolle.  

---

## Licence

To be confirmed by the project supervisors.
Until a licence is formally specified, the code and data in this repository should not be reused, redistributed, or published outside the project without permission from the authors and supervisors.
