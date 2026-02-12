# Daily-to-Hourly Temperature Downscaling (ERA5-trained RF Residual Model) + LOCA2 Batch Inference

This repository contains a reproducible pipeline for **daily-to-hourly temperature downscaling** over North America using a **Random Forest (RF) residual model** trained on **ERA5** and applied to **LOCA2-derived daily max/min** inputs. The workflow is designed for **HPC/Slurm** execution and produces per-county trained models, evaluation logs, and hourly predictions.

## Key idea

Given a daily input time series with columns:

- `date_time` (daily timestamps; only the calendar day is used)
- `daily_max` (°C)
- `daily_min` (°C)

we generate hourly temperature by:

1. Constructing hourly features from daily values and cyclical time features.
2. Predicting **residuals** around a baseline diurnal signal.
3. Optionally applying a **daily min/max enforcement** post-process.
4. Running large-scale inference on LOCA2-style synthesized ensembles via Slurm array jobs.

---

## Repository structure



- `code/`
  - `era5_temp_tocsv.py`  
    Convert ERA5 NetCDF temperature to county point CSVs (hourly).
  - `era5_dailymaxmin_from_hourly.py`  
    Compute daily max/min from hourly ERA5 point CSVs (°C).
  - `rf_residual_model_training.py`  
    Train per-county RF residual models (raw + enforced variants), save models/scalers/configs, and write evaluation logs.
  - `rf_residual_LOCA2_prediction.py`  
    Run batch downscaling for LOCA2-style daily inputs using **enforced** models; supports skipping existing outputs and handling missing Feb-29.
  - `collect_evaluation_metrics.py`  
    Parse `evaluation_log.txt` across counties (raw/enforced) into a single summary CSV.

- `file/`
  - `County_selected_revise_3.csv` (county metadata; includes `in.weather_file_latitude`, `in.weather_file_longitude`, and county IDs like `G0600850`)
  - `locations.txt` (one `lat_lon` per line, preserving original decimal precision)

- `submit/`
# Daily-to-Hourly Temperature Downscaling (ERA5-trained RF Residual Model) + LOCA2 Batch Inference

This repository contains a reproducible pipeline for **daily-to-hourly temperature downscaling** over North America using a **Random Forest (RF) residual model** trained on **ERA5** and applied to **LOCA2/CESM2-LENS2-derived daily max/min** inputs. The workflow is designed for **HPC/Slurm** execution and produces per-county trained models, evaluation logs, and hourly predictions.

## Key idea

Given a daily input time series with columns:

- `date_time` (daily timestamps; only the calendar day is used)
- `daily_max` (°C)
- `daily_min` (°C)

we generate hourly temperature by:

1. Constructing hourly features from daily values and cyclical time features.
2. Predicting **residuals** around a baseline diurnal signal.
3. Optionally applying a **daily min/max enforcement** post-process.
4. Running large-scale inference on LOCA2-style synthesized ensembles via Slurm array jobs.

---

## Repository structure (example)

> Filenames may differ slightly depending on your local organization; see the `code/` directory.

- `code/`
  - `era5_temp_tocsv.py`  
    Convert ERA5 NetCDF temperature to county point CSVs (hourly).
  - `era5_dailymaxmin_from_hourly.py`  
    Compute daily max/min from hourly ERA5 point CSVs (°C).
  - `rf_residual_model_training.py`  
    Train per-county RF residual models (raw + enforced variants), save models/scalers/configs, and write evaluation logs.
  - `rf_residual_LOCA2_prediction.py`  
    Run batch downscaling for LOCA2-style daily inputs using **enforced** models; supports skipping existing outputs and handling missing Feb-29.
  - `collect_evaluation_metrics.py`  
    Parse `evaluation_log.txt` across counties (raw/enforced) into a single summary CSV.
  - `collect_metric_distribution.py` (optional)  
    Summarize distributions (mean/min/max/quantiles/skew/kurtosis) across counties.
  - `submit_rf_residual_model_training.sh` (Slurm array training)
  - `submit_rf_residual_LOCA2_prediction.sh` (Slurm array inference)

- `file/`
  - `County_selected_revise_3.csv` (county metadata; includes `in.weather_file_latitude`, `in.weather_file_longitude`, and county IDs like `G0600850`)
  - `locations.txt` (one `lat_lon` per line, preserving original decimal precision)

---

## Data and I/O conventions

### ERA5 (training truth)
- Input NetCDF example: `era5_temperature_{year}.nc`
- Variable: `t` (Kelvin in raw ERA5); this pipeline converts and stores **Celsius** for training/evaluation consistency.

### County meta list
- `file/County_selected_revise_3.csv`
- Uses:
  - `in.weather_file_latitude`
  - `in.weather_file_longitude`
  - county ID (e.g., `G0600850`) for folder naming and outputs

### Trained model outputs
Per county:
- `rf_residual_model/{county}/raw/`
- `rf_residual_model/{county}/enforced/`

Each contains:
- `rf_resid_model.pkl`
- `scaler_residual.pkl`
- `inference_config.json`
- `evaluation_log.txt`
- `test_predictions_2016_2024.csv`
- plots under `plots/`

### LOCA2-style inference inputs
Expected pattern (example):

`/nfs/turbo/seas-mtcraig-climate/Mai/LOCA2_point_csv_2018_2062/{county}/SYNTH_ENSEMBLES/{gcm}/{SSP}/from_CESM2-LENS_SSP370/r{ensemble_id}dailymaxmin_synth.csv`

Columns:
- `date_time,daily_max,daily_min` (°C)

### Inference outputs
Written to:

`/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/rf_residual_downscaling/{county}/{gcm}/{SSP}/r{ensemble_id}_hourly_synth.csv`

---

## Method details

### Features
The training/inference feature set is defined in each county’s `inference_config.json`. A typical configuration includes:
- `daily_max`, `daily_min`
- cyclical features (sin/cos): month, day-of-week, hour
- **3-day window**: include previous-day and next-day daily min/max; if missing, copy the current day’s values.

### Target and baseline
- Baseline often uses a simple function of daily max/min (e.g., `0.5*(daily_max+daily_min)`).
- The RF predicts the **residual** between observed hourly temperature and the baseline signal.

### Post-processing (optional)
Two variants are saved:
- **raw**: no daily min/max enforcement
- **enforced**: apply a deterministic post-process so the resulting hourly series matches the daily max/min constraints

---

## Leap day (Feb 29) handling

Some synthesized daily datasets omit `02-29` even in leap years, while others include it.
The inference code includes logic to **detect missing Feb-29 and insert it** (using consistent, documented interpolation rules) so that the daily-to-hourly expansion remains aligned and complete.

---

## Skip-if-exists behavior

Inference supports **skipping outputs** when the target hourly CSV already exists. This is intended for large Slurm array runs where reruns are common.

**Note on partial outputs:** in general, partial CSVs are uncommon if you write outputs atomically (e.g., write to a temp file then rename). If your script uses atomic writes, skipping by existence is safe.

---

## Installation

Recommended: Conda environment.

```bash
conda create -n env_downscaling
pip install -U pip
pip install numpy pandas scikit-learn joblib tqdm xarray netCDF4
