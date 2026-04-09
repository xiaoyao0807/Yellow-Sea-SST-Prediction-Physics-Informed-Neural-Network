# Yellow Sea SST Prediction using Physics-Informed Neural Network (PINN)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)]()

> **Graduation Design Project**  
> An end-to-end sea surface temperature (SST) forecasting system for the Yellow Sea, built with a **Physics-Informed Neural Network (PINN)** and **JAXA GCOM-C/SGLI L3 satellite data**.  
> Supports **1–90 day** forecasting with an interactive Flask-based web dashboard.

**Key Performance:** Test set **R² = 0.9778**, **RMSE = 1.29 °C**.

---

## Table of Contents

- [Highlights](#highlights)
- [System Preview](#system-preview)
- [Quick Start](#quick-start)
- [Model & Physics Constraints](#model--physics-constraints)
- [Experimental Results](#experimental-results)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Data Source](#data-source)
- [Reproducibility Checklist & Troubleshooting](#reproducibility-checklist--troubleshooting)
- [Acknowledgments & Citation](#acknowledgments--citation)

---

## Highlights

- **Real Satellite Data Driven**: Uses JAXA GCOM-C/SGLI L3 SST V3 extracted via Google Earth Engine, covering **Jan 2018 – Apr 2026** (~2995 days).
- **Physics-Enforced Learning**: Integrates 4 physical constraints (temperature range, seasonal cycle, temporal smoothness, thermal inertia) with a warm-up training schedule.
- **End-to-End Pipeline**: Complete workflow from **data acquisition → cleaning → feature engineering → PINN training → evaluation → web service**.
- **Interactive Web Dashboard**: Built with **Flask + Chart.js**, bilingual (Chinese/English), responsive on desktop/tablet/mobile.

---

## System Preview

The web application contains four primary pages:

| Page | Functionality |
|---|---|
| **Home** | System introduction, real-time SST dashboard, overall status |
| **Forecast** | Custom 1–90 day forecast with interactive trend chart and detailed table |
| **Analysis** | Historical trends, monthly statistics, key metrics panel |
| **About** | Model architecture, physics constraints, data source and citation |

---

## Quick Start

### 1) Environment Setup

```bash
git clone https://github.com/xiaoyao0807/Yellow-Sea-SST-Prediction-Physics-Informed-Neural-Network.git
cd Yellow-Sea-SST-Prediction-Physics-Informed-Neural-Network
pip install -r requirements.txt
```

### 2) Prepare Data

Recommended:
- Export GCOM-C/SGLI L3 SST data via Google Earth Engine (see `docs/` guide).
- Place the exported CSV at:

```text
data/sgli_yellow_sea_sst_daily.csv
```

Alternative:
- Use `real_data_collector.py` for fallback/other-source collection.

### 3) Data Cleaning and Training

```bash
# Filter low-quality pixels + interpolate missing days
python prepare_sgli_data.py

# Train PINN and generate evaluation outputs
python training.py
```

### 4) Launch Web Service

```bash
python web_app_real_data.py
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Model & Physics Constraints

### Network Architecture

```text
Input (25 features)
→ Dense(128) + Tanh
→ Dense(128) + Tanh
→ Dense(64)  + Tanh
→ Dense(32)  + Tanh
→ Output (1, SST)
```

### Physics Constraints

| Constraint | Formulation | Physical Meaning |
|---|---|---|
| Temperature Range | Penalize predictions outside reasonable bounds | Physically valid Yellow Sea SST range |
| Seasonal Cycle | Enforce `corr(pred, cos(DOY)) <= -0.3` | Winter-cold / summer-warm annual behavior |
| Temporal Smoothness | Minimize `\|\|pred - 2·lag₁ + lag₂\|\|²` | Suppress unrealistic acceleration in SST change |
| Thermal Inertia | Penalize `\|pred - lag₁\| > 2 °C/day` | Mixed-layer heat capacity limits abrupt daily jumps |

### Training Strategy

- **Optimizer**: Adam (`lr = 0.001`)
- **Scheduler**: ReduceLROnPlateau (`factor = 0.5`, `patience = 20`)
- **Early Stopping**: `patience = 50`
- **Gradient Clipping**: `max_norm = 1.0`
- **Physics Warm-up**: Linearly increase physics-loss weight over the first 30 epochs

---

## Experimental Results

Evaluation on chronological train/test split (2995 days, 2018–2026):

| Metric | Value |
|---|---|
| R² (Coefficient of Determination) | **0.9778** |
| RMSE | **1.29 °C** |
| MAE | **1.00 °C** |
| MAPE | 20.05% |
| Forecast Horizon | 1–90 days |

---

## API Endpoints

Flask-based JSON APIs for frontend integration and secondary use:

| Endpoint | Method | Description | Example |
|---|---|---|---|
| `/api/health` | GET | Service health and data coverage | `/api/health` |
| `/api/data/historical` | GET | Historical records (`days` optional) | `/api/data/historical?days=180` |
| `/api/data/statistics` | GET | Statistical summary (mean/extremes/seasonality) | `/api/data/statistics` |
| `/api/predict` | GET | Future forecast (`days=1~90`) | `/api/predict?days=30` |
| `/api/model_info` | GET | Model and physics-constraint metadata | `/api/model_info` |

---

## Project Structure

```text
├── config.py
├── sgli_l3_sst_extract_gee.py
├── prepare_sgli_data.py
├── real_data_collector.py
├── data_preprocessing.py
├── pinn_model.py
├── training.py
├── evaluation.py
├── web_app_real_data.py
├── requirements.txt
│
├── data/          # generated locally
├── models/        # generated locally
├── results/       # generated locally
├── plots/         # generated locally
├── templates/
├── static/
└── docs/
```

---

## Data Source

**Primary Product:** JAXA GCOM-C/SGLI L3 Sea Surface Temperature (V3)

- **Resolution/Frequency:** ~4.6 km, daily
- **Time Coverage:** since 2018-01-22
- **Region Used in This Project:** Yellow Sea (`119°E–127°E`, `32°N–41°N`)
- **Access Platform:** Google Earth Engine  
  `JAXA/GCOM-C/L3/OCEAN/SST/V3`

Detailed extraction instructions:  
`docs/GCOM-C_SGLI_L3_SST_V3_数据接入与GitHub发布指南.md`

---

## Reproducibility Checklist & Troubleshooting

### Reproducibility Checklist

1. `data/sgli_yellow_sea_sst_daily.csv` exists and has valid date coverage.
2. `python prepare_sgli_data.py` runs successfully.
3. `python training.py` generates `models/best_model.pth` and `results/model_evaluation.csv`.
4. `python web_app_real_data.py` starts and shows `Running on http://127.0.0.1:5000`.
5. All dashboard pages load charts correctly.

### Troubleshooting

| Issue | Common Cause | Solution |
|---|---|---|
| `data file not found` during training | CSV not in `data/` | Verify `data/sgli_yellow_sea_sst_daily.csv` |
| Metrics suddenly degrade | Low-quality newly added data / too short span | Re-export data and check `valid_pixels` filtering result |
| Web starts but charts are empty | API failure / browser cache | Refresh, inspect browser console, test `/api/health` |
| GEE export fails | Task not executed / quota issue | Re-run in GEE Tasks panel, shorten export range |
| Port 5000 already in use | Existing Flask process | Stop old process or change port |

---

## Acknowledgments & Citation

- **JAXA** — GCOM-C/SGLI L3 SST V3 satellite product
- **Google Earth Engine** — Cloud data access and extraction platform
- **PyTorch** — Deep learning framework

**Satellite Data Citation**

Kurihara, Y. (2020). *GCOM-C/SGLI Sea Surface Temperature (SST) Algorithm Theoretical Basis Document (ATBD), Version 2*. JAXA.
