# Yellow Sea SST Prediction using Physics-Informed Neural Network (PINN)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)]()

> **Graduation Design Project**: A sea surface temperature (SST) forecasting system for the Yellow Sea using **Physics-Informed Neural Network (PINN)** and **JAXA GCOM-C/SGLI L3 satellite data**. Supports **1–90 day** forecasts with an interactive Flask-based web dashboard.

**Key Performance**: Test set **R² = 0.978**, **RMSE = 1.29 °C**.

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

- **Real Satellite Data Driven**: Uses JAXA GCOM-C/SGLI L3 SST V3 product, extracted via Google Earth Engine, covering **Jan 2018 – Apr 2026** (~2995 days).
- **Physics-Enforced Learning**: Implements 4 physical constraints (temperature range, seasonal cycle, temporal smoothness, thermal inertia) with a warm-up training strategy.
- **End-to-End Pipeline**: Complete workflow from **data acquisition → cleaning → feature engineering → PINN training → evaluation → web service**.
- **Interactive Web Dashboard**: Built with **Flask + Chart.js**, bilingual (Chinese/English), responsive on desktop/tablet/mobile.

---

## System Preview

The web application includes four main pages:

| Page | Functionality |
| :--- | :--- |
| **Home** | System introduction, real-time temperature dashboard, overall status |
| **Forecast** | Custom 1–90 day forecast with interactive trend curve and detailed data table |
| **Analysis** | Historical temperature trends, monthly statistics, key metrics panel |
| **About** | Model architecture, physics constraints description, data source & citation |

---

## Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/xiaoyao0807/Yellow-Sea-SST-Prediction-Physics-Informed-Neural-Network.git
cd Yellow-Sea-SST-Prediction-Physics-Informed-Neural-Network
pip install -r requirements.txt

2. Prepare Data
Recommended: Run sgli_l3_sst_extract_gee.py in Google Earth Engine Code Editor (see docs/ for detailed guide). Place the exported CSV at data/sgli_yellow_sea_sst_daily.csv.

Alternative: Use real_data_collector.py to fetch from other sources.

3. Data Preprocessing & Training

# Clean data (filter low-quality pixels, interpolate missing values)
python prepare_sgli_data.py

# Train PINN model (saves best model to models/ and generates evaluation report)
python training.py

### 4. Launch Web Service

```bash
python web_app_real_data.py

Open http://127.0.0.1:5000 in your browser.
```

Model & Physics Constraints
Network Architecture

Input (25 features) → Dense(128) → Tanh → Dense(128) → Tanh → 
Dense(64) → Tanh → Dense(32) → Tanh → Output (1, SST)


---
Physics Constraints

Constraint	Formulation	Physical Meaning

Temperature Range	Penalize predictions ∉ [-3, 31] °C	Physically reasonable SST range for Yellow Sea
Seasonal Cycle	Enforce correlation(pred, cos(DOY)) > -0.3	Capture winter-cold, summer-warm annual pattern
Temporal Smoothness	Minimize ||pred - 2·lag₁ + lag₂||²	Suppress "acceleration" of temperature change
Thermal Inertia	Enforce |pred - lag₁| ≤ 2 °C/day	Ocean mixed layer has large heat capacity

Training Strategy

Optimizer: Adam (lr = 0.001)

Scheduler: ReduceLROnPlateau (factor = 0.5, patience = 20)

Early Stopping: patience = 50

Gradient Clipping: max_norm = 1.0

Physics Warm-up: Linear increase of physics loss weight during first 30 epochs

Experimental Results

Evaluation on train/test split using 2995 days of data (2018–2026):

Metric	Value
R² (Coefficient of Determination)	0.9778
RMSE	1.29 °C
MAE	1.00 °C
MAPE	20.05%
Forecast Horizon	1 – 90 days

API Endpoints

Flask-based JSON API for frontend and secondary integration.

Endpoint	Method	Description	Example
/api/health	GET	Service health & data coverage	/api/health
/api/data/historical	GET	Historical data (optional days parameter)	/api/data/historical?days=180
/api/data/statistics	GET	Statistical metrics (mean, extremes, seasonal)	/api/data/statistics
/api/predict	GET	Future forecast (days=1~90)	/api/predict?days=30
/api/model_info	GET	Model & physics constraint description	/api/model_info
---

Project Structure

├── config.py                    # Global configuration (model, physics, paths)
├── sgli_l3_sst_extract_gee.py   # GEE data extraction script
├── prepare_sgli_data.py         # Raw CSV cleaning (filtering + interpolation)
├── real_data_collector.py       # Alternative data collector (Open-Meteo, etc.)
├── data_preprocessing.py        # Feature engineering (time/lag/physics features)
├── pinn_model.py                # PINN model + physics loss + trainer
├── training.py                  # Main training pipeline
├── evaluation.py                # Evaluation metrics calculation
├── web_app_real_data.py         # Flask web service main program
├── requirements.txt             # Dependencies
│
├── data/                        # Data directory (generated locally)
├── models/                      # Saved model weights
├── results/                     # Evaluation outputs & predictions CSV
├── plots/                       # Visualization charts
├── templates/                   # HTML templates (Chinese/English)
├── static/                      # Frontend CSS/JS assets
└── docs/                        # Detailed guides (data access, etc.)


Data Source

Primary Product: JAXA GCOM-C/SGLI L3 Sea Surface Temperature (V3)

Resolution & Frequency: ~4.6 km, daily

Time Coverage: Since 2018-01-22

Spatial Extent: Yellow Sea region (119°E–127°E, 32°N–41°N)

Access Platform: Google Earth Engine (JAXA/GCOM-C/L3/OCEAN/SST/V3)

Original data provided by Japan Aerospace Exploration Agency (JAXA). See docs/GCOM-C_SGLI_L3_SST_V3_数据接入与GitHub发布指南.md for detailed extraction steps.

Reproducibility Checklist & Troubleshooting

Reproducibility Checklist

To ensure local results match the description, follow this order:

1.data/sgli_yellow_sea_sst_daily.csv exists with complete date coverage

2.python prepare_sgli_data.py executes successfully

3.python training.py generates models/best_model.pth and results/model_evaluation.csv

4.python web_app_real_data.py starts without errors, shows Running on http://127.0.0.1:5000

5.All pages in browser load charts correctly

Troubleshooting

Issue	Common Cause	Solution
Training error: data file not found	CSV not in data/ directory	Verify path data/sgli_yellow_sea_sst_daily.csv
Sudden drop in metrics	Poor new data quality or short time span	Re-export data, ensure sufficient samples after valid_pixels filtering
Web starts but charts empty	API request failure or frontend cache	Refresh page, check browser console, visit /api/health
GEE export fails	Task not run or quota exceeded	Re-run task in GEE Tasks panel, shorten export time range
Port 5000 already in use	Another Flask process running	Kill existing process or change port before restart

Acknowledgments & Citation

JAXA — GCOM-C/SGLI L3 SST V3 satellite data product

Google Earth Engine — Cloud-based data extraction platform

PyTorch — Deep learning framework

Satellite Data Citation:

Kurihara, Y. (2020). GCOM-C/SGLI Sea Surface Temperature (SST) Algorithm Theoretical Basis Document (ATBD) (Version 2). JAXA.

Project Citation (Suggested):

[Your Name]. Yellow Sea SST Prediction Using Physics-Informed Neural Network[D]. [Your University], 2026.

Future Work

Multi-variable input (wind speed, salinity, sea surface height anomaly)

Seasonal/monthly stratified evaluation reports

Dockerized deployment with one-click startup

Baseline comparisons (LSTM / Transformer / LightGBM)

Prediction interval (confidence interval) output





- [ ] 增加置信区间（Prediction Interval）而非单点置信度
