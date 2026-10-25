# 🌦️ Global Weather Trend Forecasting & Environmental Analytics
> **Data Science Internship Assessment** | **Product Manager Accelerator (PMA)**

---

## 🎯 PM Accelerator Mission

> *"The Product Manager Accelerator Program is designed to support PM professionals through every stage of their careers. From students seeking entry-level jobs to Director-level executives looking to transition into more senior leadership roles, we help our members take their careers to the next level."*  
> — **[Product Manager Accelerator](https://www.pmaccelerator.io/)**

---

## 📌 Executive Summary

This repository contains the end-to-end Data Science assessment addressing global meteorological trends, environmental impact, anomaly detection, and predictive forecasting. Leveraging the **Global Weather Repository**, this project demonstrates robust data engineering, exploratory data analysis (EDA), time-series modeling using timestamped observations (`last_updated`), multi-model benchmarking, ensemble forecasting, and geospatial/environmental analysis.

---

## 📂 Repository Structure

```text
Data-Science-Intern-Assessment/
├── LICENSE
├── README.md                               # Comprehensive assessment documentation
├── requirements.txt                        # Python environment dependencies
└── weather-trend-forecasting/
    ├── eda_and_preprocessing.ipynb         # Data cleaning, EDA, anomaly detection & spatial analysis
    └── forecasting_and_evaluation.ipynb    # Time-series feature engineering, modeling & ensembling
```

---

## 📊 Dataset Overview

- **Source**: [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)
- **Granularity & Scope**: Real-time hourly weather and air quality observations across major world cities and countries.
- **Key Attributes**:
  - **Temporal**: `last_updated`, `last_updated_epoch`
  - **Meteorological**: `temperature_celsius`, `feels_like_celsius`, `precip_mm`, `humidity`, `wind_kph`, `pressure_mb`, `uv_index`, `visibility_km`
  - **Air Quality**: `air_quality_PM2.5`, `air_quality_PM10`, `air_quality_us-epa-index`, `air_quality_gb-defra-index`, `air_quality_Carbon_Monoxide`, `air_quality_Ozone`
  - **Geographic**: `country`, `location_name`, `latitude`, `longitude`, `timezone`

---

## 🛠️ Assessment Breakdown & Methodology

### 1. 📋 Basic Assessment

#### A. Data Cleaning & Preprocessing
- **Missing Value Imputation**: Evaluated missing rates across columns; resolved null records using median values for skewed numerical distributions and mode imputation for categorical attributes.
- **Outlier Detection & Handling**: Analyzed outliers via Interquartile Range (IQR) and Z-score methods across continuous metrics (`temperature_celsius`, `precip_mm`, `air_quality_PM2.5`). Applied Winsorization / bounded clipping to preserve signal without skewing model gradient updates.
- **Feature Normalization**: Standardized feature scales using `StandardScaler` / `MinMaxScaler` for scale-sensitive regressors and distance metrics.
- **Categorical Encoding**: Transformed categorical features (`condition_text`, `country`, `is_day`) using label and one-hot encoding.

#### B. Exploratory Data Analysis (EDA)
- **Trend & Distribution Analysis**: Evaluated global distributions of temperature, humidity, and atmospheric pressure. 
- **Temperature & Precipitation Visualizations**:
  - Histograms and Kernel Density Estimation (KDE) plots analyzing day vs. night temperature shifts.
  - Box plots identifying regional extremes and diurnal ranges.
  - Precipitation distributions highlighting heavy-rainfall clusters versus arid zones.
- **Correlation Analysis**: Correlation heatmaps revealing interactions among ambient temperature, wind speeds, humidity, and atmospheric particulate matter (`PM2.5`, `PM10`).

#### C. Model Building & Time-Series Forecasting
- **Time-Series Parsing**: Converted and decomposed `last_updated` into cyclical time components (`hour`, `day`, `day_of_week`, `is_day`, lag features).
- **Baseline Forecasting Model**: Built baseline regression forecasters (Random Forest Regressor, Ridge Regression) targeting environmental and meteorological indicators.
- **Evaluation Metrics**: Evaluated performance using standard metrics:
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Squared Error)
  - **R² Score** (Coefficient of Determination)

---

### 2. 🚀 Advanced Assessment

#### A. Advanced EDA & Anomaly Detection
- **Anomaly Detection Algorithms**: Implemented **Isolation Forest** and multivariate IQR thresholds to detect anomalous weather phenomena (e.g., sudden temperature inversions, extreme PM2.5 spikes, storm events).
- **Outlier Profiling**: Segregated identified anomalies to investigate geographic hotspots and hazardous air quality events.

#### B. Forecasting with Multiple Models & Ensembling
- **Multi-Model Comparison**: Evaluated and compared candidate models:
  1. *Random Forest Regressor* (Non-linear decision boundaries, handles tabular interactions well)
  2. *Gradient Boosting / LightGBM* (High performance on structured tabular time features)
  3. *Linear / Ridge Regression* (Regularized baseline)
- **Model Ensembling**: Built an ensemble pipeline (Weighted Average / Voting Regressor) combining tree-based estimators and regularized linear models, achieving higher generalization and lower validation MAE/RMSE.

```
+---------------------------+       +----------------------------+
|  Random Forest Regressor  |       | Gradient Boosting Regressor|
+-------------+-------------+       +--------------+-------------+
              \                                    /
               \                                  /
                v                                v
            +-----------------------------------------+
            |      Ensemble Weighted Regressor        |
            +--------------------+--------------------+
                                 |
                                 v
                       Final Forecast Output
```

#### C. Unique Analyses
1. **Climate Analysis**:
   - Analyzed long-term temperature baselines and variance across tropical, temperate, and polar climate zones.
   - Identified diurnal fluctuations across continental and maritime geographic zones.
2. **Environmental Impact & Air Quality**:
   - Explored relationships between weather conditions (humidity, wind speed, pressure, temperature) and pollution metrics (`PM2.5`, `PM10`, US-EPA Index).
   - Uncovered negative correlation between wind speed and local particulate concentration (wind dispersal effect).
3. **Feature Importance Analysis**:
   - Assessed feature contributions using **Random Forest Gini Importance (MDI)** and **Permutation Feature Importance**.
   - Identified primary drivers for atmospheric pollution and temperature dynamics.
4. **Spatial Analysis**:
   - Visualized geographic distributions with coordinate scatter maps (latitude vs. longitude).
   - Highlighted global pollution epicenters and extreme weather zones.
5. **Geographical Patterns**:
   - Compared aggregate weather metrics across countries and continents.
   - Identified macro-level patterns differentiating arid continental zones from coastal maritime regions.

---

## 📈 Key Findings & Insights

| Metric / Domain | Observation | Strategic Takeaway |
| :--- | :--- | :--- |
| **Temperature Patterns** | Concentrated around moderate bands (15°C–25°C) with clear diurnal day/night bifurcation. | Useful for energy grid load forecasting and seasonal demand planning. |
| **Air Quality (`PM2.5`)** | Significant regional variation; extreme spikes clustered in high-density industrial and urban hubs. | Identifies priority areas for environmental policy intervention and alert systems. |
| **Wind & Pollution** | Stagnant air and low wind speeds correlate strongly with elevated `PM2.5` & `PM10` levels. | Dispersion models can predict smog accumulation hours in advance. |
| **Ensemble Model** | Combining Random Forest and Gradient Boosting reduced prediction MAE relative to single-model baselines. | Ensemble approach delivers robust accuracy for real-world automated alert systems. |

---

## 💻 Setup & Execution Guide

### 1. Prerequisites
- Python 3.10+ installed
- Git installed

### 2. Clone Repository
```bash
git clone https://github.com/Omar-Mahrous-am/Data-Science-Intern-Assessment.git
cd Data-Science-Intern-Assessment
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Notebooks
Launch Jupyter Lab or Jupyter Notebook:
```bash
jupyter notebook
```
- Open `weather-trend-forecasting/eda_and_preprocessing.ipynb` to inspect data cleaning, exploratory data analysis, anomaly detection, and spatial patterns.
- Open `weather-trend-forecasting/forecasting_and_evaluation.ipynb` to review feature engineering, model training, multi-model evaluation, and ensembling.

---

## ✅ Assessment Deliverables Checklist

- [x] **PM Accelerator Mission**: Highlighted prominently at the beginning of the documentation.
- [x] **Data Cleaning & Preprocessing**: Missing values addressed, outliers handled, normalization & encoding applied.
- [x] **Exploratory Data Analysis (EDA)**: Temperature and precipitation patterns visualized; correlation analysis performed.
- [x] **Model Building**: Time series features extracted from `last_updated`; forecasting model built and evaluated with metrics.
- [x] **Advanced EDA & Anomaly Detection**: Anomaly detection implemented and analyzed.
- [x] **Multi-Model Forecasting & Ensembling**: Models compared and combined into an ensemble for enhanced forecast precision.
- [x] **Unique Analyses**:
  - [x] Climate Analysis
  - [x] Environmental Impact (Air Quality)
  - [x] Feature Importance
  - [x] Spatial Analysis
  - [x] Geographical Patterns across regions and countries
- [x] **Documentation & Reproducibility**: Structured, clear, and comprehensive `README.md` and `requirements.txt`.

---

## 📜 License
Distributed under the terms specified in the [LICENSE](file:///d:/Projects/Data-Science-Intern-Assessment/LICENSE) file.
