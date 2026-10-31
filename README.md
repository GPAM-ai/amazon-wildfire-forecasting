# Forecasting Wildfires in the Amazon: Integrating Lagged Dependence Analysis, Complex Networks, and Machine Learning

[![Conference](https://img.shields.io/badge/Accepted-IEEE%2FACM%20BDCAT%202026-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

Official repository for the paper **"Forecasting Wildfires in the Amazon: Integrating Lagged Dependence Analysis, Complex Networks, and Machine Learning"**, accepted at the IEEE/ACM International Conference on Big Data Computing, Applications and Technologies (BDCAT 2026).

**Authors:** Leticia A. Moreira¹, Lael Hayashi¹, Manoel F. Cardoso², and Didier A. Vega-Oliveros¹  
*¹Institute of Science and Technology, Federal University of São Paulo (UNIFESP), São José dos Campos, Brazil*  
*²Earth System Science Center, National Institute for Space Research (INPE), São José dos Campos, Brazil*

---

## 📌 Overview
The volume of spatiotemporal environmental data has grown significantly, requiring advanced methods to support ecological monitoring and disaster management. This repository contains the data harmonization pipeline, complex network extraction algorithms, lagged conditional-dependence analysis, and machine learning models used to forecast the continuous magnitude of forest fire outbreaks in the Amazon basin (2003–2025).

The methodological pipeline integrates three main axes:
1. **Chronological Networks (Chronnets):** Modeling macroscopic spatial fire co-occurrence over a 0.5° x 0.5° grid (1,600 cells) using NASA MODIS satellite detections (confidence > 70%).
2. **Lagged Conditional Dependence (PCMCI):** Identifying statistically supported delayed associations between macroclimatic anomalies (ENSO/ONI), local meteorology, anthropogenic exposure, and fire counts across 34 municipality-matched station cells.
3. **Multi-Horizon Magnitude Forecasting:** Benchmarking six model families—**Negative Binomial (NB2), Random Forest (RF), XGBoost, Multilayer Perceptron (MLP), SARIMAX/SARIMA, and 1-D Convolutional Neural Networks (1-D CNN)**—under a rolling chronological validation design across training windows of 12, 24, 48, and 72 months and forecast horizons from 1 to 4 months (h1, h2, h3 and h4).

---

## 📂 Repository Structure
```text
├── datasets/
│   ├── fire_graph_metrics.csv             # NASA MODIS fire hotspots and Chronnet topological metrics (1,600 grid cells, 2003–2025)
│   └── integrated_analysis.csv            # Unified panel dataset (34 municipalities, climatic, socioeconomic, macroeconomic, and fire data)
├── scripts/
│   ├── 01_network_extraction.ipynb        # Chronnet construction and topological metrics (degree, Shannon entropy, K-core, communities)
│   ├── 02_correlation.ipynb               # Annual aggregation and Spearman correlation matrix generation
│   ├── 03_causality_pcmci.ipynb           # Lagged conditional-dependence tests via Tigramite (ParCorr, GPDC, Spearman)
│   ├── 04_prediction_models/              # Multi-horizon rolling-window forecasting benchmark
│       ├── full_features/                 # Models trained with the 13 climatic, socioeconomic, and macroeconomic covariates
│       │   ├── nb2_full.py
│       │   ├── random_forest_full.py
│       │   ├── xgboost_full.py
│       │   ├── mlp_full.py
│       │   ├── sarimax_full.py
│       │   └── cnn1d_full.py
│       └── fire_history_only/             # Models trained exclusively on historical fire counts (univariate/endogenous setup)
│           ├── nb2_fire_only.py
│           ├── random_forest_fire_only.py
│           ├── xgboost_fire_only.py
│           ├── mlp_fire_only.py
│           ├── sarima_fire_only.py
│           └── cnn1d_fire_only.py
├── requirements.txt                       # Python dependencies
└── README.md
