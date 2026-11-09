https://xjtu-sy-bearing-phm-adarsh.streamlit.app/


XJTU-SY Bearing PHM Dashboard

# Industrial Bearing Prognostics & Health Management (PHM)

An end-to-end machine learning system and interactive web dashboard for **unsupervised anomaly detection**, **binary health classification**, and **Remaining Useful Life (RUL) estimation** built on the **XJTU-SY Bearing Dataset**.

---

## Project Architecture & Pipeline

```text
xjtu-sy-predictive-maintenance/
├── data/
│   ├── raw/                  # Raw vibration CSV files (XJTU-SY accelerated life tests)
│   └── processed/            # Extracted time & frequency domain features (.parquet)
├── notebooks/
│   ├── 01_eda_and_feature_extraction.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── features.py           # Signal processing engine (RMS, Kurtosis, FFT, Crest Factor)
├── saved_models/             # Trained model checkpoints (.joblib)
│   ├── xgb_classifier.joblib
│   ├── isolation_forest.joblib
│   ├── xgb_regressor.joblib
│   └── rf_regressor.joblib
├── app.py                    # Interactive Streamlit Web Dashboard
├── requirements.txt          # Python dependencies
└── README.md



Feature Engineering & Signal Processing
High-frequency vibration signals sampled at 25.6 kHz are parsed from raw 1-minute interval files. The feature extraction engine (src/features.py) computes robust statistical indicators across both horizontal and vertical axes:

Time-Domain Indicators: Root Mean Square (RMS) energy, Peak-to-Peak amplitude, Skewness, Excess Kurtosis (shock/impact indicator), Crest Factor, and Shape Factor.

Frequency-Domain Indicators (FFT): Mean spectral energy, standard deviation, peak frequency magnitude, and Spectral Centroid to track shifting vibrational frequencies as mechanical wear develops.

Models & Machine Learning Strategy
The system evaluates multiple algorithms across two core industrial tasks:

Anomaly Detection & Binary Health Classification:

Isolation Forest (Unsupervised): Trained exclusively on baseline normal operational data to detect anomalous vibration patterns without relying on failure annotations.

XGBoost Classifier (Supervised): Gradient-boosted decision tree classifier trained to distinguish between stable operation and degraded states.

Remaining Useful Life (RUL) Estimation:

XGBoost Regressor & Random Forest Regressor: Non-linear ensemble regressors trained to predict continuous degradation percentages (1.0 = healthy state down to 0.0 = complete bearing failure).
