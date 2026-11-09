import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Industrial Bearing PHM Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# --- Custom CSS to Reduce Metric Font Size ---
st.markdown("""
<style>
    /* Adjusts the main metric value font size */
    [data-testid="stMetricValue"] {
        font-size: 18px !important;
    }
    /* Adjusts the label font size above the metric */
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data and Models ---
@st.cache_data
def load_data():
    df_path = "data/processed/xjtu_sy_features.parquet"
    if os.path.exists(df_path):
        return pd.read_parquet(df_path)
    return None

@st.cache_resource
def load_or_train_models():
    os.makedirs("saved_models", exist_ok=True)
    models = {}
    
    paths = {
        "xgb_classifier": "saved_models/xgb_classifier.joblib",
        "isolation_forest": "saved_models/isolation_forest.joblib",
        "xgb_regressor": "saved_models/xgb_regressor.joblib",
        "rf_regressor": "saved_models/rf_regressor.joblib"
    }
    
    # Check if models are already saved locally
    if all(os.path.exists(p) for p in paths.values()):
        for k, p in paths.items():
            models[k] = joblib.load(p)
        return models
        
    # Otherwise, train them on the fly using processed parquet data
    df_path = "data/processed/xjtu_sy_features.parquet"
    if not os.path.exists(df_path):
        st.error("Processed dataset not found.")
        return None
        
    df = pd.read_parquet(df_path)
    df["Binary_Anomaly"] = (df["RUL_Percentage"] < 0.2).astype(int)
    drop_cols = ["Bearing", "FileIndex", "TotalLifetime", "RUL_Percentage", "Binary_Anomaly"]
    X = df.drop(columns=drop_cols)
    
    # Train Classifier
    y_class = df["Binary_Anomaly"]
    xgb_clf = XGBClassifier(n_estimators=50, random_state=42)
    xgb_clf.fit(X, y_class)
    joblib.dump(xgb_clf, paths["xgb_classifier"])
    models["xgb_classifier"] = xgb_clf
    
    # Train Isolation Forest (on normal samples)
    X_normal = X[y_class == 0]
    iso = IsolationForest(contamination=0.2, random_state=42)
    iso.fit(X_normal)
    joblib.dump(iso, paths["isolation_forest"])
    models["isolation_forest"] = iso
    
    # Train Regressors
    y_reg = df["RUL_Percentage"]
    xgb_reg = XGBRegressor(n_estimators=50, random_state=42)
    xgb_reg.fit(X, y_reg)
    joblib.dump(xgb_reg, paths["xgb_regressor"])
    models["xgb_regressor"] = xgb_reg
    
    rf_reg = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_reg.fit(X, y_reg)
    joblib.dump(rf_reg, paths["rf_regressor"])
    models["rf_regressor"] = rf_reg
    
    return models

df = load_data()
models = load_or_train_models()

# --- App Header ---
st.title("Industrial Bearing Prognostics & Health Management")
st.markdown("Predicting machinery anomalies, classifying operational health, and estimating Remaining Useful Life (RUL) using the XJTU-SY Dataset.")

if df is None:
    st.error("Processed dataset not found! Please run your feature extraction and modeling notebooks first.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("🎛️ Control Panel")
bearing_list = sorted(df["Bearing"].unique())
selected_bearing = st.sidebar.selectbox("Select Bearing Run", bearing_list)

# Filter data for selected bearing
bearing_df = df[df["Bearing"] == selected_bearing].sort_values("FileIndex")

# Time scrubber slider
max_file_idx = int(bearing_df["FileIndex"].max())
min_file_idx = int(bearing_df["FileIndex"].min())
selected_file_idx = st.sidebar.slider(
    "Scrub Lifecycle (Minutes/Files)",
    min_value=min_file_idx,
    max_value=max_file_idx,
    value=max_file_idx
)

# Extract specific row for prediction
current_row = bearing_df[bearing_df["FileIndex"] == selected_file_idx].iloc[0]
feature_cols = [c for c in df.columns if c not in ["Bearing", "FileIndex", "TotalLifetime", "RUL_Percentage", "Binary_Anomaly"]]
X_current = current_row[feature_cols].values.reshape(1, -1)

# --- Make Real-Time Predictions based on Selected Row ---
# Classifier predictions
xgb_clf_pred = models["xgb_classifier"].predict(X_current)[0] if "xgb_classifier" in models else 0
iso_pred_raw = models["isolation_forest"].predict(X_current)[0] if "isolation_forest" in models else 1
iso_clf_pred = 1 if iso_pred_raw == -1 else 0

# Regressor predictions
xgb_rul = models["xgb_regressor"].predict(X_current)[0] if "xgb_regressor" in models else 0.0
rf_rul = models["rf_regressor"].predict(X_current)[0] if "rf_regressor" in models else 0.0

# --- Top KPI Metrics Row ---
col1, col2, col3, col4 = st.columns(4)

status_color = "normal" if xgb_clf_pred == 0 else "inverse"
status_text = "🟢 Normal Operation" if xgb_clf_pred == 0 else "🔴 Anomaly / Degraded"

col1.metric("Current Health Status", status_text)
col2.metric("Isolation Forest Flag", "⚠️ Anomaly Detected" if iso_clf_pred == 1 else "✅ Normal")
col3.metric("XGBoost Predicted RUL", f"{max(0.0, xgb_rul)*100:.1f}%")
col4.metric("Random Forest Predicted RUL", f"{max(0.0, rf_rul)*100:.1f}%")

st.markdown("---")

# --- Dashboard Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📈 Vibration & Degradation Trends", 
    "🤖 Anomaly Detection Analysis", 
    "⏳ Remaining Useful Life (RUL)"
])

with tab1:
    st.subheader(f"Lifecycle Vibration Analysis for {selected_bearing}")
    st.markdown("Inspect how physical metrics like **RMS (Energy)** and **Kurtosis (Impact Shock)** evolve over time as the bearing wears down.")
    
    # Plot RMS and Kurtosis trends over time
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=bearing_df["FileIndex"], y=bearing_df["H_RMS"], name="Horizontal RMS", line=dict(color="orange")))
    fig_trend.add_trace(go.Scatter(x=bearing_df["FileIndex"], y=bearing_df["V_RMS"], name="Vertical RMS", line=dict(color="dodgerblue", dash="dash")))
    
    # Highlight current scrubber position
    fig_trend.add_vline(x=selected_file_idx, line_width=2, line_dash="dot", line_color="red", annotation_text="Selected Time")
    
    fig_trend.update_layout(title="RMS Energy Progression Over Time", xaxis_title="File Index (Minutes)", yaxis_title="RMS Value", template="plotly_white")
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.subheader("Anomaly Detection Performance")
    st.markdown("Comparing **XGBoost Supervised Classifier** vs **Isolation Forest Unsupervised Boundary Detection** across the bearing's lifecycle.")
    
    # Generate predictions for entire bearing series
    X_series = bearing_df[feature_cols]
    if "xgb_classifier" in models:
        bearing_df["XGB_Pred"] = models["xgb_classifier"].predict(X_series)
    if "isolation_forest" in models:
        bearing_df["ISO_Pred"] = np.where(models["isolation_forest"].predict(X_series) == -1, 1, 0)
        
    fig_anomaly = px.line(bearing_df, x="FileIndex", y=["H_Kurtosis", "V_Kurtosis"], title="Kurtosis Spikes (Shock Indicators) Over Time")
    fig_anomaly.add_vline(x=selected_file_idx, line_width=2, line_dash="dot", line_color="red")
    st.plotly_chart(fig_anomaly, use_container_width=True)

with tab3:
    st.subheader("Remaining Useful Life (RUL) Degradation Curves")
    st.markdown("Comparing true degradation trajectory against model estimations from **XGBoost Regressor** and **Random Forest Regressor**.")
    
    if "xgb_regressor" in models and "rf_regressor" in models:
        bearing_df["XGB_RUL_Pred"] = models["xgb_regressor"].predict(X_series)
        bearing_df["RF_RUL_Pred"] = models["rf_regressor"].predict(X_series)
        
        fig_rul = go.Figure()
        fig_rul.add_trace(go.Scatter(x=bearing_df["FileIndex"], y=bearing_df["RUL_Percentage"], name="True RUL", line=dict(color="black", width=3)))
        fig_rul.add_trace(go.Scatter(x=bearing_df["FileIndex"], y=bearing_df["XGB_RUL_Pred"], name="XGBoost Prediction", line=dict(color="green", dash="dash")))
        fig_rul.add_trace(go.Scatter(x=bearing_df["FileIndex"], y=bearing_df["RF_RUL_Pred"], name="Random Forest Prediction", line=dict(color="purple", dash="dot")))
        
        fig_rul.add_vline(x=selected_file_idx, line_width=2, line_dash="dot", line_color="red", annotation_text="Current View")
        fig_rul.update_layout(title="True vs. Predicted RUL Trajectory", xaxis_title="File Index", yaxis_title="RUL Percentage (1.0 = Healthy, 0.0 = Failure)", template="plotly_white")
        st.plotly_chart(fig_rul, use_container_width=True)
    else:
        st.warning("Regression models not found. Please ensure training script saved them to saved_models/.")