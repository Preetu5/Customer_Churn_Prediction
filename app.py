"""Interactive customer churn risk demo."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data import load_data
from src.train import main as train_model

st.set_page_config(page_title="Customer Churn Lab", page_icon="📉", layout="wide")
st.title("Customer Churn Prediction")
st.caption("An interpretable baseline built on the public IBM Telco Customer Churn dataset.")

model_path = Path("models/churn_pipeline.joblib")
if not model_path.exists():
    st.info("Train the baseline model before scoring a customer.")
    if st.button("Train model"):
        with st.spinner("Downloading data and training model…"):
            train_model()
        st.rerun()
    st.stop()

model = joblib.load(model_path)
data = load_data()
X = data.drop(columns=["customerID", "Churn"])
st.sidebar.header("Customer profile")
with st.sidebar.form("customer"):
    values = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            values[col] = st.number_input(col, value=float(X[col].median()))
        else:
            choices = sorted(X[col].dropna().astype(str).unique().tolist())
            default = "No" if "No" in choices else choices[0]
            values[col] = st.selectbox(col, choices, index=choices.index(default))
    submitted = st.form_submit_button("Estimate churn risk", type="primary")

if submitted:
    probability = float(model.predict_proba(pd.DataFrame([values]))[0, 1])
    left, right = st.columns(2)
    left.metric("Estimated churn probability", f"{probability:.1%}")
    right.metric("Risk band", "Higher" if probability >= 0.5 else "Lower")
    st.progress(probability)
    st.caption("The 50% split is a demo threshold, not a business recommendation. Validate on representative data before acting on scores.")

with st.expander("Latest held-out evaluation"):
    metrics_path = Path("reports/metrics.json")
    if metrics_path.exists():
        metrics = pd.read_json(metrics_path, typ="series").drop(labels=["confusion_matrix"]).to_frame("Value")
        st.dataframe(metrics, use_container_width=True)
    else:
        st.write("Train the model to create evaluation metrics.")
