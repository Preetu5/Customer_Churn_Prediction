"""Dataset download and validation utilities."""
from pathlib import Path

import pandas as pd
import requests

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_PATH = Path("data/raw/telco_customer_churn.csv")


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load cached IBM Telco data, downloading it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
    df = pd.read_csv(path)
    required = {"customerID", "Churn", "TotalCharges"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df
