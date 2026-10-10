# Customer Churn Prediction

A reproducible machine learning project for exploring and predicting customer churn in the IBM Telco Customer Churn dataset.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

The first run downloads the public dataset from IBM's GitHub repository. No results are asserted in advance: training writes measured metrics to `reports/metrics.json` and a fitted pipeline to `models/churn_pipeline.joblib`.

## Project layout

- `src/data.py` downloads and validates the public dataset.
- `src/features.py` builds a leakage-aware preprocessing and modeling pipeline.
- `src/train.py` trains a stratified baseline, evaluates a held-out test set, and saves artifacts.
- `app.py` provides an interactive Streamlit churn-risk demo.
- `notebooks/01_exploration.md` outlines reproducible exploratory questions.

## Dataset

IBM Telco Customer Churn: https://github.com/IBM/telco-customer-churn/blob/master/WA_Fn-UseC_-Telco-Customer-Churn.csv

The downloader caches the original CSV in `data/raw/`. Do not commit the dataset or generated model artifacts. Review the dataset's terms and attribution before redistribution.

## Method

The target is `Churn`; `customerID` is excluded. `TotalCharges` is coerced to numeric and missing values are imputed inside the training pipeline. Numeric values are median-imputed and scaled; categorical values are mode-imputed and one-hot encoded. A stratified 80/20 split is used. Reported metrics include ROC AUC, PR AUC, accuracy, precision, recall, F1, and a confusion matrix. The classifier is a class-weighted logistic regression baseline.

This is a demonstration project, not a validated customer intervention system. Threshold choice should reflect business costs and be calibrated on representative data.

## Reproducibility

Set `RANDOM_STATE` in `src/train.py` to change the seeded split. Run `python -m src.train` after installing requirements. The app prompts you to train if the model artifact does not exist.

## GitHub

After creating an empty repository in your own GitHub account, connect it locally:

```bash
git init
git add .
git commit -m "Build customer churn prediction project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-prediction.git
git push -u origin main
```

Authenticate with GitHub CLI (`gh auth login`) or your configured Git credential manager; never paste tokens or passwords into chat.
