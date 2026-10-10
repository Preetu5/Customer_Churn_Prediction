# Exploratory analysis guide

Run `python -m src.train` to fetch the dataset and establish the first reproducible baseline. Then explore the cached `data/raw/telco_customer_churn.csv` in a notebook or Python session.

Suggested questions:

1. What is the class balance, and how does churn vary by contract type and tenure?
2. How do monthly charges and service bundles differ between churned and retained customers?
3. Are there missing or unusual values in `TotalCharges` and other fields?
4. How sensitive are precision and recall to the classification threshold?
5. Does performance remain stable across customer segments and alternative stratified splits?

Keep the test set out of feature selection and threshold tuning. Report uncertainty and segment sizes alongside metrics.
