"""Shared feature contract used by training, serving, and drift monitoring."""

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "usage_gb",
]
CATEGORICAL_FEATURES = ["contract_type", "payment_method"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "churn"
MODEL_CANDIDATES = ("logistic_regression", "random_forest")
