import pandas as pd

from apex.data import generate_dataset
from apex.monitoring import evaluate_drift, numeric_psi


def test_psi_is_near_zero_for_identical_numeric_distributions():
    values = pd.Series([1, 2, 3, 4, 5, 6] * 20)
    assert numeric_psi(values, values) < 1e-8


def test_drift_monitor_reports_shifted_feature():
    reference = generate_dataset(rows=500, random_state=12).drop(columns=["churn"])
    current = reference.copy()
    current["monthly_charges"] = current["monthly_charges"] + 80
    result = evaluate_drift(reference, current, threshold=0.20)

    assert result["status"] == "DRIFT_DETECTED"
    assert "monthly_charges" in result["drifted_features"]
    assert result["features"]["monthly_charges"]["psi"] >= 0.20


def test_drift_monitor_reports_stable_same_sample():
    reference = generate_dataset(rows=500, random_state=18).drop(columns=["churn"])
    result = evaluate_drift(reference, reference.copy())
    assert result["status"] == "STABLE"
