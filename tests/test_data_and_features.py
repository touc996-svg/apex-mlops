import pandas as pd
import pytest

from apex.data import generate_dataset, load_dataset
from apex.features import build_candidates
from apex.schema import FEATURE_COLUMNS, TARGET_COLUMN


def test_synthetic_dataset_is_reproducible_and_matches_contract():
    first = generate_dataset(rows=300, random_state=7)
    second = generate_dataset(rows=300, random_state=7)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert set(first[TARGET_COLUMN].unique()) == {0, 1}


def test_dataset_loader_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"tenure_months": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_feature_pipelines_include_both_model_candidates():
    candidates = build_candidates()
    assert set(candidates) == {"logistic_regression", "random_forest"}
    assert all(hasattr(model, "fit") and hasattr(model, "predict_proba") for model in candidates.values())
