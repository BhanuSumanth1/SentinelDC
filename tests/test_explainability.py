"""Exercise frozen-model explanations without fitting or changing artifacts."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import MODEL_DIR, PROCESSED_PATH
from src.calibration import CalibratedFailureModel
from src.explainability import (FrozenTreeExplainer, FORBIDDEN, feature_factors, local_record,
                                save_global, save_live_view, verify_hashes)


@pytest.fixture(scope='module')
def frozen_inputs():
    features = json.loads((MODEL_DIR/'feature_columns.json').read_text())
    model = joblib.load(MODEL_DIR/'best_failure_model.joblib')
    rows = pd.read_csv(PROCESSED_PATH, nrows=8)
    return model, features, rows


def test_underlying_lightgbm_and_feature_order(frozen_inputs):
    model, features, rows = frozen_inputs
    helper = FrozenTreeExplainer(model, features)
    assert helper.estimator is model.base_model.named_steps['classifier']
    assert helper.estimator.__class__.__name__ == 'LGBMClassifier'
    assert helper.features == list(model.base_model.feature_names_in_)
    with pytest.raises(ValueError, match='ordering'):
        FrozenTreeExplainer(model, features[::-1])
    with pytest.raises(ValueError, match='leakage'):
        FrozenTreeExplainer(model, features+['failure_type'])


def test_explainer_never_fits_and_is_finite_additive(frozen_inputs, monkeypatch):
    model, features, rows = frozen_inputs
    before = hashlib.sha256((MODEL_DIR/'best_failure_model.joblib').read_bytes()).hexdigest()
    def forbidden(*args, **kwargs):
        pytest.fail('Explainability attempted fitting')
    monkeypatch.setattr(model.base_model, 'fit', forbidden)
    monkeypatch.setattr(model.calibrator, 'fit', forbidden)
    monkeypatch.setattr(model, 'fit_calibration', forbidden)
    for _, step in model.base_model.steps:
        monkeypatch.setattr(step, 'fit', forbidden)
    helper = FrozenTreeExplainer(model, features)
    explanation, probabilities = helper.explain(rows, batch_size=3)
    assert np.isfinite(explanation.values).all()
    assert explanation.values.shape == (8, len(features))
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    transformed = model.base_model[:-1].transform(rows[features])
    margin = helper.estimator.booster_.predict(transformed, raw_score=True)
    np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1), margin)
    assert hashlib.sha256((MODEL_DIR/'best_failure_model.joblib').read_bytes()).hexdigest() == before


def test_metadata_excluded_and_column_order_restored(frozen_inputs):
    model, features, rows = frozen_inputs
    helper = FrozenTreeExplainer(model, features)
    original, p = helper.explain(rows)
    altered = rows[rows.columns[::-1]].copy()
    for column in FORBIDDEN:
        altered[column] = 'deliberately_changed_metadata'
    changed, q = helper.explain(altered)
    np.testing.assert_array_equal(original.values, changed.values)
    np.testing.assert_array_equal(p, q)
    assert not FORBIDDEN.intersection(original.feature_names)


def test_local_direction_names_and_required_outputs(frozen_inputs, tmp_path):
    model, features, rows = frozen_inputs
    explanation, probability = FrozenTreeExplainer(model, features).explain(rows)
    metadata = dict(server_id='server_0000', timestamp='2025-01-01T00:00:00Z',
                    hybrid_risk_level='HIGH', anomaly_score=3., anomaly_percentile=.8)
    record = local_record(metadata, explanation[0], probability[0], .037, features)
    for factor in record['top_risk_drivers'] + record['protective_factors']:
        assert factor['feature'] in features
        assert factor['direction'] in ['INCREASES_RISK', 'DECREASES_RISK']
        assert (factor['shap_value'] > 0) == (factor['direction'] == 'INCREASES_RISK')
    save_live_view([record], tmp_path/'explainability_live_view.csv')
    importance = save_global(explanation, tmp_path, tmp_path)
    assert (tmp_path/'shap_global_importance.csv').exists()
    assert (tmp_path/'explainability_live_view.csv').exists()
    assert importance.mean_abs_shap.is_monotonic_decreasing
    assert set(importance.feature) == set(features)
    assert len(pd.read_csv(tmp_path/'explainability_live_view.csv')) == 1


def test_protected_hash_changes_fail(tmp_path):
    path = tmp_path/'frozen'; path.write_bytes(b'frozen')
    expected = {'frozen': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert verify_hashes(tmp_path, expected) == 1
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Protected'):
        verify_hashes(tmp_path, expected)
