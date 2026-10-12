"""Verification entry point must bypass every training/selection/evaluation stage."""
import pytest
from src import verify_frozen
import run_pipeline


def test_verify_only_dispatch_never_retrains_or_evaluates(monkeypatch):
    calls = []
    monkeypatch.setattr(verify_frozen, 'verify_frozen_artifacts', lambda: calls.append('verified'))
    def forbidden(*args, **kwargs):
        pytest.fail('Verification attempted to modify or rescore the benchmark')
    for module, name in [(run_pipeline.train, 'main'), (run_pipeline.generate_data, 'main'),
                         (run_pipeline.preprocess, 'main'), (run_pipeline.evaluate, 'select_on_validation'),
                         (run_pipeline.evaluate, 'evaluate_final_test')]:
        monkeypatch.setattr(module, name, forbidden)
    run_pipeline.main(['--verify-only'])
    assert calls == ['verified']


def test_saved_metric_mismatch_fails_clearly():
    with pytest.raises(ValueError, match='Saved metric differs'):
        verify_frozen.same_metrics({'detected_failure_events': 36}, {'detected_failure_events': 37})
