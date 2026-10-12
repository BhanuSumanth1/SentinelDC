"""Platt calibration on an early validation window; no test fitting."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import HORIZON_HOURS, TARGET
from src.train import require_both_classes


def calibration_windows(validation):
    """Early third fits calibration; purge 24h before later policy assessment."""
    timestamps = pd.to_datetime(validation.timestamp, utc=True)
    unique = timestamps.sort_values().drop_duplicates()
    boundary = unique.iloc[len(unique) // 3]
    calibration = validation[timestamps < boundary]
    policy = validation[timestamps >= boundary + pd.Timedelta(hours=HORIZON_HOURS)]
    for name, frame in [("calibration", calibration), ("policy validation", policy)]:
        require_both_classes(frame, name)
    if calibration.empty or policy.empty:
        raise ValueError("Validation is too short for calibration, 24h purge, and policy selection.")
    return calibration, policy


def logit_scores(scores):
    p = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)


class CalibratedFailureModel:
    """Store the frozen TRAIN pipeline and a validation-fitted sigmoid mapping."""
    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrator = LogisticRegression(C=1.0, max_iter=300, random_state=42)
        self.classes_ = np.array([0, 1])

    def fit_calibration(self, raw_scores, labels):
        # No class balancing: calibration must learn observed prevalence.
        self.calibrator.fit(logit_scores(raw_scores), labels)
        if self.calibrator.coef_[0, 0] <= 0:
            raise ValueError("Calibration reverses score ranking; insufficient stable calibration evidence.")
        return self

    def calibrate(self, raw_scores):
        return self.calibrator.predict_proba(logit_scores(raw_scores))[:, 1]

    def predict_components(self, features):
        """One base-model inference supplies raw and calibrated evaluations."""
        raw = self.base_model.predict_proba(features)[:, 1]
        return raw, self.calibrate(raw)

    def predict_proba(self, features):
        _, p = self.predict_components(features)
        return np.column_stack([1 - p, p])
