"""Independent normal-behavior models fitted exclusively on healthy TRAIN rows."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.config import FEATURE_COLUMNS, MODEL_DIR, PROCESSED_PATH, REPORT_DIR, SEED, TARGET
from src.train import temporal_split

ANOMALY_FEATURES = list(FEATURE_COLUMNS)
DETECTOR_NAMES = ['robust_statistical', 'isolation_forest']
ISOLATION_SAMPLE_SIZE = 50_000


class RobustStatisticalDetector(BaseEstimator):
    """Mean of the largest five clipped absolute robust feature deviations.

    Dispersion is max(1.4826*MAD, IQR/1.349, 0.1) in RobustScaler units.
    The floor handles constant/binary features; each deviation is capped at 10.
    Thus one extreme feature contributes at most 2 to a five-feature mean.
    """
    def __init__(self, top_k=5, clip=10.0, scale_floor=0.1):
        self.top_k = top_k
        self.clip = clip
        self.scale_floor = scale_floor

    def fit(self, x, y=None):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError('Robust statistics require nonempty finite training features.')
        self.center_ = np.median(x, axis=0)
        self.mad_ = np.median(np.abs(x - self.center_), axis=0)
        iqr = np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)
        self.dispersion_ = np.maximum.reduce([1.4826 * self.mad_, iqr / 1.349, np.full(x.shape[1], self.scale_floor)])
        self.n_features_in_ = x.shape[1]
        return self

    def deviations(self, x):
        return np.minimum(np.abs(np.asarray(x) - self.center_) / self.dispersion_, self.clip)

    def anomaly_score(self, x):
        deviations = self.deviations(x)
        k = min(self.top_k, deviations.shape[1])
        return np.partition(deviations, -k, axis=1)[:, -k:].mean(axis=1)

    def explain(self, x):
        deviations = self.deviations(np.asarray(x).reshape(1, -1))[0]
        k = min(self.top_k, len(deviations))
        order = np.argsort(-deviations, kind='stable')
        return {'detector': 'robust_statistical', 'overall_anomaly_score': float(deviations[order[:k]].mean()),
                'score_is_failure_probability': False,
                'aggregation': f'mean of top {k} absolute robust deviations clipped at {self.clip}',
                'largest_deviations': [{'feature': ANOMALY_FEATURES[i], 'clipped_robust_deviation': float(deviations[i]),
                    'contribution_to_overall_score': float(deviations[i] / k)} for i in order[:k]]}


def load_anomaly_splits():
    """Use exactly the saved chronological supervised splits without changing them."""
    data = pd.read_csv(PROCESSED_PATH)
    training, validation, test, split = temporal_split(data)
    if split != json.loads((REPORT_DIR / 'split_metadata.json').read_text()):
        raise ValueError('Existing chronological split metadata differs; do not silently change the benchmark.')
    return training, validation, test, split


def fit_anomaly_models(training, model_dir=MODEL_DIR):
    healthy = training.loc[training[TARGET].eq(0), ANOMALY_FEATURES]
    if healthy.empty:
        raise ValueError('No healthy TRAIN observations available.')
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', RobustScaler(quantile_range=(25, 75))),
    ])
    transformed = preprocessor.fit_transform(healthy).astype(np.float32)
    robust = RobustStatisticalDetector().fit(transformed)
    rng = np.random.default_rng(SEED)
    positions = np.sort(rng.choice(len(healthy), min(ISOLATION_SAMPLE_SIZE, len(healthy)), replace=False))
    isolation = IsolationForest(n_estimators=200, max_samples=min(4096, len(positions)),
        max_features=1.0, contamination='auto', bootstrap=False, n_jobs=-1, random_state=SEED)
    isolation.fit(transformed[positions])
    model_dir = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, model_dir / 'anomaly_preprocessor.joblib')
    joblib.dump(robust, model_dir / 'anomaly_robust_detector.joblib')
    joblib.dump(isolation, model_dir / 'isolation_forest.joblib')
    statistics = {'feature_names': ANOMALY_FEATURES, 'center_in_scaled_units': robust.center_.tolist(),
        'mad_in_scaled_units': robust.mad_.tolist(), 'dispersion_in_scaled_units': robust.dispersion_.tolist(),
        'top_k': robust.top_k, 'deviation_clip': robust.clip, 'dispersion_floor': robust.scale_floor,
        'healthy_train_medians_original_units': preprocessor.named_steps['imputer'].statistics_.tolist()}
    (model_dir / 'anomaly_statistics.json').write_text(json.dumps(statistics, indent=2) + '\n')
    metadata = {'fit_split': 'TRAIN', 'fit_filter': 'failure_within_24h == 0',
        'healthy_training_rows': len(healthy), 'isolation_training_sample_rows': len(positions),
        'isolation_max_samples_per_tree': int(isolation.max_samples_), 'seed': SEED,
        'sample_method': 'uniform without replacement over healthy TRAIN rows; chronological positions sorted after sampling',
        'sample_positions_sha256': hashlib.sha256(positions.tobytes()).hexdigest(),
        'training_start': str(training.timestamp.min()), 'training_end': str(training.timestamp.max()),
        'features': ANOMALY_FEATURES,
        'score_convention': 'larger is more anomalous; robust top-five mean, IsolationForest negative score_samples',
        'contamination_note': 'Internal contamination offset and predict/decision_function are never used as alert thresholds.'}
    (model_dir / 'anomaly_training_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return preprocessor, {'robust_statistical': robust, 'isolation_forest': isolation}, metadata


def anomaly_scores(detector_name, detector, transformed):
    scores = detector.anomaly_score(transformed) if detector_name == 'robust_statistical' else -detector.score_samples(transformed)
    if not np.isfinite(scores).all():
        raise ValueError('Anomaly scores must be finite.')
    return np.asarray(scores, dtype=float)
