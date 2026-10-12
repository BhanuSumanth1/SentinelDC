"""Post-hoc explanations of a frozen LightGBM margin, never causal diagnosis."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
import shap

FORBIDDEN = {'failure_within_24h', 'failure_type', 'failure_event_id',
             'actual_failure_timestamp', 'server_id', 'rack_id', 'switch_id', 'datacenter_id'}
DISPLAY_NAMES = {
    'cpu_utilization': 'CPU utilization', 'memory_utilization': 'Memory utilization',
    'disk_utilization': 'Disk utilization', 'gpu_utilization': 'GPU utilization',
    'temperature_c': 'Temperature', 'disk_read_mb_s': 'Disk read throughput',
    'disk_write_mb_s': 'Disk write throughput', 'network_in_mb_s': 'Inbound network throughput',
    'network_out_mb_s': 'Outbound network throughput', 'packet_loss_pct': 'Packet loss',
    'network_latency_ms': 'Network latency', 'smart_error_count': 'SMART error count',
    'fan_speed_rpm': 'Fan speed', 'power_watts': 'Power consumption', 'uptime_days': 'Server uptime',
    'hour_of_day': 'Hour of day', 'day_of_week': 'Day of week',
    'cpu_memory_interaction': 'CPU and memory interaction',
    'temperature_deviation': 'Temperature deviation', 'network_total_mb_s': 'Total network throughput',
    'disk_total_mb_s': 'Total disk throughput', 'error_rate_indicator': 'Error rate indicator',
    'high_temperature_flag': 'High temperature indicator', 'high_packet_loss_flag': 'High packet loss indicator',
    'temperature_rolling_mean_6h': 'Average temperature (6h)',
    'temperature_change_6h': 'Temperature trend (6h)',
    'packet_loss_rolling_mean_6h': 'Average packet loss (6h)',
    'latency_rolling_mean_6h': 'Average network latency (6h)',
    'smart_error_change_12h': 'SMART error growth (12h)',
    'cpu_rolling_mean_6h': 'Average CPU utilization (6h)',
    'memory_rolling_mean_6h': 'Average memory utilization (6h)',
    'power_change_6h': 'Power change (6h)',
}
DISCLAIMER = ('SHAP contributions explain the frozen base model raw log-odds, not the calibration '
              'layer. They do not establish that a feature caused a hardware failure.')


def verify_hashes(root, expected):
    changed = [name for name, digest in expected.items()
               if not (Path(root)/name).is_file()
               or hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() != digest]
    if changed:
        raise ValueError(f'Protected artifacts changed: {changed}')
    return len(expected)


class FrozenTreeExplainer:
    """Validate the saved input schema, transform only, and explain the positive margin."""
    def __init__(self, calibrated_model, features):
        self.features = list(features)
        if len(set(self.features)) != len(self.features) or FORBIDDEN.intersection(self.features):
            raise ValueError('Invalid or leakage-containing explanation features.')
        self.calibrated_model = calibrated_model
        self.pipeline = getattr(calibrated_model, 'base_model', None)
        if not isinstance(self.pipeline, Pipeline) or not isinstance(self.pipeline[-1], LGBMClassifier):
            raise ValueError('Expected saved calibrated LightGBM pipeline.')
        self.estimator = self.pipeline[-1]
        if list(self.pipeline.feature_names_in_) != self.features:
            raise ValueError('Feature ordering differs from frozen model.')
        if list(self.pipeline[:-1].get_feature_names_out()) != self.features:
            raise ValueError('Preprocessing changes feature ordering or dimension.')
        if self.estimator.n_features_in_ != len(self.features) or list(self.estimator.classes_) != [0, 1]:
            raise ValueError('Expected binary positive-class LightGBM model.')
        self.explainer = shap.TreeExplainer(self.estimator, model_output='raw',
                                           feature_perturbation='tree_path_dependent')

    def explain(self, observations, batch_size=512):
        # All identifiers/outcome fields are discarded before preprocessing and SHAP.
        inputs = observations.loc[:, self.features]
        transformed = np.asarray(self.pipeline[:-1].transform(inputs), dtype=float)
        if not np.isfinite(transformed).all():
            raise ValueError('Non-finite model input after frozen preprocessing.')
        arrays = []
        for start in range(0, len(transformed), batch_size):
            values = self.explainer.shap_values(transformed[start:start+batch_size], check_additivity=True)
            if isinstance(values, list):
                values = values[1]
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, 1]
            arrays.append(values)
        values = np.concatenate(arrays)
        base = np.asarray(self.explainer.expected_value).reshape(-1)
        base = float(base[-1])
        if values.shape != transformed.shape or not np.isfinite(values).all() or not np.isfinite(base):
            raise ValueError('SHAP shape or finite-value check failed.')
        margin = self.estimator.booster_.predict(transformed, raw_score=True)
        if not np.allclose(base + values.sum(axis=1), margin, rtol=1e-6, atol=1e-7):
            raise ValueError('SHAP additivity does not reproduce the frozen model margin.')
        probability = self.calibrated_model.predict_proba(inputs)[:, 1]
        return shap.Explanation(values=values, base_values=np.full(len(values), base),
                                data=transformed, feature_names=self.features), probability


def feature_factors(explanation, features, limit=5):
    values = np.asarray(explanation.values)
    def factor(i):
        return {'feature': features[i], 'feature_name': features[i],
                'display_name': DISPLAY_NAMES.get(features[i], features[i].replace('_', ' ').capitalize()),
                'value': float(explanation.data[i]), 'feature_value': float(explanation.data[i]),
                'shap_value': float(values[i]),
                'direction': 'INCREASES_RISK' if values[i] > 0 else 'DECREASES_RISK'}
    positive = [factor(i) for i in np.argsort(-values, kind='stable') if values[i] > 0][:limit]
    negative = [factor(i) for i in np.argsort(values, kind='stable') if values[i] < 0][:limit]
    return positive, negative


def local_record(metadata, explanation, probability, threshold, features):
    positive, negative = feature_factors(explanation, features)
    return {'server_id': str(metadata['server_id']), 'timestamp': str(metadata['timestamp']),
            'calibrated_failure_probability': float(probability), 'supervised_threshold': float(threshold),
            'hybrid_risk_level': str(metadata['hybrid_risk_level']),
            'anomaly_score': float(metadata['anomaly_score']), 'anomaly_percentile': float(metadata['anomaly_percentile']),
            'base_model_output': float(explanation.base_values),
            'explained_model_output': float(explanation.base_values + explanation.values.sum()),
            'shap_units': 'base_model_raw_log_odds', 'top_risk_drivers': positive,
            'protective_factors': negative, 'disclaimer': DISCLAIMER}


def save_global(explanation, report_dir, figure_dir):
    import matplotlib.pyplot as plt
    report_dir, figure_dir = Path(report_dir), Path(figure_dir)
    importance = pd.DataFrame({'feature': explanation.feature_names,
                               'mean_abs_shap': np.abs(explanation.values).mean(axis=0)})
    importance = importance.sort_values(['mean_abs_shap', 'feature'], ascending=[False, True]).reset_index(drop=True)
    importance['importance_rank'] = np.arange(1, len(importance)+1)
    importance.to_csv(report_dir/'shap_global_importance.csv', index=False)
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([DISPLAY_NAMES.get(f, f) for f in top.feature], top.mean_abs_shap, color='#236c9c')
    ax.set(xlabel='Mean absolute SHAP contribution (base-model log-odds)',
           title='Frozen LightGBM: global feature influence')
    fig.tight_layout(); fig.savefig(figure_dir/'shap_global_bar.png', dpi=160); plt.close(fig)
    display = shap.Explanation(values=explanation.values, base_values=explanation.base_values,
                               data=explanation.data,
                               feature_names=[DISPLAY_NAMES.get(f, f) for f in explanation.feature_names])
    np.random.seed(42)
    shap.plots.beeswarm(display, max_display=20, show=False, plot_size=(11, 9))
    plt.xlabel('SHAP contribution to base-model log-odds (not calibrated probability)')
    plt.title('Frozen LightGBM: feature values and model contributions')
    plt.tight_layout(); plt.savefig(figure_dir/'shap_summary_beeswarm.png', dpi=160); plt.close()
    rows = []
    for feature in importance.head(20).feature:
        i = explanation.feature_names.index(feature)
        x, s = explanation.data[:, i], explanation.values[:, i]
        low, high = np.quantile(x, [.25, .75])
        # Tied quartiles otherwise create identical high/low groups for sparse features.
        low_mask, high_mask = x <= low, x >= high
        if low == high:
            low_mask, high_mask = x <= low, x > high
        mean_low = float(s[low_mask].mean()) if low_mask.any() else None
        mean_high = float(s[high_mask].mean()) if high_mask.any() else None
        difference = mean_high-mean_low if mean_high is not None and mean_low is not None else None
        rows.append({'feature': feature, 'low_cutoff': low, 'high_cutoff': high,
                     'low_observations': int(low_mask.sum()), 'high_observations': int(high_mask.sum()),
                     'mean_shap_when_low': mean_low, 'mean_shap_when_high': mean_high,
                     'high_minus_low': difference,
                     'observed_direction': 'UNAVAILABLE' if difference is None else
                         'HIGHER_VALUES_MORE_RISK_INCREASING' if difference > 0 else
                         'HIGHER_VALUES_LESS_RISK_INCREASING' if difference < 0 else 'NO_DIFFERENCE',
                     'note': 'Descriptive association; correlated features and interactions matter. Tied quartiles use > cutoff for high.'})
    pd.DataFrame(rows).to_csv(report_dir/'shap_feature_direction_summary.csv', index=False)
    return importance


def save_live_view(records, path):
    rows = []
    for record in records:
        drivers = record['top_risk_drivers']
        first = drivers[0] if drivers else {}
        second = drivers[1] if len(drivers) > 1 else {}
        rows.append({'server_id': record['server_id'], 'timestamp': record['timestamp'],
                     'failure_probability': record['calibrated_failure_probability'],
                     'hybrid_risk_level': record['hybrid_risk_level'],
                     'top_risk_driver': first.get('feature'), 'top_risk_driver_value': first.get('value'),
                     'top_risk_driver_shap': first.get('shap_value'),
                     'second_risk_driver': second.get('feature'), 'second_risk_driver_shap': second.get('shap_value'),
                     'anomaly_score': record['anomaly_score'], 'anomaly_percentile': record['anomaly_percentile']})
    pd.DataFrame(rows).to_csv(path, index=False)
