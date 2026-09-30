"""Fast synthetic-data verification of the GeoAttribStack fusion constraints."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoattribstack import fit_geoattribstack  # noqa: E402


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def synthetic_problem(seed: int = 42):
    rng = np.random.default_rng(seed)
    n_samples, n_models, n_features = 360, 3, 6
    regimes = np.arange(n_samples) % n_models
    signal = rng.normal(0, 1, n_samples)
    y = 2.0 * signal + rng.normal(0, 0.08, n_samples)

    predictions = np.empty((n_samples, n_models))
    for model in range(n_models):
        scale = np.where(regimes == model, 0.15, 1.30)
        predictions[:, model] = y + rng.normal(0, scale)

    signatures = np.array(
        [
            [1.6, -0.9, -0.5, 0.2, -0.2, -0.2],
            [-0.5, 1.6, -0.9, -0.2, 0.2, -0.2],
            [-0.9, -0.5, 1.6, -0.2, -0.2, 0.2],
        ]
    )
    shap_values = np.empty((n_samples, n_models, n_features))
    for sample in range(n_samples):
        for model in range(n_models):
            shap_values[sample, model] = (
                signatures[regimes[sample]]
                + 0.04 * model
                + rng.normal(0, 0.04, n_features)
            )
    order = rng.permutation(n_samples)
    train, test = order[:270], order[270:]
    return (
        shap_values[train],
        predictions[train],
        y[train],
        shap_values[test],
        predictions[test],
        y[test],
    )


def run_quick_test() -> dict[str, float]:
    shap_train, pred_train, y_train, shap_test, pred_test, y_test = synthetic_problem()
    fitted = fit_geoattribstack(
        shap_train,
        pred_train,
        y_train,
        shap_test,
        pred_test,
        n_regimes=3,
        tau=0.7,
        seed=42,
    )

    membership = fitted["membership"]
    regime_weights = fitted["regime_weights"]
    sample_weights = fitted["sample_weights"]
    prediction = fitted["prediction"]
    equal_weight_prediction = pred_test.mean(axis=1)

    assert np.allclose(membership.sum(axis=1), 1.0, atol=1e-8)
    assert np.allclose(regime_weights.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(regime_weights >= 0)
    assert np.allclose(sample_weights.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(prediction >= pred_test.min(axis=1) - 1e-8)
    assert np.all(prediction <= pred_test.max(axis=1) + 1e-8)

    arc_rmse = rmse(y_test, prediction)
    equal_rmse = rmse(y_test, equal_weight_prediction)
    assert np.isfinite(arc_rmse)
    assert arc_rmse < equal_rmse
    return {"geoattribstack_rmse": arc_rmse, "equal_weight_rmse": equal_rmse}


if __name__ == "__main__":
    metrics = run_quick_test()
    print(f"GeoAttribStack RMSE: {metrics['geoattribstack_rmse']:.6f}")
    print(f"Equal-weight RMSE:   {metrics['equal_weight_rmse']:.6f}")
    print("QUICK TEST PASSED")
