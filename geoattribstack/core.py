"""Core attribution-regime constrained fusion operations used by GeoAttribStack."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def shap_representation(values: np.ndarray) -> np.ndarray:
    """Mean-center and L1-normalize per-model SHAP profiles.

    Parameters
    ----------
    values
        Array shaped ``(samples, models, features)``.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise ValueError("values must have shape (samples, models, features)")
    transformed = values - values.mean(axis=2, keepdims=True)
    denominator = np.sum(np.abs(transformed), axis=2, keepdims=True)
    transformed = transformed / np.maximum(denominator, 1e-12)
    return transformed.reshape(len(values), -1)


def membership_from_distances(
    distances: np.ndarray, tau: float, hard: bool = False
) -> np.ndarray:
    """Convert distances to hard or temperature-scaled soft memberships."""
    distances = np.asarray(distances, dtype=float)
    if tau <= 0:
        raise ValueError("tau must be positive")
    if hard:
        labels = np.argmin(distances, axis=1)
        membership = np.zeros_like(distances)
        membership[np.arange(len(labels)), labels] = 1.0
        return membership
    scale = np.median(np.min(distances, axis=1)) + 1e-8
    return softmax(-distances / (tau * scale), axis=1)


def fit_regime_model(
    q_train: np.ndarray,
    n_regimes: int,
    tau: float,
    seed: int = 42,
    hard: bool = False,
):
    """Fit standardized attribution regimes and return training memberships."""
    scaler = StandardScaler().fit(q_train)
    scaled = scaler.transform(q_train)
    clustering = KMeans(
        n_clusters=n_regimes, n_init=30, random_state=seed
    ).fit(scaled)
    membership = membership_from_distances(
        clustering.transform(scaled), tau=tau, hard=hard
    )
    return scaler, clustering, membership


def transform_regime_model(
    q: np.ndarray,
    scaler: StandardScaler,
    clustering: KMeans,
    tau: float,
    hard: bool = False,
) -> np.ndarray:
    """Map attribution representations to memberships in fitted regimes."""
    distances = clustering.transform(scaler.transform(q))
    return membership_from_distances(distances, tau=tau, hard=hard)


def fit_simplex_regime_weights(
    membership: np.ndarray,
    predictions: np.ndarray,
    y: np.ndarray,
    regularization: float = 0.015,
) -> np.ndarray:
    """Estimate one non-negative, sum-to-one expert vector per regime."""
    membership = np.asarray(membership, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    y = np.asarray(y, dtype=float)
    n_regimes = membership.shape[1]
    n_models = predictions.shape[1]
    y_mean, y_std = float(y.mean()), float(y.std())
    if y_std <= 0:
        raise ValueError("y must have non-zero variance")
    pred_scaled = (predictions - y_mean) / y_std
    target = (y - y_mean) / y_std

    def objective(logits_flat: np.ndarray) -> float:
        regime_weights = softmax(
            logits_flat.reshape(n_regimes, n_models), axis=1
        )
        sample_weights = membership @ regime_weights
        fused = np.sum(sample_weights * pred_scaled, axis=1)
        mse = np.mean((target - fused) ** 2)
        shrinkage = np.mean(
            (regime_weights - regime_weights.mean(axis=0, keepdims=True)) ** 2
        )
        return float(mse + regularization * shrinkage)

    result = minimize(
        objective, np.zeros(n_regimes * n_models), method="L-BFGS-B"
    )
    if not result.success:
        raise RuntimeError(f"Simplex optimization failed: {result.message}")
    return softmax(result.x.reshape(n_regimes, n_models), axis=1)


def regime_predict(
    membership: np.ndarray,
    regime_weights: np.ndarray,
    predictions: np.ndarray,
):
    """Fuse expert predictions using membership-conditioned convex weights."""
    sample_weights = membership @ regime_weights
    fused = np.sum(sample_weights * predictions, axis=1)
    return fused, sample_weights


def fit_geoattribstack(
    shap_train: np.ndarray,
    predictions_train: np.ndarray,
    y_train: np.ndarray,
    shap_test: np.ndarray,
    predictions_test: np.ndarray,
    n_regimes: int = 3,
    tau: float = 0.7,
    regularization: float = 0.015,
    seed: int = 42,
):
    """Fit the lightweight fusion core and predict an independent test set."""
    q_train = shap_representation(shap_train)
    q_test = shap_representation(shap_test)
    scaler, clustering, membership_train = fit_regime_model(
        q_train, n_regimes=n_regimes, tau=tau, seed=seed
    )
    regime_weights = fit_simplex_regime_weights(
        membership_train,
        predictions_train,
        y_train,
        regularization=regularization,
    )
    membership_test = transform_regime_model(
        q_test, scaler, clustering, tau=tau
    )
    prediction, sample_weights = regime_predict(
        membership_test, regime_weights, predictions_test
    )
    return {
        "prediction": prediction,
        "sample_weights": sample_weights,
        "membership": membership_test,
        "regime_weights": regime_weights,
        "scaler": scaler,
        "clustering": clustering,
    }
