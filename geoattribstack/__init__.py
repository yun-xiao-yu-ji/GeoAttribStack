"""Lightweight public API for the GeoAttribStack fusion core."""

from .core import (
    fit_geoattribstack,
    fit_regime_model,
    fit_simplex_regime_weights,
    membership_from_distances,
    regime_predict,
    shap_representation,
    transform_regime_model,
)

__all__ = [
    "fit_geoattribstack",
    "fit_regime_model",
    "fit_simplex_regime_weights",
    "membership_from_distances",
    "regime_predict",
    "shap_representation",
    "transform_regime_model",
]
