"""Fold-cache construction shared by the final GeoAttribStack experiment."""

from __future__ import annotations

import numpy as np

from run_routes import build_meta_features, select_base_models


def cache_fold(paths, scheme, fold, train_idx, test_idx, clean, x, y, coords, seed):
    """Create or load one leakage-free outer-fold feature cache."""
    cache_path = paths["cache"] / f"{scheme}_fold_{fold}.npz"
    if cache_path.exists():
        return np.load(cache_path, allow_pickle=False)

    x_train, y_train, coords_train = x[train_idx], y[train_idx], coords[train_idx]
    x_test, y_test, coords_test = x[test_idx], y[test_idx], coords[test_idx]
    selected, _, _ = select_base_models(
        x_train, y_train, coords_train, "spatial", seed + fold, 3
    )
    meta = build_meta_features(
        x_train,
        y_train,
        coords_train,
        x_test,
        selected,
        "spatial",
        seed + fold,
        3,
    )
    pred_train, shap_train, base_train, pred_test, shap_test, base_test = meta[:6]
    np.savez_compressed(
        cache_path,
        train_idx=train_idx,
        test_idx=test_idx,
        x_train=x_train,
        y_train=y_train,
        coords_train=coords_train,
        x_test=x_test,
        y_test=y_test,
        coords_test=coords_test,
        pred_train=pred_train,
        shap_train=shap_train,
        base_train=base_train,
        pred_test=pred_test,
        shap_test=shap_test,
        base_test=base_test,
        model_names=np.asarray(selected),
        source_rows_train=clean.source_row.iloc[train_idx].to_numpy(),
        source_rows_test=clean.source_row.iloc[test_idx].to_numpy(),
    )
    return np.load(cache_path, allow_pickle=False)
