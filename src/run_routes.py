"""Run leakage-free spatial audits (Route 2) and dynamic gating (Route 3).

The script deliberately keeps every outer test fold untouched until final scoring.
All model screening and meta-feature construction are performed inside the outer
training partition. Route 3 consumes only the spatial-fold caches created by
Route 2, so both routes are compared on identical samples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    }
)

FEATURES = ["Compress", "Ag_Den", "PRE", "Road_Distance", "River_Distance", "BLT"]
TARGET = "LS_rate"
COORDS = ["LON", "LAT"]
BASE_CANDIDATES = ["RF", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"]
REPRESENTATIONS = [
    "predictions_only",
    "shap_only",
    "hybrid_full",
    "hybrid_dropone",
    "hybrid_l1",
    "hybrid_pca",
]
COLORS = {
    "predictions_only": "#0072B2",
    "shap_only": "#999999",
    "hybrid_full": "#D55E00",
    "hybrid_dropone": "#E69F00",
    "hybrid_l1": "#009E73",
    "hybrid_pca": "#CC79A7",
}


def ensure_dirs(root: Path) -> dict[str, Path]:
    route2 = root / "route2_validation_audit"
    route3 = root / "route3_dynamic_fusion"
    paths = {
        "root": root,
        "r2": route2,
        "r2_tables": route2 / "results" / "tables",
        "r2_figures": route2 / "results" / "figures",
        "r2_cache": route2 / "results" / "fold_cache",
        "r2_models": route2 / "results" / "models",
        "r2_logs": route2 / "logs",
        "r3": route3,
        "r3_tables": route3 / "results" / "tables",
        "r3_figures": route3 / "results" / "figures",
        "r3_models": route3 / "results" / "models",
        "r3_logs": route3 / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_data(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_csv(path).replace("#NUM!", np.nan)
    needed = FEATURES + [TARGET] + COORDS
    data[needed] = data[needed].apply(pd.to_numeric, errors="coerce")
    data[needed] = data[needed].replace(-9999, np.nan)
    clean = data.dropna(subset=needed).copy().reset_index(drop=False).rename(columns={"index": "source_row"})
    x = clean[FEATURES].to_numpy(dtype=np.float64)
    y = clean[TARGET].to_numpy(dtype=np.float64)
    coords = clean[COORDS].to_numpy(dtype=np.float64)
    return clean, x, y, coords


def make_model(name: str, seed: int = 42, meta: bool = False):
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=120 if not meta else 160,
            max_depth=15 if not meta else 10,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=140,
            max_depth=17,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=180 if not meta else 160,
            max_depth=7 if not meta else 4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "LightGBM":
        return LGBMRegressor(
            n_estimators=180,
            max_depth=9,
            num_leaves=40,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "CatBoost":
        return CatBoostRegressor(
            iterations=180,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def metric_row(y: np.ndarray, pred: np.ndarray, **fields) -> dict:
    return {
        **fields,
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
    }


def make_spatial_groups(coords: np.ndarray, n_groups: int, seed: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(coords)
    return KMeans(n_clusters=n_groups, n_init=20, random_state=seed).fit_predict(scaled)


def inner_splits(coords: np.ndarray, protocol: str, n_splits: int, seed: int):
    if protocol == "random":
        return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(coords))
    groups = make_spatial_groups(coords, n_splits, seed)
    return [(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in range(n_splits)]


def outer_splits(coords: np.ndarray, protocol: str, n_splits: int, seed: int, buffer_deg: float):
    if protocol == "random":
        for fold, (train, test) in enumerate(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(coords)):
            yield fold, train, test
        return

    groups = make_spatial_groups(coords, n_splits, seed)
    for fold in range(n_splits):
        test = np.flatnonzero(groups == fold)
        train = np.flatnonzero(groups != fold)
        if protocol == "buffered_spatial":
            nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(coords[test])
            dist = nn.kneighbors(coords[train], return_distance=True)[0][:, 0]
            train = train[dist >= buffer_deg]
        yield fold, train, test


def tree_shap(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    explainer = shap.TreeExplainer(model)
    # Saabas TreeSHAP keeps the full dataset tractable for nested spatial CV.
    # The additivity audit below quantifies reconstruction precision fold by fold.
    approximate = model.__class__.__name__ not in {"LGBMRegressor", "CatBoostRegressor"}
    values = np.asarray(
        explainer.shap_values(x, approximate=approximate, check_additivity=False), dtype=np.float64
    )
    expected = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)
    base = np.full(len(x), float(expected[0]))
    return values, base


def select_base_models(x: np.ndarray, y: np.ndarray, coords: np.ndarray, protocol: str, seed: int, n_inner: int):
    splits = inner_splits(coords, protocol, n_inner, seed)
    rows = []
    for model_name in BASE_CANDIDATES:
        fold_rmse = []
        for inner_fold, (tr, va) in enumerate(splits):
            model = make_model(model_name, seed + inner_fold)
            model.fit(x[tr], y[tr])
            pred = model.predict(x[va])
            score = float(mean_squared_error(y[va], pred) ** 0.5)
            fold_rmse.append(score)
            rows.append({"candidate": model_name, "inner_fold": inner_fold, "RMSE": score})
    ranking = (
        pd.DataFrame(rows)
        .groupby("candidate", as_index=False)["RMSE"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean")
    )
    return ranking["candidate"].head(3).tolist(), pd.DataFrame(rows), ranking


def build_meta_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    coords_train: np.ndarray,
    x_test: np.ndarray,
    model_names: list[str],
    protocol: str,
    seed: int,
    n_inner: int,
):
    splits = inner_splits(coords_train, protocol, n_inner, seed)
    n_features = x_train.shape[1]
    pred_train = np.zeros((len(x_train), len(model_names)))
    shap_train = np.zeros((len(x_train), len(model_names), n_features))
    base_train = np.zeros((len(x_train), len(model_names)))
    pred_test = np.zeros((len(x_test), len(model_names)))
    shap_test = np.zeros((len(x_test), len(model_names), n_features))
    base_test = np.zeros((len(x_test), len(model_names)))
    fitted = {}

    for model_idx, model_name in enumerate(model_names):
        for inner_fold, (tr, va) in enumerate(splits):
            model = make_model(model_name, seed + 100 + inner_fold)
            model.fit(x_train[tr], y_train[tr])
            pred_train[va, model_idx] = model.predict(x_train[va])
            values, expected = tree_shap(model, x_train[va])
            shap_train[va, model_idx, :] = values
            base_train[va, model_idx] = expected

        final_model = make_model(model_name, seed + 500)
        final_model.fit(x_train, y_train)
        pred_test[:, model_idx] = final_model.predict(x_test)
        values, expected = tree_shap(final_model, x_test)
        shap_test[:, model_idx, :] = values
        base_test[:, model_idx] = expected
        fitted[model_name] = final_model

    reconstruction = {
        "train_max_abs": float(np.max(np.abs(pred_train - base_train - shap_train.sum(axis=2)))),
        "train_mean_abs": float(np.mean(np.abs(pred_train - base_train - shap_train.sum(axis=2)))),
        "test_max_abs": float(np.max(np.abs(pred_test - base_test - shap_test.sum(axis=2)))),
        "test_mean_abs": float(np.mean(np.abs(pred_test - base_test - shap_test.sum(axis=2)))),
    }
    return pred_train, shap_train, base_train, pred_test, shap_test, base_test, fitted, reconstruction


def l1_normalize(shap_values: np.ndarray) -> np.ndarray:
    denom = np.sum(np.abs(shap_values), axis=2, keepdims=True)
    return shap_values / np.maximum(denom, 1e-12)


def make_representations(pred_train, shap_train, pred_test, shap_test):
    n_train, n_models, n_features = shap_train.shape
    n_test = len(pred_test)
    raw_train = shap_train.reshape(n_train, -1)
    raw_test = shap_test.reshape(n_test, -1)
    norm_train = l1_normalize(shap_train).reshape(n_train, -1)
    norm_test = l1_normalize(shap_test).reshape(n_test, -1)
    keep = np.ones((n_models, n_features), dtype=bool)
    keep[:, -1] = False
    keep = keep.reshape(-1)

    scaler = StandardScaler().fit(raw_train)
    scaled_train = scaler.transform(raw_train)
    pca = PCA(n_components=0.95, svd_solver="full").fit(scaled_train)
    pca_train = pca.transform(scaled_train)
    pca_test = pca.transform(scaler.transform(raw_test))

    reps = {
        "predictions_only": (pred_train, pred_test),
        "shap_only": (raw_train, raw_test),
        "hybrid_full": (np.column_stack([pred_train, raw_train]), np.column_stack([pred_test, raw_test])),
        "hybrid_dropone": (
            np.column_stack([pred_train, raw_train[:, keep]]),
            np.column_stack([pred_test, raw_test[:, keep]]),
        ),
        "hybrid_l1": (np.column_stack([pred_train, norm_train]), np.column_stack([pred_test, norm_test])),
        "hybrid_pca": (np.column_stack([pred_train, pca_train]), np.column_stack([pred_test, pca_test])),
    }
    return reps, int(pca.n_components_)


def make_meta_model(name: str, seed: int):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "XGBoost":
        return make_model("XGBoost", seed=seed, meta=True)
    raise ValueError(name)


def morans_i(residuals: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    if len(residuals) <= k + 1:
        return float("nan")
    indices = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(coords).kneighbors(return_distance=False)[:, 1:]
    centered = residuals - residuals.mean()
    numerator = np.sum(centered[:, None] * centered[indices])
    denominator = np.sum(centered**2)
    return float((len(residuals) / (len(residuals) * k)) * numerator / max(denominator, 1e-12))


def conformal_audit(x_meta, y, coords, x_test, y_test, seed: int, alpha: float = 0.1):
    groups = make_spatial_groups(coords, 4, seed)
    calibration_group = int(np.argmin([np.sum(groups == g) for g in range(4)]))
    fit_idx = np.flatnonzero(groups != calibration_group)
    cal_idx = np.flatnonzero(groups == calibration_group)

    point = make_meta_model("XGBoost", seed)
    point.fit(x_meta[fit_idx], y[fit_idx])
    cal_pred = point.predict(x_meta[cal_idx])
    test_pred = point.predict(x_test)
    cal_abs = np.abs(y[cal_idx] - cal_pred)

    q_level = min(math.ceil((len(cal_idx) + 1) * (1 - alpha)) / len(cal_idx), 1.0)
    q_standard = float(np.quantile(cal_abs, q_level, method="higher"))
    standard_width = np.full(len(y_test), 2 * q_standard)
    standard_covered = np.abs(y_test - test_pred) <= q_standard

    fit_coords = coords[fit_idx]
    fit_splits = inner_splits(fit_coords, "spatial", 3, seed + 71)
    oof_pred = np.zeros(len(fit_idx))
    for tr, va in fit_splits:
        fold_model = make_meta_model("XGBoost", seed + 80)
        fold_model.fit(x_meta[fit_idx][tr], y[fit_idx][tr])
        oof_pred[va] = fold_model.predict(x_meta[fit_idx][va])
    fit_abs = np.abs(y[fit_idx] - oof_pred)
    error_model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=-1
    )
    error_model.fit(x_meta[fit_idx], fit_abs)
    beta = 0.5
    cal_scale = error_model.predict(x_meta[cal_idx]) + beta
    test_scale = error_model.predict(x_test) + beta
    q_adaptive = float(np.quantile(cal_abs / cal_scale, q_level, method="higher"))
    adaptive_margin = q_adaptive * test_scale
    adaptive_width = 2 * adaptive_margin
    adaptive_covered = np.abs(y_test - test_pred) <= adaptive_margin

    rows = [
        {
            "method": "standard_absolute_residual",
            "coverage": float(standard_covered.mean()),
            "mean_width": float(standard_width.mean()),
            "median_width": float(np.median(standard_width)),
        },
        {
            "method": "adaptive_normalized_score",
            "coverage": float(adaptive_covered.mean()),
            "mean_width": float(adaptive_width.mean()),
            "median_width": float(np.median(adaptive_width)),
        },
    ]
    details = {
        "test_pred": test_pred,
        "standard_width": standard_width,
        "standard_covered": standard_covered,
        "adaptive_width": adaptive_width,
        "adaptive_covered": adaptive_covered,
    }
    return rows, details


def savefig(fig: plt.Figure, stem: Path):
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def bootstrap_mean_ci(values: np.ndarray, seed: int = 42, n_boot: int = 10000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_stats(metrics: pd.DataFrame, protocol: str, baseline: str, meta_model: str):
    subset = metrics[(metrics["protocol"] == protocol) & (metrics["meta_model"] == meta_model)]
    pivot = subset.pivot(index="outer_fold", columns="representation", values="RMSE")
    rows = []
    if baseline not in pivot:
        return pd.DataFrame(rows)
    for method in pivot.columns:
        delta = pivot[method] - pivot[baseline]
        try:
            p_value = float(wilcoxon(delta).pvalue) if method != baseline else 1.0
        except ValueError:
            p_value = 1.0
        mean, low, high = bootstrap_mean_ci(delta.to_numpy())
        rows.append(
            {
                "protocol": protocol,
                "meta_model": meta_model,
                "method": method,
                "baseline": baseline,
                "mean_delta_RMSE": mean,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "wilcoxon_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def optimization_audit(x, y, coords, out_dir: Path, seed: int, trials: int = 10):
    groups = make_spatial_groups(coords, 3, seed)
    splits = [(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in range(3)]

    def evaluate(params):
        scores = []
        for tr, va in splits:
            model = XGBRegressor(
                objective="reg:squarederror", tree_method="hist", n_jobs=-1, random_state=seed, **params
            )
            model.fit(x[tr], y[tr])
            scores.append(mean_squared_error(y[va], model.predict(x[va])) ** 0.5)
        return float(np.mean(scores))

    default_params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3, "subsample": 1.0}
    rows = [{"optimizer": "default", "trial": 0, "RMSE": evaluate(default_params), **default_params}]

    def run_study(name, sampler):
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 320),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.25, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            }
            return evaluate(params)

        study.optimize(objective, n_trials=trials)
        for trial in study.trials:
            rows.append({"optimizer": name, "trial": trial.number, "RMSE": trial.value, **trial.params})

    run_study("random_search", optuna.samplers.RandomSampler(seed=seed))
    run_study("TPE", optuna.samplers.TPESampler(seed=seed + 101, n_startup_trials=3))
    history = pd.DataFrame(rows)
    history.to_csv(out_dir / "optimization_audit_history.csv", index=False)
    return history


def plot_route2(metrics, fold_map, conformal, optimization, fig_dir: Path):
    summary = (
        metrics.groupby(["protocol", "representation", "meta_model"], as_index=False)
        .agg(RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"))
    )
    xlabels = REPRESENTATIONS
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1), sharey=False)
    for ax, protocol in zip(axes, ["random", "spatial", "buffered_spatial"]):
        part = summary[(summary.protocol == protocol) & (summary.meta_model == "XGBoost")]
        part = part.set_index("representation").reindex(xlabels)
        xpos = np.arange(len(xlabels))
        ax.bar(xpos, part.RMSE_mean, yerr=part.RMSE_std, color=[COLORS[x] for x in xlabels], capsize=3)
        ax.set_xticks(xpos, [x.replace("hybrid_", "").replace("predictions_", "pred-") for x in xlabels], rotation=35, ha="right")
        ax.set_title(protocol.replace("_", " ").title())
        ax.set_ylabel("RMSE (mm yr$^{-1}$)")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Route 2: representation performance with a fixed XGBoost meta-learner", y=1.02)
    fig.tight_layout()
    savefig(fig, fig_dir / "R2_Fig1_validation_and_representation")

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    scatter = ax.scatter(fold_map.LON, fold_map.LAT, c=fold_map.spatial_fold, s=5, cmap="tab10", alpha=0.8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatially contiguous outer folds")
    legend = ax.legend(*scatter.legend_elements(), title="Fold", loc="best", frameon=False)
    ax.add_artist(legend)
    ax.set_aspect("equal", adjustable="box")
    savefig(fig, fig_dir / "R2_Fig2_spatial_outer_folds")

    conf_summary = conformal.groupby("method", as_index=False).agg(coverage=("coverage", "mean"), width=("mean_width", "mean"))
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    for _, row in conf_summary.iterrows():
        ax.scatter(row.width, row.coverage, s=90, label=row.method.replace("_", " "))
        ax.annotate(row.method.split("_")[0], (row.width, row.coverage), xytext=(5, 5), textcoords="offset points")
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1, label="nominal 90%")
    ax.set_xlabel("Mean interval width (mm yr$^{-1}$)")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.75, 1.01)
    ax.set_title("Spatial-fold conformal calibration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig(fig, fig_dir / "R2_Fig3_conformal_coverage_width")

    fig, ax = plt.subplots(figsize=(6.7, 4.2))
    for name, part in optimization.groupby("optimizer"):
        if name == "default":
            ax.axhline(part.RMSE.iloc[0], color="#666666", linestyle="--", label="default")
        else:
            ordered = part.sort_values("trial")
            ax.plot(ordered.trial + 1, np.minimum.accumulate(ordered.RMSE), marker="o", label=name)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best spatial-CV RMSE")
    ax.set_title("XGBoost optimization audit on a locked training fold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig(fig, fig_dir / "R2_Fig4_optimization_history")


def run_route2(data_path: Path, paths: dict[str, Path], args):
    start = time.time()
    clean, x, y, coords = load_data(data_path)
    clean[["source_row", TARGET] + COORDS + FEATURES].to_csv(paths["r2_tables"] / "cleaned_modelling_table.csv", index=False)
    sentinel_report = {
        "raw_rows": int(len(pd.read_csv(data_path))),
        "clean_rows": int(len(clean)),
        "removed_rows": int(len(pd.read_csv(data_path)) - len(clean)),
        "rule": "Replace -9999 and nonnumeric values with missing, then complete-case filter all predictors, target, and coordinates.",
    }
    (paths["r2_tables"] / "data_cleaning_report.json").write_text(json.dumps(sentinel_report, indent=2), encoding="utf-8")

    spatial_groups = make_spatial_groups(coords, args.outer_folds, args.seed)
    fold_map = clean[["source_row", "LON", "LAT"]].copy()
    fold_map["spatial_fold"] = spatial_groups
    fold_map.to_csv(paths["r2_tables"] / "spatial_outer_fold_assignments.csv", index=False)

    metric_rows = []
    selection_rows = []
    reconstruction_rows = []
    prediction_rows = []
    conformal_rows = []
    conformal_point_rows = []
    protocols = ["random", "spatial", "buffered_spatial"]

    for protocol in protocols:
        for fold, train_idx, test_idx in outer_splits(
            coords, protocol, args.outer_folds, args.seed, args.buffer_deg
        ):
            fold_start = time.time()
            x_train, y_train, coords_train = x[train_idx], y[train_idx], coords[train_idx]
            x_test, y_test, coords_test = x[test_idx], y[test_idx], coords[test_idx]
            inner_protocol = "random" if protocol == "random" else "spatial"
            selected, selection_detail, ranking = select_base_models(
                x_train, y_train, coords_train, inner_protocol, args.seed + fold, args.inner_folds
            )
            selection_detail["protocol"] = protocol
            selection_detail["outer_fold"] = fold
            selection_rows.extend(selection_detail.to_dict("records"))

            (
                pred_train,
                shap_train,
                base_train,
                pred_test,
                shap_test,
                base_test,
                fitted,
                reconstruction,
            ) = build_meta_features(
                x_train,
                y_train,
                coords_train,
                x_test,
                selected,
                inner_protocol,
                args.seed + fold,
                args.inner_folds,
            )
            reconstruction_rows.append({"protocol": protocol, "outer_fold": fold, **reconstruction})
            reps, pca_components = make_representations(pred_train, shap_train, pred_test, shap_test)

            for representation, (meta_train, meta_test) in reps.items():
                for meta_name in ["Ridge", "XGBoost"]:
                    meta_model = make_meta_model(meta_name, args.seed + fold)
                    meta_model.fit(meta_train, y_train)
                    prediction = meta_model.predict(meta_test)
                    metric_rows.append(
                        metric_row(
                            y_test,
                            prediction,
                            protocol=protocol,
                            outer_fold=fold,
                            representation=representation,
                            meta_model=meta_name,
                            n_train=len(train_idx),
                            n_test=len(test_idx),
                            pca_components=pca_components,
                            morans_I=morans_i(y_test - prediction, coords_test),
                        )
                    )
                    for local, global_idx in enumerate(test_idx):
                        prediction_rows.append(
                            {
                                "protocol": protocol,
                                "outer_fold": fold,
                                "source_row": int(clean.source_row.iloc[global_idx]),
                                "representation": representation,
                                "meta_model": meta_name,
                                "observed": y_test[local],
                                "predicted": prediction[local],
                                "LON": coords_test[local, 0],
                                "LAT": coords_test[local, 1],
                            }
                        )

            if protocol == "spatial":
                cp_train, cp_test = reps["hybrid_l1"]
                cp_rows, cp_details = conformal_audit(
                    cp_train, y_train, coords_train, cp_test, y_test, args.seed + fold
                )
                for row in cp_rows:
                    conformal_rows.append({"outer_fold": fold, **row})
                for i, global_idx in enumerate(test_idx):
                    conformal_point_rows.append(
                        {
                            "outer_fold": fold,
                            "source_row": int(clean.source_row.iloc[global_idx]),
                            "observed": y_test[i],
                            "predicted": cp_details["test_pred"][i],
                            "standard_width": cp_details["standard_width"][i],
                            "standard_covered": bool(cp_details["standard_covered"][i]),
                            "adaptive_width": cp_details["adaptive_width"][i],
                            "adaptive_covered": bool(cp_details["adaptive_covered"][i]),
                            "LON": coords_test[i, 0],
                            "LAT": coords_test[i, 1],
                        }
                    )

                np.savez_compressed(
                    paths["r2_cache"] / f"spatial_fold_{fold}.npz",
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
                for model_name, model in fitted.items():
                    joblib.dump(model, paths["r2_models"] / f"spatial_fold_{fold}_{model_name}.joblib")

            elapsed = time.time() - fold_start
            print(
                f"[Route 2] {protocol} fold {fold + 1}/{args.outer_folds}: "
                f"selected={selected}, train={len(train_idx)}, test={len(test_idx)}, {elapsed / 60:.1f} min",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    selections = pd.DataFrame(selection_rows)
    recon = pd.DataFrame(reconstruction_rows)
    predictions = pd.DataFrame(prediction_rows)
    conformal = pd.DataFrame(conformal_rows)
    conformal_points = pd.DataFrame(conformal_point_rows)
    metrics.to_csv(paths["r2_tables"] / "outer_fold_metrics.csv", index=False)
    selections.to_csv(paths["r2_tables"] / "inner_model_screening.csv", index=False)
    recon.to_csv(paths["r2_tables"] / "shap_additivity_audit.csv", index=False)
    predictions.to_csv(paths["r2_tables"] / "outer_fold_predictions.csv", index=False)
    conformal.to_csv(paths["r2_tables"] / "conformal_fold_metrics.csv", index=False)
    conformal_points.to_csv(paths["r2_tables"] / "conformal_point_results.csv", index=False)

    stats = pd.concat(
        [paired_stats(metrics, p, "predictions_only", "XGBoost") for p in protocols], ignore_index=True
    )
    stats.to_csv(paths["r2_tables"] / "paired_representation_tests.csv", index=False)

    first_spatial_train = next(outer_splits(coords, "spatial", args.outer_folds, args.seed, args.buffer_deg))[1]
    optimization = optimization_audit(
        x[first_spatial_train],
        y[first_spatial_train],
        coords[first_spatial_train],
        paths["r2_tables"],
        args.seed,
        args.optimization_trials,
    )
    plot_route2(metrics, fold_map, conformal, optimization, paths["r2_figures"])

    summary = metrics.groupby(["protocol", "representation", "meta_model"], as_index=False).agg(
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        R2_mean=("R2", "mean"),
        Moran_I_mean=("morans_I", "mean"),
    )
    summary.to_csv(paths["r2_tables"] / "performance_summary.csv", index=False)
    runtime = time.time() - start
    result = {
        "status": "completed",
        "runtime_minutes": runtime / 60,
        "data": sentinel_report,
        "best_by_protocol": {},
        "conformal": conformal.groupby("method")[["coverage", "mean_width"]].mean().to_dict("index"),
        "max_shap_reconstruction_error": float(recon.test_max_abs.max()),
    }
    for protocol in protocols:
        part = summary[summary.protocol == protocol].sort_values("RMSE_mean")
        result["best_by_protocol"][protocol] = part.iloc[0].to_dict()
    (paths["r2"] / "RESULT_SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return metrics, result


class GateNet(torch.nn.Module):
    def __init__(self, n_inputs: int, n_models: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, n_models),
        )

    def forward(self, context, predictions):
        weights = torch.softmax(self.network(context), dim=1)
        output = torch.sum(weights * predictions, dim=1)
        return output, weights


def fit_gate(context, predictions, y, coords, context_test, predictions_test, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(context)
    z = scaler.transform(context).astype(np.float32)
    z_test = scaler.transform(context_test).astype(np.float32)
    y_mean, y_std = float(y.mean()), float(y.std())
    p = ((predictions - y_mean) / y_std).astype(np.float32)
    p_test = ((predictions_test - y_mean) / y_std).astype(np.float32)
    target = ((y - y_mean) / y_std).astype(np.float32)

    val_groups = make_spatial_groups(coords, 5, seed)
    val_group = int(np.argmin([np.sum(val_groups == g) for g in range(5)]))
    train_idx = np.flatnonzero(val_groups != val_group)
    val_idx = np.flatnonzero(val_groups == val_group)

    model = GateNet(z.shape[1], p.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=2e-4)
    loss_fn = torch.nn.MSELoss()
    x_tensor = torch.from_numpy(z)
    p_tensor = torch.from_numpy(p)
    y_tensor = torch.from_numpy(target)
    best_state = None
    best_val = float("inf")
    patience = 0
    rng = np.random.default_rng(seed)

    for epoch in range(350):
        model.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(order), 2048):
            batch = order[start : start + 2048]
            optimizer.zero_grad()
            output, _ = model(x_tensor[batch], p_tensor[batch])
            loss = loss_fn(output, y_tensor[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_output, _ = model(x_tensor[val_idx], p_tensor[val_idx])
            val_loss = float(loss_fn(val_output, y_tensor[val_idx]))
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= 35:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_output, test_weights = model(torch.from_numpy(z_test), torch.from_numpy(p_test))
    prediction = test_output.numpy() * y_std + y_mean
    weights = test_weights.numpy()
    return prediction, weights, model, scaler, epoch + 1, best_val


def fit_global_convex(predictions: np.ndarray, y: np.ndarray, predictions_test: np.ndarray):
    def softmax(v):
        exp = np.exp(v - np.max(v))
        return exp / exp.sum()

    def objective(logits):
        weights = softmax(logits)
        return np.mean((y - predictions @ weights) ** 2)

    result = minimize(objective, np.zeros(predictions.shape[1]), method="L-BFGS-B")
    weights = softmax(result.x)
    return predictions_test @ weights, weights


def gate_contexts(x_train, x_test, pred_train, pred_test, shap_train, shap_test):
    norm_train = l1_normalize(shap_train).reshape(len(x_train), -1)
    norm_test = l1_normalize(shap_test).reshape(len(x_test), -1)
    scaler = StandardScaler().fit(norm_train)
    scaled = scaler.transform(norm_train)
    pca = PCA(n_components=0.90, svd_solver="full").fit(scaled)
    norm_pca_train = pca.transform(scaled)
    norm_pca_test = pca.transform(scaler.transform(norm_test))
    return {
        "gate_predictions": (pred_train, pred_test),
        "gate_raw_features": (np.column_stack([pred_train, x_train]), np.column_stack([pred_test, x_test])),
        "gate_norm_attribution": (
            np.column_stack([pred_train, norm_pca_train]),
            np.column_stack([pred_test, norm_pca_test]),
        ),
    }, int(pca.n_components_)


def route3_stats(metrics: pd.DataFrame):
    pivot = metrics.pivot(index="outer_fold", columns="method", values="RMSE")
    rows = []
    for target in ["gate_norm_attribution", "gate_raw_features", "gate_predictions"]:
        for baseline in ["gate_predictions", "traditional_stacking_xgb", "global_convex", "simple_average"]:
            if target == baseline:
                continue
            delta = pivot[target] - pivot[baseline]
            try:
                p_value = float(wilcoxon(delta).pvalue)
            except ValueError:
                p_value = 1.0
            mean, low, high = bootstrap_mean_ci(delta.to_numpy())
            rows.append(
                {
                    "method": target,
                    "baseline": baseline,
                    "mean_delta_RMSE": mean,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def plot_route3(metrics, weights, predictions, fig_dir: Path):
    order = metrics.groupby("method").RMSE.mean().sort_values().index.tolist()
    data = [metrics.loc[metrics.method == m, "RMSE"].to_numpy() for m in order]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bp = ax.boxplot(data, labels=[x.replace("_", " ") for x in order], patch_artist=True, showmeans=True)
    palette = ["#009E73", "#0072B2", "#D55E00", "#E69F00", "#56B4E9", "#CC79A7", "#999999"]
    for patch, color in zip(bp["boxes"], palette * 2):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Spatial outer-fold RMSE (mm yr$^{-1}$)")
    ax.set_title("Route 3: explicit dynamic fusion versus stacking baselines")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, fig_dir / "R3_Fig1_dynamic_fusion_comparison")

    pivot = metrics.pivot(index="outer_fold", columns="method", values="RMSE")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for method, color in [("gate_predictions", "#0072B2"), ("traditional_stacking_xgb", "#D55E00")]:
        delta = pivot["gate_norm_attribution"] - pivot[method]
        ax.plot(delta.index + 1, delta.values, marker="o", label=f"vs {method.replace('_', ' ')}", color=color)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Spatial outer fold")
    ax.set_ylabel("RMSE difference; attribution gate minus baseline")
    ax.set_title("Does attribution context improve dynamic weighting?")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig(fig, fig_dir / "R3_Fig2_foldwise_gain")

    all_gate_weights = weights[weights.method == "gate_norm_attribution"]
    example_fold = int(weights.outer_fold.min())
    gate_weights = all_gate_weights[
        (weights.method == "gate_norm_attribution") & (weights.outer_fold == example_fold)
    ]
    model_names = sorted(gate_weights.model_name.unique())
    fig, axes = plt.subplots(1, len(model_names), figsize=(13.5, 4.0), sharex=True, sharey=True)
    for ax, model_name in zip(axes, model_names):
        part = gate_weights[gate_weights.model_name == model_name]
        sc = ax.scatter(part.LON, part.LAT, c=part.weight, s=6, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(model_name)
        ax.set_xlabel("Longitude")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02, label="Mean gate weight")
    fig.suptitle(f"Spatially varying attribution-gate weights (outer fold {example_fold + 1})", y=1.01)
    savefig(fig, fig_dir / "R3_Fig3_spatial_gate_weights")

    entropy = all_gate_weights.groupby(["outer_fold", "source_row"], as_index=False).first()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.hist(entropy.weight_entropy, bins=35, color="#009E73", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Normalized gate entropy")
    ax.set_ylabel("Samples")
    ax.set_title("Confidence of dynamic model allocation")
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, fig_dir / "R3_Fig4_gate_entropy")


def run_route3(paths: dict[str, Path], args):
    start = time.time()
    metric_rows = []
    seed_rows = []
    weight_rows = []
    prediction_rows = []

    cache_files = sorted(paths["r2_cache"].glob("spatial_fold_*.npz"))
    if len(cache_files) != args.outer_folds:
        raise RuntimeError("Route 3 requires all spatial fold caches from Route 2.")

    for cache_path in cache_files:
        fold = int(cache_path.stem.rsplit("_", 1)[1])
        cache = np.load(cache_path, allow_pickle=False)
        x_train, y_train, coords_train = cache["x_train"], cache["y_train"], cache["coords_train"]
        x_test, y_test, coords_test = cache["x_test"], cache["y_test"], cache["coords_test"]
        pred_train, pred_test = cache["pred_train"], cache["pred_test"]
        shap_train, shap_test = cache["shap_train"], cache["shap_test"]
        model_names = cache["model_names"].astype(str).tolist()
        source_rows = cache["source_rows_test"]

        simple = pred_test.mean(axis=1)
        metric_rows.append(metric_row(y_test, simple, outer_fold=fold, method="simple_average"))
        global_pred, global_weights = fit_global_convex(pred_train, y_train, pred_test)
        metric_rows.append(metric_row(y_test, global_pred, outer_fold=fold, method="global_convex"))

        trad = make_meta_model("XGBoost", args.seed + fold)
        trad.fit(pred_train, y_train)
        trad_pred = trad.predict(pred_test)
        metric_rows.append(metric_row(y_test, trad_pred, outer_fold=fold, method="traditional_stacking_xgb"))

        full_train = np.column_stack([pred_train, shap_train.reshape(len(y_train), -1)])
        full_test = np.column_stack([pred_test, shap_test.reshape(len(y_test), -1)])
        original = make_meta_model("XGBoost", args.seed + fold)
        original.fit(full_train, y_train)
        original_pred = original.predict(full_test)
        metric_rows.append(metric_row(y_test, original_pred, outer_fold=fold, method="original_hybrid_xgb"))

        norm_train = l1_normalize(shap_train).reshape(len(y_train), -1)
        norm_test = l1_normalize(shap_test).reshape(len(y_test), -1)
        norm_hybrid = make_meta_model("XGBoost", args.seed + fold)
        norm_hybrid.fit(np.column_stack([pred_train, norm_train]), y_train)
        norm_hybrid_pred = norm_hybrid.predict(np.column_stack([pred_test, norm_test]))
        metric_rows.append(metric_row(y_test, norm_hybrid_pred, outer_fold=fold, method="normalized_hybrid_xgb"))

        contexts, pca_components = gate_contexts(
            x_train, x_test, pred_train, pred_test, shap_train, shap_test
        )
        for method, (context_train, context_test) in contexts.items():
            seed_predictions = []
            seed_weights = []
            for gate_seed in args.gate_seeds:
                prediction, weights, model, scaler, epochs, val_loss = fit_gate(
                    context_train,
                    pred_train,
                    y_train,
                    coords_train,
                    context_test,
                    pred_test,
                    gate_seed + fold * 100,
                )
                seed_predictions.append(prediction)
                seed_weights.append(weights)
                seed_rows.append(
                    {
                        **metric_row(y_test, prediction),
                        "outer_fold": fold,
                        "method": method,
                        "seed": gate_seed,
                        "epochs": epochs,
                        "best_validation_loss": val_loss,
                        "attribution_pca_components": pca_components,
                    }
                )
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "context_mean": scaler.mean_,
                        "context_scale": scaler.scale_,
                        "model_names": model_names,
                    },
                    paths["r3_models"] / f"fold_{fold}_{method}_seed_{gate_seed}.pt",
                )
            ensemble_prediction = np.mean(seed_predictions, axis=0)
            ensemble_weights = np.mean(seed_weights, axis=0)
            weight_std = np.std(seed_weights, axis=0)
            metric_rows.append(metric_row(y_test, ensemble_prediction, outer_fold=fold, method=method))

            entropy = -np.sum(ensemble_weights * np.log(np.maximum(ensemble_weights, 1e-12)), axis=1) / np.log(
                ensemble_weights.shape[1]
            )
            for i in range(len(y_test)):
                prediction_rows.append(
                    {
                        "outer_fold": fold,
                        "source_row": int(source_rows[i]),
                        "method": method,
                        "observed": y_test[i],
                        "predicted": ensemble_prediction[i],
                        "LON": coords_test[i, 0],
                        "LAT": coords_test[i, 1],
                    }
                )
                for j, model_name in enumerate(model_names):
                    weight_rows.append(
                        {
                            "outer_fold": fold,
                            "source_row": int(source_rows[i]),
                            "method": method,
                            "model_name": model_name,
                            "weight": ensemble_weights[i, j],
                            "weight_seed_std": weight_std[i, j],
                            "weight_entropy": entropy[i],
                            "LON": coords_test[i, 0],
                            "LAT": coords_test[i, 1],
                        }
                    )
        print(f"[Route 3] spatial fold {fold + 1}/{args.outer_folds} completed", flush=True)

    metrics = pd.DataFrame(metric_rows)
    seed_metrics = pd.DataFrame(seed_rows)
    weights = pd.DataFrame(weight_rows)
    predictions = pd.DataFrame(prediction_rows)
    stats = route3_stats(metrics)
    summary = metrics.groupby("method", as_index=False).agg(
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        R2_mean=("R2", "mean"),
    ).sort_values("RMSE_mean")

    metrics.to_csv(paths["r3_tables"] / "outer_fold_metrics.csv", index=False)
    seed_metrics.to_csv(paths["r3_tables"] / "gate_seed_stability.csv", index=False)
    weights.to_csv(paths["r3_tables"] / "dynamic_gate_weights.csv", index=False)
    predictions.to_csv(paths["r3_tables"] / "dynamic_gate_predictions.csv", index=False)
    stats.to_csv(paths["r3_tables"] / "paired_method_tests.csv", index=False)
    summary.to_csv(paths["r3_tables"] / "performance_summary.csv", index=False)
    plot_route3(metrics, weights, predictions, paths["r3_figures"])

    lookup = summary.set_index("method")
    gate = lookup.loc["gate_norm_attribution", "RMSE_mean"]
    success = bool(
        gate < lookup.loc["gate_predictions", "RMSE_mean"]
        and gate < lookup.loc["traditional_stacking_xgb", "RMSE_mean"]
    )
    raw_gate = lookup.loc["gate_raw_features", "RMSE_mean"]
    raw_gate_secondary_success = bool(
        raw_gate < lookup.loc["gate_predictions", "RMSE_mean"]
        and raw_gate < lookup.loc["traditional_stacking_xgb", "RMSE_mean"]
    )
    result = {
        "status": "completed",
        "runtime_minutes": (time.time() - start) / 60,
        "primary_success_criterion": success,
        "criterion": "Attribution gate mean spatial RMSE must be lower than prediction-only gate and traditional XGBoost stacking.",
        "secondary_raw_feature_gate_success": raw_gate_secondary_success,
        "ranking": summary.to_dict("records"),
        "mean_normalized_gate_entropy": float(
            weights.loc[weights.method == "gate_norm_attribution", "weight_entropy"].mean()
        ),
        "mean_normalized_gate_weight_seed_std": float(
            weights.loc[weights.method == "gate_norm_attribution", "weight_seed_std"].mean()
        ),
    }
    (paths["r3"] / "RESULT_SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return metrics, result


def write_readmes(paths: dict[str, Path], args, route2_result, route3_result):
    config = {
        "seed": args.seed,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "buffer_degrees": args.buffer_deg,
        "optimization_trials_each": args.optimization_trials,
        "gate_seeds": args.gate_seeds,
        "features": FEATURES,
        "target": TARGET,
        "candidate_base_models": BASE_CANDIDATES,
        "attribution_estimator": "Saabas approximation for RF/ExtraTrees/XGBoost; exact TreeSHAP for LightGBM/CatBoost; fold-level additivity audit",
    }
    (paths["root"] / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    route2_text = f"""# Route 2: validation and attribution audit

Status: {route2_result['status']}

This route removes held-out-test model selection. Candidate screening occurs only
inside each outer training partition. Random, spatial, and buffered-spatial outer
validation use the same cleaned data and report fold-level RMSE, MAE, R2, residual
Moran's I, SHAP additivity, representation ablations, and spatial conformal results.

Key machine-readable summary: `RESULT_SUMMARY.json`
Tables: `results/tables/`
Figures (PNG and PDF): `results/figures/`
Spatial fold caches consumed by Route 3: `results/fold_cache/`
"""
    (paths["r2"] / "README.md").write_text(route2_text, encoding="utf-8")
    route3_text = f"""# Route 3: explicit dynamic fusion

Status: {route3_result['status']}
Primary success criterion passed: {route3_result['primary_success_criterion']}

This route uses exactly the spatial outer folds generated by Route 2. The proposed
gate produces nonnegative sample-specific weights that sum to one. Its context is
formed from base predictions and PCA-compressed L1-normalized attribution profiles.
Prediction-only and raw-feature gates isolate whether attribution adds information.

Key machine-readable summary: `RESULT_SUMMARY.json`
Tables: `results/tables/`
Figures (PNG and PDF): `results/figures/`
Fitted gate checkpoints: `results/models/`
"""
    (paths["r3"] / "README.md").write_text(route3_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--buffer-deg", type=float, default=0.04)
    parser.add_argument("--optimization-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-seeds", type=int, nargs="+", default=[11, 42, 73])
    args = parser.parse_args()
    paths = ensure_dirs(args.output)
    total_start = time.time()
    route2_metrics, route2_result = run_route2(args.input, paths, args)
    route3_metrics, route3_result = run_route3(paths, args)
    write_readmes(paths, args, route2_result, route3_result)
    final = {
        "total_runtime_minutes": (time.time() - total_start) / 60,
        "route2": route2_result,
        "route3": route3_result,
    }
    (paths["root"] / "FINAL_RESULTS.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
