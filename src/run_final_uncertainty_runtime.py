"""Final GeoAttribStack uncertainty and runtime experiment.

This script consumes the locked confirmation-fold caches produced by the
GeoAttribStack-ARC experiment. It recreates the final ARC point predictor,
constructs spatial cross-fitted residuals inside each outer-training region,
and evaluates standard and normalized-score conformal intervals on the untouched
outer spatial test regions.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


SCHEMES = ["anti_diagonal_bands", "radial_rings"]
ALPHA = 0.10
BETA = 0.50
REGULARIZATION = 0.015


def prepare_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "tables": root / "results" / "tables",
        "figures": root / "results" / "figures",
        "summaries": root / "results" / "summaries",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def spatial_groups(coords: np.ndarray, n_groups: int, seed: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(coords)
    return KMeans(n_clusters=n_groups, n_init=20, random_state=seed).fit_predict(scaled)


def shap_representation(values: np.ndarray) -> np.ndarray:
    transformed = values - values.mean(axis=2, keepdims=True)
    denominator = np.sum(np.abs(transformed), axis=2, keepdims=True)
    transformed = transformed / np.maximum(denominator, 1e-12)
    return transformed.reshape(len(values), -1)


def membership_from_distances(distances: np.ndarray, tau: float) -> np.ndarray:
    scale = np.median(np.min(distances, axis=1)) + 1e-8
    return softmax(-distances / (tau * scale), axis=1)


def fit_regime_model(q_train: np.ndarray, n_regimes: int, tau: float, seed: int):
    scaler = StandardScaler().fit(q_train)
    scaled = scaler.transform(q_train)
    clustering = KMeans(n_clusters=n_regimes, n_init=30, random_state=seed).fit(scaled)
    membership = membership_from_distances(clustering.transform(scaled), tau)
    return scaler, clustering, membership


def transform_regime_model(
    q: np.ndarray, scaler: StandardScaler, clustering: KMeans, tau: float
) -> np.ndarray:
    distances = clustering.transform(scaler.transform(q))
    return membership_from_distances(distances, tau)


def fit_simplex_weights(
    membership: np.ndarray,
    predictions: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    n_regimes = membership.shape[1]
    n_models = predictions.shape[1]
    y_mean, y_std = float(y.mean()), float(y.std())
    pred_scaled = (predictions - y_mean) / y_std
    target = (y - y_mean) / y_std

    def objective(logits_flat: np.ndarray) -> float:
        regime_weights = softmax(logits_flat.reshape(n_regimes, n_models), axis=1)
        sample_weights = membership @ regime_weights
        fused = np.sum(sample_weights * pred_scaled, axis=1)
        mse = np.mean((target - fused) ** 2)
        shrinkage = np.mean(
            (regime_weights - regime_weights.mean(axis=0, keepdims=True)) ** 2
        )
        return float(mse + REGULARIZATION * shrinkage)

    result = minimize(
        objective,
        np.zeros(n_regimes * n_models),
        method="L-BFGS-B",
    )
    return softmax(result.x.reshape(n_regimes, n_models), axis=1)


def fit_fixed_arc(
    q_train: np.ndarray,
    pred_train: np.ndarray,
    y_train: np.ndarray,
    q_test: np.ndarray,
    pred_test: np.ndarray,
    n_regimes: int,
    tau: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler, clustering, membership_train = fit_regime_model(
        q_train, n_regimes, tau, seed
    )
    regime_weights = fit_simplex_weights(membership_train, pred_train, y_train)
    membership_test = transform_regime_model(q_test, scaler, clustering, tau)
    sample_weights = membership_test @ regime_weights
    prediction = np.sum(sample_weights * pred_test, axis=1)
    return prediction, sample_weights, membership_test


def inner_select(
    q: np.ndarray,
    predictions: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    seed: int,
) -> tuple[int, float, pd.DataFrame]:
    groups = spatial_groups(coords, 4, seed + 500)
    rows = []
    for n_regimes in range(2, 7):
        for tau in [0.35, 0.70, 1.40, 2.80]:
            scores = []
            for group in range(4):
                train = groups != group
                valid = groups == group
                pred, _, _ = fit_fixed_arc(
                    q[train],
                    predictions[train],
                    y[train],
                    q[valid],
                    predictions[valid],
                    n_regimes,
                    tau,
                    seed + group,
                )
                scores.append(mean_squared_error(y[valid], pred) ** 0.5)
            rows.append(
                {
                    "n_regimes": n_regimes,
                    "tau": tau,
                    "mean_inner_RMSE": float(np.mean(scores)),
                    "worst_inner_RMSE": float(np.max(scores)),
                    "selection_score": float(0.35 * np.mean(scores) + 0.65 * np.max(scores)),
                }
            )
    table = pd.DataFrame(rows).sort_values("selection_score")
    best = table.iloc[0]
    return int(best.n_regimes), float(best.tau), table


def finite_quantile(scores: np.ndarray, alpha: float) -> float:
    level = min(math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores), 1.0)
    return float(np.quantile(scores, level, method="higher"))


def cross_fitted_arc_predictions(
    q: np.ndarray,
    predictions: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    n_regimes: int,
    tau: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = spatial_groups(coords, 4, seed + 1700)
    oof = np.zeros(len(y))
    for group in range(4):
        train = groups != group
        valid = groups == group
        oof[valid], _, _ = fit_fixed_arc(
            q[train],
            predictions[train],
            y[train],
            q[valid],
            predictions[valid],
            n_regimes,
            tau,
            seed + 1800 + group,
        )
    return oof, groups


def normalized_conformal(
    q_train: np.ndarray,
    pred_train: np.ndarray,
    y_train: np.ndarray,
    coords_train: np.ndarray,
    q_test: np.ndarray,
    pred_test: np.ndarray,
    y_test: np.ndarray,
    test_point: np.ndarray,
    n_regimes: int,
    tau: float,
    seed: int,
) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    oof_point, groups = cross_fitted_arc_predictions(
        q_train,
        pred_train,
        y_train,
        coords_train,
        n_regimes,
        tau,
        seed,
    )
    abs_residual = np.abs(y_train - oof_point)
    scale_features_train = np.column_stack([pred_train, q_train])
    scale_features_test = np.column_stack([pred_test, q_test])
    oof_scale = np.zeros(len(y_train))
    for group in range(4):
        train = groups != group
        valid = groups == group
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=seed + 2500 + group,
            n_jobs=-1,
        )
        model.fit(scale_features_train[train], abs_residual[train])
        oof_scale[valid] = model.predict(scale_features_train[valid]) + BETA

    final_scale = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        random_state=seed + 2600,
        n_jobs=-1,
    )
    final_scale.fit(scale_features_train, abs_residual)
    test_scale = final_scale.predict(scale_features_test) + BETA

    q_standard = finite_quantile(abs_residual, ALPHA)
    q_adaptive = finite_quantile(abs_residual / oof_scale, ALPHA)
    standard_margin = np.full(len(y_test), q_standard)
    adaptive_margin = q_adaptive * test_scale

    rows = []
    details = {"point": test_point}
    for method, margin in [
        ("standard_spatial_cross_conformal", standard_margin),
        ("adaptive_normalized_spatial_cross_conformal", adaptive_margin),
    ]:
        lower = test_point - margin
        upper = test_point + margin
        covered = (y_test >= lower) & (y_test <= upper)
        rows.append(
            {
                "method": method,
                "coverage": float(covered.mean()),
                "mean_width": float(np.mean(2 * margin)),
                "median_width": float(np.median(2 * margin)),
                "width_std": float(np.std(2 * margin)),
            }
        )
        details[f"{method}_lower"] = lower
        details[f"{method}_upper"] = upper
        details[f"{method}_width"] = 2 * margin
        details[f"{method}_covered"] = covered
    return rows, details


def plot_results(
    conformal: pd.DataFrame,
    points: pd.DataFrame,
    runtime: pd.DataFrame,
    figures: Path,
) -> None:
    summary = conformal.groupby("method", as_index=False).agg(
        coverage=("coverage", "mean"),
        width=("mean_width", "mean"),
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    colors = ["#0072B2", "#009E73"]
    for (_, row), color in zip(summary.iterrows(), colors):
        label = row.method.replace("_spatial_cross_conformal", "").replace("_", " ")
        ax.scatter(row.width, row.coverage, s=110, color=color, label=label)
    ax.axhline(1 - ALPHA, color="black", linestyle="--", linewidth=1.2, label="90% target")
    ax.set_xlabel("Mean interval width")
    ax.set_ylabel("Mean spatial-fold coverage")
    ax.set_title("GeoAttribStack spatial cross-conformal performance")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / "GAS_Fig1_conformal_coverage_width")

    pivot = conformal.pivot(index=["scheme", "outer_fold"], columns="method", values="coverage")
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(pivot))
    width = 0.38
    standard = "standard_spatial_cross_conformal"
    adaptive = "adaptive_normalized_spatial_cross_conformal"
    ax.bar(x - width / 2, pivot[standard], width, color="#0072B2", label="Standard")
    ax.bar(x + width / 2, pivot[adaptive], width, color="#009E73", label="Adaptive normalized")
    ax.axhline(1 - ALPHA, color="black", linestyle="--", linewidth=1.2)
    ax.set_xticks(x, [f"{s[:3]}-{f + 1}" for s, f in pivot.index], rotation=30)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage across independent spatial holdouts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / "GAS_Fig2_foldwise_coverage")

    example = points[
        (points.scheme == "radial_rings")
        & (points.outer_fold == 0)
        & (points.method == "adaptive_normalized_spatial_cross_conformal")
    ]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    scatter = ax.scatter(
        example.LON,
        example.LAT,
        c=example.interval_width,
        cmap="YlOrRd",
        s=8,
        alpha=0.85,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial distribution of adaptive interval width")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(scatter, ax=ax, label="Prediction interval width")
    fig.tight_layout()
    save_figure(fig, figures / "GAS_Fig3_adaptive_width_map")

    stages = [
        "attribution_representation_seconds",
        "inner_selection_seconds",
        "final_arc_fit_seconds",
        "conformal_calibration_seconds",
    ]
    labels = ["Attribution representation", "Inner spatial selection", "Final ARC fit", "Conformal calibration"]
    means = [runtime[stage].mean() for stage in stages]
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.barh(labels, means, color=["#56B4E9", "#E69F00", "#009E73", "#0072B2"])
    ax.set_xlabel("Mean runtime per outer fold (s)")
    ax.set_title("GeoAttribStack computational profile")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, figures / "GAS_Fig4_runtime_profile")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    paths = prepare_dirs(args.output)
    selected = pd.read_csv(
        args.arc_root / "results" / "tables" / "selected_regime_hyperparameters.csv"
    )
    selected = selected[
        (selected.evaluation_subset == "confirmation")
        & (selected.representation == "orthogonal")
    ].set_index(["scheme", "outer_fold"])

    conformal_rows = []
    point_rows = []
    runtime_rows = []
    reproduction_rows = []
    for scheme_index, scheme in enumerate(SCHEMES):
        for fold in range(5):
            fold_started = time.perf_counter()
            cache_path = (
                args.arc_root
                / "results"
                / "confirmation_fold_cache"
                / f"{scheme}_fold_{fold}.npz"
            )
            cache = np.load(cache_path, allow_pickle=False)
            y_train, y_test = cache["y_train"], cache["y_test"]
            coords_train, coords_test = cache["coords_train"], cache["coords_test"]
            pred_train, pred_test = cache["pred_train"], cache["pred_test"]
            source_rows = cache["source_rows_test"]

            started = time.perf_counter()
            q_train = shap_representation(cache["shap_train"])
            q_test = shap_representation(cache["shap_test"])
            representation_seconds = time.perf_counter() - started

            model_seed = args.seed + fold + scheme_index * 101
            started = time.perf_counter()
            selected_k, selected_tau, selection_table = inner_select(
                q_train, pred_train, y_train, coords_train, model_seed
            )
            selection_seconds = time.perf_counter() - started
            selection_table.to_csv(
                paths["tables"] / f"inner_selection_{scheme}_fold_{fold}.csv",
                index=False,
            )

            recorded_k = int(selected.loc[(scheme, fold), "n_regimes"])
            recorded_tau = float(selected.loc[(scheme, fold), "tau"])
            if selected_k != recorded_k or not np.isclose(selected_tau, recorded_tau):
                raise RuntimeError(
                    f"Hyperparameter reproduction failed for {scheme} fold {fold}: "
                    f"computed {(selected_k, selected_tau)}, recorded {(recorded_k, recorded_tau)}"
                )

            started = time.perf_counter()
            test_point, sample_weights, membership = fit_fixed_arc(
                q_train,
                pred_train,
                y_train,
                q_test,
                pred_test,
                selected_k,
                selected_tau,
                model_seed,
            )
            final_fit_seconds = time.perf_counter() - started
            reproduction_rows.append(
                {
                    "scheme": scheme,
                    "outer_fold": fold,
                    "n_regimes": selected_k,
                    "tau": selected_tau,
                    "RMSE": float(mean_squared_error(y_test, test_point) ** 0.5),
                    "MAE": float(mean_absolute_error(y_test, test_point)),
                    "R2": float(r2_score(y_test, test_point)),
                    "mean_max_membership": float(np.max(membership, axis=1).mean()),
                    "mean_weight_entropy": float(
                        np.mean(-np.sum(sample_weights * np.log(np.maximum(sample_weights, 1e-12)), axis=1))
                    ),
                }
            )

            started = time.perf_counter()
            cp_rows, details = normalized_conformal(
                q_train,
                pred_train,
                y_train,
                coords_train,
                q_test,
                pred_test,
                y_test,
                test_point,
                selected_k,
                selected_tau,
                model_seed,
            )
            conformal_seconds = time.perf_counter() - started
            for row in cp_rows:
                conformal_rows.append({"scheme": scheme, "outer_fold": fold, **row})
                method = str(row["method"])
                lower = details[f"{method}_lower"]
                upper = details[f"{method}_upper"]
                width_values = details[f"{method}_width"]
                covered = details[f"{method}_covered"]
                for idx in range(len(y_test)):
                    point_rows.append(
                        {
                            "scheme": scheme,
                            "outer_fold": fold,
                            "source_row": int(source_rows[idx]),
                            "method": method,
                            "observed": float(y_test[idx]),
                            "predicted": float(test_point[idx]),
                            "lower": float(lower[idx]),
                            "upper": float(upper[idx]),
                            "interval_width": float(width_values[idx]),
                            "covered": bool(covered[idx]),
                            "LON": float(coords_test[idx, 0]),
                            "LAT": float(coords_test[idx, 1]),
                        }
                    )

            runtime_rows.append(
                {
                    "scheme": scheme,
                    "outer_fold": fold,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "attribution_representation_seconds": representation_seconds,
                    "inner_selection_seconds": selection_seconds,
                    "final_arc_fit_seconds": final_fit_seconds,
                    "conformal_calibration_seconds": conformal_seconds,
                    "total_seconds": time.perf_counter() - fold_started,
                }
            )
            print(f"[Final experiment] {scheme} fold {fold + 1}/5 complete", flush=True)

    conformal = pd.DataFrame(conformal_rows)
    points = pd.DataFrame(point_rows)
    runtime = pd.DataFrame(runtime_rows)
    reproduction = pd.DataFrame(reproduction_rows)
    conformal.to_csv(paths["tables"] / "conformal_fold_metrics.csv", index=False)
    points.to_csv(paths["tables"] / "conformal_point_results.csv", index=False)
    runtime.to_csv(paths["tables"] / "runtime_by_fold.csv", index=False)
    reproduction.to_csv(paths["tables"] / "arc_reproduction_metrics.csv", index=False)

    conformal_summary = conformal.groupby("method", as_index=False).agg(
        mean_coverage=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        mean_width=("mean_width", "mean"),
        median_width=("median_width", "mean"),
    )
    conformal_summary.to_csv(paths["tables"] / "conformal_summary.csv", index=False)
    runtime_summary = runtime.drop(columns=["scheme", "outer_fold"]).mean(numeric_only=True)
    runtime_summary.rename("mean").to_csv(paths["tables"] / "runtime_summary.csv")
    plot_results(conformal, points, runtime, paths["figures"])

    adaptive_name = "adaptive_normalized_spatial_cross_conformal"
    standard_name = "standard_spatial_cross_conformal"
    adaptive = conformal_summary.set_index("method").loc[adaptive_name]
    standard = conformal_summary.set_index("method").loc[standard_name]
    summary = {
        "status": "completed",
        "target_coverage": 1 - ALPHA,
        "confirmation_folds": int(len(reproduction)),
        "point_prediction": {
            "mean_RMSE": float(reproduction.RMSE.mean()),
            "mean_MAE": float(reproduction.MAE.mean()),
            "mean_R2": float(reproduction.R2.mean()),
        },
        "standard_conformal": standard.to_dict(),
        "adaptive_normalized_conformal": adaptive.to_dict(),
        "runtime_mean_seconds": runtime_summary.to_dict(),
        "hyperparameter_reproduction_passed": True,
    }
    (paths["summaries"] / "FINAL_RESULT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = f"""# Final GeoAttribStack experiment

## Point prediction

- Confirmation folds: {len(reproduction)}
- Mean RMSE: {reproduction.RMSE.mean():.3f}
- Mean MAE: {reproduction.MAE.mean():.3f}
- Mean R2: {reproduction.R2.mean():.3f}

## Spatial cross-conformal uncertainty

- Target coverage: {1 - ALPHA:.1%}
- Standard coverage: {standard.mean_coverage:.3f}
- Standard mean width: {standard.mean_width:.3f}
- Adaptive normalized coverage: {adaptive.mean_coverage:.3f}
- Adaptive normalized mean width: {adaptive.mean_width:.3f}

## Runtime

- Mean attribution representation time per fold: {runtime.attribution_representation_seconds.mean():.3f} s
- Mean inner spatial selection time per fold: {runtime.inner_selection_seconds.mean():.3f} s
- Mean final ARC fit time per fold: {runtime.final_arc_fit_seconds.mean():.3f} s
- Mean conformal calibration time per fold: {runtime.conformal_calibration_seconds.mean():.3f} s
- Mean total time per fold: {runtime.total_seconds.mean():.3f} s

All ARC hyperparameters were independently recomputed and matched the locked
confirmation experiment records.
"""
    (paths["summaries"] / "FINAL_RESULTS.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
