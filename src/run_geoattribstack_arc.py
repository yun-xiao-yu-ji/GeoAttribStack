"""GeoAttribStack-ARC: attribution-regime constrained stacking.

The additive SHAP component is projected out before regime discovery. Each
regime learns a simplex-constrained vector of base-model weights, and soft
regime membership produces sample-specific convex predictions. Previously
unused anti-diagonal bands and radial rings are reserved for confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_routes import (  # noqa: E402
    fit_gate,
    fit_global_convex,
    load_data,
    make_meta_model,
    make_spatial_groups,
    metric_row,
    savefig,
)
from cache_utils import cache_fold  # noqa: E402


DEVELOPMENT_SCHEMES = ["kmeans", "longitude_bands", "latitude_bands", "diagonal_bands"]
CONFIRMATION_SCHEMES = ["anti_diagonal_bands", "radial_rings"]
ALL_SCHEMES = DEVELOPMENT_SCHEMES + CONFIRMATION_SCHEMES
PRIMARY = "arc_soft_orthogonal"
GATE_SEEDS = [11, 42, 73]


def prepare_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "tables": root / "results" / "tables",
        "figures": root / "results" / "figures",
        "models": root / "results" / "models",
        "cache": root / "results" / "confirmation_fold_cache",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def confirmation_labels(coords: np.ndarray, scheme: str, n_folds: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(coords)
    if scheme == "anti_diagonal_bands":
        score = (scaled[:, 0] - scaled[:, 1]) / math.sqrt(2)
    elif scheme == "radial_rings":
        score = np.sqrt(np.sum(scaled**2, axis=1))
    else:
        raise ValueError(scheme)
    order = np.argsort(score, kind="mergesort")
    labels = np.empty(len(score), dtype=int)
    for fold, indices in enumerate(np.array_split(order, n_folds)):
        labels[indices] = fold
    return labels


def shap_representation(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "orthogonal":
        transformed = values - values.mean(axis=2, keepdims=True)
    elif kind == "raw_normalized":
        transformed = values.copy()
    else:
        raise ValueError(kind)
    denominator = np.sum(np.abs(transformed), axis=2, keepdims=True)
    transformed = transformed / np.maximum(denominator, 1e-12)
    return transformed.reshape(len(values), -1)


def membership_from_distances(distances: np.ndarray, tau: float, hard: bool) -> np.ndarray:
    if hard:
        labels = np.argmin(distances, axis=1)
        membership = np.zeros_like(distances)
        membership[np.arange(len(labels)), labels] = 1.0
        return membership
    scale = np.median(np.min(distances, axis=1)) + 1e-8
    return softmax(-distances / (tau * scale), axis=1)


def fit_regime_model(q_train, n_regimes: int, tau: float, hard: bool, seed: int):
    scaler = StandardScaler().fit(q_train)
    scaled = scaler.transform(q_train)
    clustering = KMeans(n_clusters=n_regimes, n_init=30, random_state=seed).fit(scaled)
    distances = clustering.transform(scaled)
    membership = membership_from_distances(distances, tau, hard)
    return scaler, clustering, membership


def transform_regime_model(q, scaler, clustering, tau: float, hard: bool):
    distances = clustering.transform(scaler.transform(q))
    return membership_from_distances(distances, tau, hard)


def fit_simplex_regime_weights(membership, predictions, y, regularization: float = 0.015):
    n_regimes = membership.shape[1]
    n_models = predictions.shape[1]
    y_mean, y_std = float(y.mean()), float(y.std())
    pred_scaled = (predictions - y_mean) / y_std
    target = (y - y_mean) / y_std

    def objective(logits_flat):
        regime_weights = softmax(logits_flat.reshape(n_regimes, n_models), axis=1)
        sample_weights = membership @ regime_weights
        fused = np.sum(sample_weights * pred_scaled, axis=1)
        mse = np.mean((target - fused) ** 2)
        shrinkage = np.mean((regime_weights - regime_weights.mean(axis=0, keepdims=True)) ** 2)
        return mse + regularization * shrinkage

    result = minimize(objective, np.zeros(n_regimes * n_models), method="L-BFGS-B")
    weights = softmax(result.x.reshape(n_regimes, n_models), axis=1)
    return weights


def regime_predict(membership, regime_weights, predictions):
    sample_weights = membership @ regime_weights
    return np.sum(sample_weights * predictions, axis=1), sample_weights


def inner_select(q, predictions, y, coords, seed: int):
    groups = make_spatial_groups(coords, 4, seed + 500)
    candidates = []
    for n_regimes in range(2, 7):
        for tau in [0.35, 0.70, 1.40, 2.80]:
            fold_scores = []
            for group in range(4):
                tr = groups != group
                va = groups == group
                scaler, clustering, membership_train = fit_regime_model(
                    q[tr], n_regimes, tau, False, seed + group
                )
                weights = fit_simplex_regime_weights(membership_train, predictions[tr], y[tr])
                membership_val = transform_regime_model(q[va], scaler, clustering, tau, False)
                prediction, _ = regime_predict(membership_val, weights, predictions[va])
                fold_scores.append(mean_squared_error(y[va], prediction) ** 0.5)
            score = 0.65 * max(fold_scores) + 0.35 * np.mean(fold_scores)
            candidates.append(
                {
                    "n_regimes": n_regimes,
                    "tau": tau,
                    "selection_score": score,
                    "mean_inner_RMSE": float(np.mean(fold_scores)),
                    "worst_inner_RMSE": float(max(fold_scores)),
                }
            )
    table = pd.DataFrame(candidates).sort_values("selection_score")
    best = table.iloc[0]
    return int(best.n_regimes), float(best.tau), table


def fit_arc_with_coords(q_train, pred_train, y_train, coords_train, q_test, pred_test, seed: int, hard=False):
    n_regimes, tau, selection = inner_select(q_train, pred_train, y_train, coords_train, seed)
    scaler, clustering, membership_train = fit_regime_model(q_train, n_regimes, tau, hard, seed)
    regime_weights = fit_simplex_regime_weights(membership_train, pred_train, y_train)
    membership_test = transform_regime_model(q_test, scaler, clustering, tau, hard)
    prediction, sample_weights = regime_predict(membership_test, regime_weights, pred_test)
    return {
        "prediction": prediction,
        "sample_weights": sample_weights,
        "membership": membership_test,
        "regime_weights": regime_weights,
        "scaler": scaler,
        "clustering": clustering,
        "n_regimes": n_regimes,
        "tau": tau,
        "selection": selection,
    }


def regime_stability(q_train, n_regimes: int, reference_labels: np.ndarray, seed: int):
    scaled = StandardScaler().fit_transform(q_train)
    scores = []
    for offset in [17, 53, 101, 211]:
        labels = KMeans(n_clusters=n_regimes, n_init=30, random_state=seed + offset).fit_predict(scaled)
        scores.append(adjusted_rand_score(reference_labels, labels))
    return float(np.mean(scores)), float(np.min(scores))


def paired_test(metrics, subset: str, target: str, baselines: list[str]):
    data = metrics[metrics.evaluation_subset == subset]
    pivot = data.pivot(index=["scheme", "outer_fold"], columns="method", values="RMSE")
    rows = []
    for baseline in baselines:
        delta = pivot[target] - pivot[baseline]
        try:
            p_value = float(wilcoxon(delta).pvalue)
        except ValueError:
            p_value = 1.0
        rng = np.random.default_rng(42)
        bootstrap = rng.choice(delta.to_numpy(), size=(20000, len(delta)), replace=True).mean(axis=1)
        rows.append(
            {
                "subset": subset,
                "method": target,
                "baseline": baseline,
                "mean_delta_RMSE": float(delta.mean()),
                "relative_change_percent": float(100 * delta.mean() / pivot[baseline].mean()),
                "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
                "wilcoxon_p": p_value,
                "folds_improved": int(np.sum(delta < 0)),
                "total_folds": int(len(delta)),
            }
        )
    return pd.DataFrame(rows)


def plot_results(metrics, selections, weights, regimes, figures: Path):
    confirmation = metrics[metrics.evaluation_subset == "confirmation"]
    order = (
        confirmation.groupby("method").RMSE.mean().sort_values().index.tolist()
    )
    data = [confirmation.loc[confirmation.method == method, "RMSE"].to_numpy() for method in order]
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    boxes = ax.boxplot(data, tick_labels=[m.replace("_", " ") for m in order], patch_artist=True, showmeans=True)
    palette = ["#009E73", "#56B4E9", "#E69F00", "#0072B2", "#CC79A7", "#D55E00", "#999999"]
    for patch, color in zip(boxes["boxes"], palette * 3):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Confirmation spatial RMSE (mm yr$^{-1}$)")
    ax.set_title("GeoAttribStack-ARC on previously unused spatial partitions")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, figures / "ARC_Fig1_confirmation_comparison")

    summary = metrics.groupby(["evaluation_subset", "method"], as_index=False).RMSE.mean()
    methods = [PRIMARY, "arc_soft_raw_normalized", "orthogonal_shap_gate", "prediction_gate", "traditional_stacking_xgb"]
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    x = np.arange(2)
    width = 0.16
    for index, method in enumerate(methods):
        part = summary[summary.method == method].set_index("evaluation_subset").reindex(["development", "confirmation"])
        ax.bar(x + (index - 2) * width, part.RMSE, width, label=method.replace("_", " "))
    ax.set_xticks(x, ["Development partitions", "Confirmation partitions"])
    ax.set_ylabel("Mean RMSE (mm yr$^{-1}$)")
    ax.set_title("Development-to-confirmation transportability")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, figures / "ARC_Fig2_development_confirmation")

    pivot = confirmation.pivot(index=["scheme", "outer_fold"], columns="method", values="RMSE")
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    for baseline, color in [("prediction_gate", "#0072B2"), ("arc_soft_raw_normalized", "#E69F00")]:
        gain = pivot[baseline] - pivot[PRIMARY]
        ax.plot(np.arange(1, len(gain) + 1), gain, marker="o", label=baseline.replace("_", " "), color=color)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Confirmation outer fold")
    ax.set_ylabel("RMSE reduction of ARC; positive favors ARC")
    ax.set_title("Does orthogonal attribution add confirmatory value?")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    savefig(fig, figures / "ARC_Fig3_foldwise_confirmation_gain")

    example = regimes[(regimes.scheme == "anti_diagonal_bands") & (regimes.outer_fold == 0)]
    fig, ax = plt.subplots(figsize=(6.5, 5.1))
    scatter = ax.scatter(example.LON, example.LAT, c=example.dominant_regime, cmap="tab10", s=7, alpha=0.85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Soft attribution regimes in a confirmation fold")
    ax.set_aspect("equal", adjustable="box")
    legend = ax.legend(*scatter.legend_elements(), title="Regime", frameon=False)
    ax.add_artist(legend)
    savefig(fig, figures / "ARC_Fig4_confirmation_regime_map")

    example_weights = weights[(weights.scheme == "anti_diagonal_bands") & (weights.outer_fold == 0)]
    matrix = example_weights.pivot(index="regime", columns="model_name", values="weight")
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    image = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), [f"Regime {idx}" for idx in matrix.index])
    ax.set_title("Simplex-constrained expert weights by attribution regime")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Weight")
    fig.tight_layout()
    savefig(fig, figures / "ARC_Fig5_regime_weight_matrix")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    confirmation_selection = selections[
        (selections.evaluation_subset == "confirmation") & (selections.representation == "orthogonal")
    ]
    counts = confirmation_selection.groupby("n_regimes").size().reindex(range(2, 7), fill_value=0)
    ax.bar(counts.index.astype(str), counts.values, color="#009E73")
    ax.set_xlabel("Selected number of attribution regimes")
    ax.set_ylabel("Outer folds")
    ax.set_title("Inner-spatial selection of model complexity")
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, figures / "ARC_Fig6_selected_regime_count")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    paths = prepare_dirs(args.output)
    started = time.time()
    clean, x, y, coords = load_data(args.input)

    metric_rows, selection_rows, weight_rows, regime_rows, prediction_rows = [], [], [], [], []
    for scheme_index, scheme in enumerate(ALL_SCHEMES):
        if scheme in CONFIRMATION_SCHEMES:
            labels = confirmation_labels(coords, scheme, args.folds)
        else:
            assignments = pd.read_csv(
                args.source_cache.parent / "tables" / "spatial_partition_assignments.csv"
            )
            labels = (
                assignments[assignments.scheme == scheme]
                .sort_values("source_row")
                .outer_fold.to_numpy(dtype=int)
            )
        subset = "confirmation" if scheme in CONFIRMATION_SCHEMES else "development"

        for fold in range(args.folds):
            fold_start = time.time()
            if scheme in DEVELOPMENT_SCHEMES:
                cache_path = args.source_cache / f"{scheme}_fold_{fold}.npz"
                cache = np.load(cache_path, allow_pickle=False)
            else:
                train_idx = np.flatnonzero(labels != fold)
                test_idx = np.flatnonzero(labels == fold)
                cache = cache_fold(
                    paths,
                    scheme,
                    fold,
                    train_idx,
                    test_idx,
                    clean,
                    x,
                    y,
                    coords,
                    args.seed + 700 + scheme_index * 101,
                )

            y_train, coords_train = cache["y_train"], cache["coords_train"]
            y_test, coords_test = cache["y_test"], cache["coords_test"]
            pred_train, pred_test = cache["pred_train"], cache["pred_test"]
            shap_train, shap_test = cache["shap_train"], cache["shap_test"]
            model_names = cache["model_names"].astype(str).tolist()
            source_rows = cache["source_rows_test"]

            def add_metric(method, prediction):
                metric_rows.append(
                    metric_row(
                        y_test,
                        prediction,
                        evaluation_subset=subset,
                        scheme=scheme,
                        outer_fold=fold,
                        method=method,
                    )
                )
                if method in {PRIMARY, "prediction_gate", "arc_soft_raw_normalized", "traditional_stacking_xgb"}:
                    for idx in range(len(y_test)):
                        prediction_rows.append(
                            {
                                "evaluation_subset": subset,
                                "scheme": scheme,
                                "outer_fold": fold,
                                "source_row": int(source_rows[idx]),
                                "method": method,
                                "observed": y_test[idx],
                                "predicted": prediction[idx],
                                "LON": coords_test[idx, 0],
                                "LAT": coords_test[idx, 1],
                            }
                        )

            add_metric("simple_average", pred_test.mean(axis=1))
            global_prediction, _ = fit_global_convex(pred_train, y_train, pred_test)
            add_metric("global_convex", global_prediction)
            traditional = make_meta_model("XGBoost", args.seed + fold)
            traditional.fit(pred_train, y_train)
            add_metric("traditional_stacking_xgb", traditional.predict(pred_test))
            full_train = np.column_stack([pred_train, shap_train.reshape(len(y_train), -1)])
            full_test = np.column_stack([pred_test, shap_test.reshape(len(y_test), -1)])
            original = make_meta_model("XGBoost", args.seed + fold)
            original.fit(full_train, y_train)
            add_metric("original_full_shap_xgb", original.predict(full_test))

            pred_gate_predictions = []
            for gate_seed in GATE_SEEDS:
                prediction, _, _, _, _, _ = fit_gate(
                    pred_train,
                    pred_train,
                    y_train,
                    coords_train,
                    pred_test,
                    pred_test,
                    gate_seed + fold + scheme_index * 101,
                )
                pred_gate_predictions.append(prediction)
            add_metric("prediction_gate", np.mean(pred_gate_predictions, axis=0))

            q_train_orth = shap_representation(shap_train, "orthogonal")
            q_test_orth = shap_representation(shap_test, "orthogonal")
            q_train_raw = shap_representation(shap_train, "raw_normalized")
            q_test_raw = shap_representation(shap_test, "raw_normalized")

            direct_predictions = []
            for gate_seed in GATE_SEEDS:
                prediction, _, _, _, _, _ = fit_gate(
                    q_train_orth,
                    pred_train,
                    y_train,
                    coords_train,
                    q_test_orth,
                    pred_test,
                    gate_seed + fold + scheme_index * 101,
                )
                direct_predictions.append(prediction)
            add_metric("orthogonal_shap_gate", np.mean(direct_predictions, axis=0))

            arc_orth = fit_arc_with_coords(
                q_train_orth,
                pred_train,
                y_train,
                coords_train,
                q_test_orth,
                pred_test,
                args.seed + fold + scheme_index * 101,
                hard=False,
            )
            add_metric(PRIMARY, arc_orth["prediction"])
            reference_labels = arc_orth["clustering"].labels_
            stability_mean, stability_min = regime_stability(
                q_train_orth,
                arc_orth["n_regimes"],
                reference_labels,
                args.seed + fold,
            )
            selection_rows.append(
                {
                    "evaluation_subset": subset,
                    "scheme": scheme,
                    "outer_fold": fold,
                    "representation": "orthogonal",
                    "n_regimes": arc_orth["n_regimes"],
                    "tau": arc_orth["tau"],
                    "stability_ARI_mean": stability_mean,
                    "stability_ARI_min": stability_min,
                }
            )
            arc_orth["selection"].to_csv(
                paths["tables"] / f"inner_selection_{scheme}_fold_{fold}_orthogonal.csv", index=False
            )

            for regime_idx in range(arc_orth["regime_weights"].shape[0]):
                for model_idx, model_name in enumerate(model_names):
                    weight_rows.append(
                        {
                            "evaluation_subset": subset,
                            "scheme": scheme,
                            "outer_fold": fold,
                            "regime": regime_idx,
                            "model_name": model_name,
                            "weight": arc_orth["regime_weights"][regime_idx, model_idx],
                        }
                    )
            for idx in range(len(y_test)):
                regime_rows.append(
                    {
                        "evaluation_subset": subset,
                        "scheme": scheme,
                        "outer_fold": fold,
                        "source_row": int(source_rows[idx]),
                        "dominant_regime": int(np.argmax(arc_orth["membership"][idx])),
                        "max_membership": float(np.max(arc_orth["membership"][idx])),
                        "LON": coords_test[idx, 0],
                        "LAT": coords_test[idx, 1],
                    }
                )

            hard_orth = fit_arc_with_coords(
                q_train_orth,
                pred_train,
                y_train,
                coords_train,
                q_test_orth,
                pred_test,
                args.seed + fold + scheme_index * 101,
                hard=True,
            )
            add_metric("arc_hard_orthogonal", hard_orth["prediction"])

            arc_raw = fit_arc_with_coords(
                q_train_raw,
                pred_train,
                y_train,
                coords_train,
                q_test_raw,
                pred_test,
                args.seed + fold + scheme_index * 101,
                hard=False,
            )
            add_metric("arc_soft_raw_normalized", arc_raw["prediction"])
            selection_rows.append(
                {
                    "evaluation_subset": subset,
                    "scheme": scheme,
                    "outer_fold": fold,
                    "representation": "raw_normalized",
                    "n_regimes": arc_raw["n_regimes"],
                    "tau": arc_raw["tau"],
                    "stability_ARI_mean": np.nan,
                    "stability_ARI_min": np.nan,
                }
            )
            print(
                f"[GeoAttribStack-ARC] {scheme} fold {fold + 1}/{args.folds} completed in "
                f"{(time.time() - fold_start) / 60:.2f} min",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    selections = pd.DataFrame(selection_rows)
    weights = pd.DataFrame(weight_rows)
    regimes = pd.DataFrame(regime_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = metrics.groupby(["evaluation_subset", "method"], as_index=False).agg(
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        R2_mean=("R2", "mean"),
    ).sort_values(["evaluation_subset", "RMSE_mean"])
    baselines = [
        "prediction_gate",
        "orthogonal_shap_gate",
        "arc_soft_raw_normalized",
        "arc_hard_orthogonal",
        "traditional_stacking_xgb",
        "original_full_shap_xgb",
        "global_convex",
    ]
    tests = pd.concat(
        [paired_test(metrics, subset, PRIMARY, baselines) for subset in ["development", "confirmation"]],
        ignore_index=True,
    )
    metrics.to_csv(paths["tables"] / "outer_fold_metrics.csv", index=False)
    summary.to_csv(paths["tables"] / "performance_summary.csv", index=False)
    tests.to_csv(paths["tables"] / "paired_tests.csv", index=False)
    selections.to_csv(paths["tables"] / "selected_regime_hyperparameters.csv", index=False)
    weights.to_csv(paths["tables"] / "regime_expert_weights.csv", index=False)
    regimes.to_csv(paths["tables"] / "test_regime_assignments.csv", index=False)
    predictions.to_csv(paths["tables"] / "selected_method_predictions.csv", index=False)
    plot_results(metrics, selections, weights, regimes, paths["figures"])

    confirmation_summary = summary[summary.evaluation_subset == "confirmation"].set_index("method")
    confirmation_tests = tests[tests.subset == "confirmation"].set_index("baseline")
    primary_rmse = float(confirmation_summary.loc[PRIMARY, "RMSE_mean"])
    prediction_rmse = float(confirmation_summary.loc["prediction_gate", "RMSE_mean"])
    raw_arc_rmse = float(confirmation_summary.loc["arc_soft_raw_normalized", "RMSE_mean"])
    success = bool(
        primary_rmse <= 0.98 * prediction_rmse
        and primary_rmse < raw_arc_rmse
        and confirmation_tests.loc["prediction_gate", "bootstrap_95_high"] < 0
        and confirmation_tests.loc["prediction_gate", "folds_improved"] >= 8
    )
    result = {
        "status": "completed",
        "runtime_minutes": (time.time() - started) / 60,
        "primary_method": PRIMARY,
        "primary_success": success,
        "success_rule": "On the 10 previously unused confirmation folds, ARC must improve prediction-only gating by at least 2%, beat raw-normalized ARC, improve at least 8 folds, and have a bootstrap 95% upper bound below zero.",
        "confirmation_primary_RMSE": primary_rmse,
        "confirmation_prediction_gate_RMSE": prediction_rmse,
        "confirmation_raw_normalized_ARC_RMSE": raw_arc_rmse,
        "mean_regime_stability_ARI": float(selections[selections.representation == "orthogonal"].stability_ARI_mean.mean()),
        "confirmation_ranking": summary[summary.evaluation_subset == "confirmation"].to_dict("records"),
        "confirmation_tests": tests[tests.subset == "confirmation"].to_dict("records"),
        "development_tests": tests[tests.subset == "development"].to_dict("records"),
    }
    (paths["root"] / "RESULT_SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    readme = f"""# GeoAttribStack-ARC

Status: completed
Primary confirmatory success: {success}

The additive SHAP component is projected out before attribution-regime
discovery. Soft regime membership selects simplex-constrained base-model
weights. Anti-diagonal bands and radial rings were reserved as 10 previously
unused confirmation folds.

- Result summary: `RESULT_SUMMARY.json`
- Fold metrics and paired tests: `results/tables/`
- Publication figures: `results/figures/`
- Confirmation fold caches: `results/confirmation_fold_cache/`
- Source: `../src/run_geoattribstack_arc.py`
"""
    (paths["root"] / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
