"""Test whether ARC benefits from the sample-to-SHAP correspondence.

The fitted base predictions, spatial confirmation folds, ARC complexity, and
all optimization settings are held fixed. Only SHAP rows are independently
permuted in the outer-training and outer-test sets. This creates a direct
negative control for attribution-guided fusion without introducing a second
prediction-only gating architecture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

CONFIRMATION_SCHEMES = ["anti_diagonal_bands", "radial_rings"]


def shap_representation(values: np.ndarray, kind: str) -> np.ndarray:
    if kind != "orthogonal":
        raise ValueError(kind)
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
    return membership_from_distances(clustering.transform(scaler.transform(q)), tau)


def fit_simplex_regime_weights(
    membership: np.ndarray,
    predictions: np.ndarray,
    y: np.ndarray,
    regularization: float = 0.015,
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
        return float(mse + regularization * shrinkage)

    result = minimize(
        objective, np.zeros(n_regimes * n_models), method="L-BFGS-B"
    )
    return softmax(result.x.reshape(n_regimes, n_models), axis=1)


def regime_predict(
    membership: np.ndarray,
    regime_weights: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sample_weights = membership @ regime_weights
    return np.sum(sample_weights * predictions, axis=1), sample_weights


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    output = args.arc_root / "results" / "shap_permutation"
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.arc_root / "results" / "tables" / "selected_regime_hyperparameters.csv")
    selected = selected[
        (selected.evaluation_subset == "confirmation")
        & (selected.representation == "orthogonal")
    ].set_index(["scheme", "outer_fold"])
    existing = pd.read_csv(args.arc_root / "results" / "tables" / "outer_fold_metrics.csv")
    true_arc = existing[
        (existing.evaluation_subset == "confirmation")
        & (existing.method == "arc_soft_orthogonal")
    ].set_index(["scheme", "outer_fold"])

    rows: list[dict[str, float | int | str]] = []
    for scheme_index, scheme in enumerate(CONFIRMATION_SCHEMES):
        for fold in range(5):
            cache_path = args.arc_root / "results" / "confirmation_fold_cache" / f"{scheme}_fold_{fold}.npz"
            cache = np.load(cache_path, allow_pickle=False)
            y_train, y_test = cache["y_train"], cache["y_test"]
            pred_train, pred_test = cache["pred_train"], cache["pred_test"]
            q_train = shap_representation(cache["shap_train"], "orthogonal")
            q_test = shap_representation(cache["shap_test"], "orthogonal")
            n_regimes = int(selected.loc[(scheme, fold), "n_regimes"])
            tau = float(selected.loc[(scheme, fold), "tau"])

            true_row = true_arc.loc[(scheme, fold)]
            rows.append(
                {
                    "scheme": scheme,
                    "outer_fold": fold,
                    "repeat": -1,
                    "method": "GeoAttribStack_ARC_true_SHAP",
                    "RMSE": float(true_row.RMSE),
                    "MAE": float(true_row.MAE),
                    "R2": float(true_row.R2),
                }
            )

            for repeat in range(args.repeats):
                rng = np.random.default_rng(
                    args.seed + scheme_index * 10000 + fold * 100 + repeat
                )
                shuffled_train = q_train[rng.permutation(len(q_train))]
                shuffled_test = q_test[rng.permutation(len(q_test))]
                model_seed = args.seed + scheme_index * 1000 + fold
                scaler, clustering, membership_train = fit_regime_model(
                    shuffled_train, n_regimes, tau, model_seed
                )
                regime_weights = fit_simplex_regime_weights(
                    membership_train, pred_train, y_train
                )
                membership_test = transform_regime_model(
                    shuffled_test, scaler, clustering, tau
                )
                prediction, _ = regime_predict(
                    membership_test, regime_weights, pred_test
                )
                score = metrics(y_test, prediction)
                rows.append(
                    {
                        "scheme": scheme,
                        "outer_fold": fold,
                        "repeat": repeat,
                        "method": "GeoAttribStack_ARC_shuffled_SHAP",
                        **score,
                    }
                )
            print(f"[SHAP permutation] {scheme} fold {fold + 1}/5 complete", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(tables / "shap_permutation_metrics.csv", index=False)

    shuffled = results[results.method == "GeoAttribStack_ARC_shuffled_SHAP"]
    shuffled_fold = shuffled.groupby(["scheme", "outer_fold"], as_index=False).agg(
        shuffled_RMSE_mean=("RMSE", "mean"),
        shuffled_RMSE_std=("RMSE", "std"),
        shuffled_MAE_mean=("MAE", "mean"),
        shuffled_R2_mean=("R2", "mean"),
    )
    true_fold = (
        results[results.method == "GeoAttribStack_ARC_true_SHAP"]
        .drop(columns=["repeat", "method"])
        .rename(columns={"RMSE": "true_RMSE", "MAE": "true_MAE", "R2": "true_R2"})
    )
    paired = true_fold.merge(shuffled_fold, on=["scheme", "outer_fold"])
    paired["delta_RMSE_true_minus_shuffled"] = paired.true_RMSE - paired.shuffled_RMSE_mean
    paired["true_SHAP_improves"] = paired.delta_RMSE_true_minus_shuffled < 0
    paired.to_csv(tables / "foldwise_true_vs_shuffled.csv", index=False)

    delta = paired.delta_RMSE_true_minus_shuffled.to_numpy()
    rng = np.random.default_rng(args.seed)
    bootstrap = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
    try:
        p_value = float(wilcoxon(delta, alternative="less").pvalue)
    except ValueError:
        p_value = 1.0

    summary = {
        "status": "completed",
        "design": "Same ARC structure and selected hyperparameters; SHAP rows independently permuted in train and test.",
        "confirmation_folds": int(len(paired)),
        "permutations_per_fold": int(args.repeats),
        "true_SHAP_mean_RMSE": float(paired.true_RMSE.mean()),
        "shuffled_SHAP_mean_RMSE": float(paired.shuffled_RMSE_mean.mean()),
        "mean_delta_RMSE_true_minus_shuffled": float(delta.mean()),
        "relative_change_percent": float(100 * delta.mean() / paired.shuffled_RMSE_mean.mean()),
        "folds_true_SHAP_improved": int((delta < 0).sum()),
        "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
        "one_sided_wilcoxon_p": p_value,
        "supports_attribution_correspondence": bool(
            delta.mean() < 0
            and (delta < 0).sum() >= 8
            and np.quantile(bootstrap, 0.975) < 0
            and p_value < 0.05
        ),
    }
    (output / "RESULT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(paired))
    ax.errorbar(
        x,
        paired.shuffled_RMSE_mean,
        yerr=paired.shuffled_RMSE_std,
        fmt="o",
        color="#D55E00",
        capsize=3,
        label="Shuffled SHAP ARC (mean +/- SD)",
    )
    ax.scatter(x, paired.true_RMSE, color="#0072B2", marker="D", s=42, label="True SHAP ARC")
    ax.set_xticks(x, [f"{s[:3]}-{f + 1}" for s, f in zip(paired.scheme, paired.outer_fold)], rotation=35)
    ax.set_ylabel("Spatial holdout RMSE")
    ax.set_title("Does sample-aligned attribution improve ARC?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / "ARC_SHAP_Fig1_true_vs_shuffled")

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    colors = np.where(delta < 0, "#009E73", "#D55E00")
    ax.bar(np.arange(len(delta)), -delta, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(delta)), [f"F{i + 1}" for i in range(len(delta))])
    ax.set_ylabel("RMSE reduction from true SHAP")
    ax.set_title("Foldwise value of the sample-SHAP correspondence")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, figures / "ARC_SHAP_Fig2_foldwise_gain")

    conclusion = (
        "SUPPORTED" if summary["supports_attribution_correspondence"] else "NOT SUPPORTED"
    )
    report = f"""# GeoAttribStack ARC: SHAP permutation test

## Result

The attribution-correspondence hypothesis is **{conclusion}** under the prespecified confirmation rule.

- True-SHAP ARC mean RMSE: {summary['true_SHAP_mean_RMSE']:.3f}
- Shuffled-SHAP ARC mean RMSE: {summary['shuffled_SHAP_mean_RMSE']:.3f}
- True minus shuffled RMSE: {summary['mean_delta_RMSE_true_minus_shuffled']:.3f}
- Relative change: {summary['relative_change_percent']:.2f}%
- Improved folds: {summary['folds_true_SHAP_improved']}/{summary['confirmation_folds']}
- One-sided Wilcoxon p: {summary['one_sided_wilcoxon_p']:.4f}
- Bootstrap 95% interval: [{summary['bootstrap_95_low']:.3f}, {summary['bootstrap_95_high']:.3f}]

The negative control retains the same ARC formulation, base predictions,
spatial folds, selected number of regimes, and temperature. It changes only
the correspondence between samples and SHAP profiles.
"""
    (output / "SHAP_PERMUTATION_RESULTS.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
