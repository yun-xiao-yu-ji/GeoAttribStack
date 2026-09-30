"""Robust spatial-group calibration for GeoAttribStack prediction intervals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_final_uncertainty_runtime import (  # noqa: E402
    ALPHA,
    BETA,
    SCHEMES,
    cross_fitted_arc_predictions,
    finite_quantile,
    fit_fixed_arc,
    save_figure,
    shap_representation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tables = args.output / "results" / "tables"
    figures = args.output / "results" / "figures"
    summaries = args.output / "results" / "summaries"
    for path in [tables, figures, summaries]:
        path.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(
        args.arc_root / "results" / "tables" / "selected_regime_hyperparameters.csv"
    )
    selected = selected[
        (selected.evaluation_subset == "confirmation")
        & (selected.representation == "orthogonal")
    ].set_index(["scheme", "outer_fold"])

    fold_rows = []
    point_rows = []
    quantile_rows = []
    for scheme_index, scheme in enumerate(SCHEMES):
        for fold in range(5):
            cache = np.load(
                args.arc_root
                / "results"
                / "confirmation_fold_cache"
                / f"{scheme}_fold_{fold}.npz",
                allow_pickle=False,
            )
            q_train = shap_representation(cache["shap_train"])
            q_test = shap_representation(cache["shap_test"])
            pred_train, pred_test = cache["pred_train"], cache["pred_test"]
            y_train, y_test = cache["y_train"], cache["y_test"]
            coords_train, coords_test = cache["coords_train"], cache["coords_test"]
            source_rows = cache["source_rows_test"]
            n_regimes = int(selected.loc[(scheme, fold), "n_regimes"])
            tau = float(selected.loc[(scheme, fold), "tau"])
            model_seed = args.seed + fold + scheme_index * 101

            test_point, _, _ = fit_fixed_arc(
                q_train,
                pred_train,
                y_train,
                q_test,
                pred_test,
                n_regimes,
                tau,
                model_seed,
            )
            oof_point, groups = cross_fitted_arc_predictions(
                q_train,
                pred_train,
                y_train,
                coords_train,
                n_regimes,
                tau,
                model_seed,
            )
            abs_residual = np.abs(y_train - oof_point)
            train_features = np.column_stack([pred_train, q_train])
            test_features = np.column_stack([pred_test, q_test])
            oof_scale = np.zeros(len(y_train))
            for group in range(4):
                train = groups != group
                valid = groups == group
                model = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=10,
                    random_state=model_seed + 2500 + group,
                    n_jobs=-1,
                )
                model.fit(train_features[train], abs_residual[train])
                oof_scale[valid] = model.predict(train_features[valid]) + BETA

            final_scale_model = RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                random_state=model_seed + 2600,
                n_jobs=-1,
            )
            final_scale_model.fit(train_features, abs_residual)
            test_scale = final_scale_model.predict(test_features) + BETA
            normalized_scores = abs_residual / oof_scale
            group_quantiles = np.asarray(
                [finite_quantile(normalized_scores[groups == group], ALPHA) for group in range(4)]
            )
            robust_quantile = float(group_quantiles.max())
            margin = robust_quantile * test_scale
            lower, upper = test_point - margin, test_point + margin
            covered = (y_test >= lower) & (y_test <= upper)
            fold_rows.append(
                {
                    "scheme": scheme,
                    "outer_fold": fold,
                    "method": "worst_group_adaptive_normalized_conformal",
                    "coverage": float(covered.mean()),
                    "mean_width": float(np.mean(2 * margin)),
                    "median_width": float(np.median(2 * margin)),
                    "minimum_group_quantile": float(group_quantiles.min()),
                    "maximum_group_quantile": robust_quantile,
                }
            )
            for group, value in enumerate(group_quantiles):
                quantile_rows.append(
                    {
                        "scheme": scheme,
                        "outer_fold": fold,
                        "inner_spatial_group": group,
                        "normalized_score_quantile": float(value),
                        "selected_as_worst_group": bool(value == robust_quantile),
                    }
                )
            for idx in range(len(y_test)):
                point_rows.append(
                    {
                        "scheme": scheme,
                        "outer_fold": fold,
                        "source_row": int(source_rows[idx]),
                        "observed": float(y_test[idx]),
                        "predicted": float(test_point[idx]),
                        "lower": float(lower[idx]),
                        "upper": float(upper[idx]),
                        "interval_width": float(2 * margin[idx]),
                        "covered": bool(covered[idx]),
                        "LON": float(coords_test[idx, 0]),
                        "LAT": float(coords_test[idx, 1]),
                    }
                )
            print(f"[Robust conformal] {scheme} fold {fold + 1}/5 complete", flush=True)

    folds = pd.DataFrame(fold_rows)
    points = pd.DataFrame(point_rows)
    quantiles = pd.DataFrame(quantile_rows)
    folds.to_csv(tables / "robust_conformal_fold_metrics.csv", index=False)
    points.to_csv(tables / "robust_conformal_point_results.csv", index=False)
    quantiles.to_csv(tables / "robust_conformal_group_quantiles.csv", index=False)

    existing = pd.read_csv(tables / "conformal_fold_metrics.csv")
    adaptive = existing[existing.method == "adaptive_normalized_spatial_cross_conformal"]
    comparison = adaptive[["scheme", "outer_fold", "coverage", "mean_width"]].merge(
        folds[["scheme", "outer_fold", "coverage", "mean_width"]],
        on=["scheme", "outer_fold"],
        suffixes=("_pooled", "_worst_group"),
    )
    comparison.to_csv(tables / "pooled_vs_robust_conformal.csv", index=False)

    summary = {
        "status": "completed",
        "target_coverage": 1 - ALPHA,
        "mean_coverage": float(folds.coverage.mean()),
        "minimum_fold_coverage": float(folds.coverage.min()),
        "folds_at_or_above_target": int((folds.coverage >= 1 - ALPHA).sum()),
        "total_folds": int(len(folds)),
        "mean_width": float(folds.mean_width.mean()),
        "median_width": float(folds.median_width.mean()),
        "width_change_vs_pooled_percent": float(
            100 * (folds.mean_width.mean() / adaptive.mean_width.mean() - 1)
        ),
    }
    (summaries / "ROBUST_CONFORMAL_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(comparison))
    width = 0.38
    ax.bar(x - width / 2, comparison.coverage_pooled, width, color="#0072B2", label="Pooled normalized")
    ax.bar(x + width / 2, comparison.coverage_worst_group, width, color="#D55E00", label="Worst-group normalized")
    ax.axhline(1 - ALPHA, color="black", linestyle="--", linewidth=1.2)
    ax.set_xticks(x, [f"{s[:3]}-{f + 1}" for s, f in zip(comparison.scheme, comparison.outer_fold)], rotation=30)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Spatially robust conformal calibration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / "GAS_Fig5_robust_conformal_coverage")

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.scatter(comparison.mean_width_pooled, comparison.coverage_pooled, color="#0072B2", s=55, label="Pooled")
    ax.scatter(comparison.mean_width_worst_group, comparison.coverage_worst_group, color="#D55E00", s=55, label="Worst-group")
    ax.axhline(1 - ALPHA, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Mean interval width")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Foldwise coverage-width trade-off")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / "GAS_Fig6_robust_coverage_width")


if __name__ == "__main__":
    main()
