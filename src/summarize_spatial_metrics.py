"""Create scheme-level and fold-level spatial prediction diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


METHOD = "adaptive_normalized_spatial_cross_conformal"


def diagnostics(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = points.loc[points["method"] == METHOD].copy()
    scheme_rows = []
    fold_rows = []

    for scheme, group in points.groupby("scheme", sort=True):
        scheme_rows.append(
            {
                "scheme": scheme,
                "n": len(group),
                "pooled_RMSE": mean_squared_error(group["observed"], group["predicted"], squared=False),
                "pooled_MAE": mean_absolute_error(group["observed"], group["predicted"]),
                "pooled_R2": r2_score(group["observed"], group["predicted"]),
                "Pearson_r": group["observed"].corr(group["predicted"]),
                "mean_bias": (group["predicted"] - group["observed"]).mean(),
                "target_sd": group["observed"].std(ddof=0),
            }
        )

        for fold, fold_group in group.groupby("outer_fold", sort=True):
            fold_rows.append(
                {
                    "scheme": scheme,
                    "outer_fold": int(fold),
                    "n": len(fold_group),
                    "target_sd": fold_group["observed"].std(ddof=0),
                    "RMSE": mean_squared_error(
                        fold_group["observed"], fold_group["predicted"], squared=False
                    ),
                    "MAE": mean_absolute_error(fold_group["observed"], fold_group["predicted"]),
                    "R2": r2_score(fold_group["observed"], fold_group["predicted"]),
                    "mean_bias": (fold_group["predicted"] - fold_group["observed"]).mean(),
                }
            )

    return pd.DataFrame(scheme_rows), pd.DataFrame(fold_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/tables/conformal_point_results.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()

    points = pd.read_csv(args.input)
    scheme_metrics, fold_metrics = diagnostics(points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scheme_metrics.to_csv(args.output_dir / "scheme_level_pooled_diagnostics.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "foldwise_spatial_diagnostics.csv", index=False)
    print(scheme_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
