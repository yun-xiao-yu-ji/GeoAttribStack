"""Generate one publication panel per file for manual manuscript assembly."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_manuscript_figures import (  # noqa: E402
    COLORS,
    FEATURE_NAMES,
    export_retained_figure,
    figure2_framework,
)
from run_final_uncertainty_runtime import (  # noqa: E402
    fit_regime_model,
    shap_representation,
    transform_regime_model,
)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 15,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_panel(fig: plt.Figure, output: Path, caption_name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{caption_name}.png", dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(output / f"{caption_name}.pdf", bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def spatial_confirmation_panels(tables: Path, output: Path) -> None:
    points = pd.read_csv(tables / "conformal_point_results.csv")
    points = points[points.method == "adaptive_normalized_spatial_cross_conformal"]
    palette = [COLORS["blue"], COLORS["green"], COLORS["gold"], COLORS["coral"], COLORS["sky"]]
    panels = [
        ("anti_diagonal_bands", "Figure_2a_Anti_diagonal_spatial_confirmation_bands"),
        ("radial_rings", "Figure_2b_Radial_spatial_confirmation_rings"),
    ]
    for scheme, filename in panels:
        fig, ax = plt.subplots(figsize=(6.3, 5.4))
        subset = points[points.scheme == scheme]
        for fold in range(5):
            fold_data = subset[subset.outer_fold == fold]
            ax.scatter(
                fold_data.LON,
                fold_data.LAT,
                s=4.2,
                color=palette[fold],
                label=f"Fold {fold + 1}",
                linewidths=0,
                rasterized=True,
            )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.legend(loc="best", markerscale=2.3, ncol=1)
        fig.tight_layout()
        save_panel(fig, output, filename)


def performance_panels(tables: Path, attribution: Path, output: Path) -> None:
    metrics = pd.read_csv(tables / "full_method_metrics.csv")
    metrics = metrics[metrics.evaluation_subset == "confirmation"]
    methods = [
        ("arc_soft_orthogonal", "GeoAttribStack"),
        ("global_convex", "Global convex"),
        ("original_full_shap_xgb", "Feature concatenation"),
        ("traditional_stacking_xgb", "Traditional stacking"),
    ]
    data = [metrics.loc[metrics.method == key, "RMSE"].to_numpy() for key, _ in methods]
    fig, ax = plt.subplots(figsize=(6.7, 5.1))
    boxes = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.62)
    for patch_box, color in zip(boxes["boxes"], [COLORS["coral"], COLORS["gray"], COLORS["sky"], COLORS["navy"]]):
        patch_box.set_facecolor(color)
        patch_box.set_alpha(0.82)
    ax.set_xticks(np.arange(1, len(methods) + 1), [label for _, label in methods], rotation=20, ha="right")
    ax.set_ylabel("Spatial holdout RMSE (mm/yr)")
    fig.tight_layout()
    save_panel(fig, output, "Figure_4a_Ensemble_performance_across_independent_spatial_confirmation_folds")

    paired = pd.read_csv(attribution / "tables" / "foldwise_true_vs_shuffled.csv")
    x = np.arange(len(paired))
    fig, ax = plt.subplots(figsize=(6.7, 5.1))
    ax.plot(x, paired.shuffled_RMSE_mean, "o-", color=COLORS["gray"], label="Shuffled SHAP", linewidth=2.0, markersize=6)
    ax.plot(x, paired.true_RMSE, "D-", color=COLORS["coral"], label="Sample-aligned SHAP", linewidth=2.1, markersize=6)
    ax.fill_between(x, paired.true_RMSE, paired.shuffled_RMSE_mean, where=paired.true_RMSE <= paired.shuffled_RMSE_mean, color=COLORS["teal"], alpha=0.15)
    ax.set_xticks(x, [f"F{i + 1}" for i in x])
    ax.set_xlabel("Independent spatial confirmation fold")
    ax.set_ylabel("Spatial holdout RMSE (mm/yr)")
    ax.legend(loc="best")
    fig.tight_layout()
    save_panel(fig, output, "Figure_4b_Attribution_correspondence_under_sample_aligned_and_shuffled_SHAP_profiles")

    pivot = metrics.pivot(index=["scheme", "outer_fold"], columns="method", values="RMSE")
    gains = pivot["traditional_stacking_xgb"] - pivot["arc_soft_orthogonal"]
    labels = [f"{scheme[:3]}-{fold + 1}" for scheme, fold in gains.index]
    fig, ax = plt.subplots(figsize=(6.7, 5.1))
    ax.bar(np.arange(len(gains)), gains, color=[COLORS["teal"] if value >= 0 else COLORS["coral"] for value in gains], width=0.72)
    ax.axhline(0, color="#30383D", linewidth=1.1)
    ax.set_xticks(np.arange(len(gains)), labels, rotation=28, ha="right")
    ax.set_xlabel("Independent spatial confirmation fold")
    ax.set_ylabel(r"RMSE gain over traditional stacking ($\Delta$ mm/yr)")
    fig.tight_layout()
    save_panel(fig, output, "Figure_4c_Foldwise_RMSE_gain_of_GeoAttribStack_over_traditional_stacking")


def spatial_prediction_panels(tables: Path, output: Path) -> None:
    points = pd.read_csv(tables / "conformal_point_results.csv")
    points = points[
        (points.scheme == "anti_diagonal_bands")
        & (points.method == "adaptive_normalized_spatial_cross_conformal")
    ].copy()
    points["residual"] = points.observed - points.predicted
    response_limit = float(np.nanpercentile(np.abs(points[["observed", "predicted"]].to_numpy()), 99.5))
    residual_limit = float(np.nanpercentile(np.abs(points.residual), 99.5))
    panels = [
        ("observed", TwoSlopeNorm(vmin=-response_limit, vcenter=0, vmax=response_limit), "Land-subsidence rate (mm/yr)", "Figure_5a_Observed_land_subsidence_under_independent_spatial_confirmation"),
        ("predicted", TwoSlopeNorm(vmin=-response_limit, vcenter=0, vmax=response_limit), "Predicted land-subsidence rate (mm/yr)", "Figure_5b_GeoAttribStack_spatial_prediction_under_independent_confirmation"),
        ("residual", TwoSlopeNorm(vmin=-residual_limit, vcenter=0, vmax=residual_limit), "Residual observed_minus_predicted (mm/yr)", "Figure_5c_Spatial_distribution_of_out_of_region_prediction_residuals"),
    ]
    for column, norm, colorbar_label, filename in panels:
        fig, ax = plt.subplots(figsize=(6.2, 5.5))
        artist = ax.scatter(points.LON, points.LAT, c=points[column], cmap="RdYlBu", norm=norm, s=4.0, linewidths=0, rasterized=True)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        colorbar = fig.colorbar(artist, ax=ax, fraction=0.050, pad=0.035)
        colorbar.set_label(colorbar_label)
        fig.tight_layout()
        save_panel(fig, output, filename)


def regime_materials(root: Path, tables: Path):
    arc_root = root / "locked_arc_input"
    cache = np.load(arc_root / "results" / "confirmation_fold_cache" / "anti_diagonal_bands_fold_0.npz", allow_pickle=False)
    selected = pd.read_csv(arc_root / "results" / "tables" / "selected_regime_hyperparameters.csv")
    row = selected[
        (selected.evaluation_subset == "confirmation")
        & (selected.representation == "orthogonal")
        & (selected.scheme == "anti_diagonal_bands")
        & (selected.outer_fold == 0)
    ].iloc[0]
    q_train = shap_representation(cache["shap_train"])
    q_test = shap_representation(cache["shap_test"])
    scaler, clustering, membership_train = fit_regime_model(q_train, int(row.n_regimes), float(row.tau), 42)
    membership_test = transform_regime_model(q_test, scaler, clustering, float(row.tau))
    coords = np.vstack([cache["coords_train"], cache["coords_test"]])
    regimes = np.concatenate([np.argmax(membership_train, axis=1), np.argmax(membership_test, axis=1)])
    representation = np.vstack([q_train, q_test])
    n_regimes = int(row.n_regimes)
    reshaped = representation.reshape(len(representation), len(cache["model_names"]), len(FEATURE_NAMES))
    prototype = np.vstack([reshaped[regimes == regime].mean(axis=(0, 1)) for regime in range(n_regimes)])
    weights = pd.read_csv(tables / "regime_expert_weights.csv")
    weights = weights[
        (weights.evaluation_subset == "confirmation")
        & (weights.scheme == "anti_diagonal_bands")
        & (weights.outer_fold == 0)
    ]
    matrix = weights.pivot(index="regime", columns="model_name", values="weight")
    model_order = [name for name in cache["model_names"] if name in matrix.columns]
    matrix = matrix.reindex(index=np.arange(n_regimes), columns=model_order)
    return cache, coords, regimes, prototype, matrix


def regime_panels(root: Path, tables: Path, output: Path) -> None:
    cache, coords, regimes, prototype, weights = regime_materials(root, tables)
    n_regimes = len(prototype)
    regime_colors = plt.get_cmap("tab10")(np.arange(n_regimes))
    test_coords = cache["coords_test"]

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    ax.scatter(coords[:, 0], coords[:, 1], c=regime_colors[regimes], s=4.0, alpha=0.88, linewidths=0, rasterized=True)
    hull = ConvexHull(test_coords)
    boundary = np.r_[hull.vertices, hull.vertices[0]]
    ax.plot(test_coords[boundary, 0], test_coords[boundary, 1], color="#2F3538", linewidth=1.5, linestyle="--", label="Independent confirmation region")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=regime_colors[idx], markeredgecolor="none", markersize=7, label=f"R{idx + 1}") for idx in range(n_regimes)]
    regime_legend = ax.legend(handles=handles, title="Attribution regime", ncol=2, loc="lower left")
    ax.add_artist(regime_legend)
    ax.legend(loc="upper right")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.tight_layout()
    save_panel(fig, output, "Figure_6a_Spatial_organization_of_dominant_attribution_regimes")

    limit = float(np.max(np.abs(prototype)))
    fig, ax = plt.subplots(figsize=(6.5, 5.3))
    image = ax.imshow(prototype, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_xticks(np.arange(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=28, ha="right")
    ax.set_yticks(np.arange(n_regimes), [f"R{idx + 1}" for idx in range(n_regimes)])
    ax.set_xlabel("Predictor")
    ax.set_ylabel("Attribution regime")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.035)
    colorbar.set_label("Mean normalized SHAP")
    fig.tight_layout()
    save_panel(fig, output, "Figure_6b_Regime_level_normalized_SHAP_attribution_prototypes")

    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    image = ax.imshow(weights.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(weights.columns)), weights.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(n_regimes), [f"R{idx + 1}" for idx in range(n_regimes)])
    ax.set_xlabel("Base learner")
    ax.set_ylabel("Attribution regime")
    for row in range(weights.shape[0]):
        for column in range(weights.shape[1]):
            value = weights.iloc[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=13, color="white" if value > 0.62 else "black")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.035)
    colorbar.set_label("Simplex-constrained weight")
    fig.tight_layout()
    save_panel(fig, output, "Figure_6c_Regime_specific_simplex_constrained_expert_weights")


def uncertainty_panels(tables: Path, output: Path) -> None:
    conformal = pd.read_csv(tables / "conformal_fold_metrics.csv")
    points = pd.read_csv(tables / "conformal_point_results.csv")
    standard = "standard_spatial_cross_conformal"
    adaptive = "adaptive_normalized_spatial_cross_conformal"
    summary = conformal.groupby("method", as_index=False).agg(coverage=("coverage", "mean"), width=("mean_width", "mean"))

    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    for method, color, label in [(standard, COLORS["navy"], "Standard"), (adaptive, COLORS["coral"], "Adaptive normalized")]:
        row = summary[summary.method == method].iloc[0]
        ax.scatter(row.width, row.coverage, s=150, color=color, label=label)
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1.3, label="90% target")
    ax.set_xlabel("Mean prediction-interval width (mm/yr)")
    ax.set_ylabel("Mean empirical coverage")
    ax.legend(loc="best")
    fig.tight_layout()
    save_panel(fig, output, "Figure_7a_Coverage_width_efficiency_of_spatial_cross_conformal_prediction")

    pivot = conformal.pivot(index=["scheme", "outer_fold"], columns="method", values="coverage")
    x = np.arange(len(pivot))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 5.1))
    ax.bar(x - width / 2, pivot[standard], width, color=COLORS["navy"], label="Standard")
    ax.bar(x + width / 2, pivot[adaptive], width, color=COLORS["coral"], label="Adaptive normalized")
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1.3, label="90% target")
    ax.set_xticks(x, [f"{scheme[:3]}-{fold + 1}" for scheme, fold in pivot.index], rotation=28, ha="right")
    ax.set_xlabel("Independent spatial confirmation fold")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1.04)
    ax.legend(loc="best")
    fig.tight_layout()
    save_panel(fig, output, "Figure_7b_Foldwise_empirical_coverage_under_two_conformal_scores")

    spatial = points[(points.scheme == "anti_diagonal_bands") & (points.method == adaptive)]
    upper = float(np.nanpercentile(spatial.interval_width, 99.0))
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    image = ax.scatter(spatial.LON, spatial.LAT, c=spatial.interval_width, cmap="YlOrRd", vmin=0, vmax=upper, s=4.0, linewidths=0, rasterized=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.035)
    colorbar.set_label("90% prediction-interval width (mm/yr)")
    fig.tight_layout()
    save_panel(fig, output, "Figure_7c_Spatial_distribution_of_adaptive_prediction_interval_width")


def supplementary_panels(root: Path, tables: Path, output: Path) -> None:
    assets = root / "assets" / "retained_figures"
    correlation = pd.read_csv(assets / "Correlation_Matrix_Data.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    image = ax.imshow(correlation.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [name.replace("_", " ") for name in correlation.columns]
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = correlation.iloc[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=13, color="white" if abs(value) > 0.62 else "black")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.035)
    colorbar.set_label("Pearson correlation coefficient")
    fig.tight_layout()
    save_panel(fig, output, "Supplementary_Figure_S1_Pearson_correlation_matrix_of_candidate_predictors")
    predictor_sources = root / "manuscript_figures" / "仅仅参考" / "figures" / "jpg"
    predictor_panels = [
        ("FigS1a_Compress.jpg", "Supplementary_Figure_S2a_Spatial_distribution_of_compressible_layer_thickness"),
        ("FigS1b_PRE.jpg", "Supplementary_Figure_S2b_Spatial_distribution_of_annual_mean_precipitation"),
        ("FigS1c_River_Distance.jpg", "Supplementary_Figure_S2c_Spatial_distribution_of_distance_to_rivers"),
        ("FigS1d_Ag_Den.jpg", "Supplementary_Figure_S2d_Spatial_distribution_of_agricultural_density"),
        ("FigS1e_BLT.jpg", "Supplementary_Figure_S2e_Spatial_distribution_of_building_volume"),
        ("FigS1f_Road_Distance.jpg", "Supplementary_Figure_S2f_Spatial_distribution_of_distance_to_roads"),
    ]
    for source, filename in predictor_panels:
        export_retained_figure(predictor_sources / source, output, filename)

    well_root = Path(r"C:\Users\Administrator\Desktop\小论文3\data\代码\data\wells\Periodic Groundwater Level Measurements\periodic_gwl_bulkdatadownload")
    precipitation = pd.read_csv(well_root / "station_monthly_precip_2017_202507.csv", low_memory=False)
    insar = pd.read_csv(well_root / "insar_avg_matched_to_stations.csv", low_memory=False)
    groundwater = pd.read_csv(well_root / "final_selected_measurements_2017_2025.csv", low_memory=False)
    stations = pd.read_csv(well_root / "final_selected_station_1.csv", low_memory=False)
    groundwater["msmt_date"] = pd.to_datetime(groundwater["msmt_date"], format="mixed", errors="coerce")
    groundwater = groundwater.dropna(subset=["msmt_date", "gwe"])
    common = set(precipitation.site_code) & set(insar.site_code) & set(groundwater.site_code) & set(stations.site_code)
    stations = stations[stations.site_code.isin(common)].sort_values("continuous")
    precip_columns = [column for column in precipitation.columns if str(column).isdigit() and len(str(column)) == 6]
    insar_columns = [column for column in insar.columns if str(column).startswith("D")]
    precip_dates = [datetime.strptime(column, "%Y%m").replace(day=15) for column in precip_columns]
    insar_dates = [datetime.strptime(column, "D%Y%m%d") for column in insar_columns]
    for station in stations.itertuples(index=False):
        site = station.site_code
        panel = str(station.continuous).strip().lower()
        precip_row = precipitation[precipitation.site_code == site].iloc[0]
        insar_row = insar[insar.site_code == site].iloc[0]
        site_groundwater = groundwater[groundwater.site_code == site].sort_values("msmt_date")
        precip_values = pd.to_numeric(precip_row[precip_columns].to_numpy(), errors="coerce")
        insar_values = pd.to_numeric(insar_row[insar_columns].to_numpy(), errors="coerce")
        precip_mask = ~pd.isna(precip_values)
        insar_mask = ~pd.isna(insar_values)

        fig, ax1 = plt.subplots(figsize=(7.4, 5.4))
        color_insar, color_groundwater, color_precip = COLORS["blue"], COLORS["green"], COLORS["coral"]
        ax1.scatter(
            np.asarray(insar_dates)[insar_mask],
            insar_values[insar_mask],
            color=color_insar,
            edgecolors="black",
            linewidth=0.25,
            s=24,
            alpha=0.90,
            zorder=4,
        )
        ax1.set_ylabel("Land subsidence (mm)", color=color_insar)
        ax1.tick_params(axis="y", colors=color_insar)
        ax1.spines["left"].set_color(color_insar)

        ax2 = ax1.twinx()
        ax2.plot(site_groundwater.msmt_date, site_groundwater.gwe, color=color_groundwater, marker="o", markersize=5, linewidth=1.8, zorder=5)
        ax2.set_ylabel("Groundwater elevation (ft)", color=color_groundwater)
        ax2.tick_params(axis="y", colors=color_groundwater)
        ax2.spines["right"].set_color(color_groundwater)

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 62))
        ax3.bar(np.asarray(precip_dates)[precip_mask], precip_values[precip_mask], width=20, color=color_precip, alpha=0.42, edgecolor="none", zorder=1)
        ax3.set_ylabel("Precipitation (mm)", color=color_precip)
        ax3.tick_params(axis="y", colors=color_precip)
        ax3.spines["right"].set_color(color_precip)
        if precip_mask.any():
            ax3.set_ylim(0, float(np.nanmax(precip_values[precip_mask])) * 1.5)

        ax1.set_zorder(3)
        ax2.set_zorder(2)
        ax3.set_zorder(1)
        ax1.patch.set_visible(False)
        ax2.patch.set_visible(False)
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.grid(axis="x", linestyle="--", alpha=0.30)
        fig.tight_layout()
        save_panel(fig, output, f"Supplementary_Figure_S3{panel}_Hydrogeological_timeseries_at_well_{site}")

    data = pd.read_csv(tables / "arc_reproduction_metrics.csv")
    labels = [f"{scheme[:3]}-{fold + 1}" for scheme, fold in zip(data.scheme, data.outer_fold)]
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(6.7, 5.0))
    ax.bar(x, data.n_regimes, color=COLORS["teal"], width=0.70)
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_xlabel("Independent spatial confirmation fold")
    ax.set_ylabel("Selected number of regimes")
    ax.set_ylim(0, max(data.n_regimes) + 1)
    fig.tight_layout()
    save_panel(fig, output, "Supplementary_Figure_S4a_Selected_number_of_attribution_regimes_across_spatial_folds")

    fig, ax = plt.subplots(figsize=(6.7, 5.0))
    ax.bar(x, data.tau, color=COLORS["gold"], width=0.70)
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_xlabel("Independent spatial confirmation fold")
    ax.set_ylabel(r"Selected membership temperature $\tau$")
    fig.tight_layout()
    save_panel(fig, output, "Supplementary_Figure_S4b_Selected_soft_membership_temperature_across_spatial_folds")

    robust = pd.read_csv(tables / "pooled_vs_robust_conformal.csv")
    x = np.arange(len(robust))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 5.1))
    ax.bar(x - width / 2, robust.coverage_pooled, width, color=COLORS["blue"], label="Pooled normalized")
    ax.bar(x + width / 2, robust.coverage_worst_group, width, color=COLORS["coral"], label="Worst-group normalized")
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1.3, label="90% target")
    ax.set_xticks(x, [f"{scheme[:3]}-{fold + 1}" for scheme, fold in zip(robust.scheme, robust.outer_fold)], rotation=28, ha="right")
    ax.set_xlabel("Independent spatial confirmation fold")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1.04)
    ax.legend(loc="best")
    fig.tight_layout()
    save_panel(fig, output, "Supplementary_Figure_S5_Spatially_robust_conformal_sensitivity_analysis")

    runtime = pd.read_csv(tables / "runtime_by_fold.csv")
    columns = ["attribution_representation_seconds", "inner_selection_seconds", "final_arc_fit_seconds", "conformal_calibration_seconds"]
    labels = ["Attribution representation", "Inner spatial selection", "Final ARC fit", "Conformal calibration"]
    values = [runtime[column].mean() for column in columns]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    bars = ax.barh(labels, values, color=[COLORS["sky"], COLORS["gold"], COLORS["teal"], COLORS["navy"]], height=0.58)
    for bar, value in zip(bars, values):
        ax.text(value + 0.5, bar.get_y() + bar.get_height() / 2, f"{value:.2f} s", va="center", fontsize=13)
    ax.set_xlabel("Mean runtime per outer fold (s)")
    ax.set_xlim(0, max(values) * 1.20)
    fig.tight_layout()
    save_panel(fig, output, "Supplementary_Figure_S6_Computational_profile_of_GeoAttribStack")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    style()
    tables = args.root / "results" / "tables"
    attribution = args.root / "results" / "attribution_validation"
    output = args.root / "manuscript_figures_single_panels"
    main_output = output / "main_text"
    supplementary_output = output / "supplementary"
    assets = args.root / "assets" / "retained_figures"

    export_retained_figure(assets / "Original_Figure_1_Study_area.jpeg", main_output, "Figure_1_Study_area_and_observation_data")
    spatial_confirmation_panels(tables, main_output)
    figure2_framework(main_output)
    performance_panels(tables, attribution, main_output)
    spatial_prediction_panels(tables, main_output)
    regime_panels(args.root, tables, main_output)
    uncertainty_panels(tables, main_output)
    supplementary_panels(args.root, tables, supplementary_output)
    print(f"Generated single-panel manuscript figures in {output}")


if __name__ == "__main__":
    main()
