"""Generate the curated GeoAttribStack manuscript figure set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_final_uncertainty_runtime import (  # noqa: E402
    fit_regime_model,
    shap_representation,
    transform_regime_model,
)


COLORS = {
    "navy": "#264653",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "coral": "#E76F51",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "gray": "#A7B0B5",
}

FEATURE_NAMES = ["Compress", "Ag_Den", "PRE", "Road_Dist.", "River_Dist.", "BLT"]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )


def save(fig: plt.Figure, output: Path, title: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{title}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{title}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_retained_figure(source: Path, output: Path, title: str) -> None:
    """Re-encode an existing manuscript figure without altering its content."""
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((3600, 3600), Image.Resampling.LANCZOS)
        rgb.save(output / f"{title}.png", dpi=(300, 300))
        width, height = rgb.size
        fig = plt.figure(figsize=(width / 300, height / 300))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(rgb)
        ax.axis("off")
        fig.savefig(output / f"{title}.pdf", dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")
        plt.close(fig)


def figure1_partitions(tables: Path, output: Path) -> None:
    points = pd.read_csv(tables / "conformal_point_results.csv")
    points = points[points.method == "adaptive_normalized_spatial_cross_conformal"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharex=True, sharey=True)
    palette = [COLORS["blue"], COLORS["green"], COLORS["gold"], COLORS["coral"], COLORS["sky"]]
    labels = [("anti_diagonal_bands", "Anti-diagonal spatial bands"), ("radial_rings", "Radial spatial rings")]
    for ax, (scheme, panel_title) in zip(axes, labels):
        part = points[points.scheme == scheme]
        for fold in range(5):
            fold_data = part[part.outer_fold == fold]
            ax.scatter(fold_data.LON, fold_data.LAT, s=1.7, color=palette[fold], label=f"Fold {fold + 1}", rasterized=True)
        ax.set_title(panel_title)
        ax.set_xlabel("Longitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
    axes[0].set_ylabel("Latitude")
    axes[1].legend(loc="upper right", markerscale=3.5, ncol=1)
    fig.tight_layout()
    save(fig, output, "Figure_2_Independent_spatial_confirmation_design")


def figure2_framework(
    output: Path,
    filename: str = "Figure_3_GeoAttribStack_attribution_informed_constrained_fusion_framework",
) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.9))
    ax.set_xlim(0.005, 0.995)
    ax.set_ylim(0.035, 0.975)
    ax.axis("off")

    def panel(x, y, w, h, step, title):
        outer = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.016",
            linewidth=1.25, edgecolor="#8C969C", facecolor="#F5F7F8",
            linestyle=(0, (3, 3)),
        )
        inner = FancyBboxPatch(
            (x + 0.014, y + 0.018), w - 0.028, h - 0.072,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=0.7, edgecolor="#C9CED1", facecolor="white",
        )
        ax.add_patch(outer)
        ax.add_patch(inner)
        ax.text(x + 0.018, y + h - 0.036, f"Step {step}: {title}", color="#1769A6", fontsize=12.2, fontweight="bold", va="center")
        return x + 0.025, y + 0.032, w - 0.05, h - 0.102

    p1 = (0.015, 0.525, 0.305, 0.425)
    p2 = (0.3475, 0.525, 0.305, 0.425)
    p3 = (0.680, 0.525, 0.305, 0.425)
    p4 = (0.515, 0.055, 0.470, 0.425)
    p5 = (0.015, 0.055, 0.470, 0.425)
    c1 = panel(*p1, 1, "Base predictions")
    c2 = panel(*p2, 2, "Local attributions")
    c3 = panel(*p3, 3, "Attribution regimes")
    c4 = panel(*p4, 4, "ARC constrained weighting")
    c5 = panel(*p5, 5, "Prediction and uncertainty")

    # Step 1: covariate nodes feed three compact learner cards.
    x, y, w, h = c1
    node_colors = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["coral"]]
    for idx, color in enumerate(node_colors):
        ax.add_patch(Circle((x + 0.035, y + 0.045 + idx * 0.046), 0.012, facecolor=color, edgecolor="white", linewidth=0.8))
    ax.text(x + 0.035, y + h - 0.025, "covariates", ha="center", fontsize=10.0, color="#44515A")
    model_x = x + 0.11
    for idx, label in enumerate(["Cat", "ET", "XGB"]):
        yy = y + 0.035 + idx * 0.067
        ax.add_patch(FancyBboxPatch((model_x, yy), 0.07, 0.042, boxstyle="round,pad=0.004", facecolor="#EAF1F8", edgecolor=COLORS["blue"], linewidth=0.8))
        ax.text(model_x + 0.035, yy + 0.021, label, ha="center", va="center", fontsize=10.0, fontweight="bold")
        ax.add_patch(FancyArrowPatch((x + 0.055, y + 0.11), (model_x - 0.004, yy + 0.021), arrowstyle="->", mutation_scale=7, linewidth=0.7, color="#778187"))
        ax.add_patch(Circle((x + w - 0.025, yy + 0.021), 0.009, facecolor=COLORS["coral"], edgecolor="none"))
        ax.add_patch(FancyArrowPatch((model_x + 0.072, yy + 0.021), (x + w - 0.038, yy + 0.021), arrowstyle="->", mutation_scale=7, linewidth=0.7, color="#778187"))
    ax.text(x + w - 0.025, y + h - 0.025, r"$\hat{y}_{ik}$", ha="center", fontsize=11.5)

    # Step 2: compact SHAP contribution chart around a zero line.
    x, y, w, h = c2
    zero_y = y + 0.11
    ax.plot([x + 0.025, x + w - 0.025], [zero_y, zero_y], color="#59656B", linewidth=0.8)
    values = [0.075, -0.045, 0.105, -0.065, 0.038]
    bar_x = np.linspace(x + 0.045, x + w - 0.045, len(values))
    for xx, value in zip(bar_x, values):
        ax.add_patch(Rectangle((xx - 0.012, zero_y if value >= 0 else zero_y + value), 0.024, abs(value), facecolor=COLORS["coral"] if value >= 0 else COLORS["blue"], edgecolor="none"))
    ax.text(x + w / 2, y + h - 0.027, "sample-level SHAP profile", ha="center", fontsize=10.2, color="#44515A")
    ax.text(x + w / 2, y + 0.012, r"$\hat{y}_{ik}=\phi_{0k}+\sum_j\phi_{ijk}$", ha="center", fontsize=11.2, color=COLORS["navy"])

    # Step 3: graphical attribution clusters.
    x, y, w, h = c3
    centers = [(x + 0.065, y + 0.14), (x + 0.145, y + 0.085), (x + 0.20, y + 0.175)]
    cluster_colors = [COLORS["blue"], COLORS["teal"], COLORS["gold"]]
    offsets = [(-0.022, 0.005), (-0.009, 0.029), (0.012, 0.018), (0.023, -0.012), (-0.014, -0.021), (0.008, -0.027)]
    for center, color in zip(centers, cluster_colors):
        ax.add_patch(Ellipse(center, 0.095, 0.085, facecolor=color, alpha=0.10, edgecolor=color, linewidth=0.9, linestyle="--"))
        for dx, dy in offsets:
            ax.add_patch(Circle((center[0] + dx, center[1] + dy), 0.008, facecolor=color, edgecolor="white", linewidth=0.4))
    ax.text(x + w / 2, y + h - 0.027, "centered + L1-normalized profiles", ha="center", fontsize=9.8, color="#44515A")
    ax.text(x + w / 2, y + 0.012, "soft regime membership", ha="center", fontsize=10.0, color="#44515A")

    # Step 4: soft membership controls simplex weights.
    x, y, w, h = c4
    left_x = x + 0.035
    for idx, (color, prob) in enumerate(zip(cluster_colors, [0.20, 0.55, 0.25])):
        yy = y + 0.055 + idx * 0.060
        ax.add_patch(Circle((left_x, yy), 0.011 + prob * 0.012, facecolor=color, edgecolor="white", linewidth=0.6))
        ax.text(left_x + 0.028, yy, rf"$\pi_{{{idx + 1}}}$={prob:.2f}", va="center", fontsize=9.8)
    ax.add_patch(FancyArrowPatch((x + 0.14, y + 0.12), (x + 0.205, y + 0.12), arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#66747B"))
    bar_left, bar_y, bar_w = x + 0.225, y + 0.093, w - 0.26
    weights = [0.22, 0.53, 0.25]
    start = bar_left
    for weight, color, label in zip(weights, [COLORS["coral"], COLORS["teal"], COLORS["blue"]], ["Cat", "ET", "XGB"]):
        ax.add_patch(Rectangle((start, bar_y), bar_w * weight, 0.055, facecolor=color, edgecolor="white", linewidth=0.6))
        ax.text(start + bar_w * weight / 2, bar_y + 0.0275, label, ha="center", va="center", fontsize=9.3, color="white", fontweight="bold")
        start += bar_w * weight
    ax.text(bar_left + bar_w / 2, bar_y + 0.083, r"$w_{ik}\geq0,\ \sum_k w_{ik}=1$", ha="center", fontsize=10.7, color=COLORS["navy"])
    ax.text(x + w / 2, y + h - 0.027, "regime-informed sample-specific weights", ha="center", fontsize=10.0, color="#44515A")

    # Step 5: weighted predictions produce a point and adaptive interval.
    x, y, w, h = c5
    pred_x = [x + 0.045, x + 0.095, x + 0.145]
    pred_y = [y + 0.075, y + 0.165, y + 0.12]
    for xx, yy, color in zip(pred_x, pred_y, [COLORS["coral"], COLORS["teal"], COLORS["blue"]]):
        ax.add_patch(Circle((xx, yy), 0.012, facecolor=color, edgecolor="white", linewidth=0.6))
        ax.add_patch(FancyArrowPatch((xx + 0.015, yy), (x + 0.225, y + 0.12), arrowstyle="->", mutation_scale=7, linewidth=0.7, color="#778187"))
    final_x, final_y = x + 0.255, y + 0.12
    ax.add_patch(Circle((final_x, final_y), 0.015, facecolor=COLORS["navy"], edgecolor="white", linewidth=0.7))
    ax.plot([final_x + 0.065, final_x + 0.065], [y + 0.055, y + 0.185], color=COLORS["coral"], linewidth=2.2)
    ax.plot([final_x + 0.05, final_x + 0.08], [y + 0.055, y + 0.055], color=COLORS["coral"], linewidth=1.4)
    ax.plot([final_x + 0.05, final_x + 0.08], [y + 0.185, y + 0.185], color=COLORS["coral"], linewidth=1.4)
    ax.add_patch(Circle((final_x + 0.065, final_y), 0.009, facecolor=COLORS["navy"], edgecolor="white", linewidth=0.5))
    ax.text(x + w / 2, y + h - 0.027, "weighted point prediction", ha="center", fontsize=10.0, color="#44515A")
    ax.text(final_x + 0.065, y + 0.018, "adaptive conformal interval", ha="center", fontsize=9.8, color="#44515A")

    def flow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color="#68747A"))

    flow((p1[0] + p1[2] + 0.005, p1[1] + p1[3] / 2), (p2[0] - 0.006, p2[1] + p2[3] / 2))
    flow((p2[0] + p2[2] + 0.005, p2[1] + p2[3] / 2), (p3[0] - 0.006, p3[1] + p3[3] / 2))
    vertical_x = p3[0] + p3[2] / 2
    flow((vertical_x, p3[1] - 0.004), (vertical_x, p4[1] + p4[3] + 0.006))
    flow((p4[0] - 0.006, p4[1] + p4[3] / 2), (p5[0] + p5[2] + 0.006, p5[1] + p5[3] / 2))

    fig.tight_layout()
    save(fig, output, filename)


def figure3_performance(tables: Path, attribution: Path, output: Path) -> None:
    metrics = pd.read_csv(tables / "full_method_metrics.csv")
    metrics = metrics[metrics.evaluation_subset == "confirmation"]
    methods = [
        ("arc_soft_orthogonal", "GeoAttribStack"),
        ("global_convex", "Global convex"),
        ("original_full_shap_xgb", "Feature concatenation"),
        ("traditional_stacking_xgb", "Traditional stacking"),
    ]
    paired = pd.read_csv(attribution / "tables" / "foldwise_true_vs_shuffled.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), gridspec_kw={"width_ratios": [1.1, 1, 0.92]})
    data = [metrics.loc[metrics.method == key, "RMSE"].to_numpy() for key, _ in methods]
    boxes = axes[0].boxplot(data, patch_artist=True, showmeans=True, widths=0.62)
    box_colors = [COLORS["coral"], COLORS["gray"], COLORS["sky"], COLORS["navy"]]
    for patch_box, color in zip(boxes["boxes"], box_colors):
        patch_box.set_facecolor(color)
        patch_box.set_alpha(0.82)
    axes[0].set_xticks(np.arange(1, len(methods) + 1), [label for _, label in methods], rotation=23, ha="right")
    axes[0].set_ylabel("Spatial holdout RMSE")
    axes[0].set_title("(a) Ensemble comparison")

    x = np.arange(len(paired))
    axes[1].plot(x, paired.shuffled_RMSE_mean, "o-", color=COLORS["gray"], label="Shuffled SHAP", linewidth=1.4, markersize=4)
    axes[1].plot(x, paired.true_RMSE, "D-", color=COLORS["coral"], label="Sample-aligned SHAP", linewidth=1.5, markersize=4)
    axes[1].fill_between(x, paired.true_RMSE, paired.shuffled_RMSE_mean, where=paired.true_RMSE <= paired.shuffled_RMSE_mean, color=COLORS["teal"], alpha=0.14)
    axes[1].set_xticks(x, [f"F{i + 1}" for i in x])
    axes[1].set_ylabel("Spatial holdout RMSE")
    axes[1].set_title("(b) Attribution correspondence")
    axes[1].legend(loc="upper right")

    confirmation = metrics.pivot(index=["scheme", "outer_fold"], columns="method", values="RMSE")
    gains = confirmation["traditional_stacking_xgb"] - confirmation["arc_soft_orthogonal"]
    bar_colors = [COLORS["teal"] if value >= 0 else COLORS["coral"] for value in gains]
    axes[2].bar(np.arange(len(gains)), gains, color=bar_colors, width=0.72)
    axes[2].axhline(0, color="#30383D", linewidth=0.9)
    axes[2].set_xticks(
        np.arange(len(gains)),
        [f"{scheme[:3]}-{fold + 1}" for scheme, fold in gains.index],
        rotation=28,
        ha="right",
    )
    axes[2].set_ylabel(r"RMSE gain ($\Delta$)")
    axes[2].set_title("(c) Gain over stacking")
    fig.tight_layout()
    save(fig, output, "Figure_4_Spatial_predictive_performance_and_attribution_correspondence")


def figure5_spatial_predictions(tables: Path, output: Path) -> None:
    points = pd.read_csv(tables / "conformal_point_results.csv")
    points = points[
        (points.scheme == "anti_diagonal_bands")
        & (points.method == "adaptive_normalized_spatial_cross_conformal")
    ].copy()
    points["residual"] = points.observed - points.predicted

    response_limit = float(np.nanpercentile(np.abs(points[["observed", "predicted"]].to_numpy()), 99.5))
    residual_limit = float(np.nanpercentile(np.abs(points.residual), 99.5))
    response_norm = TwoSlopeNorm(vmin=-response_limit, vcenter=0, vmax=response_limit)
    residual_norm = TwoSlopeNorm(vmin=-residual_limit, vcenter=0, vmax=residual_limit)
    fig, axes = plt.subplots(1, 3, figsize=(9.9, 3.75), sharex=True, sharey=True)
    panels = [
        ("observed", "(a) Observed subsidence", response_norm),
        ("predicted", "(b) GeoAttribStack prediction", response_norm),
        ("residual", "(c) Residual (observed - predicted)", residual_norm),
    ]
    artists = []
    for ax, (column, title, norm) in zip(axes, panels):
        artist = ax.scatter(
            points.LON,
            points.LAT,
            c=points[column],
            cmap="RdYlBu",
            norm=norm,
            s=2.4,
            linewidths=0,
            rasterized=True,
        )
        artists.append(artist)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
    axes[0].set_ylabel("Latitude")
    response_bar = fig.colorbar(artists[1], ax=axes[:2], fraction=0.028, pad=0.025)
    response_bar.set_label("Land-subsidence rate (mm/yr)")
    residual_bar = fig.colorbar(artists[2], ax=axes[2], fraction=0.052, pad=0.025)
    residual_bar.set_label("Residual (mm/yr)")
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.14, top=0.90, wspace=0.16)
    save(fig, output, "Figure_5_Observed_predicted_and_residual_spatial_patterns")


def figure6_regimes_and_weights(root: Path, tables: Path, output: Path) -> None:
    arc_root = root / "locked_arc_input"
    cache = np.load(
        arc_root / "results" / "confirmation_fold_cache" / "anti_diagonal_bands_fold_0.npz",
        allow_pickle=False,
    )
    selected = pd.read_csv(arc_root / "results" / "tables" / "selected_regime_hyperparameters.csv")
    row = selected[
        (selected.evaluation_subset == "confirmation")
        & (selected.representation == "orthogonal")
        & (selected.scheme == "anti_diagonal_bands")
        & (selected.outer_fold == 0)
    ].iloc[0]
    q_train = shap_representation(cache["shap_train"])
    q_test = shap_representation(cache["shap_test"])
    scaler, clustering, membership_train = fit_regime_model(
        q_train, int(row.n_regimes), float(row.tau), 42
    )
    membership_test = transform_regime_model(q_test, scaler, clustering, float(row.tau))
    coords = np.vstack([cache["coords_train"], cache["coords_test"]])
    representation = np.vstack([q_train, q_test])
    regimes = np.concatenate(
        [np.argmax(membership_train, axis=1), np.argmax(membership_test, axis=1)]
    )
    test_coords = cache["coords_test"]
    n_regimes = int(row.n_regimes)
    regime_colors = plt.get_cmap("tab10")(np.arange(n_regimes))

    # Average the three model-specific normalized SHAP blocks to obtain a
    # compact six-feature attribution prototype for each fitted regime.
    prototype = np.zeros((n_regimes, len(FEATURE_NAMES)))
    reshaped = representation.reshape(len(representation), len(cache["model_names"]), len(FEATURE_NAMES))
    for regime in range(n_regimes):
        prototype[regime] = reshaped[regimes == regime].mean(axis=(0, 1))

    weights = pd.read_csv(tables / "regime_expert_weights.csv")
    weights = weights[
        (weights.evaluation_subset == "confirmation")
        & (weights.scheme == "anti_diagonal_bands")
        & (weights.outer_fold == 0)
    ]
    weight_matrix = weights.pivot(index="regime", columns="model_name", values="weight")
    model_order = [name for name in cache["model_names"] if name in weight_matrix.columns]
    weight_matrix = weight_matrix.reindex(index=np.arange(n_regimes), columns=model_order)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.75), gridspec_kw={"width_ratios": [1.25, 1.05, 0.92]})
    ax = axes[0]
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=regime_colors[regimes],
        s=2.7,
        alpha=0.86,
        linewidths=0,
        rasterized=True,
    )
    hull = ConvexHull(test_coords)
    boundary = np.r_[hull.vertices, hull.vertices[0]]
    ax.plot(
        test_coords[boundary, 0],
        test_coords[boundary, 1],
        color="#2F3538",
        linewidth=1.1,
        linestyle="--",
        label="Independent confirmation region",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("(a) Spatial attribution regimes")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=regime_colors[idx], markeredgecolor="none", markersize=5, label=f"R{idx + 1}")
        for idx in range(n_regimes)
    ]
    regime_legend = ax.legend(handles=handles, title="Regime", ncol=2, loc="lower left", fontsize=7.0, title_fontsize=7.5)
    ax.add_artist(regime_legend)
    ax.legend(loc="upper right", frameon=False, fontsize=7.0)

    ax = axes[1]
    prototype_limit = float(np.max(np.abs(prototype)))
    prototype_image = ax.imshow(
        prototype,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-prototype_limit, vcenter=0, vmax=prototype_limit),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=32, ha="right")
    ax.set_yticks(np.arange(n_regimes), [f"R{idx + 1}" for idx in range(n_regimes)])
    ax.set_title("(b) Attribution prototypes")
    ax.set_ylabel("Attribution regime")
    prototype_bar = fig.colorbar(prototype_image, ax=ax, fraction=0.050, pad=0.035)
    prototype_bar.set_label("Mean normalized SHAP")

    ax = axes[2]
    weight_image = ax.imshow(weight_matrix.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(weight_matrix.columns)), weight_matrix.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(n_regimes), [f"R{idx + 1}" for idx in range(n_regimes)])
    ax.set_title("(c) Constrained expert weights")
    for row_idx in range(weight_matrix.shape[0]):
        for col_idx in range(weight_matrix.shape[1]):
            value = weight_matrix.iloc[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if value > 0.62 else "black",
            )
    weight_bar = fig.colorbar(weight_image, ax=ax, fraction=0.050, pad=0.035)
    weight_bar.set_label("Weight")
    fig.tight_layout()
    save(fig, output, "Figure_6_Spatial_attribution_regimes_and_constrained_expert_allocation")


def figure6_conformal(tables: Path, output: Path) -> None:
    conformal = pd.read_csv(tables / "conformal_fold_metrics.csv")
    points = pd.read_csv(tables / "conformal_point_results.csv")
    standard = "standard_spatial_cross_conformal"
    adaptive = "adaptive_normalized_spatial_cross_conformal"
    summary = conformal.groupby("method", as_index=False).agg(coverage=("coverage", "mean"), width=("mean_width", "mean"))
    pivot = conformal.pivot(index=["scheme", "outer_fold"], columns="method", values="coverage")
    spatial = points[(points.scheme == "anti_diagonal_bands") & (points.method == adaptive)]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.45), gridspec_kw={"width_ratios": [0.82, 1.28, 1.05]})
    for method, color, label in [(standard, COLORS["navy"], "Standard"), (adaptive, COLORS["coral"], "Adaptive normalized")]:
        row = summary[summary.method == method].iloc[0]
        axes[0].scatter(row.width, row.coverage, s=95, color=color, label=label)
    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1.1)
    axes[0].set_xlabel("Mean interval width")
    axes[0].set_ylabel("Mean coverage")
    axes[0].set_title("(a) Coverage-width efficiency")
    axes[0].legend(loc="lower right")

    x = np.arange(len(pivot))
    width = 0.36
    axes[1].bar(x - width / 2, pivot[standard], width, color=COLORS["navy"], label="Standard")
    axes[1].bar(x + width / 2, pivot[adaptive], width, color=COLORS["coral"], label="Adaptive normalized")
    axes[1].axhline(0.90, color="black", linestyle="--", linewidth=1.1)
    axes[1].set_xticks(x, [f"{s[:3]}-{f + 1}" for s, f in pivot.index], rotation=28, ha="right")
    axes[1].set_ylim(0, 1.04)
    axes[1].set_ylabel("Empirical coverage")
    axes[1].set_title("(b) Independent spatial holdouts")

    upper = float(np.nanpercentile(spatial.interval_width, 99.0))
    uncertainty_map = axes[2].scatter(
        spatial.LON,
        spatial.LAT,
        c=spatial.interval_width,
        cmap="YlOrRd",
        vmin=0,
        vmax=upper,
        s=2.5,
        linewidths=0,
        rasterized=True,
    )
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    axes[2].set_title("(c) Spatial interval width")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(False)
    uncertainty_bar = fig.colorbar(uncertainty_map, ax=axes[2], fraction=0.050, pad=0.035)
    uncertainty_bar.set_label("90% interval width (mm/yr)")
    fig.tight_layout()
    save(fig, output, "Figure_7_Spatially_adaptive_uncertainty_quantification")


def figure7_runtime(tables: Path, output: Path) -> None:
    runtime = pd.read_csv(tables / "runtime_by_fold.csv")
    columns = [
        "attribution_representation_seconds",
        "inner_selection_seconds",
        "final_arc_fit_seconds",
        "conformal_calibration_seconds",
    ]
    labels = ["Attribution representation", "Inner spatial selection", "Final ARC fit", "Conformal calibration"]
    values = [runtime[column].mean() for column in columns]
    fig, ax = plt.subplots(figsize=(5.9, 3.4))
    bars = ax.barh(labels, values, color=[COLORS["sky"], COLORS["gold"], COLORS["teal"], COLORS["navy"]], height=0.58)
    for bar, value in zip(bars, values):
        ax.text(value + 0.5, bar.get_y() + bar.get_height() / 2, f"{value:.2f} s", va="center", fontsize=8.5)
    ax.set_xlabel("Mean runtime per outer fold (s)")
    ax.set_title("Computational profile of GeoAttribStack")
    ax.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    save(fig, output, "Supplementary_Figure_S6_Computational_profile_of_GeoAttribStack")


def supplementary_hyperparameter_stability(tables: Path, output: Path) -> None:
    data = pd.read_csv(tables / "arc_reproduction_metrics.csv").copy()
    labels = [f"{scheme[:3]}-{fold + 1}" for scheme, fold in zip(data.scheme, data.outer_fold)]
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25))
    axes[0].bar(x, data.n_regimes, color=COLORS["teal"], width=0.70)
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].set_ylabel("Selected number of regimes")
    axes[0].set_title("(a) Regime-count stability")
    axes[0].set_ylim(0, max(data.n_regimes) + 1)
    axes[1].bar(x, data.tau, color=COLORS["gold"], width=0.70)
    axes[1].set_xticks(x, labels, rotation=28, ha="right")
    axes[1].set_ylabel(r"Selected temperature $\tau$")
    axes[1].set_title("(b) Membership-temperature stability")
    fig.tight_layout()
    save(fig, output, "Supplementary_Figure_S4_Selected_regime_hyperparameter_stability")


def supplementary_robust(tables: Path, output: Path) -> None:
    data = pd.read_csv(tables / "pooled_vs_robust_conformal.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    x = np.arange(len(data))
    width = 0.36
    ax.bar(x - width / 2, data.coverage_pooled, width, color=COLORS["blue"], label="Pooled normalized")
    ax.bar(x + width / 2, data.coverage_worst_group, width, color=COLORS["coral"], label="Worst-group normalized")
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1.1)
    ax.set_xticks(x, [f"{s[:3]}-{f + 1}" for s, f in zip(data.scheme, data.outer_fold)], rotation=28, ha="right")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Spatially robust conformal sensitivity analysis")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, output, "Supplementary_Figure_S5_Spatially_robust_conformal_sensitivity_analysis")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    style()
    tables = args.root / "results" / "tables"
    attribution = args.root / "results" / "attribution_validation"
    output = args.root / "manuscript_figures_reorganized"
    main_output = output / "main_text"
    supplementary_output = output / "supplementary"
    assets = args.root / "assets" / "retained_figures"
    main_output.mkdir(parents=True, exist_ok=True)
    supplementary_output.mkdir(parents=True, exist_ok=True)

    export_retained_figure(
        assets / "Original_Figure_1_Study_area.jpeg",
        main_output,
        "Figure_1_Study_area_and_observation_data",
    )
    export_retained_figure(
        assets / "Original_Supplementary_Pearson_Correlation_Matrix.png",
        supplementary_output,
        "Supplementary_Figure_S1_Pearson_correlation_matrix",
    )
    export_retained_figure(
        assets / "Original_Supplementary_Predictor_Layers.png",
        supplementary_output,
        "Supplementary_Figure_S2_Spatial_distribution_of_predictor_layers",
    )
    export_retained_figure(
        assets / "Original_Figure_7_Hydrogeological_consistency.jpeg",
        supplementary_output,
        "Supplementary_Figure_S3_Hydrogeological_consistency",
    )

    figure1_partitions(tables, main_output)
    figure2_framework(main_output)
    figure3_performance(tables, attribution, main_output)
    figure5_spatial_predictions(tables, main_output)
    figure6_regimes_and_weights(args.root, tables, main_output)
    figure6_conformal(tables, main_output)
    supplementary_hyperparameter_stability(tables, supplementary_output)
    supplementary_robust(tables, supplementary_output)
    figure7_runtime(tables, supplementary_output)
    print(f"Generated manuscript figures in {output}")


if __name__ == "__main__":
    main()
