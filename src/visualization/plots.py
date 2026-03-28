"""
Visualization functions for GT-ADF results.

"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Style constants (matching paper figures)
# -----------------------------------------------------------------------
MODEL_COLORS = {
    "SVM": "#8ecae6",
    "Random Forest": "#219ebc",
    "CNN-IDS": "#023047",
    "LSTM": "#ffb703",
    "GNN": "#fb8500",
    "TADDY": "#9b5de5",
    "GraphBERT": "#f15bb5",
    "Federated GNN-IDS": "#00bbf9",
    "GT-ADF (Proposed)": "#e63946",
}

FIGURE_DPI = 150
FIGURE_SIZE = (9, 5)


def _save_or_show(fig, save_path: Optional[str] = None):
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Figure saved: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 7: Detection Accuracy Comparison
# -----------------------------------------------------------------------

def plot_accuracy_comparison(
    results: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Detection Accuracy Analysis",
) -> plt.Figure:
    """
    Bar chart comparing detection accuracy across models.

    Args:
        results: {model_name: accuracy_percentage}
        save_path: Optional file path to save the figure.
    """
    models = list(results.keys())
    accuracies = [results[m] * 100 if results[m] <= 1.0 else results[m] for m in models]
    colors = [MODEL_COLORS.get(m, "#adb5bd") for m in models]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.bar(models, accuracies, color=colors, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylim(75, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    _save_or_show(fig, save_path)
    return fig


# -----------------------------------------------------------------------
# Figure 8: Precision / Recall / F1 Grouped Bar Chart
# -----------------------------------------------------------------------

def plot_precision_recall_f1(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Precision, Recall, F1-Score Analysis",
) -> plt.Figure:
    """
    Grouped bar chart for Precision, Recall, and F1-Score.

    Args:
        results: {model_name: {'precision': x, 'recall': y, 'f1': z}}
    """
    models = list(results.keys())
    metrics = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1-Score"]
    metric_colors = ["#4361ee", "#f72585", "#4cc9f0"]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        values = []
        for m in models:
            v = results[m].get(metric, 0.0)
            values.append(v * 100 if v <= 1.0 else v)
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(75, 100)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    _save_or_show(fig, save_path)
    return fig


# -----------------------------------------------------------------------
# Figure 10: ROC Curves
# -----------------------------------------------------------------------

def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "ROC Curves Analysis",
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.

    Args:
        y_true: True binary labels [N].
        y_scores: {model_name: probability_scores [N]}
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for model_name, scores in y_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        color = MODEL_COLORS.get(model_name, "#adb5bd")
        lw = 2.5 if "GT-ADF" in model_name else 1.5
        ax.plot(
            fpr, tpr,
            color=color, linewidth=lw,
            label=f"{model_name} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    _save_or_show(fig, save_path)
    return fig


# -----------------------------------------------------------------------
# Figure 11: Scalability Analysis
# -----------------------------------------------------------------------

def plot_scalability(
    network_sizes: List[int],
    model_accuracies: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Scalability Analysis",
) -> plt.Figure:
    """
    Line chart: detection accuracy vs. number of EV charging stations.

    Args:
        network_sizes: X-axis (number of stations).
        model_accuracies: {model_name: [accuracy at each network size]}
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for model_name, accs in model_accuracies.items():
        accs_pct = [a * 100 if a <= 1.0 else a for a in accs]
        color = MODEL_COLORS.get(model_name, "#adb5bd")
        lw = 2.5 if "GT-ADF" in model_name else 1.5
        marker = "o" if "GT-ADF" in model_name else "^"
        ax.plot(
            network_sizes, accs_pct,
            color=color, linewidth=lw, marker=marker, markersize=5,
            label=model_name,
        )

    ax.axhline(y=94, color="gray", linestyle="--", linewidth=1.0, alpha=0.5,
               label="94% threshold")
    ax.set_xlabel("Number of EV Charging Stations", fontsize=11)
    ax.set_ylabel("Detection Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(70, 100)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    _save_or_show(fig, save_path)
    return fig


# -----------------------------------------------------------------------
# Figure 12: Processing Time Comparison
# -----------------------------------------------------------------------

def plot_processing_time(
    model_times: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Processing Time Comparison (ms)",
) -> plt.Figure:
    """Bar chart of inference latency per model."""
    models = list(model_times.keys())
    times = list(model_times.values())
    colors = [MODEL_COLORS.get(m, "#adb5bd") for m in models]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.bar(models, times, color=colors, edgecolor="white")

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{t:.1f}ms",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    _save_or_show(fig, save_path)
    return fig


# -----------------------------------------------------------------------
# Figure 13: Attention Weight Heatmap
# -----------------------------------------------------------------------

def plot_attention_heatmap(
    attention_weights: np.ndarray,
    node_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Attention Weight Distribution",
) -> plt.Figure:
    """
    Heatmap of aggregated attention coefficients between nodes.

    Args:
        attention_weights: [N, N] matrix of attention values.
        node_labels: Optional list of node name strings.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        attention_weights,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=node_labels or "auto",
        yticklabels=node_labels or "auto",
        linewidths=0.3,
        annot=(attention_weights.shape[0] <= 10),
        fmt=".2f",
        cbar_kws={"label": "Attention Weight"},
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Target Node (j)", fontsize=10)
    ax.set_ylabel("Source Node (i)", fontsize=10)

    _save_or_show(fig, save_path)
    return fig


# -----------------------------------------------------------------------
# Figure 14: Cross-dataset Validation
# -----------------------------------------------------------------------

def plot_cross_dataset_validation(
    dataset_results: Dict[str, Dict[str, float]],
    metric: str = "f1",
    save_path: Optional[str] = None,
    title: str = "Cross-Dataset Validation (F1-Score)",
) -> plt.Figure:
    """
    Grouped bar chart comparing GT-ADF vs. baselines across datasets.

    Args:
        dataset_results: {dataset_name: {model_name: score}}
        metric: Metric key to plot.
    """
    datasets = list(dataset_results.keys())
    all_models = list({m for d in dataset_results.values() for m in d.keys()})

    x = np.arange(len(datasets))
    width = 0.8 / len(all_models)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, model_name in enumerate(all_models):
        values = [
            dataset_results[d].get(model_name, 0.0) for d in datasets
        ]
        values_pct = [v * 100 if v <= 1.0 else v for v in values]
        color = MODEL_COLORS.get(model_name, "#adb5bd")
        offset = (i - len(all_models) / 2 + 0.5) * width
        ax.bar(x + offset, values_pct, width * 0.9, label=model_name,
               color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("F1-Score (%)", fontsize=11)
    ax.set_ylim(75, 100)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    _save_or_show(fig, save_path)
    return fig


# -----------------------------------------------------------------------
# Utility: generate all paper figures from results dict
# -----------------------------------------------------------------------

def generate_all_figures(results: dict, output_dir: str = "results/figures") -> None:
    """
    Generate and save all main result figures.

    Args:
        results: Dictionary containing all model results.
        output_dir: Directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Figure 7: Accuracy
    if "accuracy" in results:
        plot_accuracy_comparison(
            results["accuracy"],
            save_path=os.path.join(output_dir, "fig7_accuracy_comparison.png"),
        )

    # Figure 8: Precision/Recall/F1
    if "prf" in results:
        plot_precision_recall_f1(
            results["prf"],
            save_path=os.path.join(output_dir, "fig8_precision_recall_f1.png"),
        )

    # Figure 10: ROC curves
    if "roc" in results:
        plot_roc_curves(
            results["roc"]["y_true"],
            results["roc"]["y_scores"],
            save_path=os.path.join(output_dir, "fig10_roc_curves.png"),
        )

    # Figure 11: Scalability
    if "scalability" in results:
        plot_scalability(
            results["scalability"]["network_sizes"],
            results["scalability"]["model_accuracies"],
            save_path=os.path.join(output_dir, "fig11_scalability.png"),
        )

    # Figure 12: Processing time
    if "latency" in results:
        plot_processing_time(
            results["latency"],
            save_path=os.path.join(output_dir, "fig12_processing_time.png"),
        )

    # Figure 14: Cross-dataset
    if "cross_dataset" in results:
        plot_cross_dataset_validation(
            results["cross_dataset"],
            save_path=os.path.join(output_dir, "fig14_cross_dataset.png"),
        )

    logger.info(f"All figures saved to {output_dir}")
