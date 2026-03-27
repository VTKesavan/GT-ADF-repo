"""
Evaluation metrics for GT-ADF.

Computes:
    - Accuracy
    - Precision (macro)
    - Recall (macro)
    - F1-Score (macro)
    - ROC-AUC (macro OvR)
    - Matthews Correlation Coefficient (MCC)
    - Negative Predictive Value (NPV)
    - Per-class precision, recall, F1
    - Confusion matrix
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_score: Optional[np.ndarray] = None,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_score: Predicted probabilities [N, C] (optional, for AUC).
        average: Averaging strategy for multi-class metrics.

    Returns:
        Dictionary of metric name → value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}

    # Standard classification metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision_macro"] = float(
        precision_score(y_true, y_pred, average=average, zero_division=0)
    )
    metrics["recall_macro"] = float(
        recall_score(y_true, y_pred, average=average, zero_division=0)
    )
    metrics["f1_macro"] = float(
        f1_score(y_true, y_pred, average=average, zero_division=0)
    )
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    # NPV: TN / (TN + FN) for binary case
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        metrics["tpr"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # ROC-AUC
    if y_score is not None:
        try:
            n_classes = y_score.shape[1] if y_score.ndim > 1 else 2
            if n_classes == 2 and y_score.ndim > 1:
                auc = roc_auc_score(y_true, y_score[:, 1])
            else:
                auc = roc_auc_score(
                    y_true, y_score, multi_class="ovr", average=average
                )
            metrics["roc_auc"] = float(auc)
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics["roc_auc"] = 0.0

    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    return_scores: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the GT-ADF model on a data loader.

    Args:
        model: Trained GT-ADF model.
        loader: DataLoader with test graphs.
        device: Compute device.
        return_scores: If True, compute ROC-AUC using softmax probabilities.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _, anomaly_scores = model(batch.x, batch.edge_index, batch.batch)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch.y.cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_scores.append(probs)

    scores_array = np.concatenate(all_scores, axis=0) if return_scores else None
    metrics = compute_metrics(all_labels, all_preds, y_score=scores_array)

    # Print classification report
    report = classification_report(
        all_labels, all_preds, zero_division=0
    )
    logger.info(f"\nClassification Report:\n{report}")

    return metrics


def print_metrics_table(metrics: Dict[str, float], dataset_name: str = "") -> None:
    """Pretty-print metrics in a table format."""
    title = f"Evaluation Results{' — ' + dataset_name if dataset_name else ''}"
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    key_metrics = [
        ("Accuracy",  "accuracy"),
        ("Precision (macro)", "precision_macro"),
        ("Recall (macro)",    "recall_macro"),
        ("F1-Score (macro)",  "f1_macro"),
        ("ROC-AUC",           "roc_auc"),
        ("MCC",               "mcc"),
        ("NPV",               "npv"),
    ]
    for label, key in key_metrics:
        if key in metrics:
            print(f"  {label:<25} {metrics[key]:.4f}")
    print(f"{'='*55}\n")
