"""
src.evaluation.metrics  ·  Metric computation, ROC curves, and confusion matrices.
==================================================================================
Provides:
    - compute_classification_metrics: Accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
    - aggregate_video_metrics: Frame-to-video probability pooling and video-level evaluation.
    - plot_roc_curve: Matplotlib ROC curve plot with AUC score.
    - plot_confusion_matrix: Annotated confusion matrix heatmap (counts + %).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "compute_classification_metrics",
    "aggregate_video_metrics",
    "plot_roc_curve",
    "plot_confusion_matrix",
]

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Metrics Computation
# ═════════════════════════════════════════════════════════════════

def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """
    Compute binary classification metrics.

    Parameters
    ----------
    y_true : Sequence[int] | np.ndarray
        Ground truth binary labels (0 = Real, 1 = Fake).
    y_prob : Sequence[float] | np.ndarray
        Predicted probabilities for the positive class (Fake).
    threshold : float, default=0.5
        Decision threshold for binary prediction.

    Returns
    -------
    dict[str, float | int]
        Dictionary with accuracy, precision, recall, f1, auc, and confusion matrix counts.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.5,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "samples": 0,
        }

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC requires at least one sample in each class
    unique_classes = np.unique(y_true)
    if len(unique_classes) >= 2:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "samples": len(y_true),
    }


# ═════════════════════════════════════════════════════════════════
#  Video-Level Metrics Aggregation
# ═════════════════════════════════════════════════════════════════

def aggregate_video_metrics(
    df_predictions: pd.DataFrame,
    aggregation_method: str = "mean",
    threshold: float = 0.5,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """
    Aggregate per-frame predictions by video_id to produce video-level metrics.

    Parameters
    ----------
    df_predictions : pd.DataFrame
        Must contain 'video_id', 'label', and 'probability'.
    aggregation_method : str, default="mean"
        Pooling function: 'mean', 'max', or 'median'.
    threshold : float, default=0.5
        Decision threshold.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        (video_metrics_dict, df_video_predictions)
    """
    if "video_id" not in df_predictions.columns or df_predictions.empty:
        return compute_classification_metrics([], []), pd.DataFrame()

    # Group by video_id
    grouped = df_predictions.groupby("video_id")
    video_records = []

    for video_id, group in grouped:
        true_label = int(group["label"].iloc[0])
        probs = group["probability"].to_numpy()

        if aggregation_method == "max":
            v_prob = float(np.max(probs))
        elif aggregation_method == "median":
            v_prob = float(np.median(probs))
        else:  # mean
            v_prob = float(np.mean(probs))

        v_pred = int(v_prob >= threshold)
        video_records.append({
            "video_id": video_id,
            "label": true_label,
            "probability": v_prob,
            "prediction": v_pred,
            "num_frames": len(group),
        })

    df_video = pd.DataFrame(video_records)
    metrics = compute_classification_metrics(
        y_true=df_video["label"].to_numpy(),
        y_prob=df_video["probability"].to_numpy(),
        threshold=threshold,
    )
    return metrics, df_video


# ═════════════════════════════════════════════════════════════════
#  Visualisation: ROC Curve
# ═════════════════════════════════════════════════════════════════

def plot_roc_curve(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    save_path: str | Path,
    title: str = "Receiver Operating Characteristic (ROC)",
) -> None:
    """Plot and save ROC curve."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    plt.figure(figsize=(7, 6))

    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
    else:
        plt.plot([0, 1], [0, 1], color="#2563eb", lw=2, label="ROC Curve (Insufficient classes)")

    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ═════════════════════════════════════════════════════════════════
#  Visualisation: Confusion Matrix
# ═════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    save_path: str | Path,
    class_names: tuple[str, str] = ("Real", "Fake"),
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save annotated confusion matrix with counts and percentages."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = np.sum(cm)
    norm_cm = cm.astype(float) / total if total > 0 else cm.astype(float)

    # Format annotations: "count\n(percentage%)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({norm_cm[i, j] * 100:.1f}%)"

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"fontsize": 13, "fontweight": "bold"},
    )
    plt.xlabel("Predicted Label", fontsize=12, labelpad=8)
    plt.ylabel("True Ground Truth Label", fontsize=12, labelpad=8)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
