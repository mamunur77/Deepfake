"""
src.evaluation  ·  Evaluation pipelines, metrics, and visualization.
"""

from .evaluate import evaluate_checkpoint, run_evaluation
from .metrics import (
    aggregate_video_metrics,
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)

__all__ = [
    "evaluate_checkpoint",
    "run_evaluation",
    "compute_classification_metrics",
    "aggregate_video_metrics",
    "plot_roc_curve",
    "plot_confusion_matrix",
]
