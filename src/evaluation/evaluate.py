"""
src.evaluation.evaluate  ·  Model Evaluation, Metrics & Cross-Dataset Benchmark.
================================================================================
Loads a trained checkpoint and evaluates it on test data:
    - Reports: Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
    - Video Aggregation: Computes video-level scores by aggregating frame probabilities.
    - Cross-Dataset Benchmark: --test_data enables evaluating on external datasets (e.g., Celeb-DF).
    - Output Artifacts in outputs/{run_name}/:
        • metrics.json (or metrics_cross_{dataset}.json)
        • metrics.csv
        • predictions.csv
        • roc_curve.png
        • confusion_matrix.png

Usage:
    # Standard evaluation on in-domain test set
    python src/evaluation/evaluate.py --checkpoint outputs/vit_b16_finetune/best_model.pt

    # Cross-dataset benchmark (Train on FaceForensics++, Test on Celeb-DF)
    python src/evaluation/evaluate.py --checkpoint outputs/vit_b16_finetune/best_model.pt --test_data data/splits_celebdf/test.csv

    # Debug mode (safe on local laptop)
    python src/evaluation/evaluate.py --checkpoint outputs/efficientnet_b0/best_model.pt --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CachedEmbeddingDataset, DeepfakeDataset
from src.data.transforms import get_val_transforms
from src.evaluation.metrics import (
    aggregate_video_metrics,
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.models.factory import build_model
from src.models.video_model import build_video_model
from src.utils.config import add_config_args, apply_debug_overrides, load_config
from src.utils.safeguard import check_environment
from src.utils.seed import seed_everything

__all__ = ["evaluate_checkpoint", "run_evaluation"]

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Dataset Loader for Evaluation
# ═════════════════════════════════════════════════════════════════

def _load_eval_dataset_and_loader(
    test_csv_path: Path,
    cfg: Any,
    batch_size: int = 32,
    num_workers: int = 2,
    is_debug: bool = False,
) -> tuple[DataLoader, pd.DataFrame]:
    """Build evaluation DataLoader and return underlying DataFrame."""
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {test_csv_path}")

    df = pd.read_csv(test_csv_path)
    if df.empty:
        raise ValueError(f"Evaluation CSV at {test_csv_path} is empty.")

    if is_debug:
        max_s = getattr(getattr(cfg, "debug", None), "max_samples", 50)
        df = df.iloc[:max_s].reset_index(drop=True)

    mode = getattr(getattr(cfg, "data", None), "mode", "image")
    num_frames = getattr(getattr(cfg, "data", None), "num_frames", 16)
    img_size = getattr(getattr(cfg, "data", None), "img_size", 224)

    # Branch A: Cached embedding mode
    if mode in {"cached_embedding", "embedding", "frozen_embeddings"}:
        dataset = CachedEmbeddingDataset(df, num_frames=num_frames, is_train=False)
    # Branch B: Raw frames (image or video mode)
    else:
        val_tf = get_val_transforms(img_size)
        dataset = DeepfakeDataset(df, mode=mode, num_frames=num_frames, transform=val_tf, is_train=False)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if is_debug else num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader, df


# ═════════════════════════════════════════════════════════════════
#  Core Evaluation Routine
# ═════════════════════════════════════════════════════════════════

def evaluate_checkpoint(
    checkpoint_path: str | Path,
    cfg: Any,
    test_csv_path: str | Path,
    output_dir: str | Path | None = None,
    dataset_name: str = "in_domain",
    device: str | torch.device = "auto",
    batch_size: int = 32,
    is_debug: bool = False,
) -> dict[str, Any]:
    """
    Run full evaluation on a model checkpoint and save all artifacts.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Resolve target device
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    # Resolve output directory
    if output_dir is None:
        out_dir = checkpoint_path.parent
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_csv = Path(test_csv_path)

    # 1. Build DataLoader
    loader, df_test = _load_eval_dataset_and_loader(
        test_csv_path=test_csv,
        cfg=cfg,
        batch_size=batch_size,
        is_debug=is_debug,
    )

    # 2. Build Model
    model_type = getattr(getattr(cfg, "model", None), "type", "image")
    data_mode = getattr(getattr(cfg, "data", None), "mode", "image")

    if model_type == "video" or data_mode in {"video", "cached_embedding"}:
        model = build_video_model(cfg=cfg).to(dev)
    else:
        model = build_model(cfg=cfg).to(dev)

    # 3. Load Checkpoint state_dict
    print(f"[evaluate] Loading checkpoint weights: {checkpoint_path} ...")
    state_dict = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    all_probs: list[float] = []
    all_targets: list[int] = []

    print(f"[evaluate] Running inference on {len(loader.dataset)} samples ({dev}) ...")
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating", leave=True):
            inputs = inputs.to(dev)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            t_vals = targets.numpy().flatten().astype(int)

            all_probs.extend(probs.tolist())
            all_targets.extend(t_vals.tolist())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    # 5. Compute Frame/Sample Metrics
    sample_metrics = compute_classification_metrics(y_true=y_true, y_prob=y_prob)

    # 6. Save Predictions Table
    df_pred = df_test.copy().iloc[: len(y_prob)].reset_index(drop=True)
    df_pred["probability"] = y_prob
    df_pred["prediction"] = y_pred

    pred_csv_name = (
        "predictions.csv" if dataset_name == "in_domain" else f"predictions_cross_{dataset_name}.csv"
    )
    df_pred.to_csv(out_dir / pred_csv_name, index=False)

    # 7. Compute Video-Level Metrics (if video_id present)
    has_video_id = "video_id" in df_pred.columns
    if has_video_id:
        video_metrics, df_video_pred = aggregate_video_metrics(df_pred, aggregation_method="mean")
        video_csv_name = (
            "video_predictions.csv"
            if dataset_name == "in_domain"
            else f"video_predictions_cross_{dataset_name}.csv"
        )
        df_video_pred.to_csv(out_dir / video_csv_name, index=False)
    else:
        video_metrics = sample_metrics

    # 8. Save Visualisations (ROC Curve and Confusion Matrix)
    prefix = "" if dataset_name == "in_domain" else f"cross_{dataset_name}_"
    roc_path = out_dir / f"{prefix}roc_curve.png"
    cm_path = out_dir / f"{prefix}confusion_matrix.png"

    plot_roc_curve(
        y_true=y_true,
        y_prob=y_prob,
        save_path=roc_path,
        title=f"ROC Curve - {cfg.experiment.name} ({dataset_name})",
    )
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        save_path=cm_path,
        title=f"Confusion Matrix - {cfg.experiment.name} ({dataset_name})",
    )

    # 9. Assemble Summary & Export JSON / CSV
    results: dict[str, Any] = {
        "experiment_name": getattr(getattr(cfg, "experiment", None), "name", "experiment"),
        "backbone": getattr(getattr(cfg, "model", None), "backbone", "unknown"),
        "mode": getattr(getattr(cfg, "model", None), "mode", "finetune"),
        "checkpoint": str(checkpoint_path.resolve()),
        "test_dataset": dataset_name,
        "test_csv": str(test_csv.resolve()),
        "frame_metrics": sample_metrics,
        "video_metrics": video_metrics,
    }

    metrics_json_name = "metrics.json" if dataset_name == "in_domain" else f"metrics_cross_{dataset_name}.json"
    json_path = out_dir / metrics_json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Flat CSV export
    flat_row = {
        "experiment": results["experiment_name"],
        "backbone": results["backbone"],
        "dataset": dataset_name,
        "frame_acc": sample_metrics["accuracy"],
        "frame_auc": sample_metrics["auc"],
        "frame_f1": sample_metrics["f1"],
        "frame_prec": sample_metrics["precision"],
        "frame_rec": sample_metrics["recall"],
        "video_acc": video_metrics["accuracy"],
        "video_auc": video_metrics["auc"],
        "video_f1": video_metrics["f1"],
    }
    metrics_csv_name = "metrics.csv" if dataset_name == "in_domain" else f"metrics_cross_{dataset_name}.csv"
    pd.DataFrame([flat_row]).to_csv(out_dir / metrics_csv_name, index=False)

    # 10. Print Formatted Terminal Summary
    print("\n" + "=" * 70)
    print(f"  Evaluation Results: {results['experiment_name']}  ({dataset_name})")
    print("=" * 70)
    print(f"  Checkpoint : {checkpoint_path.name}")
    print(f"  Test Data  : {test_csv.name} ({sample_metrics['samples']} samples)")
    print("-" * 70)
    print(f"  {'Metric':<18} | {'Frame-Level':<14} | {'Video-Level':<14}")
    print("-" * 70)
    print(f"  {'Accuracy':<18} | {sample_metrics['accuracy']*100:>13.2f}% | {video_metrics['accuracy']*100:>13.2f}%")
    print(f"  {'ROC-AUC':<18} | {sample_metrics['auc']:>14.4f} | {video_metrics['auc']:>14.4f}")
    print(f"  {'F1-Score':<18} | {sample_metrics['f1']:>14.4f} | {video_metrics['f1']:>14.4f}")
    print(f"  {'Precision':<18} | {sample_metrics['precision']:>14.4f} | {video_metrics['precision']:>14.4f}")
    print(f"  {'Recall':<18} | {sample_metrics['recall']:>14.4f} | {video_metrics['recall']:>14.4f}")
    print("-" * 70)
    print(f"  Confusion Matrix (Frame) : TN={sample_metrics['tn']} | FP={sample_metrics['fp']} | FN={sample_metrics['fn']} | TP={sample_metrics['tp']}")
    print(f"  ROC Plot Saved to        : {roc_path.resolve()}")
    print(f"  Metrics JSON Saved to    : {json_path.resolve()}")
    print("=" * 70 + "\n")

    return results


# ═════════════════════════════════════════════════════════════════
#  CLI Runner & Cross-Dataset Dispatcher
# ═════════════════════════════════════════════════════════════════

def run_evaluation(
    checkpoint: str | Path,
    config: str | Path | None = None,
    test_data: str | Path | None = None,
    split_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    debug: bool = False,
    batch_size: int = 32,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)

    # 1. Resolve configuration
    if config is not None:
        cfg = load_config(config)
    else:
        # Check adjacent config.yaml saved with checkpoint
        adj_cfg = checkpoint_path.parent / "config.yaml"
        if adj_cfg.exists():
            cfg = load_config(adj_cfg)
        else:
            cfg = load_config("configs/default.yaml")

    if debug:
        cfg.debug.enabled = True
    cfg = apply_debug_overrides(cfg)
    seed_everything(getattr(getattr(cfg, "experiment", None), "seed", 42))

    # 2. Resolve Test Split & Cross-Dataset Name
    dataset_name = "in_domain"
    if test_data is not None:
        test_path = Path(test_data)
        dataset_name = test_path.stem.replace("test", "").replace("_", "") or "cross_test"
        if "celeb" in str(test_data).lower():
            dataset_name = "celeb_df"
        elif "dfdc" in str(test_data).lower():
            dataset_name = "dfdc"
        elif "faceforensics" in str(test_data).lower() or "ff" in str(test_data).lower():
            dataset_name = "faceforensics"
    elif split_csv is not None:
        test_path = Path(split_csv)
    else:
        split_dir = Path(getattr(getattr(cfg, "data", None), "split_dir", "data/splits"))
        test_path = split_dir / "test.csv"

    # Enforce hardware safeguard
    check_environment(50 if debug else 100)

    return evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        test_csv_path=test_path,
        output_dir=output_dir,
        dataset_name=dataset_name,
        batch_size=batch_size,
        is_debug=debug,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Deepfake ViT Model Checkpoint")
    parser = add_config_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint (.pt).")
    parser.add_argument("--test_data", "--test-data", type=str, default=None, help="Cross-dataset test CSV (e.g. Celeb-DF).")
    parser.add_argument("--split-csv", type=str, default=None, help="Explicit split CSV to evaluate.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for metrics and plots.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    args = parser.parse_args()

    run_evaluation(
        checkpoint=args.checkpoint,
        config=args.config if args.config != "configs/default.yaml" else None,
        test_data=args.test_data,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        debug=args.debug,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
