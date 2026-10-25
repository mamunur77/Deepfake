"""
src.explain.explain  ·  Visual Explanations: Grad-CAM & Attention Rollout.
========================================================================
Features:
    - Single Image / Video Frame Explanation:
        Generates and saves Grad-CAM and Attention Rollout heatmap overlays.
    - Publication-Quality Grid Generation:
        Produces a comparative figure displaying:
            • Correct Real   (True Negative)
            • Correct Fake   (True Positive)
            • Misclassified  (False Positive / False Negative)
        Columns: Original Face Crop | Grad-CAM Heatmap | Attention Rollout
    - Hardware-safe: Limits local inference to a small number of samples.

Usage:
    # Single image explanation
    python src/explain/explain.py --checkpoint outputs/vit_b16_finetune/best_model.pt --image data/processed/real/face_test_real/frame_0000.jpg

    # Generate 3x3 explanation grid from validation set
    python src/explain/explain.py --checkpoint outputs/vit_b16_finetune/best_model.pt --split-csv data/splits/val.csv --grid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.face_extract import create_mtcnn, detect_and_crop_face, extract_frames_from_video
from src.data.transforms import get_val_transforms
from src.explain.attention_rollout import compute_attention_rollout
from src.explain.gradcam import compute_gradcam, overlay_heatmap
from src.models.factory import build_model
from src.models.video_model import build_video_model
from src.utils.config import add_config_args, apply_debug_overrides, load_config
from src.utils.safeguard import check_environment

__all__ = [
    "explain_single_image",
    "generate_explanation_grid",
    "run_explain_pipeline",
]


def _load_model_for_explain(
    checkpoint_path: str | Path,
    cfg: Any,
    device: torch.device,
) -> nn.Module:
    """Instantiate and load checkpoint weights for explainability."""
    checkpoint_path = Path(checkpoint_path)
    model_type = getattr(getattr(cfg, "model", None), "type", "image")
    data_mode = getattr(getattr(cfg, "data", None), "mode", "image")

    if model_type == "video" or data_mode in {"video", "cached_embedding"}:
        model = build_video_model(cfg=cfg).to(device)
    else:
        model = build_model(cfg=cfg).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def explain_single_image(
    model: nn.Module,
    image_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    img_size: int = 224,
    item_name: str = "sample",
) -> dict[str, Any]:
    """
    Generate and save Grad-CAM and Attention Rollout overlays for one image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read Image
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image at: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (img_size, img_size):
        rgb = cv2.resize(rgb, (img_size, img_size))

    # 2. Transform to Tensor
    transform = get_val_transforms(img_size=img_size)
    tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)

    # 3. Model Prediction
    with torch.no_grad():
        logits = model(tensor)
        prob = float(torch.sigmoid(logits).item())
        pred = int(prob >= 0.5)

    # 4. Grad-CAM
    cam_mask = compute_gradcam(model, tensor, target_category=1, device=device)[0]
    cam_overlay = overlay_heatmap(rgb, cam_mask, alpha=0.5)

    # 5. Attention Rollout (if Transformer)
    is_transformer = hasattr(model, "blocks") or (
        hasattr(model, "backbone") and hasattr(model.backbone, "blocks")
    )
    if is_transformer:
        rollout_mask = compute_attention_rollout(model, tensor, device=device)[0]
        rollout_overlay = overlay_heatmap(rgb, rollout_mask, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS)
    else:
        rollout_mask = cam_mask
        rollout_overlay = cam_overlay

    # 6. Save Individual Overlays
    cv2.imwrite(
        str(output_dir / f"{item_name}_gradcam.jpg"),
        cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(output_dir / f"{item_name}_rollout.jpg"),
        cv2.cvtColor(rollout_overlay, cv2.COLOR_RGB2BGR),
    )

    # 7. Side-by-side Figure
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Input Face\nProb(Fake)={prob:.3f}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cam_overlay)
    axes[1].set_title("Grad-CAM Overlay", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(rollout_overlay)
    axes[2].set_title("Attention Rollout", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    comparison_path = output_dir / f"{item_name}_explanation.png"
    plt.savefig(comparison_path, dpi=300)
    plt.close()

    print(f"[explain] Saved visual explanation -> {comparison_path.resolve()}")
    return {
        "probability": prob,
        "prediction": pred,
        "gradcam_path": str(output_dir / f"{item_name}_gradcam.jpg"),
        "comparison_path": str(comparison_path),
    }


def generate_explanation_grid(
    model: nn.Module,
    samples_df: pd.DataFrame,
    output_path: str | Path,
    device: torch.device,
    img_size: int = 224,
) -> None:
    """
    Generate a 3-row grid comparing:
        Row 1: Correct Real (TN)
        Row 2: Correct Fake (TP)
        Row 3: Misclassified (FP or FN)
    Columns: Original Face | Grad-CAM | Attention Rollout
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform = get_val_transforms(img_size=img_size)

    # Predict probabilities for samples in DataFrame
    evaluated_records = []
    for _, row in samples_df.iterrows():
        fp = row["frame_path"]
        label = int(row["label"])
        try:
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (img_size, img_size):
                rgb = cv2.resize(rgb, (img_size, img_size))

            tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                prob = float(torch.sigmoid(model(tensor)).item())
                pred = int(prob >= 0.5)

            evaluated_records.append({
                "frame_path": fp,
                "rgb": rgb,
                "tensor": tensor,
                "label": label,
                "prob": prob,
                "pred": pred,
            })
        except Exception:
            continue

    if not evaluated_records:
        print("[explain] No valid samples found for grid generation.")
        return

    # Categorize into Correct Real, Correct Fake, Misclassified
    correct_real = [r for r in evaluated_records if r["label"] == 0 and r["pred"] == 0]
    correct_fake = [r for r in evaluated_records if r["label"] == 1 and r["pred"] == 1]
    misclassified = [r for r in evaluated_records if r["label"] != r["pred"]]

    # Fallback if specific category is empty in small test set
    row_candidates = [
        ("Correct Real (True Negative)", correct_real[0] if correct_real else evaluated_records[0]),
        ("Correct Fake (True Positive)", correct_fake[0] if correct_fake else evaluated_records[-1]),
        ("Misclassified Example", misclassified[0] if misclassified else evaluated_records[0]),
    ]

    is_transformer = hasattr(model, "blocks") or (
        hasattr(model, "backbone") and hasattr(model.backbone, "blocks")
    )

    fig, axes = plt.subplots(3, 3, figsize=(11, 10))

    for row_idx, (title_prefix, item) in enumerate(row_candidates):
        rgb = item["rgb"]
        tensor = item["tensor"]
        prob = item["prob"]
        true_lbl = "Real" if item["label"] == 0 else "Fake"
        pred_lbl = "Real" if item["pred"] == 0 else "Fake"

        # Grad-CAM
        cam_mask = compute_gradcam(model, tensor, target_category=1, device=device)[0]
        cam_overlay = overlay_heatmap(rgb, cam_mask, alpha=0.5)

        # Attention Rollout
        if is_transformer:
            rollout_mask = compute_attention_rollout(model, tensor, device=device)[0]
            rollout_overlay = overlay_heatmap(rgb, rollout_mask, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS)
        else:
            rollout_overlay = cam_overlay

        # Column 0: Original
        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_title(
            f"{title_prefix}\nTrue: {true_lbl} | Pred: {pred_lbl} ({prob:.2f})",
            fontsize=10,
            fontweight="bold",
        )
        axes[row_idx, 0].axis("off")

        # Column 1: Grad-CAM
        axes[row_idx, 1].imshow(cam_overlay)
        axes[row_idx, 1].set_title("Grad-CAM Heatmap", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Column 2: Rollout
        axes[row_idx, 2].imshow(rollout_overlay)
        axes[row_idx, 2].set_title("Attention Rollout", fontsize=10, fontweight="bold")
        axes[row_idx, 2].axis("off")

    plt.suptitle("Deepfake Explainability Matrix (ViT Grad-CAM & Attention Rollout)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[explain] Generated 3x3 explanation grid -> {output_path.resolve()}\n")


def run_explain_pipeline(
    checkpoint: str | Path,
    config: str | Path | None = None,
    image_path: str | Path | None = None,
    video_path: str | Path | None = None,
    split_csv: str | Path | None = None,
    grid: bool = False,
    output_dir: str | Path | None = None,
    limit: int = 6,
    debug: bool = False,
) -> None:
    checkpoint_path = Path(checkpoint)

    # 1. Resolve configuration
    if config is not None:
        cfg = load_config(config)
    else:
        adj_cfg = checkpoint_path.parent / "config.yaml"
        cfg = load_config(adj_cfg) if adj_cfg.exists() else load_config("configs/default.yaml")

    if debug:
        cfg.debug.enabled = True
    cfg = apply_debug_overrides(cfg)

    # Enforce hardware safeguard
    check_environment(limit if not debug else 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir) if output_dir else checkpoint_path.parent / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[explain] Loading model on {device} ...")
    model = _load_model_for_explain(checkpoint_path, cfg, device)
    img_size = getattr(getattr(cfg, "data", None), "img_size", 224)

    # ── Single Image ──────────────────────────────────────────
    if image_path is not None:
        explain_single_image(
            model=model,
            image_path=image_path,
            output_dir=out_dir,
            device=device,
            img_size=img_size,
            item_name=Path(image_path).stem,
        )

    # ── Single Video (Extract first face) ──────────────────────
    elif video_path is not None:
        print(f"[explain] Extracting sample frame from video: {video_path}")
        frames = extract_frames_from_video(video_path, n_frames=1)
        if not frames:
            print(f"[explain] Error: Could not read video at {video_path}")
            return
        mtcnn = create_mtcnn(str(device))
        face = detect_and_crop_face(frames[0][1], mtcnn, margin_pct=20, output_size=img_size)
        if face is None:
            face = frames[0][1].resize((img_size, img_size))

        temp_img = out_dir / "temp_video_frame.jpg"
        face.save(temp_img)
        explain_single_image(
            model=model,
            image_path=temp_img,
            output_dir=out_dir,
            device=device,
            img_size=img_size,
            item_name=f"video_{Path(video_path).stem}",
        )

    # ── Explanation Grid (or Default) ─────────────────────────
    if grid or (image_path is None and video_path is None):
        csv_file = Path(split_csv) if split_csv else Path("data/splits/val.csv")
        if not csv_file.exists():
            csv_file = Path("data/splits/train.csv")

        if csv_file.exists():
            df = pd.read_csv(csv_file).iloc[:limit]
            grid_out = out_dir / "explainability_grid.png"
            generate_explanation_grid(
                model=model,
                samples_df=df,
                output_path=grid_out,
                device=device,
                img_size=img_size,
            )
        else:
            print(f"[explain] Notice: Split CSV not found at {csv_file}. Skipping grid.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake Explainability (Grad-CAM & Rollout)")
    parser = add_config_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--image", type=str, default=None, help="Path to single image.")
    parser.add_argument("--video", type=str, default=None, help="Path to single video.")
    parser.add_argument("--split-csv", type=str, default=None, help="Split CSV to draw grid examples from.")
    parser.add_argument("--grid", action="store_true", help="Generate 3x3 comparison grid.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for visualisations.")
    parser.add_argument("--limit", type=int, default=6, help="Limit number of local samples.")
    args = parser.parse_args()

    run_explain_pipeline(
        checkpoint=args.checkpoint,
        config=args.config if args.config != "configs/default.yaml" else None,
        image_path=args.image,
        video_path=args.video,
        split_csv=args.split_csv,
        grid=args.grid,
        output_dir=args.output_dir,
        limit=args.limit,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
