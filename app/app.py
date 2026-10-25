"""
app/app.py  ·  Interactive Deepfake Detection Web Application (Gradio).
======================================================================
Deepfake Image and Video Detection Using Vision Transformers:
A Comparative Study of Supervised (ViT, DeiT) and Self-Supervised (DINOv2) Pretraining.

Features:
    - Image Tab:
        • MTCNN face detection with 20% margin.
        • Graceful handling when no face is detected.
        • Single logit inference -> Real/Fake verdict with confidence %.
        • Dual visual explainability: Grad-CAM heatmap & Attention Rollout overlays.
    - Video Tab:
        • Uniform sampling of 16 frames across video duration.
        • Per-frame MTCNN face extraction and prediction.
        • Interactive Matplotlib temporal plot of per-frame fake probabilities.
        • Aggregated video-level prediction and confidence score.
        • High-attention frame inspection with Grad-CAM heatmap.
    - Backbone & Checkpoint Switcher:
        • Compare DINOv2, DeiT-S, ViT-B, and EfficientNet backbones.
        • Runs on CPU by default (compatible with free-tier Hugging Face Spaces).

Hugging Face Spaces:
    - Compatible with Gradio SDK (app_file: app/app.py).
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.data.face_extract import create_mtcnn, detect_and_crop_face, extract_frames_from_video
from src.data.transforms import get_val_transforms
from src.explain.attention_rollout import compute_attention_rollout
from src.explain.gradcam import compute_gradcam, overlay_heatmap
from src.models.factory import build_model
from src.models.video_model import build_video_model
from src.utils.config import load_config


# ═════════════════════════════════════════════════════════════════
#  Global Model & MTCNN Cache
# ═════════════════════════════════════════════════════════════════

class AppState:
    """Singleton to cache detector and loaded models for low latency."""
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.mtcnn = create_mtcnn(device=device)
        self.val_transform = get_val_transforms(img_size=224)
        self.current_model: nn.Module | None = None
        self.current_model_name: str = ""
        self.cfg: Any = None

    def load_active_model(
        self,
        checkpoint_path: str | Path | None = None,
        backbone: str = "dinov2_vits14",
    ) -> tuple[nn.Module, str]:
        """Load requested model checkpoint or initialize backbone."""
        # 1. Resolve checkpoint path
        cand_paths = []
        if checkpoint_path:
            cand_paths.append(Path(checkpoint_path))
        cand_paths.extend([
            PROJECT_ROOT / "outputs" / "dinov2_frozen" / "best_model.pt",
            PROJECT_ROOT / "outputs" / "deit_small_finetune" / "best_model.pt",
            PROJECT_ROOT / "outputs" / "efficientnet_b0" / "best_model.pt",
            PROJECT_ROOT / "outputs" / "video_vit_temporal" / "best_model.pt",
        ])

        found_ckpt = None
        for p in cand_paths:
            if p.exists():
                found_ckpt = p
                break

        # 2. Load associated config if available
        if found_ckpt and (found_ckpt.parent / "config.yaml").exists():
            cfg = load_config(found_ckpt.parent / "config.yaml")
        elif (PROJECT_ROOT / "configs" / "app_config.yaml").exists():
            cfg = load_config(PROJECT_ROOT / "configs" / "app_config.yaml")
        else:
            cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")

        self.cfg = cfg
        bb = getattr(getattr(cfg, "model", None), "backbone", backbone)
        model_type = getattr(getattr(cfg, "model", None), "type", "image")

        print(f"[app] Loading model '{bb}' (ckpt: {found_ckpt}) on {self.device}...")

        if model_type == "video":
            model = build_video_model(cfg=cfg).to(self.device)
        else:
            model = build_model(
                backbone=bb,
                pretrained=True,
                mode=getattr(getattr(cfg, "model", None), "mode", "finetune"),
            ).to(self.device)

        if found_ckpt and found_ckpt.exists():
            try:
                state_dict = torch.load(found_ckpt, map_location=self.device)
                model.load_state_dict(state_dict)
                status = f"Loaded checkpoint: {found_ckpt.name} ({bb})"
            except Exception as e:
                status = f"Pretrained weights ({bb}) [Checkpoint load notice: {e}]"
        else:
            status = f"Pretrained base weights ({bb}) [No fine-tuned checkpoint found]"

        model.eval()
        self.current_model = model
        self.current_model_name = bb
        return model, status


# Global instance default to CPU
app_state = AppState(device="cpu")


# ═════════════════════════════════════════════════════════════════
#  Inference Pipeline: Single Image
# ═════════════════════════════════════════════════════════════════

def predict_single_image(
    input_image: Image.Image | np.ndarray | None,
    threshold: float = 0.5,
) -> tuple[str, dict[str, float] | None, Image.Image | None, Image.Image | None, Image.Image | None]:
    """
    Execute full inference pipeline for one image:
        1. MTCNN Face Detection
        2. Albumentations Preprocessing
        3. Model Forward Pass
        4. Grad-CAM & Attention Rollout heatmaps
    """
    if input_image is None:
        return (
            "<div style='padding: 12px; background: #2d3748; color: #cbd5e0; border-radius: 8px;'>Please upload an image to begin analysis.</div>",
            None,
            None,
            None,
            None,
        )

    # Convert to PIL Image
    if isinstance(input_image, np.ndarray):
        img_pil = Image.fromarray(input_image.astype(np.uint8))
    else:
        img_pil = input_image.convert("RGB")

    # 1. MTCNN Face Detection with 20% margin
    face_crop = detect_and_crop_face(
        image=img_pil,
        mtcnn=app_state.mtcnn,
        margin_pct=20,
        output_size=224,
    )

    if face_crop is None:
        error_html = (
            "<div style='padding: 16px; background: #fff5f5; border: 1px solid #feb2b2; border-radius: 8px; color: #c53030; font-weight: 600;'>"
            "⚠️ No Face Detected: MTCNN could not locate a human face in the uploaded image.<br>"
            "<span style='font-size: 0.85em; font-weight: normal; color: #742a2a;'>"
            "Please ensure the image contains a clear, unobstructed facial region.</span></div>"
        )
        return error_html, None, None, None, None

    # 2. Preprocess
    face_rgb = np.array(face_crop)
    tensor = app_state.val_transform(image=face_rgb)["image"].unsqueeze(0).to(app_state.device)

    # 3. Predict
    model = app_state.current_model
    if model is None:
        model, _ = app_state.load_active_model()

    with torch.no_grad():
        logits = model(tensor)
        prob_fake = float(torch.sigmoid(logits).item())
        prob_real = 1.0 - prob_fake

    is_fake = prob_fake >= threshold
    confidence = (prob_fake if is_fake else prob_real) * 100.0

    # 4. Grad-CAM
    cam_mask = compute_gradcam(
        model=model,
        input_tensor=tensor,
        target_category=1,
        device=app_state.device,
    )[0]
    cam_overlay_rgb = overlay_heatmap(face_rgb, cam_mask, alpha=0.5, colormap=cv2.COLORMAP_JET)
    cam_overlay_pil = Image.fromarray(cam_overlay_rgb)

    # 5. Attention Rollout (if Vision Transformer)
    is_transformer = hasattr(model, "blocks") or (
        hasattr(model, "backbone") and hasattr(model.backbone, "blocks")
    )
    if is_transformer:
        try:
            rollout_mask = compute_attention_rollout(model=model, input_tensor=tensor, device=app_state.device)[0]
            rollout_overlay_rgb = overlay_heatmap(face_rgb, rollout_mask, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS)
            rollout_overlay_pil = Image.fromarray(rollout_overlay_rgb)
        except Exception:
            rollout_overlay_pil = cam_overlay_pil
    else:
        rollout_overlay_pil = cam_overlay_pil

    # 6. Badge Output
    if is_fake:
        verdict_color = "#e53e3e"
        bg_color = "#fff5f5"
        border_color = "#feb2b2"
        badge_text = "FAKE / DEEPFAKE DETECTED"
    else:
        verdict_color = "#38a169"
        bg_color = "#f0fff4"
        border_color = "#9ae6b4"
        badge_text = "REAL / AUTHENTIC FACE"

    verdict_html = (
        f"<div style='padding: 16px; background: {bg_color}; border: 2px solid {border_color}; border-radius: 10px;'>"
        f"<div style='font-size: 1.25rem; font-weight: 800; color: {verdict_color}; margin-bottom: 4px;'>{badge_text}</div>"
        f"<div style='font-size: 1rem; color: #2d3748;'>"
        f"Confidence: <b>{confidence:.1f}%</b> | "
        f"P(Fake): <code style='background: #edf2f7; padding: 2px 6px; border-radius: 4px;'>{prob_fake:.4f}</code> | "
        f"P(Real): <code style='background: #edf2f7; padding: 2px 6px; border-radius: 4px;'>{prob_real:.4f}</code>"
        f"</div></div>"
    )

    label_dict = {"Real": prob_real, "Fake": prob_fake}
    return verdict_html, label_dict, face_crop, cam_overlay_pil, rollout_overlay_pil


# ═════════════════════════════════════════════════════════════════
#  Inference Pipeline: Video & Temporal Sampling
# ═════════════════════════════════════════════════════════════════

def plot_video_scores(
    frame_indices: list[int],
    probabilities: list[float],
    threshold: float = 0.5,
) -> Image.Image:
    """Generate high-contrast publication quality line plot of per-frame scores."""
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=200)

    x = np.arange(1, len(probabilities) + 1)
    y = np.array(probabilities)

    # Gradient fill between threshold and curve
    ax.axhline(threshold, color="#e53e3e", linestyle="--", linewidth=1.5, label=f"Decision Threshold ({threshold})")
    ax.fill_between(x, threshold, y, where=(y >= threshold), color="#e53e3e", alpha=0.2, interpolate=True)
    ax.fill_between(x, 0.0, y, where=(y < threshold), color="#38a169", alpha=0.15, interpolate=True)

    # Line plot with markers
    ax.plot(x, y, color="#3182ce", linewidth=2.0, marker="o", markersize=6, label="P(Fake)")

    # Color specific markers
    for xi, yi in zip(x, y):
        color = "#e53e3e" if yi >= threshold else "#38a169"
        ax.plot(xi, yi, marker="o", color=color, markersize=7)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.5, len(probabilities) + 0.5)
    ax.set_xticks(x)
    ax.set_xlabel("Sampled Video Frame (Uniform 16-frame grid)", fontsize=9, fontweight="bold")
    ax.set_ylabel("P(Fake)", fontsize=9, fontweight="bold")
    ax.set_title("Temporal Deepfake Probability Trajectory", fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def predict_video(
    video_path: str | None,
    n_sample_frames: int = 16,
    threshold: float = 0.5,
) -> tuple[str, dict[str, float] | None, Image.Image | None, Image.Image | None, Image.Image | None]:
    """
    Execute video inference pipeline:
        1. Extract 16 frames uniformly
        2. MTCNN detect + crop face on each frame
        3. Score each face
        4. Aggregate video score & generate per-frame temporal plot
        5. Return Grad-CAM for the highest-anomaly frame
    """
    if not video_path:
        return (
            "<div style='padding: 12px; background: #2d3748; color: #cbd5e0; border-radius: 8px;'>Please upload a video file (.mp4, .avi, .mov).</div>",
            None,
            None,
            None,
            None,
        )

    frames = extract_frames_from_video(video_path, n_frames=n_sample_frames)
    if not frames:
        error_html = (
            "<div style='padding: 16px; background: #fff5f5; border: 1px solid #feb2b2; border-radius: 8px; color: #c53030; font-weight: 600;'>"
            "⚠️ Video Error: Unable to extract video frames from the file.<br>"
            "<span style='font-size: 0.85em; font-weight: normal; color: #742a2a;'>"
            "Please ensure the file is a valid video format (MP4, AVI, WebM).</span></div>"
        )
        return error_html, None, None, None, None

    model = app_state.current_model
    if model is None:
        model, _ = app_state.load_active_model()

    valid_frame_indices: list[int] = []
    frame_probabilities: list[float] = []
    detected_faces: list[Image.Image] = []
    tensors: list[torch.Tensor] = []

    for idx, frame_pil in frames:
        face_crop = detect_and_crop_face(
            image=frame_pil,
            mtcnn=app_state.mtcnn,
            margin_pct=20,
            output_size=224,
        )
        if face_crop is not None:
            face_rgb = np.array(face_crop)
            t = app_state.val_transform(image=face_rgb)["image"].unsqueeze(0).to(app_state.device)
            with torch.no_grad():
                prob_fake = float(torch.sigmoid(model(t)).item())

            valid_frame_indices.append(idx)
            frame_probabilities.append(prob_fake)
            detected_faces.append(face_crop)
            tensors.append(t)

    # Graceful fallback: If zero faces detected across all frames
    if not frame_probabilities:
        error_html = (
            "<div style='padding: 16px; background: #fff5f5; border: 1px solid #feb2b2; border-radius: 8px; color: #c53030; font-weight: 600;'>"
            f"⚠️ No Faces Detected: MTCNN could not detect any faces across all {len(frames)} sampled video frames.<br>"
            "<span style='font-size: 0.85em; font-weight: normal; color: #742a2a;'>"
            "Please upload a video where the subject's face is clearly visible to the camera.</span></div>"
        )
        return error_html, None, None, None, None

    # Video-level aggregation: Mean of valid frames
    agg_fake_prob = float(np.mean(frame_probabilities))
    agg_real_prob = 1.0 - agg_fake_prob
    is_fake = agg_fake_prob >= threshold
    confidence = (agg_fake_prob if is_fake else agg_real_prob) * 100.0

    # Generate temporal trajectory plot
    plot_img = plot_video_scores(valid_frame_indices, frame_probabilities, threshold=threshold)

    # Find highest suspicious frame for Grad-CAM
    suspicious_idx = int(np.argmax(frame_probabilities))
    suspicious_face = detected_faces[suspicious_idx]
    suspicious_tensor = tensors[suspicious_idx]

    cam_mask = compute_gradcam(
        model=model,
        input_tensor=suspicious_tensor,
        target_category=1,
        device=app_state.device,
    )[0]
    cam_overlay_rgb = overlay_heatmap(np.array(suspicious_face), cam_mask, alpha=0.5, colormap=cv2.COLORMAP_JET)
    cam_overlay_pil = Image.fromarray(cam_overlay_rgb)

    # HTML Verdict
    if is_fake:
        verdict_color = "#e53e3e"
        bg_color = "#fff5f5"
        border_color = "#feb2b2"
        badge_text = "VIDEO VERDICT: DEEPFAKE DETECTED"
    else:
        verdict_color = "#38a169"
        bg_color = "#f0fff4"
        border_color = "#9ae6b4"
        badge_text = "VIDEO VERDICT: AUTHENTIC / REAL"

    verdict_html = (
        f"<div style='padding: 16px; background: {bg_color}; border: 2px solid {border_color}; border-radius: 10px;'>"
        f"<div style='font-size: 1.25rem; font-weight: 800; color: {verdict_color}; margin-bottom: 4px;'>{badge_text}</div>"
        f"<div style='font-size: 1rem; color: #2d3748;'>"
        f"Video Confidence: <b>{confidence:.1f}%</b> | "
        f"Mean P(Fake): <code style='background: #edf2f7; padding: 2px 6px; border-radius: 4px;'>{agg_fake_prob:.4f}</code> | "
        f"Analyzed: <b>{len(frame_probabilities)}/{len(frames)}</b> frames with detected faces"
        f"</div></div>"
    )

    label_dict = {"Real": agg_real_prob, "Fake": agg_fake_prob}
    return verdict_html, label_dict, plot_img, suspicious_face, cam_overlay_pil


# ═════════════════════════════════════════════════════════════════
#  Gradio UI Assembly
# ═════════════════════════════════════════════════════════════════

def build_gradio_app() -> gr.Blocks:
    """Build modern, publication-ready multi-tab Gradio app."""
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: auto !important; }
    .header-box { text-align: center; padding: 20px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: white; border-radius: 12px; margin-bottom: 20px; }
    .header-box p { font-size: 0.95rem; color: #94a3b8; margin: 0; }
    .tab-nav button { font-weight: 600; font-size: 1.05rem; }
    """

    gradio_major = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 4
    blocks_kwargs: dict[str, Any] = {"title": "Deepfake ViT & DINOv2 Detector"}
    if gradio_major < 6:
        blocks_kwargs["theme"] = theme
        blocks_kwargs["css"] = custom_css

    with gr.Blocks(**blocks_kwargs) as demo:
        # Header banner
        gr.HTML(
            """
            <div class="header-box">
                <h1>🛡️ Deepfake Detection with Vision Transformers</h1>
                <p>Comparative Study of Supervised (ViT-B, DeiT-S) and Self-Supervised (DINOv2) Pretraining with Grad-CAM & Attention Rollout Explainability</p>
            </div>
            """
        )

        # Model Status Bar
        with gr.Accordion("⚙️ Model Architecture & Inference Settings", open=False):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Backbone Architecture",
                    choices=[
                        "dinov2_vits14",
                        "deit_small_patch16_224",
                        "vit_base_patch16_224",
                        "efficientnet_b0",
                    ],
                    value="dinov2_vits14",
                    interactive=True,
                )
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Classification Threshold P(Fake)",
                )
                model_status_text = gr.Textbox(
                    label="Current Status",
                    value="Initializing...",
                    interactive=False,
                )

        # ── TAB 1: Single Image Inference ─────────────────────────
        with gr.Tab("🖼️ Image Detection"):
            with gr.Row():
                with gr.Column(scale=5):
                    image_input = gr.Image(
                        label="Upload Face Image",
                        type="pil",
                        sources=["upload", "clipboard", "webcam"],
                    )
                    image_analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

                with gr.Column(scale=7):
                    image_verdict_box = gr.HTML(label="Verdict")
                    image_label_output = gr.Label(label="Class Probability", num_top_classes=2)

                    with gr.Row():
                        image_crop_output = gr.Image(label="MTCNN Face Crop (224x224)", type="pil")
                        image_cam_output = gr.Image(label="Grad-CAM Heatmap Overlay", type="pil")
                        image_rollout_output = gr.Image(label="Attention Rollout (ViT)", type="pil")

            # Connect Image Events
            image_analyze_btn.click(
                fn=predict_single_image,
                inputs=[image_input, threshold_slider],
                outputs=[
                    image_verdict_box,
                    image_label_output,
                    image_crop_output,
                    image_cam_output,
                    image_rollout_output,
                ],
            )

        # ── TAB 2: Video Detection ────────────────────────────────
        with gr.Tab("🎥 Video Detection"):
            with gr.Row():
                with gr.Column(scale=5):
                    video_input = gr.Video(label="Upload Video File (.mp4, .avi, .mov)")
                    video_frames_slider = gr.Slider(
                        minimum=8,
                        maximum=32,
                        value=16,
                        step=4,
                        label="Sampled Frames (Uniform Temporal Grid)",
                    )
                    video_analyze_btn = gr.Button("⚡ Analyze Video Frames", variant="primary", size="lg")

                with gr.Column(scale=7):
                    video_verdict_box = gr.HTML(label="Video Verdict")
                    video_label_output = gr.Label(label="Aggregated Video Probability", num_top_classes=2)
                    video_plot_output = gr.Image(label="Per-Frame Probability Trajectory", type="pil")

                    with gr.Row():
                        video_suspicious_face = gr.Image(label="Highest-Risk Face Crop", type="pil")
                        video_suspicious_cam = gr.Image(label="Grad-CAM of Anomaly Frame", type="pil")

            # Connect Video Events
            video_analyze_btn.click(
                fn=predict_video,
                inputs=[video_input, video_frames_slider, threshold_slider],
                outputs=[
                    video_verdict_box,
                    video_label_output,
                    video_plot_output,
                    video_suspicious_face,
                    video_suspicious_cam,
                ],
            )

        # Dynamic Model Switching Event
        def on_model_change(selected_backbone: str) -> str:
            _, status = app_state.load_active_model(backbone=selected_backbone)
            return status

        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[model_status_text],
        )

        # Initial App Load Callback
        demo.load(
            fn=lambda: app_state.load_active_model()[1],
            outputs=[model_status_text],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Deepfake ViT Gradio Web App")
    parser.add_argument("--config", type=str, default="configs/app_config.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt).")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda).")
    parser.add_argument("--port", type=int, default=7860, help="Local port for server.")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link.")
    args = parser.parse_args()

    # Initialize app state
    app_state.device = torch.device(args.device)
    app_state.mtcnn = create_mtcnn(device=args.device)
    _, status = app_state.load_active_model(checkpoint_path=args.checkpoint)
    print(f"[app] {status}")

    demo = build_gradio_app()
    gradio_major = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 4
    launch_kwargs: dict[str, Any] = {
        "server_name": "0.0.0.0",
        "server_port": args.port,
        "share": args.share,
    }
    if gradio_major >= 6:
        theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate")
        launch_kwargs["theme"] = theme
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
