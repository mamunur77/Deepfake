---
title: Deepfake ViT & DINOv2 Detector
emoji: 🛡️
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
---

# Deepfake Detection Web Application (Gradio)
## Vision Transformers (ViT, DeiT) & Self-Supervised DINOv2

This application provides real-time deepfake detection across both single images and full video clips, complete with visual explainability via Grad-CAM and Attention Rollout.

---

### 🌟 Key Features

1. **🖼️ Image Analysis Tab**:
   - Automatic MTCNN face detection with 20% margin.
   - Graceful handling when no human face is detected.
   - Binary Real/Fake classification with confidence percentages.
   - Side-by-side visual explanations:
     - **Grad-CAM**: Highlights regions of highest activation contributing to the fake prediction.
     - **Attention Rollout**: Traces $[CLS]$-to-patch multi-layer attention flow across transformer blocks.

2. **🎥 Video Analysis Tab**:
   - Uniform sampling of 16 frames across the video duration.
   - Per-frame MTCNN face extraction and model inference.
   - **Temporal Trajectory Plot**: High-contrast interactive graph showing frame-by-frame fake probabilities against the 0.5 decision boundary.
   - Aggregated video prediction score and confidence metrics.
   - Highest-anomaly frame inspection with Grad-CAM heatmap overlay.

3. **⚙️ Multi-Backbone Comparison**:
   - Dynamically switch between:
     - `DINOv2-S/14` (Self-supervised, ViT-Small patch 14)
     - `DeiT-S/16` (Supervised Distilled ViT-Small patch 16)
     - `ViT-B/16` (Supervised ImageNet-pretrained ViT-Base patch 16)
     - `EfficientNet-B0` (CNN baseline)
   - Operates on CPU by default with minimal memory footprint (`map_location="cpu"`).

---

### 🚀 Running Locally

```bash
# 1. From repository root, install dependencies:
pip install -r app/requirements.txt

# 2. Launch the Gradio web server:
python app/app.py

# 3. Specify custom checkpoint or port if desired:
python app/app.py --checkpoint outputs/dinov2_frozen/best_model.pt --port 7860
```
Open `http://localhost:7860` in your web browser.

---

### 🌐 Deploying to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Choose **Gradio** as the Space SDK.
3. Push the repository files to the Space git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push space main
   ```
4. The Space will automatically install dependencies from `requirements.txt` and launch `app/app.py`.
