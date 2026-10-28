"""
scripts.setup_demo_samples  ·  Download real and AI-generated face images for the Gradio demo gallery
and verify predictions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import urllib.request
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.face_extract import create_mtcnn, detect_and_crop_face
from src.data.transforms import get_val_transforms
from src.models.factory import build_model
from src.utils.config import load_config


def main() -> None:
    demo_dir = "app/demo_samples"
    os.makedirs(demo_dir, exist_ok=True)

    # 1. Download 2 AI-generated faces from ThisPersonDoesNotExist
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for i in range(1, 3):
        fake_path = os.path.join(demo_dir, f"ai_generated_face_{i:02d}.jpg")
        if not os.path.exists(fake_path):
            print(f"Downloading AI face sample {i}...")
            req = urllib.request.Request("https://thispersondoesnotexist.com", headers=headers)
            with urllib.request.urlopen(req) as resp, open(fake_path, "wb") as f:
                f.write(resp.read())

    # 2. Copy Real camera portraits from local processed data
    import shutil
    real_src1 = "data/processed/real/face_test_real/frame_0000.jpg"
    real_src2 = "data/processed/real/face_test_real/frame_0004.jpg"
    shutil.copyfile(real_src1, os.path.join(demo_dir, "real_camera_portrait_01.jpg"))
    shutil.copyfile(real_src2, os.path.join(demo_dir, "real_camera_portrait_02.jpg"))

    print("\nVerifying samples with DINOv2 model:")
    mtcnn = create_mtcnn("cpu")
    val_tf = get_val_transforms(224)

    cfg = load_config("configs/dinov2_partial.yaml")
    model = build_model(cfg=cfg, backbone="dinov2_vitb14", pretrained=False)
    ckpt = torch.load("outputs/dinov2_partial/best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()

    for fname in sorted(os.listdir(demo_dir)):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        fpath = os.path.join(demo_dir, fname)
        img = Image.open(fpath).convert("RGB")
        crop = detect_and_crop_face(img, mtcnn, margin_pct=20, output_size=224)
        if crop is None:
            print(f"  {fname}: No face detected")
            continue
        tensor = val_tf(image=np.array(crop))["image"].unsqueeze(0)
        with torch.no_grad():
            prob_fake = float(torch.sigmoid(model(tensor)).item())
        verdict = "FAKE" if prob_fake >= 0.5 else "REAL"
        conf = prob_fake if verdict == "FAKE" else (1.0 - prob_fake)
        print(f"  {fname:30s} -> {verdict:4s} (Confidence: {conf*100:.1f}%, P_fake: {prob_fake:.4f})")


if __name__ == "__main__":
    main()
