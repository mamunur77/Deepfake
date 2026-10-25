#!/usr/bin/env python3
"""
Create 4 small synthetic sample videos (2 real, 2 fake) for end-to-end testing.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

def make_face_image(size=224, bg=(200, 200, 200), mouth_offset=0):
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    # Head
    draw.ellipse([cx-50, cy-70, cx+50, cy+70], fill=(225, 190, 160))
    # Eyes
    draw.ellipse([cx-30, cy-20, cx-10, cy-5], fill=(255, 255, 255))
    draw.ellipse([cx-24, cy-17, cx-16, cy-9], fill=(50, 40, 30))
    draw.ellipse([cx+10, cy-20, cx+30, cy-5], fill=(255, 255, 255))
    draw.ellipse([cx+16, cy-17, cx+24, cy-9], fill=(50, 40, 30))
    # Mouth
    draw.arc([cx-20, cy+20+mouth_offset, cx+20, cy+40+mouth_offset], 0, 180, fill=(180, 80, 80), width=3)
    return img

def create_video(path: Path, n_frames=10, bg=(180, 190, 200), mouth_delta=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (224, 224))
    for i in range(n_frames):
        img = make_face_image(size=224, bg=bg, mouth_offset=(i % 3) * mouth_delta)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    print(f"Created: {path}")

def main():
    root = Path("data/raw")
    create_video(root / "real" / "sample_real_01.mp4", bg=(170, 190, 210), mouth_delta=0)
    create_video(root / "real" / "sample_real_02.mp4", bg=(180, 200, 220), mouth_delta=1)
    create_video(root / "fake" / "sample_fake_01.mp4", bg=(210, 180, 180), mouth_delta=2)
    create_video(root / "fake" / "sample_fake_02.mp4", bg=(220, 170, 170), mouth_delta=3)
    print("Done creating 4 test videos.")

if __name__ == "__main__":
    main()
