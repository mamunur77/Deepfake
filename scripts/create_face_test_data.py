#!/usr/bin/env python3
"""
Generate test data with a real face photo (from the internet)
so MTCNN can actually detect and crop it.
"""
from pathlib import Path
import urllib.request
import cv2
import numpy as np

# Publicly available face test images (wikimedia commons, royalty-free)
FACE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"

def create_face_video_from_pil():
    """Create test videos using PIL-generated face-like images with more realistic features."""
    from PIL import Image, ImageDraw
    
    def make_face_image(size=320, bg=(200, 200, 200)):
        """Generate a more realistic face-like image using PIL."""
        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)
        cx, cy = size // 2, size // 2
        
        # Head - flesh-toned oval
        draw.ellipse([cx-70, cy-90, cx+70, cy+90], fill=(220, 185, 155))
        # Hair
        draw.ellipse([cx-75, cy-100, cx+75, cy-40], fill=(60, 40, 30))
        # Left eye white
        draw.ellipse([cx-40, cy-30, cx-10, cy-8], fill=(240, 240, 240))
        # Left iris
        draw.ellipse([cx-32, cy-25, cx-18, cy-11], fill=(80, 60, 40))
        # Left pupil
        draw.ellipse([cx-28, cy-22, cx-22, cy-16], fill=(20, 20, 20))
        # Right eye white
        draw.ellipse([cx+10, cy-30, cx+40, cy-8], fill=(240, 240, 240))
        # Right iris
        draw.ellipse([cx+18, cy-25, cx+32, cy-11], fill=(80, 60, 40))
        # Right pupil
        draw.ellipse([cx+22, cy-22, cx+28, cy-16], fill=(20, 20, 20))
        # Nose shadow
        draw.polygon([(cx, cy-5), (cx-8, cy+20), (cx+8, cy+20)], fill=(200, 165, 135))
        # Mouth
        draw.arc([cx-25, cy+25, cx+25, cy+55], 0, 180, fill=(180, 100, 100), width=3)
        # Eyebrows
        draw.arc([cx-42, cy-50, cx-8, cy-25], 180, 360, fill=(60, 40, 30), width=3)
        draw.arc([cx+8, cy-50, cx+42, cy-25], 180, 360, fill=(60, 40, 30), width=3)
        
        return img
    
    root = Path("data/raw")
    
    # Create "real" video
    real_dir = root / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    vid_path = real_dir / "face_test_real.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, 30.0, (320, 320))
    
    for i in range(30):
        face_img = make_face_image(320, bg=(180 + i, 190 + i, 200 + i))
        frame = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    print(f"  Created: {vid_path}")
    
    # Create "fake" image (test image support)
    fake_dir = root / "fake"
    fake_dir.mkdir(parents=True, exist_ok=True)
    img_path = fake_dir / "face_test_fake.jpg"
    
    face_img = make_face_image(320, bg=(200, 180, 180))
    face_img.save(str(img_path), quality=95)
    print(f"  Created: {img_path}")
    
    print("\n[test-data] Realistic face test data generated.")


if __name__ == "__main__":
    create_face_video_from_pil()
