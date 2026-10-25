#!/usr/bin/env python3
"""
scripts/create_test_data.py  ·  Generate tiny sample videos for testing.
========================================================================
Creates 2 short videos (30 frames each, ~1 second at 30 fps):
    data/raw/real/sample_real_001.mp4
    data/raw/fake/sample_fake_001.mp4

Each video contains a simple face-like oval drawing so that MTCNN has
*something* to try detecting (it may or may not succeed — either
outcome exercises a valid code path).

Usage:
    python scripts/create_test_data.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def draw_synthetic_face(frame: np.ndarray, jitter: int = 0) -> np.ndarray:
    """
    Draw a crude face (oval + eyes + mouth) on *frame* so MTCNN has
    a chance of detecting it.  *jitter* shifts features slightly
    between frames to simulate motion.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2 + jitter, h // 2 + jitter

    # Head oval (skin-ish colour)
    cv2.ellipse(frame, (cx, cy), (80, 100), 0, 0, 360, (200, 170, 140), -1)
    # Left eye
    cv2.circle(frame, (cx - 30, cy - 25), 12, (255, 255, 255), -1)
    cv2.circle(frame, (cx - 30, cy - 25), 6, (50, 40, 30), -1)
    # Right eye
    cv2.circle(frame, (cx + 30, cy - 25), 12, (255, 255, 255), -1)
    cv2.circle(frame, (cx + 30, cy - 25), 6, (50, 40, 30), -1)
    # Nose
    cv2.line(frame, (cx, cy - 10), (cx - 5, cy + 10), (170, 140, 110), 2)
    # Mouth
    cv2.ellipse(frame, (cx, cy + 35), (25, 10), 0, 0, 180, (150, 80, 80), 2)

    return frame


def generate_video(
    output_path: Path,
    n_frames: int = 30,
    size: tuple[int, int] = (320, 320),
    bg_colour: tuple[int, int, int] = (100, 120, 140),
) -> None:
    """Write a tiny MP4 with a synthetic face."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, size)

    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), bg_colour, dtype=np.uint8)
        jitter = int(5 * np.sin(i * 0.3))       # slight sway
        draw_synthetic_face(frame, jitter=jitter)
        writer.write(frame)

    writer.release()
    print(f"  Created: {output_path}  ({n_frames} frames)")


def main() -> None:
    root = Path("data/raw")
    print("[test-data] Generating sample videos ...\n")

    generate_video(
        root / "real" / "sample_real_001.mp4",
        n_frames=30, bg_colour=(110, 130, 150),
    )
    generate_video(
        root / "fake" / "sample_fake_001.mp4",
        n_frames=30, bg_colour=(140, 100, 100),
    )

    print(f"\n[test-data] Done.  Files written under {root}/")


if __name__ == "__main__":
    main()
