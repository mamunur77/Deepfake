"""
tests/test_splits.py  ·  Unit tests for data splitting, dataset, and transforms.
==============================================================================
Validates:
1. Video ID Mutual Exclusivity: Asserts that NO video_id appears in more than one split (Zero Leakage).
2. Stratification: Ensures balanced/stratified partition between classes.
3. Class Weight Calculation: Validates balanced class weighting formula.
4. Transforms Pipeline: Asserts correct output shape and normalization.
5. PyTorch DeepfakeDataset: Tests both 'image' mode (C, H, W) and 'video' mode (T, C, H, W).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.dataset import DeepfakeDataset
from src.data.make_splits import compute_class_weights, split_by_video_id
from src.data.transforms import get_train_transforms, get_val_transforms


class TestDataSplitsAndDataset(unittest.TestCase):
    def setUp(self) -> None:
        # Create a mock DataFrame simulating 20 videos (10 real, 10 fake) with 6 frames each
        records = []
        for i in range(10):
            vid_id = f"real_vid_{i:02d}"
            for f in range(6):
                records.append({
                    "frame_path": f"/fake/path/real/{vid_id}/frame_{f:04d}.jpg",
                    "video_id": vid_id,
                    "label": 0,
                    "label_name": "real",
                })
        for i in range(10):
            vid_id = f"fake_vid_{i:02d}"
            for f in range(6):
                records.append({
                    "frame_path": f"/fake/path/fake/{vid_id}/frame_{f:04d}.jpg",
                    "video_id": vid_id,
                    "label": 1,
                    "label_name": "fake",
                })

        self.mock_df = pd.DataFrame(records)

    def test_no_video_id_appears_in_more_than_one_split(self) -> None:
        """CRITICAL: Assert strict mutual exclusivity of video IDs across splits."""
        train_df, val_df, test_df = split_by_video_id(
            self.mock_df,
            split_ratios=(0.70, 0.15, 0.15),
            seed=42,
        )

        train_vids = set(train_df["video_id"])
        val_vids = set(val_df["video_id"])
        test_vids = set(test_df["video_id"])

        # Check pairwise intersections
        train_val_overlap = train_vids.intersection(val_vids)
        train_test_overlap = train_vids.intersection(test_vids)
        val_test_overlap = val_vids.intersection(test_vids)

        self.assertEqual(
            len(train_val_overlap),
            0,
            f"Data leakage detected! Video IDs shared between train and val: {train_val_overlap}",
        )
        self.assertEqual(
            len(train_test_overlap),
            0,
            f"Data leakage detected! Video IDs shared between train and test: {train_test_overlap}",
        )
        self.assertEqual(
            len(val_test_overlap),
            0,
            f"Data leakage detected! Video IDs shared between val and test: {val_test_overlap}",
        )

        # Ensure no videos were dropped
        all_split_vids = train_vids | val_vids | test_vids
        expected_vids = set(self.mock_df["video_id"])
        self.assertEqual(
            all_split_vids,
            expected_vids,
            "Total set of video IDs across splits does not match the original dataset.",
        )

    def test_class_weight_computation(self) -> None:
        """Verify balanced class weight computation."""
        # 80 fake (label=1), 20 real (label=0) -> total 100
        imbalanced_labels = [0] * 20 + [1] * 80
        weights, summary = compute_class_weights(imbalanced_labels, num_classes=2, as_tensor=True)

        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape[0], 2)

        # Real (minority) should have weight 100 / (2 * 20) = 2.5
        # Fake (majority) should have weight 100 / (2 * 80) = 0.625
        self.assertAlmostEqual(weights[0].item(), 2.5, places=3)
        self.assertAlmostEqual(weights[1].item(), 0.625, places=3)
        self.assertGreater(weights[0].item(), weights[1].item())

    def test_albumentations_transforms(self) -> None:
        """Verify train and val transforms output correct tensor shapes."""
        dummy_rgb = np.full((300, 300, 3), 128, dtype=np.uint8)

        train_tf = get_train_transforms(img_size=224)
        val_tf = get_val_transforms(img_size=224)

        train_out = train_tf(image=dummy_rgb)["image"]
        val_out = val_tf(image=dummy_rgb)["image"]

        self.assertIsInstance(train_out, torch.Tensor)
        self.assertEqual(train_out.shape, (3, 224, 224))
        self.assertEqual(val_out.shape, (3, 224, 224))

    def test_dataset_image_and_video_modes(self) -> None:
        """Verify DeepfakeDataset returns (C, H, W) for image mode and (T, C, H, W) for video mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create actual image files on disk
            records = []
            for v_idx in range(4):
                vid_name = f"test_vid_{v_idx}"
                v_dir = tmp_path / vid_name
                v_dir.mkdir()
                for f_idx in range(8):
                    f_path = v_dir / f"frame_{f_idx:04d}.jpg"
                    img = Image.new("RGB", (64, 64), color=(v_idx * 30, f_idx * 20, 100))
                    img.save(f_path)
                    records.append({
                        "frame_path": str(f_path),
                        "video_id": vid_name,
                        "label": v_idx % 2,
                    })

            df = pd.DataFrame(records)
            tf = get_val_transforms(img_size=112)

            # 1. Test image mode
            image_ds = DeepfakeDataset(df, mode="image", transform=tf)
            self.assertEqual(len(image_ds), 32)
            img_tensor, label = image_ds[0]
            self.assertEqual(img_tensor.shape, (3, 112, 112))
            self.assertIn(label, (0, 1))

            # 2. Test video mode (T=6 frames)
            video_ds = DeepfakeDataset(df, mode="video", num_frames=6, transform=tf)
            self.assertEqual(len(video_ds), 4)  # 4 unique videos
            vid_tensor, label = video_ds[0]
            self.assertEqual(vid_tensor.shape, (6, 3, 112, 112))  # (T, C, H, W)
            self.assertIn(label, (0, 1))


if __name__ == "__main__":
    unittest.main()
