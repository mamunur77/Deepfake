"""
tests/test_video_model.py  ·  Unit tests for temporal head & video classifier.
==============================================================================
Validates:
1. TemporalHead forward pass: (B, T, D) -> (B, 1).
2. VideoClassifier end-to-end forward pass: (B, T, C, H, W) -> (B, 1).
3. CachedEmbeddingDataset loading: returns (T, D) tensor and label.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
from pathlib import Path

from src.models.video_model import TemporalHead, VideoClassifier, build_video_model
from src.data.dataset import CachedEmbeddingDataset


class TestVideoModel(unittest.TestCase):
    def test_temporal_head_forward(self) -> None:
        """Verify TemporalHead processes (B, T, D) embeddings to single logit (B, 1)."""
        B, T, D = 2, 8, 384
        head = TemporalHead(embed_dim=D, num_heads=6, num_layers=2, dropout=0.0)
        head.eval()

        dummy_emb = torch.randn(B, T, D)
        with torch.no_grad():
            out = head(dummy_emb)

        self.assertEqual(out.shape, (B, 1))
        self.assertFalse(torch.isnan(out).any())

    def test_video_classifier_end_to_end(self) -> None:
        """Verify VideoClassifier processes (B, T, C, H, W) to single logit (B, 1)."""
        B, T = 2, 4
        model = VideoClassifier(
            backbone_name="deit_small_patch16_224",
            pretrained=False,
            freeze_backbone=True,
            num_heads=6,
            num_layers=2,
            dropout=0.0,
        )
        model.eval()

        dummy_video = torch.randn(B, T, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy_video)

        self.assertEqual(out.shape, (B, 1))
        self.assertFalse(torch.isnan(out).any())

    def test_cached_embedding_dataset(self) -> None:
        """Verify CachedEmbeddingDataset loads .npy and samples/pads to (num_frames, D)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create synthetic .npy files with 5 frames and D=128
            records = []
            for i in range(4):
                npy_file = tmp_path / f"vid_{i}.npy"
                data = np.random.randn(5, 128).astype(np.float32)
                np.save(npy_file, data)
                records.append({
                    "embedding_path": str(npy_file),
                    "video_id": f"vid_{i}",
                    "label": i % 2,
                })

            df = pd.DataFrame(records)
            ds = CachedEmbeddingDataset(df, num_frames=10)
            self.assertEqual(len(ds), 4)

            # Check padded tensor shape: (T=10, D=128)
            tensor, label = ds[0]
            self.assertEqual(tensor.shape, (10, 128))
            self.assertIn(label, (0, 1))


if __name__ == "__main__":
    unittest.main()
