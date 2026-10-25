"""
tests/test_explain.py  ·  Unit tests for Grad-CAM & Attention Rollout.
====================================================================
Validates:
1. ViT Grad-CAM with vit_reshape_transform outputs normalized heatmap [0, 1].
2. ViT Attention Rollout outputs normalized attention mask [0, 1].
3. DINOv2 Grad-CAM with vit_reshape_transform.
4. DINOv2 Attention Rollout.
5. Heatmap overlay blending.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from src.explain.attention_rollout import compute_attention_rollout
from src.explain.gradcam import compute_gradcam, overlay_heatmap, vit_reshape_transform
from src.models.factory import build_model


class TestExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cpu")
        cls.dummy_input = torch.randn(1, 3, 224, 224, device=cls.device)

    def test_vit_gradcam_and_rollout(self) -> None:
        """Verify Grad-CAM and Attention Rollout on Vision Transformer (ViT/DeiT)."""
        model = build_model(backbone="deit_small_patch16_224", pretrained=False)
        model.to(self.device)
        model.eval()

        # 1. Grad-CAM
        cam_mask = compute_gradcam(model, self.dummy_input, target_category=1, device=self.device)
        self.assertEqual(cam_mask.shape, (1, 224, 224))
        self.assertGreaterEqual(float(cam_mask.min()), 0.0)
        self.assertLessEqual(float(cam_mask.max()), 1.0)

        # 2. Attention Rollout
        rollout_mask = compute_attention_rollout(model, self.dummy_input, device=self.device)
        self.assertEqual(rollout_mask.shape, (1, 224, 224))
        self.assertGreaterEqual(float(rollout_mask.min()), 0.0)
        self.assertLessEqual(float(rollout_mask.max()), 1.0)

    def test_dinov2_gradcam_and_rollout(self) -> None:
        """Verify Grad-CAM and Attention Rollout on DINOv2 backbone."""
        model = build_model(backbone="dinov2_vits14", pretrained=False)
        model.to(self.device)
        model.eval()

        # 1. Grad-CAM
        cam_mask = compute_gradcam(model, self.dummy_input, target_category=1, device=self.device)
        self.assertEqual(cam_mask.shape, (1, 224, 224))
        self.assertGreaterEqual(float(cam_mask.min()), 0.0)
        self.assertLessEqual(float(cam_mask.max()), 1.0)

        # 2. Attention Rollout
        rollout_mask = compute_attention_rollout(model, self.dummy_input, device=self.device)
        self.assertEqual(rollout_mask.shape, (1, 224, 224))
        self.assertGreaterEqual(float(rollout_mask.min()), 0.0)
        self.assertLessEqual(float(rollout_mask.max()), 1.0)

    def test_vit_base_gradcam(self) -> None:
        """Verify Grad-CAM on standard ViT-Base."""
        model = build_model(backbone="vit_base_patch16_224", pretrained=False)
        model.to(self.device)
        model.eval()

        cam_mask = compute_gradcam(model, self.dummy_input, target_category=1, device=self.device)
        self.assertEqual(cam_mask.shape, (1, 224, 224))
        self.assertGreaterEqual(float(cam_mask.min()), 0.0)
        self.assertLessEqual(float(cam_mask.max()), 1.0)

    def test_cnn_gradcam(self) -> None:
        """Verify Grad-CAM on CNN baseline (EfficientNet-B0)."""
        model = build_model(backbone="efficientnet_b0", pretrained=False)
        model.to(self.device)
        model.eval()

        cam_mask = compute_gradcam(model, self.dummy_input, target_category=1, device=self.device)
        self.assertEqual(cam_mask.shape, (1, 224, 224))
        self.assertGreaterEqual(float(cam_mask.min()), 0.0)
        self.assertLessEqual(float(cam_mask.max()), 1.0)

    def test_overlay_heatmap(self) -> None:
        """Verify overlay blending produces valid uint8 RGB image."""
        img = np.full((224, 224, 3), 120, dtype=np.uint8)
        mask = np.random.rand(224, 224).astype(np.float32)

        blended = overlay_heatmap(img, mask, alpha=0.5)
        self.assertEqual(blended.shape, (224, 224, 3))
        self.assertEqual(blended.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
