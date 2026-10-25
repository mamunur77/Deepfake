"""
tests/test_models.py  ·  Unit tests for model factory & backbones.
==================================================================
Validates:
1. Forward pass on CPU for each backbone on a random batch of size 2 (B=2, C=3, H=224, W=224).
   Asserts output shape is [2, 1] (single logit for real/fake).
   Backbones tested:
     - vit_base_patch16_224   (timm)
     - deit_small_patch16_224 (timm)
     - dinov2_vits14          (torch.hub facebookresearch/dinov2)
     - dinov2_vitb14          (torch.hub facebookresearch/dinov2)
     - efficientnet_b0        (timm CNN baseline)
2. Training modes ("frozen", "finetune", "partial"):
   Asserts parameter freeze/unfreeze behavior matches mode specifications.
"""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from src.models.factory import (
    SUPPORTED_BACKBONES,
    build_model,
    count_parameters,
    apply_training_mode,
)


class TestModelFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cpu")
        cls.dummy_input = torch.randn(2, 3, 224, 224, device=cls.device)

    def test_forward_pass_all_backbones(self) -> None:
        """Run forward pass for each backbone on a random batch of size 2 on CPU."""
        for backbone in sorted(SUPPORTED_BACKBONES):
            with self.subTest(backbone=backbone):
                # Build model without downloading full pretrained weights for fast CPU test
                model = build_model(
                    backbone=backbone,
                    pretrained=False,
                    mode="finetune",
                    drop_rate=0.0,
                )
                model.to(self.device)
                model.eval()

                with torch.no_grad():
                    logits = model(self.dummy_input)

                # Assert single logit output shape [B, 1]
                self.assertEqual(
                    logits.shape,
                    (2, 1),
                    f"Expected output shape (2, 1) for {backbone}, got {logits.shape}",
                )
                # Assert no NaNs or Infs
                self.assertFalse(torch.isnan(logits).any())
                self.assertFalse(torch.isinf(logits).any())

    def test_training_modes_freeze_finetune_partial(self) -> None:
        """Verify layer trainability across frozen, finetune, and partial modes."""
        backbone = "vit_base_patch16_224"

        # 1. Full Fine-tuning
        m_finetune = build_model(backbone=backbone, pretrained=False, mode="finetune")
        trainable_ft, total_ft = count_parameters(m_finetune)
        self.assertEqual(trainable_ft, total_ft, "In 'finetune' mode, all parameters must be trainable.")

        # 2. Frozen (Linear Probe)
        m_frozen = build_model(backbone=backbone, pretrained=False, mode="frozen")
        trainable_fr, total_fr = count_parameters(m_frozen)
        self.assertLess(trainable_fr, total_fr)
        # In ViT-B/16, head is Linear(768, 1) -> 768 weight + 1 bias = 769 parameters
        self.assertEqual(trainable_fr, 769, "In 'frozen' mode, only head parameters should be trainable.")

        # 3. Partial Fine-tuning (unfreeze last 2 blocks)
        m_partial = build_model(
            backbone=backbone,
            pretrained=False,
            mode="partial",
            unfreeze_blocks=2,
        )
        trainable_pt, total_pt = count_parameters(m_partial)
        # Trainable params should be strictly between frozen (769) and total (~86M)
        self.assertGreater(trainable_pt, trainable_fr)
        self.assertLess(trainable_pt, total_ft)

        # Verify initial blocks are frozen while trailing blocks are trainable
        self.assertFalse(any(p.requires_grad for p in m_partial.blocks[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in m_partial.blocks[-1].parameters()))


if __name__ == "__main__":
    unittest.main()
