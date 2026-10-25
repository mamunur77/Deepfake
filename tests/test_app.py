"""
tests/test_app.py  ·  Unit tests for Gradio Web Application.
============================================================
Validates:
1. AppState initializes detector and loads active model on CPU.
2. predict_single_image handles 'no face detected' gracefully.
3. predict_single_image produces valid predictions, crops, and Grad-CAM overlays for faces.
4. plot_video_scores returns a valid PIL Image chart.
5. build_gradio_app builds a complete Gradio Blocks application.
"""

from __future__ import annotations

import unittest
import numpy as np
from PIL import Image
import torch

from app.app import app_state, build_gradio_app, plot_video_scores, predict_single_image


class TestGradioApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app_state.device = torch.device("cpu")
        model, status = app_state.load_active_model()
        cls.model = model
        cls.status = status

    def test_model_loaded_on_cpu(self) -> None:
        """Verify model is properly initialized on CPU."""
        self.assertIsNotNone(app_state.current_model)
        self.assertEqual(app_state.device.type, "cpu")

    def test_predict_single_image_no_face(self) -> None:
        """Verify graceful handling when no human face is in the image."""
        blank_canvas = Image.new("RGB", (224, 224), color=(255, 255, 255))
        verdict, labels, crop, cam, rollout = predict_single_image(blank_canvas)

        self.assertIn("No Face Detected", verdict)
        self.assertIsNone(labels)
        self.assertIsNone(crop)
        self.assertIsNone(cam)
        self.assertIsNone(rollout)

    def test_predict_single_image_none_input(self) -> None:
        """Verify graceful handling when input is None."""
        verdict, labels, crop, cam, rollout = predict_single_image(None)
        self.assertIn("Please upload an image", verdict)
        self.assertIsNone(labels)

    def test_predict_single_image_with_face(self) -> None:
        """Verify end-to-end inference and Grad-CAM heatmap on a face crop."""
        test_face_path = "data/processed/real/face_test_real/frame_0000.jpg"
        face_img = Image.open(test_face_path)

        verdict, labels, crop, cam, rollout = predict_single_image(face_img)

        self.assertIsNotNone(labels)
        self.assertIn("Real", labels)
        self.assertIn("Fake", labels)
        self.assertIsNotNone(crop)
        self.assertEqual(crop.size, (224, 224))
        self.assertIsNotNone(cam)
        self.assertIsInstance(cam, Image.Image)

    def test_plot_video_scores(self) -> None:
        """Verify video temporal trajectory plot generation."""
        indices = list(range(16))
        probs = [0.1 * (i % 10) for i in range(16)]

        chart_img = plot_video_scores(indices, probs, threshold=0.5)
        self.assertIsInstance(chart_img, Image.Image)
        self.assertGreater(chart_img.width, 100)
        self.assertGreater(chart_img.height, 100)

    def test_build_gradio_app(self) -> None:
        """Verify Gradio app structure."""
        demo = build_gradio_app()
        self.assertIsNotNone(demo)


if __name__ == "__main__":
    unittest.main()
