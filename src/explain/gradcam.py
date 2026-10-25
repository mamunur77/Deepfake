"""
src.explain.gradcam  ·  Grad-CAM visualisations for ViT models.
===============================================================
Responsibilities:
    - Wrap pytorch-grad-cam around any ViT backbone.
    - Generate heatmap overlay on input face crops.
    - Save visualisations to outputs/ or return for Gradio display.

TODO (Step 6):
    - Implement explain_image(model, image, target_layer, cfg) → np.ndarray.
"""

__all__: list[str] = []
