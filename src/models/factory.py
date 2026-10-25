"""
src.models.factory  ·  Model construction from config.
=====================================================
Responsibilities:
    - Build a ViT backbone from timm (supervised ImageNet weights).
    - Build a ViT backbone from HuggingFace (DINO / MAE self-supervised).
    - Replace the classifier head for binary deepfake detection.
    - Return model + target_layer name (for Grad-CAM).

TODO (Step 3):
    - Implement build_model(cfg) → (nn.Module, str).
"""

__all__: list[str] = []
