"""
src.explain  ·  Visual explainability: ViT Grad-CAM & Attention Rollout.
"""

from .attention_rollout import compute_attention_rollout
from .explain import (
    explain_single_image,
    generate_explanation_grid,
    run_explain_pipeline,
)
from .gradcam import (
    compute_gradcam,
    get_gradcam_target_layer,
    overlay_heatmap,
    vit_reshape_transform,
)

__all__ = [
    "compute_gradcam",
    "get_gradcam_target_layer",
    "overlay_heatmap",
    "vit_reshape_transform",
    "compute_attention_rollout",
    "explain_single_image",
    "generate_explanation_grid",
    "run_explain_pipeline",
]
