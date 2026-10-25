"""
src.models  ·  Model construction via timm & torch.hub.
"""

from .factory import (
    DINOV2_MODELS,
    SUPPORTED_BACKBONES,
    DINOv2Classifier,
    apply_training_mode,
    build_model,
    count_parameters,
    print_parameter_summary,
)
from .video_model import (
    TemporalHead,
    VideoClassifier,
    build_feature_extractor,
    build_video_model,
)

__all__ = [
    "DINOV2_MODELS",
    "SUPPORTED_BACKBONES",
    "DINOv2Classifier",
    "apply_training_mode",
    "build_model",
    "count_parameters",
    "print_parameter_summary",
    "TemporalHead",
    "VideoClassifier",
    "build_feature_extractor",
    "build_video_model",
]
