"""
TRIAD: TRi-space Invariant ADaptation
Main package with three orthogonal innovations

Three Innovations (Tri-space):
  C1: D³LR — Dynamic Depth-Decay Learning Rate (optimization space)
  C2: SAOD — Source-Anchored Output Distillation (function space)
  C3: AIS  — Augmentation-Invariant Selection (sample selection space)

Engineering stabilizers (not claimed as contributions):
    - EATA-style entropy filtering + entropy weighting
    - EATA-style anti-redundancy filtering

Supports:
- BatchNorm adaptation (ResNet, WideResNet, etc.)
- GroupNorm adaptation (ResNet-GN for TTA)
- LayerNorm adaptation (ViT, Swin, DeiT, etc.)
- Hybrid BN+LN adaptation
"""

from .triad import (
    TRIAD,
    DELTA,
    create_triad,
    create_delta,
    configure_model,
    collect_all_norm_params,
    softmax_entropy,
    entropy_filtered_loss,
    update_model_probs,
    update_ema,
    SAMOptimizer,
)
from .triad_f1 import TRIAD as TRIAD_F1, create_triad as create_triad_f1
from .triad_f2 import TRIAD as TRIAD_F2, create_triad as create_triad_f2
from .triad_f3 import TRIAD as TRIAD_F3, create_triad as create_triad_f3
from .triad_f4 import TRIAD as TRIAD_F4, create_triad as create_triad_f4

__version__ = "4.0.0"
__all__ = [
    # Main class
    "TRIAD",
    "DELTA",
    "create_triad",
    "create_delta",
    "TRIAD_F1",
    "TRIAD_F2",
    "TRIAD_F3",
    "TRIAD_F4",
    "create_triad_f1",
    "create_triad_f2",
    "create_triad_f3",
    "create_triad_f4",
    # Utilities
    "softmax_entropy",
    "entropy_filtered_loss",
    "SAMOptimizer",
    "collect_all_norm_params",
    "update_model_probs",
    "configure_model",
]
