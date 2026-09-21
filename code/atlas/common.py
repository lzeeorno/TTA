"""
Core utilities for ATLAS TTA family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantile_selection_mask(
    scores: torch.Tensor,
    quantile: float = 0.5,
    *,
    higher_is_better: bool = True,
) -> torch.Tensor:
    """Select entities relative to the current unlabeled score distribution."""
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("selection quantile must be in [0, 1]")
    flat = scores.detach().reshape(-1)
    if flat.numel() == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    threshold_q = float(quantile) if higher_is_better else 1.0 - float(quantile)
    threshold = torch.quantile(flat.float(), threshold_q).to(scores)
    return scores >= threshold if higher_is_better else scores <= threshold


def destroy_sensitivity(raw_support: torch.Tensor, destroy_support: torch.Tensor) -> torch.Tensor:
    """Positive confidence drop under a structure-breaking view (larger is stronger support)."""
    return (raw_support - destroy_support).clamp(min=0.0, max=1.0)


def insufficient_destroy_penalty(
    sensitivity: torch.Tensor,
    minimum_drop: float = 0.2,
) -> torch.Tensor:
    """Risk incurred when a destroy view fails to reduce target-class support."""
    return (float(minimum_drop) - sensitivity).clamp_min(0.0)


def selected_entity_mean(
    losses: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean over selected entities; weights scale terms but never the denominator."""
    if losses.numel() == 0:
        raise ValueError("selected_entity_mean requires at least one selected entity")
    if weights is not None:
        losses = losses * weights.detach().to(losses)
    return losses.sum() / losses.numel()


def class_centered_anchor_loss(
    logits: torch.Tensor,
    class_center: torch.Tensor,
) -> torch.Tensor:
    """Scale-normalized direction whose descent step moves predictions to the center."""
    probs = logits.softmax(dim=1)
    center = class_center.detach()
    with torch.no_grad():
        entropy_term = -(probs * logits).sum(1, keepdim=True)
        gradient_scale = ((logits + entropy_term + 1.0) * probs).abs().sum(
            1, keepdim=True
        ).clamp_min(1e-6)
    centered_direction = (probs - center) / gradient_scale.detach()
    return (centered_direction * logits).sum(dim=1)


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    return -(probs * logits.log_softmax(dim=1)).sum(dim=1)


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    num_classes = max(logits.shape[1], 2)
    return (softmax_entropy(logits) / math.log(num_classes)).clamp_(0.0, 1.0)


@torch.jit.script
def consistency_cross_entropy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-view consistency regularization between two prediction tensors."""
    return -(x.softmax(1) * y.log_softmax(1)).sum(1).mean()


def asymmetric_clip_ratio(
    ratio: torch.Tensor,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
) -> torch.Tensor:
    return ratio.clamp(1.0 - eps_low, 1.0 + eps_high)


def arc_surrogate(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
    enabled: bool = True,
) -> torch.Tensor:
    if not enabled:
        return ratio * advantage
    clipped = asymmetric_clip_ratio(ratio, eps_low=eps_low, eps_high=eps_high)
    return torch.minimum(ratio * advantage, clipped * advantage)


def compute_sos_penalty(
    shift_score: torch.Tensor,
    safe_margin: float = 0.35,
    hard_margin: float = 0.75,
) -> torch.Tensor:
    if hard_margin <= safe_margin:
        raise ValueError("hard_margin must be greater than safe_margin")

    penalty = torch.zeros_like(shift_score)
    soft_region = (shift_score > safe_margin) & (shift_score <= hard_margin)
    penalty[soft_region] = (
        (shift_score[soft_region] - safe_margin)
        / (hard_margin - safe_margin)
    )
    penalty[shift_score > hard_margin] = 1.0
    return penalty.clamp_(0.0, 1.0)


def sos_weight_from_score(
    shift_score: torch.Tensor,
    safe_margin: float = 0.35,
    hard_margin: float = 0.75,
) -> torch.Tensor:
    return 1.0 - compute_sos_penalty(
        shift_score,
        safe_margin=safe_margin,
        hard_margin=hard_margin,
    )


def build_photometric_view(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        return x
    mean = x.mean(dim=(2, 3), keepdim=True)
    view = 0.85 * x + 0.15 * mean
    view = view + 0.01 * torch.randn_like(view)
    return view.clamp_(0.0, 1.0)


def build_structural_view(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    if x.ndim != 4:
        return x
    _, _, h, w = x.shape
    target_h = max(8, int(h * ratio))
    target_w = max(8, int(w * ratio))
    if target_h >= h or target_w >= w:
        return x
    down = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)


def build_destroy_view(x: torch.Tensor, patch: Optional[int] = None) -> torch.Tensor:
    if x.ndim != 4:
        return x
    b, c, h, w = x.shape
    patch = patch or (4 if min(h, w) <= 64 else 16)
    patch = max(1, min(patch, min(h, w)))
    usable_h = (h // patch) * patch
    usable_w = (w // patch) * patch
    if usable_h == 0 or usable_w == 0:
        return x

    crop = x[:, :, :usable_h, :usable_w]
    num_h = usable_h // patch
    num_w = usable_w // patch
    patches = crop.view(b, c, num_h, patch, num_w, patch)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
        b, num_h * num_w, c, patch, patch
    )
    perm = torch.stack(
        [torch.randperm(num_h * num_w, device=x.device) for _ in range(b)],
        dim=0,
    )
    shuffled = patches[torch.arange(b, device=x.device).unsqueeze(1), perm]
    out = shuffled.reshape(b, num_h, num_w, c, patch, patch)
    out = out.permute(0, 3, 1, 4, 2, 5).reshape(b, c, usable_h, usable_w)
    if usable_h != h or usable_w != w:
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    return out


def build_vit_plpd_view(x: torch.Tensor, patch_len: int = 4) -> torch.Tensor:
    """Build a DeYO-style patch-shuffle view for ViT PLPD scoring.

    `patch_len` follows the DeYO convention: the image is split into a
    `patch_len x patch_len` grid, and the grid cells are shuffled.
    For 224x224 inputs and `patch_len=4`, this yields 56x56 shuffled regions.
    """
    if x.ndim != 4:
        return x

    b, c, h, w = x.shape
    patch_len = max(1, int(patch_len))
    usable_h = (h // patch_len) * patch_len
    usable_w = (w // patch_len) * patch_len
    if usable_h == 0 or usable_w == 0:
        return x

    if usable_h != h or usable_w != w:
        crop = F.interpolate(
            x,
            size=(usable_h, usable_w),
            mode="bilinear",
            align_corners=False,
        )
    else:
        crop = x

    patch_h = usable_h // patch_len
    patch_w = usable_w // patch_len
    patches = crop.view(b, c, patch_len, patch_h, patch_len, patch_w)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
        b, patch_len * patch_len, c, patch_h, patch_w
    )
    perm = torch.stack(
        [torch.randperm(patch_len * patch_len, device=x.device) for _ in range(b)],
        dim=0,
    )
    shuffled = patches[torch.arange(b, device=x.device).unsqueeze(1), perm]
    out = shuffled.reshape(b, patch_len, patch_len, c, patch_h, patch_w)
    out = out.permute(0, 3, 1, 4, 2, 5).reshape(b, c, usable_h, usable_w)
    if usable_h != h or usable_w != w:
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    return out


def weighted_majority_vote(
    predictions: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    num_classes: int,
) -> torch.Tensor:
    if len(predictions) != len(weights):
        raise ValueError("predictions and weights must have the same length")
    votes = torch.zeros(
        predictions[0].shape[0],
        num_classes,
        device=predictions[0].device,
        dtype=weights[0].dtype,
    )
    for pred, weight in zip(predictions, weights):
        votes.scatter_add_(1, pred.unsqueeze(1), weight.unsqueeze(1))
    return votes.argmax(dim=1)


def merge_metric(
    previous: float,
    value: float,
    momentum: float = 0.9,
) -> float:
    if math.isnan(previous):
        return value
    return momentum * previous + (1.0 - momentum) * value


def configure_model(model: nn.Module, adapt_type: str = "bn") -> nn.Module:
    model.train()
    model.requires_grad_(False)
    adapt_type = adapt_type.lower()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if adapt_type in {"bn", "all", "bn+ln"}:
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
        elif isinstance(module, nn.GroupNorm):
            if adapt_type in {"gn", "all", "gn+ln"}:
                module.requires_grad_(True)
        elif isinstance(module, nn.LayerNorm):
            if adapt_type in {"ln", "all", "gn+ln", "bn+ln"}:
                module.requires_grad_(True)
    return model


def collect_norm_params(model: nn.Module, adapt_type: str = "bn") -> Tuple[List[nn.Parameter], List[str]]:
    adapt_type = adapt_type.lower()
    if adapt_type == "bn":
        norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    elif adapt_type == "gn":
        norm_types = (nn.GroupNorm,)
    elif adapt_type == "ln":
        norm_types = (nn.LayerNorm,)
    elif adapt_type == "gn+ln":
        norm_types = (nn.GroupNorm, nn.LayerNorm)
    elif adapt_type == "bn+ln":
        norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)
    else:
        norm_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.LayerNorm,
        )

    params: List[nn.Parameter] = []
    names: List[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, norm_types):
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in {"weight", "bias"} and param.requires_grad:
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")
    return params, names


def looks_like_vit_family(model: nn.Module) -> bool:
    child_names = {name for name, _ in model.named_children()}

    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type in {"vit", "swin", "beit", "segformer", "mix_transformer"}:
        return True

    if "blocks" in child_names or hasattr(model, "patch_embed"):
        return True
    if hasattr(model, "cls_token") or hasattr(model, "pos_embed"):
        return True

    if "segformer" in child_names or hasattr(model, "segformer"):
        return True
    if "decode_head" in child_names and hasattr(model, "segformer"):
        return True

    segformer_core = getattr(model, "segformer", None)
    if segformer_core is not None:
        segformer_children = {name for name, _ in segformer_core.named_children()}
        if "encoder" in segformer_children:
            return True

    return False


def looks_like_vit(model: nn.Module) -> bool:
    return looks_like_vit_family(model)


def infer_vit_token_count(model: nn.Module) -> int:
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "num_patches"):
        num_patches = int(getattr(patch_embed, "num_patches"))
        has_cls = 1 if hasattr(model, "cls_token") else 0
        return num_patches + has_cls
    return 197


def active_unit_count(
    batch_size: int,
    mode: str,
    token_count: int = 1,
    spatial_shape: Optional[Tuple[int, int]] = None,
    prompt_views: int = 1,
) -> int:
    if mode == "token":
        return max(1, batch_size * token_count)
    if mode == "pixel":
        if spatial_shape is None:
            raise ValueError("spatial_shape is required for pixel mode")
        return max(1, batch_size * spatial_shape[0] * spatial_shape[1])
    if mode == "prompt_view":
        return max(1, batch_size * prompt_views * token_count)
    return max(1, batch_size)


@dataclass
class DIGEntry:
    tensor: torch.Tensor


class DIGBuffer:
    def __init__(self, target_size: int = 8):
        self.target_size = max(1, int(target_size))
        self.entries: List[DIGEntry] = []

    def reset(self) -> None:
        self.entries.clear()

    def add(self, tensors: Iterable[torch.Tensor]) -> int:
        available = self.target_size - len(self.entries)
        if available <= 0:
            return 0

        added = 0
        for tensor in tensors:
            if available <= 0:
                break
            self.entries.append(DIGEntry(tensor=tensor.detach().cpu()))
            added += 1
            available -= 1
        return added

    def ready(self) -> bool:
        return len(self.entries) >= self.target_size

    def pop_ready_batch(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self.ready():
            return None
        batch_entries = self.entries[: self.target_size]
        self.entries = self.entries[self.target_size :]
        batch = torch.stack([entry.tensor for entry in batch_entries], dim=0)
        return batch.to(device, non_blocking=True)

    def __len__(self) -> int:
        return len(self.entries)
