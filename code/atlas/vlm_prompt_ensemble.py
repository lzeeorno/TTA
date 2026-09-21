from __future__ import annotations

import os
import sys
from typing import Iterable, Sequence

import torch

TTA_VLM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "baselines", "tta-vlm-main"))
if TTA_VLM_DIR not in sys.path:
    sys.path.insert(0, TTA_VLM_DIR)

from clip import tokenize
from data.imagnet_prompts import imagenet_templates


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(getattr(model, "device", "cpu"))


def build_prompt_ensemble_text_features(
    model,
    classnames: Sequence[str],
    templates: Iterable[str] | None = None,
    chunk_size: int = 256,
) -> torch.Tensor:
    if not hasattr(model, "token_embedding"):
        raise RuntimeError("Prompt ensemble source requires model.token_embedding.")

    templates = list(templates or imagenet_templates)
    if not templates:
        raise RuntimeError("Prompt ensemble source requires at least one template.")

    device = _model_device(model)
    dtype = model.dtype
    normalized_classnames = [name.replace("_", " ") for name in classnames]
    feature_sum = None

    for template in templates:
        chunks = []
        for start in range(0, len(normalized_classnames), chunk_size):
            names = normalized_classnames[start : start + chunk_size]
            prompts = [template.format(name) for name in names]
            tokenized = torch.cat([tokenize(prompt) for prompt in prompts]).to(device)
            with torch.no_grad():
                prompt_embeddings = model.token_embedding(tokenized).type(dtype)
                text_features = model.text_encoder(prompt_embeddings, tokenized)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            chunks.append(text_features)
        template_features = torch.cat(chunks, dim=0)
        feature_sum = template_features if feature_sum is None else feature_sum + template_features

    text_features = feature_sum / len(templates)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.detach()


def source_ensemble_logits(model, images: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    image_features = model.image_encoder(images.type(model.dtype))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return model.logit_scale.exp() * image_features @ text_features.t()
