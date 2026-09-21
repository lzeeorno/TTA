"""
Episodic VLM branch of ATLAS for Table 6 prompt-only adaptation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

import importlib.util
import os

import torch

DEM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "baselines", "DEM-main"))
_ADADEM_SPEC = importlib.util.spec_from_file_location(
    "_atlas_vlm_adadem", os.path.join(DEM_DIR, "adadem.py")
)
if _ADADEM_SPEC is None or _ADADEM_SPEC.loader is None:
    raise ImportError(f"Could not load VLM AdaDEM loss from {DEM_DIR}")
_ADADEM_MODULE = importlib.util.module_from_spec(_ADADEM_SPEC)
_ADADEM_SPEC.loader.exec_module(_ADADEM_MODULE)
AdaDEM = _ADADEM_MODULE.AdaDEM

from .vlm_prompt_ensemble import build_prompt_ensemble_text_features, source_ensemble_logits


def avg_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    avg_logits = log_probs.logsumexp(dim=0) - torch.log(
        torch.tensor(log_probs.shape[0], device=log_probs.device, dtype=log_probs.dtype)
    )
    avg_logits = torch.clamp(avg_logits, min=torch.finfo(avg_logits.dtype).min)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def select_confident_samples(logits: torch.Tensor, top: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    selected = max(1, int(batch_entropy.size(0) * float(top)))
    selected = min(selected, batch_entropy.size(0))
    idx = torch.argsort(batch_entropy, descending=False)[:selected]
    return logits[idx], idx


class ATLASInstance:
    """Episodic prompt adaptation used by Table 6 ATLAS rows.

    This adapter follows the same per-sample reset contract as TPT/AdaDEM in
    `instance_tta.py`, so prompt updates cannot drift across the benchmark
    stream. CoOp initialization is preserved as the source prompt state because
    `instance_tta.py` loads it before constructing this trainer.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimizer = None
        self.optim_state = None
        self.prompt_param_names = []
        self.source_prompt_state = {}
        self.adadem = None
        self.entropy_weight = 0.35
        self.anchor_weight = 0.02
        self.view_consistency_weight = 0.05
        self.source_output_mode = "auto"
        self.output_fusion = "confidence"
        self.adapted_output_weight = 1.0
        self.source_output_weight = 1.0
        self.view_output_weight = 0.5
        self.fusion_temperature = 2.0
        self.source_ensemble_text_features = None
        self.update_count = 0
        self.source_anchor_loss = 0.0
        self.view_consistency_loss = 0.0
        self.entropy_loss = 0.0
        self.loss_mode = "adadem_entropy_anchor_consistency"
        self.method_variant = "full"
        self.selection_quantile = 0.5

    def prepare_model_and_optimization(self, args) -> None:
        self.method_variant = str(getattr(args, "atlas_method_variant", "full")).lower()
        if self.method_variant not in {"core", "full"}:
            raise ValueError(
                "ATLAS VLM method_variant must be one of: core, full; "
                f"got {self.method_variant!r}"
            )
        self.selection_quantile = float(getattr(args, "atlas_selection_quantile", 0.5))
        if not 0.0 <= self.selection_quantile <= 1.0:
            raise ValueError("ATLAS selection quantile must be in [0, 1]")
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

        trainable = []
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name:
                param.requires_grad_(True)
                trainable.append(param)
                self.prompt_param_names.append(name)
                self.source_prompt_state[name] = param.detach().clone()

        if not trainable:
            raise RuntimeError("ATLAS instance could not find prompt learner parameters.")

        self.optimizer = torch.optim.AdamW(trainable, args.lr)
        self.optim_state = deepcopy(self.optimizer.state_dict())
        self.adadem = AdaDEM(pi=float(getattr(args, "atlas_adadem_pi", 0.1)), reduction="mean")
        self.entropy_weight = float(getattr(args, "atlas_entropy_weight", 0.35))
        self.anchor_weight = float(getattr(args, "atlas_source_anchor_weight", 0.02))
        self.view_consistency_weight = float(getattr(args, "atlas_view_consistency_weight", 0.05))
        self.source_output_mode = str(getattr(args, "atlas_source_output", "auto")).lower()
        if self.source_output_mode == "auto":
            self.source_output_mode = "prompt" if getattr(args, "load", None) else "ensemble"
        self.output_fusion = str(getattr(args, "atlas_output_fusion", "confidence")).lower()
        self.adapted_output_weight = float(getattr(args, "atlas_adapted_output_weight", 1.0))
        self.source_output_weight = float(getattr(args, "atlas_source_output_weight", 1.0))
        self.view_output_weight = float(getattr(args, "atlas_view_output_weight", 0.5))
        self.fusion_temperature = float(getattr(args, "atlas_fusion_temperature", 2.0))
        if self.method_variant == "core":
            self.entropy_weight = 0.0
            self.view_consistency_weight = 0.0
            self.output_fusion = "adapted"
            self.loss_mode = "adadem_anchor"

        if self.source_output_mode == "ensemble":
            self.source_ensemble_text_features = build_prompt_ensemble_text_features(
                self.model,
                self.model.classnames,
                chunk_size=int(getattr(args, "prompt_ensemble_chunk_size", 256)),
            )
        elif self.source_output_mode not in {"prompt", "none"}:
            raise ValueError(
                "atlas_source_output must be one of auto, ensemble, prompt, none; "
                f"got {self.source_output_mode!r}"
            )
        if self.output_fusion not in {"confidence", "mean", "adapted"}:
            raise ValueError(
                "atlas_output_fusion must be one of confidence, mean, adapted; "
                f"got {self.output_fusion!r}"
            )

    def pre_adaptation(self) -> None:
        self._restore_source_prompt()
        self.optimizer.load_state_dict(self.optim_state)
        if self.adadem is not None:
            self.adadem.reset()

    def adaptation_process(self, image, images, args) -> Dict[str, torch.Tensor]:
        if image is None:
            image = images[:1]
        if images.ndim != 4:
            raise ValueError("ATLAS instance expects a 4D tensor of prompt views.")

        selected_idx = None
        for _ in range(int(getattr(args, "tta_steps", 1))):
            logits = self.model(images)
            if selected_idx is not None:
                selected_logits = logits[selected_idx]
            else:
                selected_fraction = float(getattr(args, "selection_p", 0.1))
                if self.method_variant == "core":
                    selected_fraction = 1.0 - self.selection_quantile
                selected_logits, selected_idx = select_confident_samples(logits, selected_fraction)

            loss = self.adadem(selected_logits) if self.adadem is not None else selected_logits.new_zeros(())
            if self.entropy_weight > 0.0:
                entropy_loss = avg_entropy(selected_logits)
                loss = loss + self.entropy_weight * entropy_loss
                self.entropy_loss = float(entropy_loss.detach().item())

            if self.anchor_weight > 0.0:
                with torch.no_grad():
                    source_logits = self._forward_anchor_source_logits(images)[selected_idx]
                    source_probs = source_logits.softmax(dim=1)
                anchor_loss = -(source_probs * selected_logits.log_softmax(dim=1)).sum(dim=1).mean()
                loss = loss + self.anchor_weight * anchor_loss
                self.source_anchor_loss = float(anchor_loss.detach().item())

            if self.view_consistency_weight > 0.0 and logits.shape[0] > 1:
                original_probs = logits[:1].detach().softmax(dim=1)
                view_logits = logits[1:]
                consistency = -(original_probs.expand_as(view_logits) * view_logits.log_softmax(dim=1)).sum(dim=1).mean()
                loss = loss + self.view_consistency_weight * consistency
                self.view_consistency_loss = float(consistency.detach().item())

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.update_count += 1

        with torch.no_grad():
            adapted_logits = self.model(image)
            source_logits = self._forward_output_source_logits(image)
            view_logits = self._forward_view_consensus_logits(images)
            output = self._fuse_output(adapted_logits, source_logits, view_logits)

        return {"output": output.detach()}

    def get_stats(self) -> Dict:
        return {
            "atlas_runtime": "episodic",
            "atlas_method_variant": self.method_variant,
            "atlas_selection_quantile": self.selection_quantile,
            "atlas_loss_mode": self.loss_mode,
            "atlas_update_count": self.update_count,
            "atlas_entropy_weight": self.entropy_weight,
            "atlas_source_anchor_weight": self.anchor_weight,
            "atlas_view_consistency_weight": self.view_consistency_weight,
            "atlas_source_output": self.source_output_mode,
            "atlas_output_fusion": self.output_fusion,
            "atlas_adapted_output_weight": self.adapted_output_weight,
            "atlas_source_output_weight": self.source_output_weight,
            "atlas_view_output_weight": self.view_output_weight,
            "atlas_fusion_temperature": self.fusion_temperature,
            "atlas_entropy_loss": self.entropy_loss,
            "atlas_source_anchor_loss": self.source_anchor_loss,
            "atlas_view_consistency_loss": self.view_consistency_loss,
        }

    @staticmethod
    def fuse_logits_by_confidence(
        adapted_logits: torch.Tensor,
        source_logits: torch.Tensor | None = None,
        view_logits: torch.Tensor | None = None,
        *,
        adapted_weight: float = 1.0,
        source_weight: float = 1.0,
        view_weight: float = 0.5,
        temperature: float = 2.0,
    ) -> torch.Tensor:
        candidates = [
            (adapted_logits, max(0.0, float(adapted_weight))),
            (source_logits, max(0.0, float(source_weight))),
            (view_logits, max(0.0, float(view_weight))),
        ]
        probs_sum = None
        weights_sum = None
        for logits, base_weight in candidates:
            if logits is None or base_weight <= 0.0:
                continue
            probs = logits.softmax(dim=1)
            num_classes = max(logits.shape[1], 2)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
            confidence = (1.0 - entropy / torch.log(torch.tensor(num_classes, device=logits.device, dtype=logits.dtype))).clamp(0.0, 1.0)
            weight = base_weight * confidence.clamp_min(1e-4).pow(float(temperature))
            probs_sum = probs * weight if probs_sum is None else probs_sum + probs * weight
            weights_sum = weight if weights_sum is None else weights_sum + weight

        if probs_sum is None or weights_sum is None:
            return adapted_logits
        fused_probs = (probs_sum / weights_sum.clamp_min(1e-8)).clamp_min(1e-8)
        return fused_probs.log()

    def _restore_source_prompt(self) -> None:
        named_params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, source in self.source_prompt_state.items():
                named_params[name].copy_(source.to(device=named_params[name].device, dtype=named_params[name].dtype))

        prompt_learner = getattr(self.model, "prompt_learner", None)
        if prompt_learner is not None and hasattr(prompt_learner, "ctx_init_state") and hasattr(prompt_learner, "ctx"):
            prompt_learner.ctx_init_state = prompt_learner.ctx.detach().clone()

    def _forward_source_logits(self, images: torch.Tensor) -> torch.Tensor:
        named_params = dict(self.model.named_parameters())
        backups = {}
        try:
            for name, source in self.source_prompt_state.items():
                param = named_params[name]
                backups[name] = param.detach().clone()
                param.data.copy_(source.to(device=param.device, dtype=param.dtype))
            return self.model(images)
        finally:
            for name, backup in backups.items():
                named_params[name].data.copy_(backup)

    def _forward_anchor_source_logits(self, images: torch.Tensor) -> torch.Tensor:
        if self.source_output_mode == "ensemble" and self.source_ensemble_text_features is not None:
            return source_ensemble_logits(self.model, images, self.source_ensemble_text_features)
        return self._forward_source_logits(images)

    def _forward_output_source_logits(self, images: torch.Tensor) -> torch.Tensor | None:
        if self.source_output_mode == "none":
            return None
        if self.source_output_mode == "ensemble":
            return source_ensemble_logits(self.model, images, self.source_ensemble_text_features)
        return self._forward_source_logits(images)

    def _forward_view_consensus_logits(self, images: torch.Tensor) -> torch.Tensor | None:
        if images.ndim != 4 or images.shape[0] <= 1:
            return None
        logits = self.model(images)
        probs = logits.softmax(dim=1).mean(dim=0, keepdim=True).clamp_min(1e-8)
        return probs.log()

    def _fuse_output(
        self,
        adapted_logits: torch.Tensor,
        source_logits: torch.Tensor | None,
        view_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.output_fusion == "adapted":
            return adapted_logits
        if self.output_fusion == "mean":
            candidates = [
                (adapted_logits, max(0.0, self.adapted_output_weight)),
                (source_logits, max(0.0, self.source_output_weight)),
                (view_logits, max(0.0, self.view_output_weight)),
            ]
            probs_sum = None
            total_weight = 0.0
            for logits, weight in candidates:
                if logits is None or weight <= 0.0:
                    continue
                probs_sum = logits.softmax(dim=1) * weight if probs_sum is None else probs_sum + logits.softmax(dim=1) * weight
                total_weight += weight
            if probs_sum is None or total_weight <= 0.0:
                return adapted_logits
            return (probs_sum / total_weight).clamp_min(1e-8).log()
        return self.fuse_logits_by_confidence(
            adapted_logits=adapted_logits,
            source_logits=source_logits,
            view_logits=view_logits,
            adapted_weight=self.adapted_output_weight,
            source_weight=self.source_output_weight,
            view_weight=self.view_output_weight,
            temperature=self.fusion_temperature,
        )
