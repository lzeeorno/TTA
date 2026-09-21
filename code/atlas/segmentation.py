"""
Segmentation branch of ATLAS TTA family.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

import math

import torch
import torch.nn.functional as F

from .common import (
    active_unit_count,
    arc_surrogate,
    class_centered_anchor_loss,
    destroy_sensitivity,
    insufficient_destroy_penalty,
    build_photometric_view,
    build_structural_view,
    build_vit_plpd_view,
    collect_norm_params,
    configure_model,
    looks_like_vit_family,
    merge_metric,
    normalized_entropy,
    quantile_selection_mask,
    selected_entity_mean,
    softmax_entropy,
    sos_weight_from_score,
)


class ATLASSegmentationAdapter:
    def __init__(
        self,
        model,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        arc_config: Optional[Dict] = None,
        dig_config: Optional[Dict] = None,
        uan_config: Optional[Dict] = None,
        sos_config: Optional[Dict] = None,
        vit_ln_config: Optional[Dict] = None,
        no_arc: bool = False,
        no_dig: bool = False,
        no_uan: bool = False,
        no_sos: bool = False,
        sane_normalizer: str = "selected",
        optimizer_name: str = "sgd",
        optim_momentum: float = 0.9,
        method_variant: str = "full",
    ):
        self.method_variant = str(method_variant).lower()
        if self.method_variant not in {"core", "full"}:
            raise ValueError(
                "ATLAS segmentation method_variant must be one of: core, full; "
                f"got {method_variant!r}"
            )
        self.use_branch_refinements = self.method_variant == "full"
        if sane_normalizer not in {"selected", "total"}:
            raise ValueError(f"Unsupported sane_normalizer: {sane_normalizer}")
        self.arc_enabled = not no_arc
        self.dig_enabled = not no_dig
        self.uan_enabled = not no_uan
        self.sos_enabled = not no_sos
        self.sane_normalizer = sane_normalizer

        self.arc_config = {"eps_low": 0.2, "eps_high": 0.28}
        if arc_config:
            self.arc_config.update(arc_config)
        self.dig_config = {"std_threshold": 0.03}
        if dig_config:
            self.dig_config.update(dig_config)
        self.uan_mode = (uan_config or {}).get("mode", "pixel") if self.uan_enabled else "sample"
        self.sos_config = {"safe_margin": 0.30, "hard_margin": 0.70}
        if sos_config:
            self.sos_config.update(sos_config)
        self.vit_ln_config = {
            "optimizer": optimizer_name,
            "momentum": optim_momentum,
            "pi": 0.1,
            "entropy_margin_scale": 0.4,
            "plpd_threshold": 0.2,
            "patch_len": 4,
            "probe_policy": "always",
            "shared_role_only": False,
            "target_selected_ratio": 0.15,
            "source_conf_threshold": 0.0,
            "source_conf_tolerance": 0.15,
            "source_anchor_weight": 0.03,
            "predict_after_update": True,
            "output_hflip": True,
            "output_hflip_weight": 1.0,
            "output_scale": 2.0,
            "output_scale_weight": 1.0,
            "adapt_decode_bn": False,
            "selection_quantile": 0.5,
        }
        if vit_ln_config:
            self.vit_ln_config.update(vit_ln_config)
        if not self.use_branch_refinements:
            self.vit_ln_config.update(
                {
                    "shared_role_only": True,
                    "output_hflip": False,
                    "output_scale_weight": 0.0,
                    "adapt_decode_bn": False,
                }
            )
        self.ln_probe_policy = str(self.vit_ln_config.get("probe_policy", "always")).lower()
        if self.ln_probe_policy not in {"always", "alternate", "off"}:
            raise ValueError(
                "vit_ln.probe_policy must be one of: always, alternate, off"
            )
        self.ln_shared_role_only = bool(
            self.vit_ln_config.get("shared_role_only", False)
        )
        self.optimizer_name = str(self.vit_ln_config.get("optimizer", optimizer_name)).lower()
        if self.optimizer_name not in {"sgd", "adam"}:
            raise ValueError(f"Unsupported segmentation ATLAS optimizer: {self.optimizer_name}")
        self.optim_momentum = float(self.vit_ln_config.get("momentum", optim_momentum))
        self.num_classes = int(getattr(getattr(model, "config", None), "num_labels", 19))
        self.vit_ln_family_mode = looks_like_vit_family(model)

        self.model = configure_model(model, adapt_type="ln")
        self.ln_adapt_decode_bn = (
            bool(self.vit_ln_config.get("adapt_decode_bn", True))
            if self.vit_ln_family_mode else False
        )
        if self.ln_adapt_decode_bn:
            for module_name, module in self.model.named_modules():
                if "decode_head" not in module_name:
                    continue
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
                    if module.weight is not None:
                        module.weight.requires_grad_(True)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)
        self.source_model = deepcopy(self.model)
        self.source_model.eval()
        for param in self.source_model.parameters():
            param.requires_grad_(False)

        self.params, _ = collect_norm_params(self.model, adapt_type="ln")
        if self.ln_adapt_decode_bn:
            seen_params = {id(param) for param in self.params}
            for module_name, module in self.model.named_modules():
                if "decode_head" not in module_name:
                    continue
                if not isinstance(module, torch.nn.BatchNorm2d):
                    continue
                for param in (module.weight, module.bias):
                    if param is not None and param.requires_grad and id(param) not in seen_params:
                        self.params.append(param)
                        seen_params.add(id(param))
        if not self.params:
            raise RuntimeError("ATLAS segmentation found no LayerNorm affine parameters.")
        if self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.params,
                lr=lr,
                momentum=self.optim_momentum,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(self.params, lr=lr, weight_decay=weight_decay)

        self.vit_ctta_core = "seg_source_safe_adadem_plpd" if self.vit_ln_family_mode else "none"
        self.ln_pi = float(self.vit_ln_config["pi"]) if self.vit_ln_family_mode else 0.0
        self.ln_entropy_margin = (
            float(self.vit_ln_config["entropy_margin_scale"]) * math.log(max(self.num_classes, 2))
            if self.vit_ln_family_mode else 0.0
        )
        self.ln_plpd_threshold = float(self.vit_ln_config["plpd_threshold"]) if self.vit_ln_family_mode else 0.0
        self.ln_patch_len = int(self.vit_ln_config["patch_len"]) if self.vit_ln_family_mode else 0
        self.ln_target_selected_ratio = (
            float(self.vit_ln_config["target_selected_ratio"]) if self.vit_ln_family_mode else 1.0
        )
        self.ln_source_conf_threshold = (
            float(self.vit_ln_config["source_conf_threshold"]) if self.vit_ln_family_mode else 0.0
        )
        self.selection_quantile = float(self.vit_ln_config["selection_quantile"])
        if not 0.0 <= self.selection_quantile <= 1.0:
            raise ValueError("ATLAS selection quantile must be in [0, 1]")
        self.ln_source_conf_tolerance = (
            float(self.vit_ln_config["source_conf_tolerance"]) if self.vit_ln_family_mode else 0.0
        )
        self.ln_source_anchor_weight = (
            float(self.vit_ln_config["source_anchor_weight"]) if self.vit_ln_family_mode else 0.0
        )
        if self.ln_shared_role_only:
            self.ln_source_anchor_weight = 0.0
        self.ln_predict_after_update = (
            bool(self.vit_ln_config.get("predict_after_update", True))
            if self.vit_ln_family_mode else False
        )
        self.ln_output_hflip = (
            bool(self.vit_ln_config.get("output_hflip", True))
            if self.vit_ln_family_mode else False
        )
        self.ln_output_hflip_weight = (
            float(self.vit_ln_config.get("output_hflip_weight", 1.0))
            if self.vit_ln_family_mode else 0.0
        )
        self.ln_output_scale = (
            float(self.vit_ln_config.get("output_scale", 2.0))
            if self.vit_ln_family_mode else 1.0
        )
        self.ln_output_scale_weight = (
            float(self.vit_ln_config.get("output_scale_weight", 1.0))
            if self.vit_ln_family_mode else 0.0
        )
        if self.ln_shared_role_only:
            self.ln_output_hflip = False
            self.ln_output_scale_weight = 0.0
        self.ln_class_template = None
        self.ln_template_initialized = False

        self.step_count = 0
        self.update_count = 0
        self.skip_count = 0
        self.dig_kept_groups = 0
        self.dig_buffer_fills = 0
        self.source_anchor_agreement = float("nan")
        self.group_reward_std = float("nan")
        self.sos_mean_penalty = float("nan")
        self.last_ln_entropy_keep_ratio = 0.0
        self.last_ln_plpd_keep_ratio = 0.0
        self.last_ln_selected_ratio = 0.0
        self.last_ln_plpd_mean = 0.0
        self.last_ln_adadem_loss = 0.0
        self.last_ln_source_anchor_loss = 0.0
        self.last_ln_loss_normalizer = 0.0
        self.last_ln_selected_count = 0
        self.last_ln_total_units = 0
        self.last_ln_source_agreement = 0.0
        self.last_ln_source_safe_ratio = 0.0
        self.last_ln_probe_drop = 0.0
        self.last_ln_source_confidence = 0.0
        self.last_ln_hflip_agreement = 0.0
        self.last_ln_scale_agreement = 0.0
        self._ln_stats_steps = 0
        self._ln_selected_ratio_sum = 0.0
        self._ln_source_agreement_sum = 0.0
        self._ln_probe_drop_sum = 0.0
        self.ln_probe_forward_count = 0
        self.ln_probe_skipped_count = 0
        self.last_ln_probe_used = False

    def _should_run_ln_probe(self) -> bool:
        if self.ln_probe_policy == "always":
            run_probe = True
        elif self.ln_probe_policy == "off":
            run_probe = False
        else:
            run_probe = (self.step_count % 2) == 0
        if run_probe:
            self.ln_probe_forward_count += 1
        else:
            self.ln_probe_skipped_count += 1
        self.last_ln_probe_used = run_probe
        return run_probe

    def _forward_logits(self, pixel_values: torch.Tensor, require_grad: bool) -> torch.Tensor:
        if require_grad:
            return self.model(pixel_values=pixel_values).logits
        with torch.no_grad():
            return self.model(pixel_values=pixel_values).logits

    def _forward_source_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.source_model(pixel_values=pixel_values).logits

    def _apply_ln_output_views(
        self,
        pixel_values: torch.Tensor,
        base_logits: torch.Tensor,
        input_size,
    ) -> torch.Tensor:
        use_hflip = self.ln_output_hflip and self.ln_output_hflip_weight > 0.0
        use_scale = (
            self.ln_output_scale_weight > 0.0
            and self.ln_output_scale > 0.0
            and abs(self.ln_output_scale - 1.0) > 1e-6
        )
        if not use_hflip and not use_scale:
            self.last_ln_hflip_agreement = 0.0
            self.last_ln_scale_agreement = 0.0
            return base_logits.detach()

        with torch.no_grad():
            base_probs = base_logits.detach().softmax(dim=1)
            base_pred = base_probs.argmax(dim=1)
            weighted_probs = base_probs
            weight_sum = 1.0

            if use_hflip:
                hflip_values = torch.flip(pixel_values, dims=[-1])
                hflip_logits = self._forward_logits(hflip_values, require_grad=False)
                hflip_logits = F.interpolate(
                    hflip_logits,
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                )
                hflip_logits = torch.flip(hflip_logits, dims=[-1])
                hflip_probs = hflip_logits.softmax(dim=1)
                hflip_pred = hflip_probs.argmax(dim=1)
                self.last_ln_hflip_agreement = float((base_pred == hflip_pred).float().mean().item())
                hflip_weight = max(float(self.ln_output_hflip_weight), 0.0)
                weighted_probs = weighted_probs + hflip_weight * hflip_probs
                weight_sum += hflip_weight
            else:
                self.last_ln_hflip_agreement = 0.0

            if use_scale:
                scaled_values = F.interpolate(
                    pixel_values,
                    scale_factor=float(self.ln_output_scale),
                    mode="bilinear",
                    align_corners=False,
                )
                scale_logits = self._forward_logits(scaled_values, require_grad=False)
                scale_logits = F.interpolate(
                    scale_logits,
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                )
                scale_probs = scale_logits.softmax(dim=1)
                scale_pred = scale_probs.argmax(dim=1)
                self.last_ln_scale_agreement = float((base_pred == scale_pred).float().mean().item())
                scale_weight = max(float(self.ln_output_scale_weight), 0.0)
                weighted_probs = weighted_probs + scale_weight * scale_probs
                weight_sum += scale_weight
            else:
                self.last_ln_scale_agreement = 0.0

            fused_probs = weighted_probs / max(weight_sum, 1e-6)
            return fused_probs.clamp_min(1e-8).log()

    def _ensure_ln_state(self, device: torch.device, dtype: torch.dtype) -> None:
        if not self.vit_ln_family_mode:
            return
        if (
            self.ln_class_template is None
            or self.ln_class_template.device != device
            or self.ln_class_template.dtype != dtype
            or self.ln_class_template.shape != (self.num_classes, self.num_classes)
        ):
            self.ln_class_template = torch.full(
                (self.num_classes, self.num_classes),
                1.0 / float(self.num_classes),
                device=device,
                dtype=dtype,
            )

    def _update_ln_class_template(
        self,
        probs: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        force_init: bool = False,
    ) -> None:
        if not self.vit_ln_family_mode or probs.numel() == 0:
            return
        self._ensure_ln_state(probs.device, probs.dtype)
        momentum = 0.0 if force_init or not self.ln_template_initialized else 1.0 - float(self.ln_pi)
        detached_probs = probs.detach()
        detached_weights = None
        if sample_weights is not None:
            detached_weights = sample_weights.detach().to(device=probs.device, dtype=probs.dtype).view(-1)
        pseudo_labels = detached_probs.argmax(dim=1)
        for label in pseudo_labels.unique():
            label_mask = pseudo_labels == label
            label_probs = detached_probs[label_mask]
            if detached_weights is None:
                label_mean = label_probs.mean(dim=0)
            else:
                label_weights = detached_weights[label_mask]
                label_weights = label_weights / label_weights.sum().clamp_min(1e-6)
                label_mean = (label_probs * label_weights.unsqueeze(1)).sum(dim=0)
            self.ln_class_template[label] = (
                momentum * self.ln_class_template[label]
                + (1.0 - momentum) * label_mean
            )
        self.ln_template_initialized = True

    def _compute_ln_adadem_loss(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        self._ensure_ln_state(logits.device, logits.dtype)
        pseudo_labels = probs.argmax(dim=1)
        template = self.ln_class_template[pseudo_labels].detach()
        return class_centered_anchor_loss(logits, template)

    def _cap_ln_mask_by_score(
        self,
        candidate_mask: torch.Tensor,
        score: torch.Tensor,
        total_units: int,
    ) -> torch.Tensor:
        selected_count = int(candidate_mask.sum().item())
        if selected_count == 0:
            return candidate_mask
        target_count = int(round(float(total_units) * self.ln_target_selected_ratio))
        target_count = max(1, min(selected_count, target_count))
        if selected_count <= target_count:
            return candidate_mask

        flat_mask = candidate_mask.reshape(-1)
        flat_score = score.reshape(-1)
        candidate_indices = flat_mask.nonzero(as_tuple=False).view(-1)
        candidate_scores = flat_score[candidate_indices]
        keep_relative = torch.topk(candidate_scores, k=target_count, largest=True).indices
        keep_indices = candidate_indices[keep_relative]
        flat_final = torch.zeros_like(flat_mask)
        flat_final[keep_indices] = True
        return flat_final.view_as(candidate_mask)

    def _clear_ln_step_stats(
        self,
        *,
        entropy_keep_ratio: float,
        plpd_keep_ratio: float,
        selected_ratio: float,
        plpd_mean: float,
        source_safe_ratio: float,
        total_units: int,
    ) -> None:
        self.last_ln_entropy_keep_ratio = entropy_keep_ratio
        self.last_ln_plpd_keep_ratio = plpd_keep_ratio
        self.last_ln_selected_ratio = selected_ratio
        self.last_ln_plpd_mean = plpd_mean
        self.last_ln_adadem_loss = 0.0
        self.last_ln_source_anchor_loss = 0.0
        self.last_ln_loss_normalizer = 0.0
        self.last_ln_selected_count = 0
        self.last_ln_total_units = total_units
        self.last_ln_source_agreement = 0.0
        self.last_ln_source_safe_ratio = source_safe_ratio
        self.last_ln_probe_drop = 0.0
        self.last_ln_source_confidence = 0.0

    def _record_ln_step_stats(
        self,
        *,
        selected_ratio: float,
        source_agreement: float,
        probe_drop: float,
    ) -> None:
        self._ln_stats_steps += 1
        self._ln_selected_ratio_sum += selected_ratio
        self._ln_source_agreement_sum += source_agreement
        self._ln_probe_drop_sum += probe_drop

    def _adapt_batch_vit_ln(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        raw_logits = self._forward_logits(pixel_values, require_grad=True)
        input_size = pixel_values.shape[-2:]
        raw_logits = F.interpolate(raw_logits, size=input_size, mode="bilinear", align_corners=False)
        raw_probs = raw_logits.softmax(dim=1)
        raw_entropy = softmax_entropy(raw_logits)
        entropy_mask = raw_entropy < self.ln_entropy_margin
        if self.ln_shared_role_only:
            entropy_mask = torch.ones_like(raw_entropy, dtype=torch.bool)
        entropy_keep_ratio = float(entropy_mask.float().mean().item())
        total_units = int(raw_logits.shape[0] * raw_logits.shape[2] * raw_logits.shape[3])

        if int(entropy_mask.sum().item()) == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.step_count += 1
            self.skip_count += 1
            self._clear_ln_step_stats(
                entropy_keep_ratio=entropy_keep_ratio,
                plpd_keep_ratio=0.0,
                selected_ratio=0.0,
                plpd_mean=0.0,
                source_safe_ratio=0.0,
                total_units=total_units,
            )
            return self._apply_ln_output_views(pixel_values, raw_logits, input_size)

        with torch.no_grad():
            source_logits = self._forward_source_logits(pixel_values)
            source_logits = F.interpolate(source_logits, size=input_size, mode="bilinear", align_corners=False)
            source_probs = source_logits.softmax(dim=1)
            probe_used = self._should_run_ln_probe()
            if probe_used:
                plpd_logits = self._forward_logits(
                    build_vit_plpd_view(pixel_values, patch_len=self.ln_patch_len),
                    require_grad=False,
                )
                plpd_logits = F.interpolate(plpd_logits, size=input_size, mode="bilinear", align_corners=False)
                plpd_probs = plpd_logits.softmax(dim=1)

        top1 = raw_probs.argmax(dim=1, keepdim=True)
        source_top1 = source_probs.argmax(dim=1, keepdim=True)
        raw_conf = raw_probs.gather(1, top1).squeeze(1)
        source_conf = source_probs.gather(1, top1).squeeze(1)
        source_top_conf = source_probs.gather(1, source_top1).squeeze(1)
        source_agreement_mask = top1.squeeze(1) == source_top1.squeeze(1)
        source_safe_mask = source_agreement_mask
        if self.ln_shared_role_only:
            source_safe_mask = torch.ones_like(source_agreement_mask, dtype=torch.bool)
        else:
            if self.ln_source_conf_threshold > 0.0:
                source_safe_mask = source_safe_mask & (source_top_conf >= self.ln_source_conf_threshold)
            if self.ln_source_conf_tolerance >= 0.0:
                source_safe_mask = source_safe_mask & (
                    raw_conf >= source_top_conf - self.ln_source_conf_tolerance
                )
        source_safe_ratio = float(source_safe_mask.float().mean().item())

        if probe_used:
            plpd_scores = raw_probs.gather(1, top1).squeeze(1) - plpd_probs.gather(1, top1).squeeze(1)
        else:
            plpd_scores = raw_entropy.new_zeros(raw_entropy.shape)
        entropy_plpd_mask = (
            plpd_scores[entropy_mask] > self.ln_plpd_threshold
            if probe_used else torch.ones_like(plpd_scores[entropy_mask], dtype=torch.bool)
        )
        plpd_keep_ratio = (
            float(entropy_plpd_mask.float().mean().item())
            if entropy_plpd_mask.numel() > 0 else 0.0
        )
        plpd_mean = (
            float(plpd_scores[entropy_mask].mean().item())
            if int(entropy_mask.sum().item()) > 0 else 0.0
        )

        candidate_mask = entropy_mask & source_safe_mask
        if probe_used:
            candidate_mask = candidate_mask & (plpd_scores > self.ln_plpd_threshold)
        entropy_confidence = 1.0 - (raw_entropy / math.log(max(self.num_classes, 2))).clamp(0.0, 1.0)
        selection_score = (
            0.35 * raw_conf.detach()
            + 0.35 * source_conf.detach()
            + 0.20 * (plpd_scores.detach() - self.ln_plpd_threshold).clamp_min(0.0)
            + 0.10 * entropy_confidence.detach()
        )
        if self.ln_shared_role_only:
            evidence = [raw_conf.detach(), entropy_confidence.detach()]
            if probe_used:
                evidence.append(plpd_scores.detach().clamp(0.0, 1.0))
            selection_score = torch.stack(evidence, dim=-1).mean(dim=-1)
            candidate_mask = torch.ones_like(selection_score, dtype=torch.bool)
        elif not probe_used:
            selection_score = (
                0.45 * raw_conf.detach()
                + 0.40 * source_conf.detach()
                + 0.15 * entropy_confidence.detach()
            )
        if self.ln_shared_role_only:
            final_mask = quantile_selection_mask(
                selection_score, self.selection_quantile
            )
        else:
            final_mask = self._cap_ln_mask_by_score(
                candidate_mask=candidate_mask,
                score=selection_score,
                total_units=total_units,
            )
        selected_ratio = float(final_mask.float().mean().item())
        if int(final_mask.sum().item()) == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.step_count += 1
            self.skip_count += 1
            self._clear_ln_step_stats(
                entropy_keep_ratio=entropy_keep_ratio,
                plpd_keep_ratio=plpd_keep_ratio,
                selected_ratio=selected_ratio,
                plpd_mean=plpd_mean,
                source_safe_ratio=source_safe_ratio,
                total_units=total_units,
            )
            return self._apply_ln_output_views(pixel_values, raw_logits, input_size)

        selected_logits = raw_logits.permute(0, 2, 3, 1)[final_mask]
        selected_probs = raw_probs.permute(0, 2, 3, 1)[final_mask]
        selected_source_probs = source_probs.permute(0, 2, 3, 1)[final_mask]
        selected_scores = selection_score[final_mask]
        selected_count = int(final_mask.sum().item())
        selected_source_agreement = float(source_agreement_mask[final_mask].float().mean().item())
        selected_probe_drop = float(plpd_scores[final_mask].mean().item())
        selected_source_confidence = float(source_conf[final_mask].mean().item())

        if not self.ln_template_initialized:
            self._update_ln_class_template(
                selected_source_probs.detach(),
                sample_weights=selected_scores.detach(),
                force_init=True,
            )

        per_unit_loss = self._compute_ln_adadem_loss(selected_logits, selected_probs)
        if self.uan_enabled:
            loss_normalizer = float(selected_count if self.sane_normalizer == "selected" else max(total_units, 1))
        else:
            loss_normalizer = float(
                active_unit_count(
                    batch_size=int(raw_logits.shape[0]),
                    mode=self.uan_mode,
                    spatial_shape=input_size,
                )
            )
        adadem_loss = (
            selected_entity_mean(per_unit_loss)
            if self.uan_enabled and self.sane_normalizer == "selected"
            else per_unit_loss.sum() / loss_normalizer
        )
        if self.ln_source_anchor_weight != 0.0:
            source_anchor_per_unit = F.kl_div(
                F.log_softmax(selected_logits, dim=1),
                selected_source_probs.detach(),
                reduction="none",
            ).sum(dim=1)
            source_anchor_loss = (
                selected_entity_mean(source_anchor_per_unit)
                if self.uan_enabled and self.sane_normalizer == "selected"
                else source_anchor_per_unit.sum() / loss_normalizer
            )
        else:
            source_anchor_loss = selected_logits.new_zeros(())
        loss = adadem_loss + self.ln_source_anchor_weight * source_anchor_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._update_ln_class_template(
            selected_probs.detach(),
            sample_weights=selected_scores.detach(),
        )

        self.step_count += 1
        self.update_count += 1
        self.dig_kept_groups += selected_count
        self.last_ln_entropy_keep_ratio = entropy_keep_ratio
        self.last_ln_plpd_keep_ratio = plpd_keep_ratio
        self.last_ln_selected_ratio = selected_ratio
        self.last_ln_plpd_mean = plpd_mean
        self.last_ln_adadem_loss = float(adadem_loss.detach().item())
        self.last_ln_source_anchor_loss = float(source_anchor_loss.detach().item())
        self.last_ln_loss_normalizer = loss_normalizer
        self.last_ln_selected_count = selected_count
        self.last_ln_total_units = total_units
        self.last_ln_source_agreement = selected_source_agreement
        self.last_ln_source_safe_ratio = source_safe_ratio
        self.last_ln_probe_drop = selected_probe_drop
        self.last_ln_source_confidence = selected_source_confidence
        self.source_anchor_agreement = merge_metric(self.source_anchor_agreement, selected_source_agreement)
        self.group_reward_std = float("nan")
        self.sos_mean_penalty = float("nan")
        self._record_ln_step_stats(
            selected_ratio=selected_ratio,
            source_agreement=selected_source_agreement,
            probe_drop=selected_probe_drop,
        )
        if self.ln_predict_after_update:
            post_logits = self._forward_logits(pixel_values, require_grad=False)
            post_logits = F.interpolate(post_logits, size=input_size, mode="bilinear", align_corners=False)
            return self._apply_ln_output_views(pixel_values, post_logits, input_size)
        return self._apply_ln_output_views(pixel_values, raw_logits, input_size)

    def _adapt_batch_legacy(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raw_logits = self._forward_logits(pixel_values, require_grad=True)
        with torch.no_grad():
            source_logits = self._forward_logits(pixel_values, require_grad=False)
            weak_logits = self._forward_logits(build_photometric_view(pixel_values), require_grad=False)
            structural_logits = self._forward_logits(build_structural_view(pixel_values), require_grad=False)

        raw_probs = raw_logits.softmax(dim=1)
        source_probs = source_logits.softmax(dim=1)
        weak_probs = weak_logits.softmax(dim=1)
        structural_probs = structural_logits.softmax(dim=1)

        target = weighted_pixel_vote(raw_probs, source_probs, weak_probs, structural_probs)

        rewards = torch.stack(
            [
                pixel_reward(raw_probs, source_probs, target, raw_logits),
                pixel_reward(weak_probs, source_probs, target, weak_logits),
                pixel_reward(structural_probs, source_probs, target, structural_logits),
            ],
            dim=1,
        )
        reward_std = rewards.std(dim=1, unbiased=False)
        raw_advantage = (rewards[:, 0] - rewards.mean(dim=1)) / reward_std.clamp_min(1e-4)

        raw_pred = raw_probs.argmax(dim=1)
        source_pred = source_probs.argmax(dim=1)
        weak_pred = weak_probs.argmax(dim=1)
        structural_pred = structural_probs.argmax(dim=1)
        entropy_score = normalized_entropy(
            raw_logits.permute(0, 2, 3, 1).reshape(-1, raw_logits.shape[1])
        ).view(raw_logits.shape[0], -1).mean(dim=1)
        anchor_disagree = (raw_pred != source_pred).float().mean(dim=(1, 2))
        view_inconsistency = (
            1.0
            - torch.stack(
                [
                    (raw_pred == weak_pred).float().mean(dim=(1, 2)),
                    (raw_pred == structural_pred).float().mean(dim=(1, 2)),
                ],
                dim=1,
            ).mean(dim=1)
        )
        structural_sensitivity = destroy_sensitivity(
            raw_probs.gather(1, target.unsqueeze(1)).squeeze(1).mean(dim=(1, 2)),
            structural_probs.gather(1, target.unsqueeze(1)).squeeze(1).mean(dim=(1, 2)),
        )
        structural_risk = insufficient_destroy_penalty(structural_sensitivity, 0.2)
        shift_score = (
            0.40 * entropy_score
            + 0.25 * anchor_disagree
            + 0.20 * view_inconsistency
            + 0.15 * structural_risk
        ).clamp(0.0, 1.0)
        sos_weight = (
            sos_weight_from_score(
                shift_score,
                safe_margin=self.sos_config["safe_margin"],
                hard_margin=self.sos_config["hard_margin"],
            )
            if self.sos_enabled else torch.ones_like(shift_score)
        )
        informative_mask = reward_std > self.dig_config["std_threshold"] if self.dig_enabled else torch.ones_like(reward_std, dtype=torch.bool)
        final_mask = informative_mask & (sos_weight > 0.0)
        if final_mask.sum() == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.step_count += 1
            self.skip_count += 1
            return raw_logits.detach()

        target_prob = raw_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        ref_prob = source_probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        ratio = target_prob / ref_prob
        per_pixel_surrogate = arc_surrogate(
            ratio=ratio[final_mask],
            advantage=raw_advantage[final_mask].view(-1, 1, 1),
            eps_low=self.arc_config["eps_low"],
            eps_high=self.arc_config["eps_high"],
            enabled=self.arc_enabled,
        )
        weight = sos_weight[final_mask].view(-1, 1, 1)
        loss = -(weight * per_pixel_surrogate).sum()
        normalization_batch = int(final_mask.sum().item())
        if self.sane_normalizer == "total":
            normalization_batch = int(target.shape[0])
        loss = loss / float(
            active_unit_count(
                batch_size=normalization_batch,
                mode=self.uan_mode,
                spatial_shape=target.shape[-2:],
            )
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        self.update_count += 1
        self.dig_kept_groups += int(final_mask.sum().item())
        anchor_agree = float((raw_pred[final_mask] == source_pred[final_mask]).float().mean().item())
        self.source_anchor_agreement = merge_metric(self.source_anchor_agreement, anchor_agree)
        self.group_reward_std = merge_metric(self.group_reward_std, float(reward_std[final_mask].mean().item()))
        self.sos_mean_penalty = merge_metric(self.sos_mean_penalty, float((1.0 - sos_weight[final_mask]).mean().item()))
        self.last_ln_entropy_keep_ratio = 0.0
        self.last_ln_plpd_keep_ratio = 0.0
        self.last_ln_selected_ratio = 0.0
        self.last_ln_plpd_mean = 0.0
        self.last_ln_adadem_loss = 0.0
        return raw_logits.detach()

    def adapt_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.vit_ln_family_mode:
            return self._adapt_batch_vit_ln(pixel_values)
        return self._adapt_batch_legacy(pixel_values)

    def get_stats(self) -> Dict:
        return {
            "method_variant": self.method_variant,
            "branch_refinements": self.use_branch_refinements,
            "step_count": self.step_count,
            "update_count": self.update_count,
            "skip_count": self.skip_count,
            "arc_enabled": self.arc_enabled,
            "uan_enabled": self.uan_enabled,
            "no_uan": not self.uan_enabled,
            "dig_kept_groups": self.dig_kept_groups,
            "dig_buffer_fills": self.dig_buffer_fills,
            "optimizer": self.optimizer_name,
            "optimizer_momentum": self.optim_momentum if self.optimizer_name == "sgd" else 0.0,
            "uan_mode": self.uan_mode,
            "sane_normalizer": self.sane_normalizer if self.uan_enabled else None,
            "sos_mean_penalty": self.sos_mean_penalty if not math.isnan(self.sos_mean_penalty) else 0.0,
            "source_anchor_agreement": self.source_anchor_agreement if not math.isnan(self.source_anchor_agreement) else 0.0,
            "group_reward_std": self.group_reward_std if not math.isnan(self.group_reward_std) else 0.0,
            "ln_mode": self.vit_ln_family_mode,
            "ln_continual_mode": self.vit_ln_family_mode,
            "vit_ctta_core": self.vit_ctta_core,
            "ln_probe_policy": self.ln_probe_policy,
            "ln_probe_forward_count": self.ln_probe_forward_count,
            "ln_probe_skipped_count": self.ln_probe_skipped_count,
            "last_ln_probe_used": self.last_ln_probe_used,
            "ln_shared_role_only": self.ln_shared_role_only,
            "ln_effective_pi": self.ln_pi,
            "ln_effective_entropy_margin_scale": (
                self.ln_entropy_margin / math.log(max(self.num_classes, 2))
                if self.vit_ln_family_mode else 0.0
            ),
            "ln_effective_plpd_threshold": self.ln_plpd_threshold,
            "selection_quantile": self.selection_quantile,
            "ln_effective_patch_len": self.ln_patch_len,
            "ln_adapt_decode_bn": self.ln_adapt_decode_bn,
            "ln_predict_after_update": self.ln_predict_after_update,
            "ln_output_hflip": self.ln_output_hflip,
            "ln_output_hflip_weight": self.ln_output_hflip_weight,
            "ln_output_scale": self.ln_output_scale,
            "ln_output_scale_weight": self.ln_output_scale_weight,
            "last_ln_entropy_keep_ratio": self.last_ln_entropy_keep_ratio,
            "last_ln_plpd_keep_ratio": self.last_ln_plpd_keep_ratio,
            "last_ln_selected_ratio": self.last_ln_selected_ratio,
            "last_ln_plpd_mean": self.last_ln_plpd_mean,
            "last_ln_adadem_loss": self.last_ln_adadem_loss,
            "last_ln_source_anchor_loss": self.last_ln_source_anchor_loss,
            "last_ln_loss_normalizer": self.last_ln_loss_normalizer,
            "last_ln_selected_count": self.last_ln_selected_count,
            "last_ln_total_units": self.last_ln_total_units,
            "last_ln_source_agreement": self.last_ln_source_agreement,
            "last_ln_source_safe_ratio": self.last_ln_source_safe_ratio,
            "last_ln_probe_drop": self.last_ln_probe_drop,
            "last_ln_source_confidence": self.last_ln_source_confidence,
            "last_ln_hflip_agreement": self.last_ln_hflip_agreement,
            "last_ln_scale_agreement": self.last_ln_scale_agreement,
            "mean_ln_selected_ratio": (
                self._ln_selected_ratio_sum / self._ln_stats_steps
                if self._ln_stats_steps > 0 else 0.0
            ),
            "mean_ln_source_agreement": (
                self._ln_source_agreement_sum / self._ln_stats_steps
                if self._ln_stats_steps > 0 else 0.0
            ),
            "mean_ln_probe_drop": (
                self._ln_probe_drop_sum / self._ln_stats_steps
                if self._ln_stats_steps > 0 else 0.0
            ),
        }


def weighted_pixel_vote(
    raw_probs: torch.Tensor,
    source_probs: torch.Tensor,
    weak_probs: torch.Tensor,
    structural_probs: torch.Tensor,
) -> torch.Tensor:
    vote_stack = (
        1.25 * source_probs
        + raw_probs
        + weak_probs
        + structural_probs
    )
    return vote_stack.argmax(dim=1)


def pixel_reward(
    probs: torch.Tensor,
    source_probs: torch.Tensor,
    target: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    target_prob = probs.gather(1, target.unsqueeze(1)).squeeze(1).mean(dim=(1, 2))
    anchor_agree = (probs.argmax(dim=1) == source_probs.argmax(dim=1)).float().mean(dim=(1, 2))
    ent_score = 1.0 - normalized_entropy(logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])).view(logits.shape[0], -1).mean(dim=1)
    return 0.50 * target_prob + 0.25 * anchor_agree + 0.25 * ent_score
