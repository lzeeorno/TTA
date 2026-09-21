"""
Classification branch of ATLAS TTA family.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import importlib.util
import math
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    DIGBuffer,
    active_unit_count,
    arc_surrogate,
    build_destroy_view,
    build_photometric_view,
    build_structural_view,
    build_vit_plpd_view,
    collect_norm_params,
    consistency_cross_entropy,
    class_centered_anchor_loss,
    configure_model,
    infer_vit_token_count,
    looks_like_vit_family,
    merge_metric,
    normalized_entropy,
    quantile_selection_mask,
    destroy_sensitivity,
    insufficient_destroy_penalty,
    softmax_entropy,
    sos_weight_from_score,
    weighted_majority_vote,
)


def _clone_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return deepcopy(value)


def _move_optimizer_state_to_param_device(optimizer: torch.optim.Optimizer) -> None:
    for param, state in optimizer.state.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=param.device)


ATLAS_VIEW_NAMES = ("source", "raw", "photometric", "structural", "destroy")
ATLAS_VIEW_ALIASES = {
    "weak": "photometric",
}


def aggregate_evidence(
    evidence: torch.Tensor,
    mode: str = "equal_mean",
    legacy_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Aggregate normalized evidence along the final dimension."""
    if mode == "equal_mean":
        return evidence.mean(dim=-1)
    if mode == "legacy_weighted":
        if legacy_weights is None:
            raise ValueError("legacy_weighted aggregation requires legacy_weights")
        return (evidence * legacy_weights.to(evidence)).sum(dim=-1)
    raise ValueError(f"Unknown ATLAS evidence aggregation: {mode!r}")


class ATLAS(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        arc_config: Optional[Dict] = None,
        dig_config: Optional[Dict] = None,
        uan_config: Optional[Dict] = None,
        sos_config: Optional[Dict] = None,
        view_config: Optional[Dict] = None,
        adapt_type: str = "bn",
        adaptation_mode: str = "continual",
        scenario: str = "normal",
        lr: float = 2.5e-4,
        optimizer_name: str = "sgd",
        optim_momentum: float = 0.9,
        optim_weight_decay: float = 0.0,
        vit_ln_config: Optional[Dict] = None,
        num_classes: int = 1000,
        no_arc: bool = False,
        no_dig: bool = False,
        no_uan: bool = False,
        no_sos: bool = False,
        ablation_row: Optional[str] = None,
        reward_config: Optional[Dict] = None,
        selection_config: Optional[Dict] = None,
        method_variant: str = "full",
    ):
        super().__init__()
        self.method_variant = str(method_variant).lower()
        if self.method_variant not in {"core", "full"}:
            raise ValueError(
                "ATLAS method_variant must be one of: core, full; "
                f"got {method_variant!r}"
            )
        self.use_branch_refinements = self.method_variant == "full"
        self.selection_quantile = float((selection_config or {}).get("quantile", 0.5))
        if not 0.0 <= self.selection_quantile <= 1.0:
            raise ValueError("ATLAS selection quantile must be in [0, 1]")
        self.adapt_type = adapt_type
        self.adaptation_mode = adaptation_mode
        self.scenario = scenario
        self.num_classes = num_classes
        self.atlas_ablation_row = str(ablation_row).upper() if ablation_row else None

        self.arc_enabled = not no_arc
        self.dig_enabled = not no_dig
        self.uan_enabled = not no_uan
        self.sos_enabled = not no_sos
        self._bn_standard_mode = (
            str(adapt_type).lower() == "bn"
            and str(adaptation_mode).lower() == "standard"
            and str(scenario).lower() == "normal"
        )
        self._bn_continual_mode = (
            str(adapt_type).lower() == "bn"
            and str(adaptation_mode).lower() == "continual"
            and str(scenario).lower() == "normal"
        )
        self._gn_standard_mode = (
            str(adapt_type).lower() == "gn"
            and str(adaptation_mode).lower() == "standard"
            and str(scenario).lower() == "normal"
        )
        self._gn_continual_mode = (
            str(adapt_type).lower() == "gn"
            and str(adaptation_mode).lower() == "continual"
            and str(scenario).lower() == "normal"
        )
        self._gn_mode = self._gn_standard_mode or self._gn_continual_mode
        self._ln_vit_mode = (
            str(adapt_type).lower() == "ln"
            and looks_like_vit_family(model)
        )
        self._ln_standard_mode = (
            self._ln_vit_mode
            and str(adaptation_mode).lower() == "standard"
        )
        self._ln_continual_mode = (
            self._ln_vit_mode
            and str(adaptation_mode).lower() == "continual"
        )
        self._ln_mode = self._ln_standard_mode or self._ln_continual_mode

        self.arc_config = {
            "eps_low": 0.2,
            "eps_high": 0.28,
        }
        if arc_config:
            self.arc_config.update(arc_config)

        self.dig_config = {
            "std_threshold": 0.05,
            "buffer_target": 8,
            "buffer_scenarios": ["bs1", "label_shifts", "mix_shifts"],
        }
        if dig_config:
            self.dig_config.update(dig_config)

        self._is_vit = looks_like_vit_family(model)
        default_uan_mode = "token" if self._is_vit else "sample"
        self.vit_ctta_core = "adadem_plpd" if self._ln_vit_mode else "none"
        self.uan_config = {
            "mode": default_uan_mode,
        }
        if uan_config:
            self.uan_config.update(uan_config)

        self.sos_config = {
            "safe_margin": 0.35,
            "hard_margin": 0.75,
            "entropy_weight": 0.35,
            "anchor_weight": 0.25,
            "view_weight": 0.20,
            "destroy_weight": 0.20,
            "destroy_support_margin": 0.20,
        }
        if sos_config:
            self.sos_config.update(sos_config)

        reward_names = ("target", "entropy", "source", "consensus")
        reward_config = dict(reward_config or {})
        self.reward_aggregation = str(
            reward_config.pop("aggregation", "equal_mean")
        ).lower()
        legacy_weights = reward_config.pop("legacy_weights", None)
        # Flat weights remain readable only for historical config reproduction.
        if reward_config:
            if set(reward_config) != set(reward_names):
                unknown = ", ".join(sorted(set(reward_config) - set(reward_names)))
                raise ValueError(f"Unknown ATLAS reward setting(s): {unknown}")
            legacy_weights = reward_config
            self.reward_aggregation = "legacy_weighted"
        if self.reward_aggregation not in {"equal_mean", "legacy_weighted"}:
            raise ValueError("ATLAS reward aggregation must be equal_mean or legacy_weighted")
        self.legacy_reward_weights = None
        if self.reward_aggregation == "legacy_weighted":
            if not isinstance(legacy_weights, dict) or set(legacy_weights) != set(reward_names):
                raise ValueError("legacy_weighted aggregation requires four legacy_weights")
            reward_values = [float(legacy_weights[name]) for name in reward_names]
            if any(not math.isfinite(value) or value < 0.0 for value in reward_values):
                raise ValueError("ATLAS reward weights must be finite and non-negative")
            if not math.isclose(sum(reward_values), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("ATLAS reward weights must sum to 1")
            self.legacy_reward_weights = dict(zip(reward_names, reward_values))

        self.view_config = dict(view_config or {})
        self._explicit_active_views = self.view_config.get("active") is not None
        if self._explicit_active_views:
            self.active_views = self._normalize_active_views(self.view_config.get("active"))
        else:
            self.active_views = ATLAS_VIEW_NAMES
        self.source_view_active = "source" in self.active_views
        self.raw_view_active = "raw" in self.active_views
        self.photometric_view_active = "photometric" in self.active_views
        self.structural_view_active = "structural" in self.active_views
        self.destroy_view_active = "destroy" in self.active_views

        self.bn_standard_reliable_margin = 0.4 * math.log(max(self.num_classes, 2))
        self.bn_standard_entropy_weight = 0.50 if self._bn_standard_mode else 0.0
        self.bn_standard_consistency_weight = 0.01 if self._bn_standard_mode else 0.0
        self.bn_standard_probe_count = 16 if self._bn_standard_mode else 0
        self.bn_standard_probe_entropy_momentum = 0.9
        self.bn_standard_template_momentum = 0.9
        self.use_bn_standard_d3lr = self._bn_standard_mode and self.use_branch_refinements
        self.bn_standard_d3lr_alpha_max = 6.0 if self._bn_standard_mode else 1.0
        self.bn_standard_d3lr_kappa = 0.5 if self._bn_standard_mode else 1.0
        self.bn_standard_d3lr_h_bar = (
            0.31 * math.log(max(self.num_classes, 2)) if self._bn_standard_mode else 0.0
        )
        self.bn_standard_d3lr_current_alpha = 1.0
        self.bn_standard_trusted_hard_boost = 0.08 if self._bn_standard_mode else 0.0
        self.bn_standard_class_template = None
        self.bn_standard_probe_entropy_ema = None
        self.bn_continual_selection_margin = 0.50 if self._bn_continual_mode else 1.0
        self.bn_continual_reweight_margin = 0.40 if self._bn_continual_mode else 1.0
        self.bn_continual_plpd_margin = 0.12 if self._bn_continual_mode else 0.0
        self.bn_continual_min_fraction = 0.20 if self._bn_continual_mode else 0.0
        self.bn_continual_anchor_margin = 0.80 if self._bn_continual_mode else 0.0
        self.bn_continual_hard_anchor_margin = 0.75 if self._bn_continual_mode else 0.0
        self.bn_continual_plpd_veto_margin = 0.74 if self._bn_continual_mode else 0.0
        self.bn_continual_batch_alignment_margin = 0.42 if self._bn_continual_mode else 1.0
        self.bn_continual_veto_margin = 0.24 if self._bn_continual_mode else 1.0
        self.bn_continual_arc_safe_margin = 0.18 if self._bn_continual_mode else 1.0
        self.bn_continual_arc_hard_margin = 0.34 if self._bn_continual_mode else 1.0
        self.bn_continual_arc_floor = 0.40 if self._bn_continual_mode else 1.0
        self.bn_continual_trusted_template_debias = 0.12 if self._bn_continual_mode else 0.0
        self.bn_continual_trusted_hard_boost = 0.14 if self._bn_continual_mode else 0.0
        self.bn_continual_plpd_sharpness = 0.55 if self._bn_continual_mode else 0.0
        self.bn_continual_rescue_plpd_margin = 0.64 if self._bn_continual_mode else 1.0
        self.bn_continual_rescue_support_ceiling = 0.84 if self._bn_continual_mode else 1.0
        self.bn_continual_rescue_entropy_ceiling = 0.50 if self._bn_continual_mode else 1.0
        self.bn_continual_core_entropy_mix = 0.40 if self._bn_continual_mode else 1.0
        self.bn_continual_core_plpd_mix = 0.60 if self._bn_continual_mode else 1.0
        self.bn_continual_marginal_momentum = 0.9
        self.bn_continual_balance_floor = 0.35
        self.bn_continual_prior_power = 0.5
        self.bn_continual_anchor_weight = 0.30 if self._bn_continual_mode else 0.0
        self.use_bn_continual_d3lr = False
        self.bn_continual_d3lr_alpha_max = 1.0
        self.bn_continual_d3lr_kappa = 1.0
        self.bn_continual_d3lr_h_bar = 0.0
        self.bn_continual_d3lr_current_alpha = 1.0
        self.bn_continual_prob_ema = None
        self.bn_continual_class_prior = None
        self.bn_continual_class_template = None
        self.use_gn_unified_67b = self._gn_mode
        self.gn_entropy_margin = 0.78 if self._gn_mode else 1.0
        self.gn_keep_target = (
            0.55 if self._gn_continual_mode
            else (0.70 if self._gn_standard_mode else 0.0)
        )
        self.gn_min_fraction = 0.18 if self._gn_mode else 0.0
        self.gn_score_floor = 0.24 if self._gn_mode else 0.0
        self.gn_plpd_threshold = 0.20 if self.use_gn_unified_67b else 0.0
        self.gn_entropy_weight = 1.0 if self.use_gn_unified_67b else 0.0
        self.gn_plpd_weight = 1.0 if self.use_gn_unified_67b else 0.0
        self.gn_anchor_weight = 0.15 if self._gn_mode else 0.0
        self.gn_struct_weight = 0.20 if self._gn_mode else 0.0
        self.gn_arc_floor = 0.60 if self._gn_mode else 1.0
        self.gn_d_margin = 0.05 if self.use_gn_unified_67b else 0.0
        self.gn_model_probs_momentum = 0.9
        self.gn_distill_temp = 5.0 if self.use_gn_unified_67b else 1.0
        self.gn_distill_weight = 0.8 if self.use_gn_unified_67b else 0.0
        self.gn_structural_weight = (
            0.0005 * float(self.num_classes) if self.use_gn_unified_67b else 0.0
        )
        self.use_gn_continual_d3lr = self.use_gn_unified_67b
        self.gn_d3lr_alpha_max = 50.0 if self.use_gn_unified_67b else 1.0
        self.gn_d3lr_kappa = 0.5 if self.use_gn_unified_67b else 1.0
        self.gn_d3lr_h_bar = (
            0.2 * math.log(max(self.num_classes, 2)) if self.use_gn_unified_67b else 0.0
        )
        self.gn_d3lr_current_alpha = 1.0
        self.gn_distill_warmup_steps = 64 if self.use_gn_unified_67b else 0
        self.gn_structural_warmup_steps = 32 if self.use_gn_unified_67b else 0
        self.gn_redundancy_warmup_steps = 32 if self.use_gn_unified_67b else 0
        self.gn_distill_min_factor = 0.25 if self.use_gn_unified_67b else 0.0
        self.gn_structural_min_factor = 0.25 if self.use_gn_unified_67b else 0.0
        self.gn_current_model_probs = None
        self.vit_ln_config = {
            "optimizer": "sgd",
            "lr": 0.05,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "pi": 0.1,
            "entropy_margin_scale": 0.4,
            "plpd_threshold": 0.2,
            "patch_len": 4,
            "probe_policy": "always",
            "shared_role_only": False,
            "source_calib_max_steps": 8,
            "source_entropy_alpha": 0.3,
            "source_entropy_abs_scale": 1.2,
            "snow_entropy_margin_scale": 0.5,
            "snow_plpd_threshold": 0.2,
            "snow_patch_len": 4,
            "standard_plpd_warmup_steps": 64,
            "standard_warmup_entropy_margin_scale": 0.5,
            "standard_recover_selected_ratio": 0.60,
            "standard_recover_active_classes": 16,
            "standard_recover_patience": 8,
            "standard_source_anchor_weight": 0.25,
            "lowtemp_corruptions": ["snow"],
            "lowtemp_snow_always": True,
            "lowtemp_collapse_base_selected_ratio": 0.80,
            "lowtemp_collapse_entropy_active_classes": 24,
            "lowtemp_collapse_probe_steps": 32,
            "lowtemp_patch_len": 4,
            "lowtemp_phase1_entropy_margin_scale": 0.5,
            "lowtemp_phase1_source_anchor_weight": 0.65,
            "lowtemp_phase1_min_steps": 64,
            "lowtemp_transition_selected_ratio": 0.85,
            "lowtemp_transition_active_classes": 16,
            "lowtemp_transition_patience": 2,
            "lowtemp_phase2_source_anchor_weight": 0.25,
            "lowtemp_phase2_plpd_temperature": 0.1,
            "bs1": {
                "enabled": True,
                "candidate": 4,
                "lr": 5e-4,
                "param_scope": "deyo_vit",
                "buffer_all_samples": True,
                "loss": "deyo_anchor",
                "entropy_filter_scale": 0.4,
                "entropy_weight_margin_scale": 0.4,
                "plpd_threshold": 0.2,
                "patch_len": 4,
                "reweight_ent": 1.0,
                "reweight_plpd": 1.0,
                "source_anchor_weight": 0.25,
                "topology_weight": 0.0,
            },
            "natural_shift": {
                "enabled": False,
                "entropy_filter_scale": 0.5,
                "entropy_weight_margin_scale": 0.4,
                "plpd_threshold": 0.2,
                "patch_len": 4,
                "reweight_ent": 1.0,
                "reweight_plpd": 1.0,
                "source_anchor_weight": 0.0,
                "topology_weight": 0.05,
            },
        }
        if vit_ln_config:
            default_natural_shift_config = dict(self.vit_ln_config["natural_shift"])
            default_bs1_config = dict(self.vit_ln_config["bs1"])
            self.vit_ln_config.update(vit_ln_config)
            if isinstance(vit_ln_config.get("bs1"), dict):
                default_bs1_config.update(vit_ln_config["bs1"])
                self.vit_ln_config["bs1"] = default_bs1_config
            if isinstance(vit_ln_config.get("natural_shift"), dict):
                default_natural_shift_config.update(vit_ln_config["natural_shift"])
                self.vit_ln_config["natural_shift"] = default_natural_shift_config
        self.ln_probe_policy = str(self.vit_ln_config.get("probe_policy", "always")).lower()
        if self.ln_probe_policy not in {"always", "alternate", "off"}:
            raise ValueError(
                "vit_ln.probe_policy must be one of: always, alternate, off"
            )
        self.ln_shared_role_only = (
            not self.use_branch_refinements
            or bool(self.vit_ln_config.get("shared_role_only", False))
        )
        bs1_config = self.vit_ln_config.get("bs1", {})
        self.ln_bs1_enabled = (
            self.use_branch_refinements
            and self._ln_vit_mode
            and str(self.scenario).lower() == "bs1"
            and bool(bs1_config.get("enabled", True))
        )
        self.ln_bs1_candidate = (
            int(bs1_config.get("candidate", 4))
            if self.ln_bs1_enabled else 0
        )
        self.ln_bs1_candidate = max(0, min(4, self.ln_bs1_candidate))
        self.ln_bs1_lr = (
            float(bs1_config.get("lr", 5e-4))
            if self.ln_bs1_enabled and self.ln_bs1_candidate >= 1 else 0.0
        )
        self.ln_bs1_buffer_all_samples = (
            self.ln_bs1_enabled
            and self.ln_bs1_candidate >= 3
            and bool(bs1_config.get("buffer_all_samples", True))
        )
        self.ln_bs1_loss_name = str(bs1_config.get("loss", "deyo_anchor")).lower()
        self.ln_bs1_loss_enabled = (
            self.ln_bs1_enabled
            and self.ln_bs1_candidate >= 4
            and self.ln_bs1_loss_name in {"deyo", "deyo_anchor", "deyo_plpd"}
        )
        self.ln_bs1_entropy_margin = (
            float(bs1_config.get("entropy_filter_scale", 0.4))
            * math.log(max(self.num_classes, 2))
            if self.ln_bs1_loss_enabled else 0.0
        )
        self.ln_bs1_entropy_weight_margin = (
            float(bs1_config.get("entropy_weight_margin_scale", 0.4))
            * math.log(max(self.num_classes, 2))
            if self.ln_bs1_loss_enabled else 0.0
        )
        self.ln_bs1_plpd_threshold = (
            float(bs1_config.get("plpd_threshold", 0.2))
            if self.ln_bs1_loss_enabled else 0.0
        )
        self.ln_bs1_patch_len = (
            int(bs1_config.get("patch_len", 4))
            if self.ln_bs1_loss_enabled else 0
        )
        self.ln_bs1_reweight_ent = (
            float(bs1_config.get("reweight_ent", 1.0))
            if self.ln_bs1_loss_enabled else 0.0
        )
        self.ln_bs1_reweight_plpd = (
            float(bs1_config.get("reweight_plpd", 1.0))
            if self.ln_bs1_loss_enabled else 0.0
        )
        self.ln_bs1_source_anchor_weight = (
            float(bs1_config.get("source_anchor_weight", 0.25))
            if self.ln_bs1_loss_enabled else 0.0
        )
        self.ln_bs1_topology_weight = (
            float(bs1_config.get("topology_weight", 0.0))
            if self.ln_bs1_loss_enabled else 0.0
        )
        self.ln_pi = float(self.vit_ln_config["pi"]) if self._ln_vit_mode else 0.0
        self.ln_entropy_margin = (
            float(self.vit_ln_config["entropy_margin_scale"]) * math.log(max(self.num_classes, 2))
            if self._ln_vit_mode else 0.0
        )
        self.ln_plpd_threshold = float(self.vit_ln_config["plpd_threshold"]) if self._ln_vit_mode else 0.0
        self.ln_patch_len = int(self.vit_ln_config["patch_len"]) if self._ln_vit_mode else 0
        ln_param_scope = str(self.vit_ln_config.get("param_scope", "full_ln")).lower()
        if self.ln_bs1_enabled and self.ln_bs1_candidate >= 2:
            ln_param_scope = str(bs1_config.get("param_scope", "deyo_vit")).lower()
        self.ln_param_scope = (
            ln_param_scope
            if self._ln_vit_mode else "none"
        )
        natural_shift_config = self.vit_ln_config.get("natural_shift", {})
        self.ln_natural_shift_enabled = (
            self.use_branch_refinements
            and self._ln_continual_mode
            and bool(natural_shift_config.get("enabled", False))
        )
        self.ln_natural_entropy_margin = (
            float(natural_shift_config.get("entropy_filter_scale", 0.5))
            * math.log(max(self.num_classes, 2))
            if self.ln_natural_shift_enabled else 0.0
        )
        self.ln_natural_entropy_weight_margin = (
            float(natural_shift_config.get("entropy_weight_margin_scale", 0.4))
            * math.log(max(self.num_classes, 2))
            if self.ln_natural_shift_enabled else 0.0
        )
        self.ln_natural_plpd_threshold = (
            float(natural_shift_config.get("plpd_threshold", 0.2))
            if self.ln_natural_shift_enabled else 0.0
        )
        self.ln_natural_patch_len = (
            int(natural_shift_config.get("patch_len", 4))
            if self.ln_natural_shift_enabled else 0
        )
        self.ln_natural_reweight_ent = (
            float(natural_shift_config.get("reweight_ent", 1.0))
            if self.ln_natural_shift_enabled else 0.0
        )
        self.ln_natural_reweight_plpd = (
            float(natural_shift_config.get("reweight_plpd", 1.0))
            if self.ln_natural_shift_enabled else 0.0
        )
        self.ln_natural_source_anchor_weight = (
            float(natural_shift_config.get("source_anchor_weight", 0.0))
            if self.ln_natural_shift_enabled else 0.0
        )
        self.ln_natural_topology_weight = (
            float(natural_shift_config.get("topology_weight", 0.05))
            if self.ln_natural_shift_enabled else 0.0
        )
        if self.ln_natural_shift_enabled:
            self.vit_ctta_core = "deyo_plpd"
        if self.ln_bs1_loss_enabled:
            self.vit_ctta_core = "bs1_deyo_plpd"
        self.ln_snow_entropy_margin = (
            float(self.vit_ln_config["snow_entropy_margin_scale"]) * math.log(max(self.num_classes, 2))
            if self._ln_standard_mode else 0.0
        )
        self.ln_snow_plpd_threshold = (
            float(self.vit_ln_config["snow_plpd_threshold"]) if self._ln_standard_mode else 0.0
        )
        self.ln_snow_patch_len = int(self.vit_ln_config["snow_patch_len"]) if self._ln_standard_mode else 0
        self.ln_source_calib_steps = 0
        self.ln_source_calib_max_steps = (
            int(self.vit_ln_config.get("source_calib_max_steps", 8))
            if self._ln_vit_mode else 0
        )
        self.ln_source_entropy_alpha = (
            float(self.vit_ln_config.get("source_entropy_alpha", 0.3))
            if self._ln_vit_mode else 0.0
        )
        self.ln_source_entropy_abs_scale = (
            float(self.vit_ln_config.get("source_entropy_abs_scale", 1.2))
            if self._ln_vit_mode else 1.0
        )
        self.ln_standard_plpd_warmup_steps = (
            int(self.vit_ln_config["standard_plpd_warmup_steps"]) if self._ln_standard_mode else 0
        )
        self.ln_standard_warmup_entropy_margin = (
            float(self.vit_ln_config["standard_warmup_entropy_margin_scale"])
            * math.log(max(self.num_classes, 2))
            if self._ln_standard_mode else 0.0
        )
        self.ln_standard_recover_selected_ratio = (
            float(self.vit_ln_config["standard_recover_selected_ratio"]) if self._ln_standard_mode else 0.0
        )
        self.ln_standard_recover_active_classes = (
            int(self.vit_ln_config["standard_recover_active_classes"]) if self._ln_standard_mode else 0
        )
        self.ln_standard_recover_patience = (
            int(self.vit_ln_config["standard_recover_patience"]) if self._ln_standard_mode else 0
        )
        self.ln_standard_source_anchor_weight = (
            float(self.vit_ln_config["standard_source_anchor_weight"]) if self._ln_standard_mode else 0.0
        )
        self.ln_standard_base_selected_floor = 0.30 if self._ln_standard_mode else 0.0
        self.ln_standard_warmup_updates = 0
        self.ln_standard_recovery_streak = 0
        self.current_corruption = "unknown"
        self.ln_lowtemp_corruptions = (
            tuple(str(name).lower() for name in self.vit_ln_config["lowtemp_corruptions"])
            if self._ln_standard_mode else tuple()
        )
        self.ln_lowtemp_snow_always = bool(self.vit_ln_config["lowtemp_snow_always"]) if self._ln_standard_mode else False
        self.ln_lowtemp_collapse_base_selected_ratio = (
            float(self.vit_ln_config["lowtemp_collapse_base_selected_ratio"]) if self._ln_standard_mode else 0.0
        )
        self.ln_lowtemp_collapse_entropy_active_classes = (
            int(self.vit_ln_config["lowtemp_collapse_entropy_active_classes"]) if self._ln_standard_mode else 0
        )
        self.ln_lowtemp_collapse_probe_steps = (
            int(self.vit_ln_config["lowtemp_collapse_probe_steps"]) if self._ln_standard_mode else 0
        )
        self.ln_lowtemp_patch_len = int(self.vit_ln_config["lowtemp_patch_len"]) if self._ln_standard_mode else 0
        self.ln_lowtemp_phase1_entropy_margin = (
            float(self.vit_ln_config["lowtemp_phase1_entropy_margin_scale"]) * math.log(max(self.num_classes, 2))
            if self._ln_standard_mode else 0.0
        )
        self.ln_lowtemp_phase1_source_anchor_weight = (
            float(self.vit_ln_config["lowtemp_phase1_source_anchor_weight"]) if self._ln_standard_mode else 0.0
        )
        self.ln_lowtemp_phase1_min_steps = (
            int(self.vit_ln_config["lowtemp_phase1_min_steps"]) if self._ln_standard_mode else 0
        )
        self.ln_lowtemp_transition_selected_ratio = (
            float(self.vit_ln_config["lowtemp_transition_selected_ratio"]) if self._ln_standard_mode else 0.0
        )
        self.ln_lowtemp_transition_active_classes = (
            int(self.vit_ln_config["lowtemp_transition_active_classes"]) if self._ln_standard_mode else 0
        )
        self.ln_lowtemp_transition_patience = (
            int(self.vit_ln_config["lowtemp_transition_patience"]) if self._ln_standard_mode else 0
        )
        self.ln_lowtemp_phase2_source_anchor_weight = (
            float(self.vit_ln_config["lowtemp_phase2_source_anchor_weight"]) if self._ln_standard_mode else 0.0
        )
        self.ln_lowtemp_phase2_plpd_temperature = (
            float(self.vit_ln_config["lowtemp_phase2_plpd_temperature"]) if self._ln_standard_mode else 1.0
        )
        self.ln_lowtemp_branch_active = False
        self.ln_lowtemp_phase = "disabled"
        self.ln_lowtemp_decision_locked = False
        self.ln_lowtemp_probe_updates = 0
        self.ln_lowtemp_phase1_updates = 0
        self.ln_lowtemp_transition_streak = 0
        self.ln_lowtemp_trigger_reason = "none"
        self.ln_class_template = None
        self.ln_template_initialized = False
        self.last_probe_reliability = 0.0
        self.last_bn_entropy_loss = 0.0
        self.last_bn_consistency_loss = 0.0
        self.last_bn_standard_debias = 1.0
        self.last_bn_standard_trusted_boost = 1.0
        self.last_bn_standard_entropy_core = 0.0
        self.last_bn_standard_entropy_keep_ratio = 0.0
        self.last_bn_standard_arc_reg = 0.0
        self.last_bn_standard_d3lr_alpha = 1.0
        self.last_bn_standard_d3lr_scale = 1.0
        self.last_bn_standard_aug_mode = "none"
        self.last_bn_standard_teacher_mode = "none"
        self.last_bn_standard_consistency_factor = 0.0
        self.last_bn_standard_consistency_mix = 0.0
        self.last_bn_standard_aux_mix = 0.0
        self.last_bn_standard_probe_entropy = 0.0
        self.last_bn_standard_ais_soft_weight = 0.0
        self.last_bn_standard_accepted_entropy = 0.0
        self.last_bn_standard_d3lr_driver = "none"
        self.last_bn_continual_plpd = 0.0
        self.last_bn_continual_harmful = 0.0
        self.last_bn_continual_veto = 0.0
        self.last_bn_continual_anchor = 0.0
        self.last_bn_continual_batch_alignment = 0.0
        self.last_bn_continual_balance = 0.0
        self.last_bn_continual_arc_guard = 1.0
        self.last_bn_continual_debias = 1.0
        self.last_bn_continual_d3lr_alpha = 1.0
        self.last_bn_continual_d3lr_driver = "none"
        self.last_bn_continual_rescue = 0.0
        self.last_bn_continual_rescue_plpd_edge = 0.0
        self.last_bn_continual_rescue_anchor_edge = 0.0
        self.last_gn_plpd = 0.0
        self.last_gn_anchor = 0.0
        self.last_gn_keep_ratio = 0.0
        self.last_gn_redundancy_skip = 0.0
        self.last_gn_arc_guard = 1.0
        self.last_gn_batch_alignment = 0.0
        self.last_gn_stage1_ratio = 0.0
        self.last_gn_nonredundant_ratio = 0.0
        self.last_gn_entropy_loss = 0.0
        self.last_gn_distill_loss = 0.0
        self.last_gn_distill_factor = 0.0
        self.last_gn_structural_factor = 0.0
        self.last_gn_redundancy_active = 0.0
        self.last_gn_d3lr_alpha = 1.0
        self.last_gn_raw_entropy = 0.0
        self.last_ln_entropy_keep_ratio = 0.0
        self.last_ln_plpd_keep_ratio = 0.0
        self.last_ln_selected_ratio = 0.0
        self.last_ln_plpd_mean = 0.0
        self.last_ln_adadem_loss = 0.0
        self.last_ln_topology_loss = 0.0
        self.last_ln_topology_active_classes = 0
        self.last_ln_topology_weight = 0.0
        self.last_ln_loss_mode = "none"
        self.last_ln_active_classes = 0
        self.last_ln_selection_mode = "none"
        self.last_ln_entropy_active_classes = 0
        self.last_ln_source_anchor_loss = 0.0
        self.last_ln_param_scope = self.ln_param_scope
        self.last_ln_natural_shift_profile = "none"
        self.last_ln_natural_shift_rescue_active = False
        self.last_ln_deyo_coeff_mean = 0.0
        self.last_ln_deyo_coeff_max = 0.0
        self.last_ln_lowtemp_trigger_reason = "none"
        self.last_ln_lowtemp_plpd_soft_mean = 0.0
        self.last_ln_source_entropy_mean = 0.0
        self.last_ln_entropy_calibration_active = False
        self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
        self.last_ln_calibrated_entropy_keep_ratio = 0.0
        self.last_ln_entropy_abs_override_ratio = 0.0
        self.last_ln_base_selected_ratio = 0.0
        self._bn_standard_soft_transform = None
        self._bn_standard_strong_transform = None
        self._ln_aggregate_steps = 0
        self._ln_entropy_keep_sum = 0.0
        self._ln_plpd_keep_sum = 0.0
        self._ln_selected_ratio_sum = 0.0
        self._ln_sketch_rescue_steps = 0
        self._ln_sketch_rescue_active_classes_sum = 0.0
        self._ln_deyo_coeff_mean_sum = 0.0
        self._ln_deyo_coeff_max_sum = 0.0
        self._ln_selected_active_classes_sum = 0.0

        self.model = configure_model(model, adapt_type=adapt_type)
        self.source_model = deepcopy(self.model)
        self.source_model.eval()
        for param in self.source_model.parameters():
            param.requires_grad_(False)

        self._arch_type = self._detect_architecture()
        self._max_depth = self._infer_max_depth()
        self.params, self.param_names = collect_norm_params(self.model, adapt_type=adapt_type)
        if self._gn_mode:
            self.params, self.param_names = self._filter_gn_params(
                self.params,
                self.param_names,
            )
        if self._ln_vit_mode and self.use_branch_refinements:
            self.params, self.param_names = self._filter_ln_vit_params(
                self.params,
                self.param_names,
            )
        if not self.params:
            raise ValueError(f"ATLAS found no adaptable parameters for adapt_type={adapt_type!r}")

        optimizer_name = optimizer_name.lower()
        effective_optimizer_name = optimizer_name
        effective_lr = lr
        effective_momentum = optim_momentum
        effective_weight_decay = optim_weight_decay
        if self._ln_vit_mode:
            effective_lr = (
                self.ln_bs1_lr
                if self.ln_bs1_enabled and self.ln_bs1_candidate >= 1
                else float(self.vit_ln_config["lr"])
            )
            effective_optimizer_name = str(self.vit_ln_config["optimizer"]).lower()
            effective_momentum = float(self.vit_ln_config["momentum"])
            effective_weight_decay = float(self.vit_ln_config["weight_decay"])
        elif self._bn_standard_mode and self.use_branch_refinements:
            effective_lr = min(float(lr), 3e-4)
            if effective_optimizer_name == "sgd":
                effective_optimizer_name = "adam"
        elif self._bn_continual_mode and self.use_branch_refinements:
            effective_lr = min(float(lr), 3.5e-4)
            if effective_optimizer_name == "sgd":
                effective_optimizer_name = "adam"
        elif self._gn_mode and self.use_branch_refinements:
            effective_lr = min(float(lr), 1.5e-4)
            if effective_optimizer_name == "sgd":
                effective_optimizer_name = "adam"
        self.optimizer_name = effective_optimizer_name
        optimizer_params = self.params
        if self.use_bn_standard_d3lr or self.use_bn_continual_d3lr:
            optimizer_params = self._build_bn_standard_d3lr_param_groups(effective_lr)
        elif self.use_gn_continual_d3lr:
            optimizer_params = self._build_gn_continual_d3lr_param_groups(effective_lr)

        if effective_optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                optimizer_params,
                lr=effective_lr,
                weight_decay=effective_weight_decay,
            )
        elif effective_optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                optimizer_params,
                lr=effective_lr,
                weight_decay=effective_weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                optimizer_params,
                lr=effective_lr,
                momentum=effective_momentum,
                weight_decay=effective_weight_decay,
            )

        self.lr = effective_lr
        self.token_count = infer_vit_token_count(self.model) if self._is_vit else 1
        requested_uan_mode = self.uan_config["mode"]
        if requested_uan_mode == "auto":
            requested_uan_mode = default_uan_mode
        self.uan_mode = (
            requested_uan_mode
            if self.uan_enabled and self.use_branch_refinements
            else "sample"
        )
        self.buffering_enabled = (
            self.use_branch_refinements
            and self.dig_enabled
            and self.scenario in set(self.dig_config["buffer_scenarios"])
        )
        self.dig_buffer = DIGBuffer(target_size=self.dig_config["buffer_target"])

        self._supports_reset = str(self.adaptation_mode).lower() == "standard"
        self.model_state = _clone_to_cpu(self.model.state_dict()) if self._supports_reset else None
        self.optimizer_state = _clone_to_cpu(self.optimizer.state_dict()) if self._supports_reset else None

        self.step_count = 0
        self.update_count = 0
        self.skip_count = 0
        self.dig_kept_groups = 0
        self.dig_buffer_fills = 0
        self.source_anchor_agreement = float("nan")
        self.group_reward_std = float("nan")
        self.sos_mean_penalty = float("nan")
        self.last_selected_ratio = 0.0
        self.last_advantage_mean = 0.0
        self.ln_probe_forward_count = 0
        self.ln_probe_skipped_count = 0
        self.last_ln_probe_used = False

    def reset(self) -> None:
        if self.model_state is not None:
            self.model.load_state_dict(self.model_state, strict=True)
        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)
            _move_optimizer_state_to_param_device(self.optimizer)
        self.dig_buffer.reset()
        self.step_count = 0
        self.update_count = 0
        self.skip_count = 0
        self.dig_kept_groups = 0
        self.dig_buffer_fills = 0
        self.source_anchor_agreement = float("nan")
        self.group_reward_std = float("nan")
        self.sos_mean_penalty = float("nan")
        self.last_selected_ratio = 0.0
        self.last_advantage_mean = 0.0
        self.ln_probe_forward_count = 0
        self.ln_probe_skipped_count = 0
        self.last_ln_probe_used = False
        self.last_probe_reliability = 0.0
        self.last_bn_entropy_loss = 0.0
        self.last_bn_consistency_loss = 0.0
        self.last_bn_standard_debias = 1.0
        self.last_bn_standard_trusted_boost = 1.0
        self.last_bn_standard_entropy_core = 0.0
        self.last_bn_standard_entropy_keep_ratio = 0.0
        self.last_bn_standard_arc_reg = 0.0
        self.bn_standard_d3lr_current_alpha = 1.0
        self.last_bn_standard_d3lr_alpha = 1.0
        self.last_bn_standard_d3lr_scale = 1.0
        self.last_bn_standard_aug_mode = "none"
        self.last_bn_standard_teacher_mode = "none"
        self.last_bn_standard_consistency_factor = 0.0
        self.last_bn_standard_consistency_mix = 0.0
        self.last_bn_standard_aux_mix = 0.0
        self.last_bn_standard_probe_entropy = 0.0
        self.last_bn_standard_ais_soft_weight = 0.0
        self.last_bn_standard_accepted_entropy = 0.0
        self.last_bn_standard_d3lr_driver = "none"
        self.bn_continual_prob_ema = None
        self.bn_continual_class_prior = None
        self.bn_continual_class_template = None
        self.bn_continual_d3lr_current_alpha = 1.0
        self.last_bn_continual_plpd = 0.0
        self.last_bn_continual_harmful = 0.0
        self.last_bn_continual_veto = 0.0
        self.last_bn_continual_anchor = 0.0
        self.last_bn_continual_batch_alignment = 0.0
        self.last_bn_continual_balance = 0.0
        self.last_bn_continual_arc_guard = 1.0
        self.last_bn_continual_debias = 1.0
        self.last_bn_continual_d3lr_alpha = 1.0
        self.last_bn_continual_d3lr_driver = "none"
        self.last_bn_continual_rescue = 0.0
        self.last_bn_continual_rescue_plpd_edge = 0.0
        self.last_bn_continual_rescue_anchor_edge = 0.0
        self.gn_current_model_probs = None
        self.last_gn_plpd = 0.0
        self.last_gn_anchor = 0.0
        self.last_gn_keep_ratio = 0.0
        self.last_gn_redundancy_skip = 0.0
        self.last_gn_arc_guard = 1.0
        self.last_gn_batch_alignment = 0.0
        self.last_gn_stage1_ratio = 0.0
        self.last_gn_nonredundant_ratio = 0.0
        self.last_gn_entropy_loss = 0.0
        self.last_gn_distill_loss = 0.0
        self.last_gn_distill_factor = 0.0
        self.last_gn_structural_factor = 0.0
        self.last_gn_redundancy_active = 0.0
        self.gn_d3lr_current_alpha = 1.0
        self.last_gn_d3lr_alpha = 1.0
        self.last_gn_raw_entropy = 0.0
        self.ln_class_template = None
        self.ln_template_initialized = False
        self.ln_source_calib_steps = 0
        self.ln_standard_warmup_updates = 0
        self.ln_standard_recovery_streak = 0
        self.ln_lowtemp_branch_active = False
        self.ln_lowtemp_phase = "disabled"
        self.ln_lowtemp_decision_locked = False
        self.ln_lowtemp_probe_updates = 0
        self.ln_lowtemp_phase1_updates = 0
        self.ln_lowtemp_transition_streak = 0
        self.ln_lowtemp_trigger_reason = "none"
        self.last_ln_entropy_keep_ratio = 0.0
        self.last_ln_plpd_keep_ratio = 0.0
        self.last_ln_selected_ratio = 0.0
        self.last_ln_plpd_mean = 0.0
        self.last_ln_adadem_loss = 0.0
        self.last_ln_topology_loss = 0.0
        self.last_ln_topology_active_classes = 0
        self.last_ln_topology_weight = 0.0
        self.last_ln_loss_mode = "none"
        self.last_ln_active_classes = 0
        self.last_ln_selection_mode = "none"
        self.last_ln_entropy_active_classes = 0
        self.last_ln_source_anchor_loss = 0.0
        self.last_ln_param_scope = self.ln_param_scope
        self.last_ln_natural_shift_profile = "none"
        self.last_ln_natural_shift_rescue_active = False
        self.last_ln_deyo_coeff_mean = 0.0
        self.last_ln_deyo_coeff_max = 0.0
        self.last_ln_source_entropy_mean = 0.0
        self.last_ln_entropy_calibration_active = False
        self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
        self.last_ln_calibrated_entropy_keep_ratio = 0.0
        self.last_ln_entropy_abs_override_ratio = 0.0
        self.last_ln_base_selected_ratio = 0.0
        self.last_ln_lowtemp_trigger_reason = "none"
        self.last_ln_lowtemp_plpd_soft_mean = 0.0
        self._ln_aggregate_steps = 0
        self._ln_entropy_keep_sum = 0.0
        self._ln_plpd_keep_sum = 0.0
        self._ln_selected_ratio_sum = 0.0
        self._ln_sketch_rescue_steps = 0
        self._ln_sketch_rescue_active_classes_sum = 0.0
        self._ln_deyo_coeff_mean_sum = 0.0
        self._ln_deyo_coeff_max_sum = 0.0
        self._ln_selected_active_classes_sum = 0.0
        self.bn_standard_class_template = None
        self.bn_standard_probe_entropy_ema = None

    def _normalize_active_views(self, active_views) -> Tuple[str, ...]:
        if isinstance(active_views, str):
            raw_names = [item.strip() for item in active_views.split(",")]
        else:
            raw_names = list(active_views or [])

        normalized_names: List[str] = []
        seen = set()
        for name in raw_names:
            candidate = str(name).strip().lower()
            if not candidate:
                continue
            candidate = ATLAS_VIEW_ALIASES.get(candidate, candidate)
            if candidate not in ATLAS_VIEW_NAMES:
                raise ValueError(
                    f"Unsupported ATLAS view '{candidate}'. Expected one of {ATLAS_VIEW_NAMES}."
                )
            if candidate in seen:
                continue
            seen.add(candidate)
            normalized_names.append(candidate)

        if not normalized_names:
            raise ValueError("ATLAS active views cannot be empty.")
        if "raw" not in seen:
            raise ValueError("ATLAS active views must include 'raw'.")

        return tuple(normalized_names)

    @staticmethod
    def _neutral_support(targets: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            targets.shape[0],
            device=targets.device,
            dtype=reference.dtype,
        )

    @staticmethod
    def _neutral_anchor(predictions: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            predictions.shape[0],
            device=predictions.device,
            dtype=torch.float32,
        )

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

    def get_stats(self) -> Dict:
        return {
            "method_variant": self.method_variant,
            "branch_refinements": self.use_branch_refinements,
            "step_count": self.step_count,
            "update_count": self.update_count,
            "skip_count": self.skip_count,
            "arc_enabled": self.arc_enabled,
            "dig_kept_groups": self.dig_kept_groups,
            "dig_buffer_fills": self.dig_buffer_fills,
            "uan_mode": self.uan_mode,
            "sos_mean_penalty": self.sos_mean_penalty if not math.isnan(self.sos_mean_penalty) else 0.0,
            "source_anchor_agreement": self.source_anchor_agreement if not math.isnan(self.source_anchor_agreement) else 0.0,
            "group_reward_std": self.group_reward_std if not math.isnan(self.group_reward_std) else 0.0,
            "last_selected_ratio": self.last_selected_ratio,
            "last_advantage_mean": self.last_advantage_mean,
            "buffer_size": len(self.dig_buffer),
            "bn_standard_mode": self._bn_standard_mode,
            "bn_continual_mode": self._bn_continual_mode,
            "optimizer_name": self.optimizer_name,
            "effective_lr": self.lr,
            "ln_mode": self._ln_mode,
            "ln_vit_mode": self._ln_vit_mode,
            "ln_standard_mode": self._ln_standard_mode,
            "ln_continual_mode": self._ln_continual_mode,
            "ln_adaptation_mode": (
                "standard" if self._ln_standard_mode
                else ("continual" if self._ln_continual_mode else "disabled")
            ),
            "ln_bs1_safe_enabled": self.ln_bs1_enabled,
            "ln_bs1_candidate": self.ln_bs1_candidate,
            "ln_bs1_buffer_all_samples": self.ln_bs1_buffer_all_samples,
            "ln_bs1_loss_enabled": self.ln_bs1_loss_enabled,
            "ln_probe_policy": self.ln_probe_policy,
            "ln_probe_forward_count": self.ln_probe_forward_count,
            "ln_probe_skipped_count": self.ln_probe_skipped_count,
            "last_ln_probe_used": self.last_ln_probe_used,
            "ln_shared_role_only": self.ln_shared_role_only,
            "ln_effective_pi": self.ln_pi,
            "ln_effective_entropy_margin_scale": (
                self.ln_entropy_margin / math.log(max(self.num_classes, 2))
                if self._ln_vit_mode else 0.0
            ),
            "ln_effective_plpd_threshold": self.ln_plpd_threshold,
            "ln_effective_patch_len": self.ln_patch_len,
            "current_corruption": self.current_corruption,
            "lowtemp_branch_active": self.ln_lowtemp_branch_active,
            "lowtemp_phase": self.ln_lowtemp_phase,
            "vit_ctta_core": self.vit_ctta_core,
            "atlas_ablation_row": self.atlas_ablation_row or "default",
            "view_selection_mode": "explicit" if self._explicit_active_views else "legacy",
            "active_views": list(self.active_views),
            "reward_aggregation": self.reward_aggregation,
            "legacy_reward_weights": deepcopy(self.legacy_reward_weights),
            "selection_quantile": self.selection_quantile,
            "last_probe_reliability": self.last_probe_reliability,
            "last_bn_entropy_loss": self.last_bn_entropy_loss,
            "last_bn_consistency_loss": self.last_bn_consistency_loss,
            "last_bn_standard_debias": self.last_bn_standard_debias,
            "last_bn_standard_trusted_boost": self.last_bn_standard_trusted_boost,
            "last_bn_standard_entropy_core": self.last_bn_standard_entropy_core,
            "last_bn_standard_entropy_keep_ratio": self.last_bn_standard_entropy_keep_ratio,
            "last_bn_standard_arc_reg": self.last_bn_standard_arc_reg,
            "last_bn_standard_d3lr_alpha": self.last_bn_standard_d3lr_alpha,
            "last_bn_standard_d3lr_scale": self.last_bn_standard_d3lr_scale,
            "last_bn_standard_aug_mode": self.last_bn_standard_aug_mode,
            "last_bn_standard_teacher_mode": self.last_bn_standard_teacher_mode,
            "last_bn_standard_consistency_factor": self.last_bn_standard_consistency_factor,
            "last_bn_standard_consistency_mix": self.last_bn_standard_consistency_mix,
            "last_bn_standard_aux_mix": self.last_bn_standard_aux_mix,
            "last_bn_standard_probe_entropy": self.last_bn_standard_probe_entropy,
            "last_bn_standard_ais_soft_weight": self.last_bn_standard_ais_soft_weight,
            "last_bn_standard_accepted_entropy": self.last_bn_standard_accepted_entropy,
            "last_bn_standard_d3lr_driver": self.last_bn_standard_d3lr_driver,
            "last_bn_continual_plpd": self.last_bn_continual_plpd,
            "last_bn_continual_harmful": self.last_bn_continual_harmful,
            "last_bn_continual_veto": self.last_bn_continual_veto,
            "last_bn_continual_anchor": self.last_bn_continual_anchor,
            "last_bn_continual_batch_alignment": self.last_bn_continual_batch_alignment,
            "last_bn_continual_balance": self.last_bn_continual_balance,
            "last_bn_continual_arc_guard": self.last_bn_continual_arc_guard,
            "last_bn_continual_debias": self.last_bn_continual_debias,
            "last_bn_continual_d3lr_alpha": self.last_bn_continual_d3lr_alpha,
            "last_bn_continual_d3lr_driver": self.last_bn_continual_d3lr_driver,
            "last_bn_continual_rescue": self.last_bn_continual_rescue,
            "last_bn_continual_rescue_plpd_edge": self.last_bn_continual_rescue_plpd_edge,
            "last_bn_continual_rescue_anchor_edge": self.last_bn_continual_rescue_anchor_edge,
            "gn_standard_mode": self._gn_standard_mode,
            "gn_continual_mode": self._gn_continual_mode,
            "last_gn_plpd": self.last_gn_plpd,
            "last_gn_anchor": self.last_gn_anchor,
            "last_gn_keep_ratio": self.last_gn_keep_ratio,
            "last_gn_redundancy_skip": self.last_gn_redundancy_skip,
            "last_gn_arc_guard": self.last_gn_arc_guard,
            "last_gn_batch_alignment": self.last_gn_batch_alignment,
            "last_gn_stage1_ratio": self.last_gn_stage1_ratio,
            "last_gn_nonredundant_ratio": self.last_gn_nonredundant_ratio,
            "last_gn_entropy_loss": self.last_gn_entropy_loss,
            "last_gn_distill_loss": self.last_gn_distill_loss,
            "last_gn_distill_factor": self.last_gn_distill_factor,
            "last_gn_structural_factor": self.last_gn_structural_factor,
            "last_gn_redundancy_active": self.last_gn_redundancy_active,
            "last_gn_d3lr_alpha": self.last_gn_d3lr_alpha,
            "last_gn_raw_entropy": self.last_gn_raw_entropy,
            "last_ln_entropy_keep_ratio": self.last_ln_entropy_keep_ratio,
            "last_ln_plpd_keep_ratio": self.last_ln_plpd_keep_ratio,
            "last_ln_selected_ratio": self.last_ln_selected_ratio,
            "last_ln_plpd_mean": self.last_ln_plpd_mean,
            "last_ln_adadem_loss": self.last_ln_adadem_loss,
            "last_ln_topology_loss": self.last_ln_topology_loss,
            "last_ln_topology_active_classes": self.last_ln_topology_active_classes,
            "last_ln_topology_weight": self.last_ln_topology_weight,
            "last_ln_loss_mode": self.last_ln_loss_mode,
            "last_ln_active_classes": self.last_ln_active_classes,
            "last_ln_selection_mode": self.last_ln_selection_mode,
            "last_ln_entropy_active_classes": self.last_ln_entropy_active_classes,
            "last_ln_source_anchor_loss": self.last_ln_source_anchor_loss,
            "last_ln_param_scope": self.last_ln_param_scope,
            "last_ln_natural_shift_profile": self.last_ln_natural_shift_profile,
            "last_ln_natural_shift_rescue_active": self.last_ln_natural_shift_rescue_active,
            "last_ln_deyo_coeff_mean": self.last_ln_deyo_coeff_mean,
            "last_ln_deyo_coeff_max": self.last_ln_deyo_coeff_max,
            "last_ln_source_entropy_mean": self.last_ln_source_entropy_mean,
            "last_ln_entropy_calibration_active": self.last_ln_entropy_calibration_active,
            "last_ln_entropy_calibration_alpha": self.last_ln_entropy_calibration_alpha,
            "last_ln_calibrated_entropy_keep_ratio": self.last_ln_calibrated_entropy_keep_ratio,
            "last_ln_entropy_abs_override_ratio": self.last_ln_entropy_abs_override_ratio,
            "last_ln_base_selected_ratio": self.last_ln_base_selected_ratio,
            "last_ln_lowtemp_trigger_reason": self.last_ln_lowtemp_trigger_reason,
            "last_ln_lowtemp_plpd_soft_mean": self.last_ln_lowtemp_plpd_soft_mean,
            "mean_ln_entropy_keep_ratio": (
                self._ln_entropy_keep_sum / self._ln_aggregate_steps
                if self._ln_aggregate_steps > 0 else 0.0
            ),
            "mean_ln_plpd_keep_ratio": (
                self._ln_plpd_keep_sum / self._ln_aggregate_steps
                if self._ln_aggregate_steps > 0 else 0.0
            ),
            "mean_ln_selected_ratio": (
                self._ln_selected_ratio_sum / self._ln_aggregate_steps
                if self._ln_aggregate_steps > 0 else 0.0
            ),
            "mean_ln_sketch_rescue_step_ratio": (
                float(self._ln_sketch_rescue_steps) / self._ln_aggregate_steps
                if self._ln_aggregate_steps > 0 else 0.0
            ),
            "mean_ln_sketch_rescue_active_classes": (
                self._ln_sketch_rescue_active_classes_sum / self._ln_sketch_rescue_steps
                if self._ln_sketch_rescue_steps > 0 else 0.0
            ),
            "mean_ln_deyo_coeff_mean": (
                self._ln_deyo_coeff_mean_sum / self._ln_aggregate_steps
                if self._ln_aggregate_steps > 0 else 0.0
            ),
            "mean_ln_deyo_coeff_max": (
                self._ln_deyo_coeff_max_sum / self._ln_aggregate_steps
                if self._ln_aggregate_steps > 0 else 0.0
            ),
            "mean_ln_selected_active_classes": (
                self._ln_selected_active_classes_sum / self._ln_aggregate_steps
                if self._ln_aggregate_steps > 0 else 0.0
            ),
        }

    def set_current_corruption(self, corruption: str) -> None:
        self.current_corruption = str(corruption)

    def _use_ln_natural_shift_bridge(self) -> bool:
        return bool(self.ln_natural_shift_enabled)

    def _record_ln_aggregate_stats(self) -> None:
        self._ln_aggregate_steps += 1
        self._ln_entropy_keep_sum += float(self.last_ln_entropy_keep_ratio)
        self._ln_plpd_keep_sum += float(self.last_ln_plpd_keep_ratio)
        self._ln_selected_ratio_sum += float(self.last_ln_selected_ratio)
        if bool(self.last_ln_natural_shift_rescue_active):
            self._ln_sketch_rescue_steps += 1
            self._ln_sketch_rescue_active_classes_sum += float(
                self.last_ln_entropy_active_classes
            )
        self._ln_deyo_coeff_mean_sum += float(self.last_ln_deyo_coeff_mean)
        self._ln_deyo_coeff_max_sum += float(self.last_ln_deyo_coeff_max)
        self._ln_selected_active_classes_sum += float(self.last_ln_active_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_outputs = self.model(x)
        self._maybe_adapt(x)
        return current_outputs.detach()

    def _maybe_adapt(self, x: torch.Tensor) -> None:
        update_batch = None
        if self.buffering_enabled:
            if self.ln_bs1_buffer_all_samples:
                added_count = self.dig_buffer.add(x)
                self.dig_buffer_fills += added_count
            else:
                info_mask = self._precompute_informative_mask(x)
                if info_mask.any():
                    added_count = self.dig_buffer.add(x[info_mask])
                    self.dig_buffer_fills += added_count
            update_batch = self.dig_buffer.pop_ready_batch(x.device)
            if update_batch is None:
                self.skip_count += 1
                self.step_count += 1
                return
        else:
            update_batch = x

        adapted = self._adapt_on_batch(update_batch)
        if self._ln_mode:
            self._record_ln_aggregate_stats()
        if not adapted:
            self.skip_count += 1
        self.step_count += 1

    def _precompute_informative_mask(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            raw_logits = self.model(x)
            group_std, sos_weight, _, _, _, _, _ = self._compute_group_signals(x, raw_logits)
        return (
            (group_std > self.dig_config["std_threshold"])
            & (sos_weight > 0.0)
        )

    def _filter_gn_params(self, params, names):
        filtered_params = []
        filtered_names = []
        for param, name in zip(params, names):
            lname = name.lower()
            if lname.startswith("layer4."):
                continue
            if lname.startswith(("fc.", "head.", "classifier.", "global_pool.")):
                continue
            if lname in {"norm.weight", "norm.bias"}:
                continue
            filtered_params.append(param)
            filtered_names.append(name)
        return filtered_params, filtered_names

    def _filter_ln_vit_params(self, params, names):
        scope = self.ln_param_scope
        if scope in {"full", "full_ln", "all"}:
            return params, names

        if scope != "deyo_vit":
            return params, names

        filtered_params = []
        filtered_names = []
        for param, name in zip(params, names):
            lname = name.lower()
            if any(f"blocks.{idx}." in lname for idx in (9, 10, 11)):
                continue
            if lname.startswith("norm.") or lname in {"norm.weight", "norm.bias"}:
                continue
            if lname.startswith(("head.", "fc.", "classifier.")):
                continue
            filtered_params.append(param)
            filtered_names.append(name)
        return filtered_params, filtered_names

    def _filter_ln_params_by_scope(self, params, names):
        return self._filter_ln_vit_params(params, names)

    def _ensure_gn_state(self, device: torch.device, dtype: torch.dtype) -> None:
        if not self.use_gn_unified_67b:
            return
        if self.gn_current_model_probs is None:
            return
        if self.gn_current_model_probs.shape[0] != self.num_classes:
            self.gn_current_model_probs = None
            return
        if (
            self.gn_current_model_probs.device != device
            or self.gn_current_model_probs.dtype != dtype
        ):
            self.gn_current_model_probs = self.gn_current_model_probs.to(
                device=device,
                dtype=dtype,
            )

    def _update_gn_history(self, probs: torch.Tensor) -> None:
        if not self.use_gn_unified_67b or probs.numel() == 0:
            return
        self._ensure_gn_state(probs.device, probs.dtype)
        batch_mean = probs.detach().mean(dim=0)
        if self.gn_current_model_probs is None:
            self.gn_current_model_probs = batch_mean
            return
        momentum = self.gn_model_probs_momentum
        self.gn_current_model_probs = (
            momentum * self.gn_current_model_probs
            + (1.0 - momentum) * batch_mean
        )

    def _get_gn_continual_factors(self) -> Tuple[float, float, bool]:
        if not self.use_gn_unified_67b:
            return 1.0, 1.0, True

        distill_factor = min(
            1.0,
            float(self.step_count + 1)
            / max(float(self.gn_distill_warmup_steps), 1.0),
        )
        structural_factor = min(
            1.0,
            float(self.step_count + 1)
            / max(float(self.gn_structural_warmup_steps), 1.0),
        )
        distill_factor = max(self.gn_distill_min_factor, distill_factor)
        structural_factor = max(self.gn_structural_min_factor, structural_factor)
        redundancy_active = self.step_count >= self.gn_redundancy_warmup_steps
        return distill_factor, structural_factor, redundancy_active

    def _select_top_fraction_mask(
        self,
        scores: torch.Tensor,
        target_keep: float,
        min_keep: int,
        floor: float,
    ) -> torch.Tensor:
        if scores.numel() == 0:
            return torch.zeros(0, device=scores.device, dtype=torch.bool)

        keep_count = max(1, int(math.ceil(scores.numel() * target_keep)))
        keep_count = min(scores.numel(), max(min_keep, keep_count))
        keep_indices = torch.topk(scores, k=keep_count, largest=True).indices
        keep_mask = torch.zeros(scores.numel(), device=scores.device, dtype=torch.bool)
        keep_mask[keep_indices] = True
        if floor > 0.0:
            keep_mask |= scores >= floor
        if int(keep_mask.sum().item()) < min_keep:
            keep_mask.zero_()
            keep_mask[keep_indices] = True
        return keep_mask

    def _ensure_bn_standard_state(self, device: torch.device, dtype: torch.dtype) -> None:
        if not self._bn_standard_mode:
            return
        if (
            self.bn_standard_class_template is None
            or self.bn_standard_class_template.device != device
            or self.bn_standard_class_template.dtype != dtype
            or self.bn_standard_class_template.shape[0] != self.num_classes
            or self.bn_standard_class_template.shape[1] != self.num_classes
        ):
            self.bn_standard_class_template = torch.full(
                (self.num_classes, self.num_classes),
                1.0 / float(self.num_classes),
                device=device,
                dtype=dtype,
            )

    def _update_bn_standard_history(self, probs: torch.Tensor) -> None:
        if not self._bn_standard_mode or probs.numel() == 0:
            return
        self._ensure_bn_standard_state(probs.device, probs.dtype)
        momentum = self.bn_standard_template_momentum
        pseudo_label = probs.detach().argmax(dim=1)
        for label in pseudo_label.unique():
            label_mask = pseudo_label == label
            label_mean = probs.detach()[label_mask].mean(dim=0)
            self.bn_standard_class_template[label] = (
                momentum * self.bn_standard_class_template[label]
                + (1.0 - momentum) * label_mean
            )

    def _update_bn_standard_probe_entropy(self, probe_entropy: float) -> float:
        if not self._bn_standard_mode:
            return 0.0
        if self.bn_standard_probe_entropy_ema is None:
            self.bn_standard_probe_entropy_ema = float(probe_entropy)
        else:
            m = self.bn_standard_probe_entropy_momentum
            self.bn_standard_probe_entropy_ema = (
                m * float(self.bn_standard_probe_entropy_ema)
                + (1.0 - m) * float(probe_entropy)
            )
        return float(self.bn_standard_probe_entropy_ema)

    def _ensure_bn_continual_state(self, device: torch.device, dtype: torch.dtype) -> None:
        if not self._bn_continual_mode:
            return
        if (
            self.bn_continual_prob_ema is None
            or self.bn_continual_prob_ema.device != device
            or self.bn_continual_prob_ema.dtype != dtype
            or self.bn_continual_prob_ema.shape[0] != self.num_classes
        ):
            self.bn_continual_prob_ema = torch.full(
                (self.num_classes,),
                1.0 / float(self.num_classes),
                device=device,
                dtype=dtype,
            )
        if (
            self.bn_continual_class_prior is None
            or self.bn_continual_class_prior.device != device
            or self.bn_continual_class_prior.dtype != dtype
            or self.bn_continual_class_prior.shape[0] != self.num_classes
        ):
            self.bn_continual_class_prior = torch.full(
                (self.num_classes,),
                1.0 / float(self.num_classes),
                device=device,
                dtype=dtype,
            )
        if (
            self.bn_continual_class_template is None
            or self.bn_continual_class_template.device != device
            or self.bn_continual_class_template.dtype != dtype
            or self.bn_continual_class_template.shape[0] != self.num_classes
            or self.bn_continual_class_template.shape[1] != self.num_classes
        ):
            self.bn_continual_class_template = torch.full(
                (self.num_classes, self.num_classes),
                1.0 / float(self.num_classes),
                device=device,
                dtype=dtype,
            )

    def _compute_bn_continual_diversity(
        self,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.bn_continual_prob_ema is None:
            ones = torch.ones(probs.shape[0], device=probs.device, dtype=probs.dtype)
            return ones, probs.new_tensor(0.0)

        history = self.bn_continual_prob_ema.to(device=probs.device, dtype=probs.dtype)
        sample_cos = F.cosine_similarity(
            probs,
            history.unsqueeze(0).expand_as(probs),
            dim=1,
        ).clamp(-1.0, 1.0)
        diversity_weight = (1.15 - sample_cos).clamp_(0.35, 1.0)

        batch_alignment = F.cosine_similarity(
            probs.mean(dim=0, keepdim=True),
            history.unsqueeze(0),
            dim=1,
        ).clamp(-1.0, 1.0)
        return diversity_weight, batch_alignment

    def _update_bn_continual_history(self, probs: torch.Tensor) -> None:
        if not self._bn_continual_mode or probs.numel() == 0:
            return
        self._ensure_bn_continual_state(probs.device, probs.dtype)
        batch_mean = probs.detach().mean(dim=0)
        momentum = self.bn_continual_marginal_momentum
        self.bn_continual_prob_ema = (
            momentum * self.bn_continual_prob_ema
            + (1.0 - momentum) * batch_mean
        )
        self.bn_continual_class_prior = (
            momentum * self.bn_continual_class_prior
            + (1.0 - momentum) * batch_mean
        )
        pseudo_label = probs.detach().argmax(dim=1)
        for label in pseudo_label.unique():
            label_mask = pseudo_label == label
            label_mean = probs.detach()[label_mask].mean(dim=0)
            self.bn_continual_class_template[label] = (
                momentum * self.bn_continual_class_template[label]
                + (1.0 - momentum) * label_mean
            )

    def _update_bn_continual_d3lr(self, batch_entropy: float) -> None:
        if not self.use_bn_continual_d3lr:
            return
        L = max(float(self._max_depth), 1.0)
        H_bar = self.bn_continual_d3lr_h_bar
        kappa = max(self.bn_continual_d3lr_kappa, 1e-8)
        sigmoid_val = 1.0 / (1.0 + math.exp(-(batch_entropy - H_bar) / kappa))
        alpha_t = 1.0 + (self.bn_continual_d3lr_alpha_max - 1.0) * sigmoid_val
        self.bn_continual_d3lr_current_alpha = alpha_t
        self.last_bn_continual_d3lr_alpha = alpha_t
        for group in self.optimizer.param_groups:
            depth = group.get("depth", None)
            if depth is None:
                continue
            group["lr"] = self.lr * (alpha_t ** ((L - float(depth)) / L))

    def _detect_architecture(self) -> str:
        child_names = {name for name, _ in self.model.named_children()}
        if "block1" in child_names:
            return "wideresnet"
        if "blocks" in child_names:
            return "vit"
        return "resnet"

    def _infer_max_depth(self) -> int:
        if self._arch_type == "wideresnet":
            return 3
        if self._arch_type == "vit":
            blocks = getattr(self.model, "blocks", None)
            if blocks is not None and len(blocks) > 0:
                return len(blocks)
            return 12
        return 4

    def _get_layer_depth(self, param_name: str) -> int:
        if self._arch_type == "wideresnet":
            if param_name.startswith("block1"):
                return 1
            if param_name.startswith("block2"):
                return 2
            if param_name.startswith("block3") or param_name.startswith("bn1"):
                return 3
            return 2
        if self._arch_type == "vit":
            if param_name.startswith("patch_embed") or param_name.startswith("norm_pre"):
                return 0
            if param_name.startswith("norm"):
                return self._max_depth
            if param_name.startswith("blocks."):
                parts = param_name.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    return min(int(parts[1]) + 1, self._max_depth)
            return max(1, self._max_depth // 2)
        if param_name.startswith("bn1") or param_name.startswith("gn1"):
            return 0
        if param_name.startswith("layer1"):
            return 1
        if param_name.startswith("layer2"):
            return 2
        if param_name.startswith("layer3"):
            return 3
        if param_name.startswith("layer4"):
            return 4
        return 2

    def _build_bn_standard_d3lr_param_groups(self, base_lr: float):
        depth_params = {}
        for param, name in zip(self.params, self.param_names):
            depth = self._get_layer_depth(name)
            depth_params.setdefault(depth, []).append(param)

        param_groups = []
        for depth in sorted(depth_params.keys()):
            params = depth_params[depth]
            if not params:
                continue
            param_groups.append(
                {
                    "params": params,
                    "lr": base_lr,
                    "depth": depth,
                }
            )
        return param_groups

    def _build_gn_continual_d3lr_param_groups(self, base_lr: float):
        depth_params = {}
        for param, name in zip(self.params, self.param_names):
            depth = self._get_layer_depth(name)
            depth_params.setdefault(depth, []).append(param)

        param_groups = []
        for depth in sorted(depth_params.keys()):
            params = depth_params[depth]
            if not params:
                continue
            param_groups.append(
                {
                    "params": params,
                    "lr": base_lr,
                    "depth": depth,
                }
            )
        return param_groups

    def _update_bn_standard_d3lr(self, batch_entropy: float) -> None:
        if not self.use_bn_standard_d3lr:
            return
        L = max(float(self._max_depth), 1.0)
        H_bar = self.bn_standard_d3lr_h_bar
        kappa = max(self.bn_standard_d3lr_kappa, 1e-8)
        sigmoid_val = 1.0 / (1.0 + math.exp(-(batch_entropy - H_bar) / kappa))
        alpha_t = 1.0 + (self.bn_standard_d3lr_alpha_max - 1.0) * sigmoid_val
        self.bn_standard_d3lr_current_alpha = alpha_t
        self.last_bn_standard_d3lr_alpha = alpha_t
        self.last_bn_standard_d3lr_scale = alpha_t
        for group in self.optimizer.param_groups:
            depth = group.get("depth", None)
            if depth is None:
                continue
            group["lr"] = self.lr * (alpha_t ** ((L - float(depth)) / L))

    def _update_gn_continual_d3lr(self, batch_entropy: float) -> None:
        if not self.use_gn_continual_d3lr:
            self.gn_d3lr_current_alpha = 1.0
            self.last_gn_d3lr_alpha = 1.0
            return
        L = max(float(self._max_depth), 1.0)
        H_bar = self.gn_d3lr_h_bar
        kappa = max(self.gn_d3lr_kappa, 1e-8)
        sigmoid_val = 1.0 / (1.0 + math.exp(-(batch_entropy - H_bar) / kappa))
        alpha_t = 1.0 + (self.gn_d3lr_alpha_max - 1.0) * sigmoid_val
        self.gn_d3lr_current_alpha = alpha_t
        self.last_gn_d3lr_alpha = alpha_t
        for group in self.optimizer.param_groups:
            depth = group.get("depth", None)
            if depth is None:
                continue
            group["lr"] = self.lr * (alpha_t ** ((L - float(depth)) / L))

    def _build_bn_standard_view(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            return x
        view = torch.flip(x, dims=[3])
        if x.shape[-1] > 4:
            view = torch.roll(view, shifts=1, dims=3)
        if x.shape[-2] > 4:
            view = torch.roll(view, shifts=1, dims=2)
        view = 0.75 * view + 0.25 * x.mean(dim=(2, 3), keepdim=True)
        view = view + 0.01 * torch.randn_like(view)
        with torch.no_grad():
            min_val = float(x.detach().amin().item())
            max_val = float(x.detach().amax().item())
        if min_val >= 0.0 and max_val <= 1.5:
            view = view.clamp_(0.0, 1.0)
        return view

    def _build_bn_standard_aux_view(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            return x
        _, _, h, w = x.shape
        patch_size = max(1, int(min(h, w) * 0.125))
        row_start = max((h - patch_size) // 2, 0)
        col_start = max((w - patch_size) // 2, 0)
        row_end = min(row_start + patch_size, h)
        col_end = min(col_start + patch_size, w)

        x_aux = x.clone()
        fill = x.mean(dim=(2, 3), keepdim=True)
        x_aux[:, :, row_start:row_end, col_start:col_end] = fill.expand(
            -1, -1, row_end - row_start, col_end - col_start
        )
        return x_aux

    def _get_bn_standard_soft_transform(self):
        if self._bn_standard_soft_transform is not None:
            return self._bn_standard_soft_transform

        cotta_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "baselines",
            "cotta",
            "cifar",
            "cotta.py",
        )
        cotta_dir = os.path.dirname(cotta_path)
        if cotta_dir not in sys.path:
            sys.path.append(cotta_dir)
        spec = importlib.util.spec_from_file_location("argo_bn_standard_cotta_soft", cotta_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load CoTTA transforms from {cotta_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._bn_standard_soft_transform = module.get_tta_transforms(
            gaussian_std=0.005,
            soft=True,
        )
        return self._bn_standard_soft_transform

    def _get_bn_standard_strong_transform(self):
        if self._bn_standard_strong_transform is not None:
            return self._bn_standard_strong_transform

        cotta_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "baselines",
            "cotta",
            "cifar",
            "cotta.py",
        )
        cotta_dir = os.path.dirname(cotta_path)
        if cotta_dir not in sys.path:
            sys.path.append(cotta_dir)
        spec = importlib.util.spec_from_file_location("argo_bn_standard_cotta_strong", cotta_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load CoTTA transforms from {cotta_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._bn_standard_strong_transform = module.get_tta_transforms(
            gaussian_std=0.005,
            soft=False,
        )
        return self._bn_standard_strong_transform

    def _build_bn_standard_surgeon_view(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            return x
        return self._get_bn_standard_strong_transform()(x)

    def _ensure_ln_state(self, device: torch.device, dtype: torch.dtype) -> None:
        if not self._ln_mode:
            return
        if (
            self.ln_class_template is None
            or self.ln_class_template.device != device
            or self.ln_class_template.dtype != dtype
            or self.ln_class_template.shape[0] != self.num_classes
            or self.ln_class_template.shape[1] != self.num_classes
        ):
            self.ln_class_template = torch.full(
                (self.num_classes, self.num_classes),
                1.0 / float(self.num_classes),
                device=device,
                dtype=dtype,
            )
            self.ln_template_initialized = False

    def _update_ln_class_template(
        self,
        probs: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> None:
        if not self._ln_mode or probs.numel() == 0:
            return
        self._ensure_ln_state(probs.device, probs.dtype)
        momentum = 1.0 - float(self.ln_pi) if self.ln_template_initialized else 0.0
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

    def _compute_ln_adadem_loss(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_ln_state(logits.device, logits.dtype)
        if not self.ln_template_initialized:
            self._update_ln_class_template(probs.detach())
        pseudo_labels = probs.argmax(dim=1)
        template = self.ln_class_template[pseudo_labels].detach()
        return class_centered_anchor_loss(logits, template)

    def _compute_ln_topology_loss(
        self,
        current_probs: torch.Tensor,
        source_probs: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        if current_probs.numel() == 0 or source_probs.numel() == 0:
            zero = current_probs.new_zeros(())
            return zero, 0

        pseudo_labels = current_probs.detach().argmax(dim=1)
        unique_labels = pseudo_labels.unique(sorted=True)
        active_classes = int(unique_labels.numel())
        if active_classes < 2:
            zero = current_probs.new_zeros(())
            return zero, active_classes

        current_centroids = []
        source_centroids = []
        for label in unique_labels:
            label_mask = pseudo_labels == label
            label_current = current_probs[label_mask]
            label_source = source_probs[label_mask]
            if sample_weights is None:
                current_centroids.append(label_current.mean(dim=0))
                source_centroids.append(label_source.mean(dim=0))
            else:
                label_weights = sample_weights[label_mask].to(
                    device=current_probs.device,
                    dtype=current_probs.dtype,
                )
                label_weights = label_weights / label_weights.sum().clamp_min(1e-6)
                current_centroids.append((label_current * label_weights.unsqueeze(1)).sum(dim=0))
                source_centroids.append((label_source * label_weights.unsqueeze(1)).sum(dim=0))

        current_centroids = F.normalize(torch.stack(current_centroids, dim=0), dim=1)
        source_centroids = F.normalize(torch.stack(source_centroids, dim=0), dim=1)
        sim_current = current_centroids @ current_centroids.transpose(0, 1)
        sim_source = source_centroids @ source_centroids.transpose(0, 1)
        topology_loss = F.mse_loss(sim_current, sim_source)
        return topology_loss, active_classes

    def _adapt_on_batch_ln_vit_legacy_continual(self, x: torch.Tensor) -> bool:
        """Exact 69C-style LN-ViT continual path for ImageNet-C stability."""
        raw_logits = self.model(x)
        raw_probs = raw_logits.softmax(dim=1)
        raw_entropy = softmax_entropy(raw_logits)
        with torch.no_grad():
            source_logits = self.source_model(x)
            source_probs = source_logits.softmax(dim=1)
            source_entropy = softmax_entropy(source_logits)
            probe_used = self._should_run_ln_probe()
            if probe_used:
                plpd_logits = self.model(build_vit_plpd_view(x, patch_len=self.ln_patch_len))
                plpd_probs = plpd_logits.softmax(dim=1)

        top1 = raw_probs.argmax(dim=1, keepdim=True)
        if probe_used:
            plpd_scores = (
                raw_probs.gather(1, top1) - plpd_probs.gather(1, top1)
            ).reshape(-1)
        else:
            plpd_scores = raw_entropy.new_zeros(raw_entropy.shape)
        base_entropy_mask = raw_entropy < self.ln_entropy_margin
        base_final_mask = (
            base_entropy_mask & (plpd_scores > self.ln_plpd_threshold)
            if probe_used else base_entropy_mask
        )
        if self.ln_shared_role_only:
            confidence = 1.0 - (raw_entropy / math.log(max(self.num_classes, 2))).clamp(0.0, 1.0)
            reliability = confidence
            if probe_used:
                reliability = 0.5 * (confidence + plpd_scores.clamp(0.0, 1.0))
            base_final_mask = quantile_selection_mask(
                reliability, self.selection_quantile
            )
            base_entropy_mask = base_final_mask
        base_selected_ratio = float(base_final_mask.float().mean().item())

        ablation_row = self.atlas_ablation_row
        use_all_samples = ablation_row in {"A0", "A2"}
        strict_deyo_selection = ablation_row == "B1"

        calibration_active = (
            not use_all_samples
            and not strict_deyo_selection
            and not self.ln_shared_role_only
            and self.ln_source_calib_steps < self.ln_source_calib_max_steps
            and base_selected_ratio < 0.5
        )
        calibrated_entropy = raw_entropy
        entropy_abs_override = raw_entropy < (
            self.ln_entropy_margin * self.ln_source_entropy_abs_scale
        )
        if calibration_active:
            calibrated_entropy = (
                raw_entropy - self.ln_source_entropy_alpha * source_entropy.detach()
            )
            entropy_mask = (calibrated_entropy < self.ln_entropy_margin) | entropy_abs_override
        else:
            entropy_mask = base_entropy_mask

        entropy_keep_ratio = float(entropy_mask.float().mean().item())
        calibrated_entropy_keep_ratio = (
            float((calibrated_entropy < self.ln_entropy_margin).float().mean().item())
            if calibration_active else entropy_keep_ratio
        )
        entropy_abs_override_ratio = (
            float(entropy_abs_override.float().mean().item())
            if calibration_active else 0.0
        )
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
        entropy_active_classes = (
            int(raw_probs[entropy_mask].detach().argmax(dim=1).unique(sorted=True).numel())
            if int(entropy_mask.sum().item()) > 0 else 0
        )

        if use_all_samples:
            final_mask = torch.ones_like(base_entropy_mask, dtype=torch.bool)
            entropy_active_classes = int(raw_probs.detach().argmax(dim=1).unique(sorted=True).numel())
            selection_mode = "all_samples"
        elif strict_deyo_selection:
            final_mask = base_final_mask
            selection_mode = "deyo_entropy_plpd" if probe_used else "deyo_entropy_source"
        elif self.ln_shared_role_only:
            final_mask = base_final_mask
            selection_mode = "unlabeled_quantile"
        else:
            final_mask = (
                entropy_mask & (plpd_scores > self.ln_plpd_threshold)
                if probe_used else entropy_mask
            )
            selection_mode = "entropy_plpd" if probe_used else "entropy_source"

        selected_ratio = float(final_mask.float().mean().item())
        self.ln_source_calib_steps += 1

        if int(final_mask.sum().item()) == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.last_selected_ratio = 0.0
            self.last_advantage_mean = 0.0
            self.last_ln_entropy_keep_ratio = entropy_keep_ratio
            self.last_ln_plpd_keep_ratio = plpd_keep_ratio
            self.last_ln_selected_ratio = selected_ratio
            self.last_ln_plpd_mean = plpd_mean
            self.last_ln_adadem_loss = 0.0
            self.last_ln_topology_loss = 0.0
            self.last_ln_topology_active_classes = 0
            self.last_ln_topology_weight = 0.0
            self.last_ln_loss_mode = "no_update"
            self.last_ln_active_classes = 0
            self.last_ln_selection_mode = selection_mode
            self.last_ln_entropy_active_classes = entropy_active_classes
            self.last_ln_source_anchor_loss = 0.0
            self.last_ln_param_scope = self.ln_param_scope
            self.last_ln_natural_shift_profile = "none"
            self.last_ln_natural_shift_rescue_active = False
            self.last_ln_deyo_coeff_mean = 0.0
            self.last_ln_deyo_coeff_max = 0.0
            self.last_ln_source_entropy_mean = float(source_entropy.mean().item())
            self.last_ln_entropy_calibration_active = calibration_active
            self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
            self.last_ln_calibrated_entropy_keep_ratio = calibrated_entropy_keep_ratio
            self.last_ln_entropy_abs_override_ratio = entropy_abs_override_ratio
            self.last_ln_base_selected_ratio = base_selected_ratio
            self.last_ln_lowtemp_trigger_reason = "none"
            self.last_ln_lowtemp_plpd_soft_mean = 0.0
            return False

        selected_logits = raw_logits[final_mask]
        selected_probs = raw_probs[final_mask]
        selected_source_probs = source_probs[final_mask]
        active_classes = int(
            selected_probs.detach().argmax(dim=1).unique(sorted=True).numel()
        )

        self.optimizer.zero_grad(set_to_none=True)
        topology_weight = 0.0
        source_anchor_loss = selected_logits.new_zeros(())

        if ablation_row == "A0":
            adadem_loss = selected_logits.new_zeros(())
            topology_loss = selected_logits.new_zeros(())
            topology_active_classes = 0
            loss = softmax_entropy(selected_logits).mean(0)
            loss_mode = "entropy_all"
        elif ablation_row == "A1":
            adadem_loss = selected_logits.new_zeros(())
            topology_loss = selected_logits.new_zeros(())
            topology_active_classes = 0
            loss = softmax_entropy(selected_logits).mean(0)
            loss_mode = "entropy_selected"
        else:
            per_sample_loss = self._compute_ln_adadem_loss(selected_logits, selected_probs)
            adadem_loss = per_sample_loss.mean(0)
            if ablation_row == "B2" or self.ln_shared_role_only:
                topology_loss = selected_logits.new_zeros(())
                topology_active_classes = 0
                loss = adadem_loss
                loss_mode = "adadem_shared_roles" if self.ln_shared_role_only else "adadem_only"
            else:
                topology_loss, topology_active_classes = self._compute_ln_topology_loss(
                    selected_probs,
                    selected_source_probs,
                )
                topology_weight = 0.05
                loss = adadem_loss + topology_weight * topology_loss
                loss_mode = "adadem_topology_all" if ablation_row == "A2" else "adadem_topology"

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if ablation_row != "A0" and ablation_row != "A1":
            self._update_ln_class_template(selected_probs.detach())

        self.update_count += 1
        selected_count = int(final_mask.sum().item())
        self.dig_kept_groups += selected_count
        self.last_selected_ratio = selected_ratio
        self.last_advantage_mean = 0.0
        self.last_ln_entropy_keep_ratio = entropy_keep_ratio
        self.last_ln_plpd_keep_ratio = plpd_keep_ratio
        self.last_ln_selected_ratio = selected_ratio
        self.last_ln_plpd_mean = plpd_mean
        self.last_ln_adadem_loss = float(adadem_loss.detach().item())
        self.last_ln_topology_loss = float(topology_loss.detach().item())
        self.last_ln_topology_active_classes = topology_active_classes
        self.last_ln_topology_weight = topology_weight
        self.last_ln_loss_mode = loss_mode
        self.last_ln_active_classes = active_classes
        self.last_ln_selection_mode = selection_mode
        self.last_ln_entropy_active_classes = entropy_active_classes
        self.last_ln_source_anchor_loss = float(source_anchor_loss.detach().item())
        self.last_ln_param_scope = self.ln_param_scope
        self.last_ln_natural_shift_profile = "none"
        self.last_ln_natural_shift_rescue_active = False
        self.last_ln_deyo_coeff_mean = 0.0
        self.last_ln_deyo_coeff_max = 0.0
        self.last_ln_source_entropy_mean = float(source_entropy.mean().item())
        self.last_ln_entropy_calibration_active = calibration_active
        self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
        self.last_ln_calibrated_entropy_keep_ratio = calibrated_entropy_keep_ratio
        self.last_ln_entropy_abs_override_ratio = entropy_abs_override_ratio
        self.last_ln_base_selected_ratio = base_selected_ratio
        self.last_ln_lowtemp_trigger_reason = "none"
        self.last_ln_lowtemp_plpd_soft_mean = 0.0
        self.last_probe_reliability = 0.0
        self.last_bn_entropy_loss = 0.0
        self.last_bn_consistency_loss = 0.0
        self.source_anchor_agreement = float("nan")
        self.group_reward_std = float("nan")
        self.sos_mean_penalty = float("nan")
        return True

    def _compute_ln_deyo_coeff(
        self,
        entropy: torch.Tensor,
        plpd_scores: torch.Tensor,
    ) -> torch.Tensor:
        coeff = entropy.new_zeros(entropy.shape)
        if self.ln_natural_reweight_ent != 0.0:
            coeff = coeff + self.ln_natural_reweight_ent * torch.exp(
                self.ln_natural_entropy_weight_margin - entropy.detach()
            )
        if self.ln_natural_reweight_plpd != 0.0:
            coeff = coeff + self.ln_natural_reweight_plpd * torch.exp(
                plpd_scores.detach()
            )
        if self.ln_natural_reweight_ent == 0.0 and self.ln_natural_reweight_plpd == 0.0:
            coeff = torch.ones_like(entropy)
        return coeff

    def _compute_ln_bs1_coeff(
        self,
        entropy: torch.Tensor,
        plpd_scores: torch.Tensor,
    ) -> torch.Tensor:
        coeff = entropy.new_zeros(entropy.shape)
        if self.ln_bs1_reweight_ent != 0.0:
            coeff = coeff + self.ln_bs1_reweight_ent * torch.exp(
                self.ln_bs1_entropy_weight_margin - entropy.detach()
            )
        if self.ln_bs1_reweight_plpd != 0.0:
            coeff = coeff + self.ln_bs1_reweight_plpd * torch.exp(
                plpd_scores.detach()
            )
        if self.ln_bs1_reweight_ent == 0.0 and self.ln_bs1_reweight_plpd == 0.0:
            coeff = torch.ones_like(entropy)
        return coeff

    def _adapt_on_batch_ln_vit_bs1(
        self,
        *,
        raw_logits: torch.Tensor,
        raw_probs: torch.Tensor,
        raw_entropy: torch.Tensor,
        source_probs: torch.Tensor,
        source_entropy: torch.Tensor,
        plpd_scores: torch.Tensor,
        probe_used: bool = True,
    ) -> bool:
        entropy_mask = raw_entropy < self.ln_bs1_entropy_margin
        final_mask = (
            entropy_mask & (plpd_scores > self.ln_bs1_plpd_threshold)
            if probe_used else entropy_mask
        )
        selected_ratio = float(final_mask.float().mean().item())
        entropy_keep_ratio = float(entropy_mask.float().mean().item())
        entropy_plpd_mask = (
            plpd_scores[entropy_mask] > self.ln_bs1_plpd_threshold
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
        entropy_active_classes = (
            int(raw_probs[entropy_mask].detach().argmax(dim=1).unique(sorted=True).numel())
            if int(entropy_mask.sum().item()) > 0 else 0
        )
        self.ln_source_calib_steps += 1

        if int(final_mask.sum().item()) == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.last_selected_ratio = 0.0
            self.last_advantage_mean = 0.0
            self.last_ln_entropy_keep_ratio = entropy_keep_ratio
            self.last_ln_plpd_keep_ratio = plpd_keep_ratio
            self.last_ln_selected_ratio = selected_ratio
            self.last_ln_plpd_mean = plpd_mean
            self.last_ln_adadem_loss = 0.0
            self.last_ln_topology_loss = 0.0
            self.last_ln_topology_active_classes = 0
            self.last_ln_topology_weight = 0.0
            self.last_ln_loss_mode = "no_update"
            self.last_ln_active_classes = 0
            self.last_ln_selection_mode = "bs1_entropy_plpd"
            self.last_ln_entropy_active_classes = entropy_active_classes
            self.last_ln_source_anchor_loss = 0.0
            self.last_ln_param_scope = self.ln_param_scope
            self.last_ln_natural_shift_profile = "bs1_safe"
            self.last_ln_natural_shift_rescue_active = False
            self.last_ln_deyo_coeff_mean = 0.0
            self.last_ln_deyo_coeff_max = 0.0
            self.last_ln_source_entropy_mean = float(source_entropy.mean().item())
            self.last_ln_entropy_calibration_active = False
            self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
            self.last_ln_calibrated_entropy_keep_ratio = entropy_keep_ratio
            self.last_ln_entropy_abs_override_ratio = 0.0
            self.last_ln_base_selected_ratio = selected_ratio
            self.last_ln_lowtemp_trigger_reason = "none"
            self.last_ln_lowtemp_plpd_soft_mean = 0.0
            return False

        selected_logits = raw_logits[final_mask]
        selected_probs = raw_probs[final_mask]
        selected_source_probs = source_probs[final_mask]
        selected_entropy = raw_entropy[final_mask]
        selected_plpd_scores = plpd_scores[final_mask]
        active_classes = int(
            selected_probs.detach().argmax(dim=1).unique(sorted=True).numel()
        )

        coeff = self._compute_ln_bs1_coeff(selected_entropy, selected_plpd_scores)
        if not probe_used and self.ln_bs1_reweight_ent != 0.0:
            coeff = self.ln_bs1_reweight_ent * torch.exp(
                self.ln_bs1_entropy_weight_margin - selected_entropy.detach()
            )
        weighted_entropy = selected_entropy * coeff
        deyo_loss = weighted_entropy.mean(0)

        if self.ln_bs1_topology_weight != 0.0:
            topology_loss, topology_active_classes = self._compute_ln_topology_loss(
                selected_probs,
                selected_source_probs,
                sample_weights=coeff.detach(),
            )
        else:
            topology_loss = selected_logits.new_zeros(())
            topology_active_classes = 0

        if self.ln_bs1_source_anchor_weight != 0.0:
            source_anchor_loss = F.kl_div(
                F.log_softmax(selected_logits, dim=1),
                selected_source_probs.detach(),
                reduction="batchmean",
            )
        else:
            source_anchor_loss = selected_logits.new_zeros(())

        self.optimizer.zero_grad(set_to_none=True)
        loss = (
            deyo_loss
            + self.ln_bs1_topology_weight * topology_loss
            + self.ln_bs1_source_anchor_weight * source_anchor_loss
        )
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._update_ln_class_template(selected_probs.detach(), sample_weights=coeff.detach())

        self.update_count += 1
        selected_count = int(final_mask.sum().item())
        self.dig_kept_groups += selected_count
        self.last_selected_ratio = selected_ratio
        self.last_advantage_mean = 0.0
        self.last_ln_entropy_keep_ratio = entropy_keep_ratio
        self.last_ln_plpd_keep_ratio = plpd_keep_ratio
        self.last_ln_selected_ratio = selected_ratio
        self.last_ln_plpd_mean = plpd_mean
        self.last_ln_adadem_loss = float(deyo_loss.detach().item())
        self.last_ln_topology_loss = float(topology_loss.detach().item())
        self.last_ln_topology_active_classes = topology_active_classes
        self.last_ln_topology_weight = self.ln_bs1_topology_weight
        loss_mode_parts = ["bs1_deyo"]
        if self.ln_bs1_topology_weight != 0.0:
            loss_mode_parts.append("topology")
        if self.ln_bs1_source_anchor_weight != 0.0:
            loss_mode_parts.append("source_anchor")
        self.last_ln_loss_mode = "_".join(loss_mode_parts)
        self.last_ln_active_classes = active_classes
        self.last_ln_selection_mode = "bs1_entropy_plpd"
        self.last_ln_entropy_active_classes = entropy_active_classes
        self.last_ln_source_anchor_loss = float(source_anchor_loss.detach().item())
        self.last_ln_param_scope = self.ln_param_scope
        self.last_ln_natural_shift_profile = "bs1_safe"
        self.last_ln_natural_shift_rescue_active = False
        self.last_ln_deyo_coeff_mean = float(coeff.mean().detach().item())
        self.last_ln_deyo_coeff_max = float(coeff.max().detach().item())
        self.last_ln_source_entropy_mean = float(source_entropy.mean().item())
        self.last_ln_entropy_calibration_active = False
        self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
        self.last_ln_calibrated_entropy_keep_ratio = entropy_keep_ratio
        self.last_ln_entropy_abs_override_ratio = 0.0
        self.last_ln_base_selected_ratio = selected_ratio
        self.last_ln_lowtemp_trigger_reason = "none"
        self.last_ln_lowtemp_plpd_soft_mean = 0.0
        self.last_probe_reliability = 0.0
        self.last_bn_entropy_loss = 0.0
        self.last_bn_consistency_loss = 0.0
        self.source_anchor_agreement = float("nan")
        self.group_reward_std = float("nan")
        self.sos_mean_penalty = float("nan")
        return True

    def _adapt_on_batch_ln_vit_legacy_standard(
        self,
        *,
        raw_logits: torch.Tensor,
        raw_probs: torch.Tensor,
        source_probs: torch.Tensor,
        source_entropy: torch.Tensor,
        entropy_mask: torch.Tensor,
        entropy_keep_ratio: float,
        calibrated_entropy_keep_ratio: float,
        entropy_abs_override_ratio: float,
        plpd_scores: torch.Tensor,
        plpd_keep_ratio: float,
        plpd_mean: float,
        base_selected_ratio: float,
        calibration_active: bool,
    ) -> bool:
        entropy_active_classes = (
            int(raw_probs[entropy_mask].detach().argmax(dim=1).unique(sorted=True).numel())
            if int(entropy_mask.sum().item()) > 0 else 0
        )
        if self.ln_shared_role_only:
            confidence = 1.0 - (
                softmax_entropy(raw_logits) / math.log(max(self.num_classes, 2))
            ).clamp(0.0, 1.0)
            reliability = 0.5 * (confidence + plpd_scores.clamp(0.0, 1.0))
            final_mask = quantile_selection_mask(reliability, self.selection_quantile)
            entropy_mask = final_mask
            entropy_keep_ratio = float(final_mask.float().mean().item())
            selection_mode = "unlabeled_quantile"
        else:
            final_mask = entropy_mask & (plpd_scores > self.ln_plpd_threshold)
            selection_mode = "entropy_plpd"
        selected_ratio = float(final_mask.float().mean().item())
        self.ln_source_calib_steps += 1

        if int(final_mask.sum().item()) == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.last_selected_ratio = 0.0
            self.last_advantage_mean = 0.0
            self.last_ln_entropy_keep_ratio = entropy_keep_ratio
            self.last_ln_plpd_keep_ratio = plpd_keep_ratio
            self.last_ln_selected_ratio = selected_ratio
            self.last_ln_plpd_mean = plpd_mean
            self.last_ln_adadem_loss = 0.0
            self.last_ln_topology_loss = 0.0
            self.last_ln_topology_active_classes = 0
            self.last_ln_topology_weight = 0.0
            self.last_ln_loss_mode = "no_update"
            self.last_ln_active_classes = 0
            self.last_ln_selection_mode = selection_mode
            self.last_ln_entropy_active_classes = entropy_active_classes
            self.last_ln_source_anchor_loss = 0.0
            self.last_ln_source_entropy_mean = float(source_entropy.mean().item())
            self.last_ln_entropy_calibration_active = calibration_active
            self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
            self.last_ln_calibrated_entropy_keep_ratio = calibrated_entropy_keep_ratio
            self.last_ln_entropy_abs_override_ratio = entropy_abs_override_ratio
            self.last_ln_base_selected_ratio = base_selected_ratio
            self.last_ln_lowtemp_trigger_reason = "none"
            self.last_ln_lowtemp_plpd_soft_mean = 0.0
            return False

        selected_logits = raw_logits[final_mask]
        selected_probs = raw_probs[final_mask]
        selected_source_probs = source_probs[final_mask]
        active_classes = int(
            selected_probs.detach().argmax(dim=1).unique(sorted=True).numel()
        )
        per_sample_loss = self._compute_ln_adadem_loss(selected_logits, selected_probs)
        adadem_loss = per_sample_loss.mean(0)
        topology_loss, topology_active_classes = self._compute_ln_topology_loss(
            selected_probs,
            selected_source_probs,
        )
        source_anchor_loss = selected_logits.new_zeros(())
        self.optimizer.zero_grad(set_to_none=True)
        loss = adadem_loss + 0.05 * topology_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._update_ln_class_template(selected_probs.detach())

        self.update_count += 1
        selected_count = int(final_mask.sum().item())
        self.dig_kept_groups += selected_count
        self.last_selected_ratio = selected_ratio
        self.last_advantage_mean = 0.0
        self.last_ln_entropy_keep_ratio = entropy_keep_ratio
        self.last_ln_plpd_keep_ratio = plpd_keep_ratio
        self.last_ln_selected_ratio = selected_ratio
        self.last_ln_plpd_mean = plpd_mean
        self.last_ln_adadem_loss = float(adadem_loss.detach().item())
        self.last_ln_topology_loss = float(topology_loss.detach().item())
        self.last_ln_topology_active_classes = topology_active_classes
        self.last_ln_topology_weight = 0.05
        self.last_ln_loss_mode = "adadem_topology"
        self.last_ln_active_classes = active_classes
        self.last_ln_selection_mode = selection_mode
        self.last_ln_entropy_active_classes = entropy_active_classes
        self.last_ln_source_anchor_loss = float(source_anchor_loss.detach().item())
        self.last_ln_source_entropy_mean = float(source_entropy.mean().item())
        self.last_ln_entropy_calibration_active = calibration_active
        self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
        self.last_ln_calibrated_entropy_keep_ratio = calibrated_entropy_keep_ratio
        self.last_ln_entropy_abs_override_ratio = entropy_abs_override_ratio
        self.last_ln_base_selected_ratio = base_selected_ratio
        self.last_ln_lowtemp_trigger_reason = "none"
        self.last_ln_lowtemp_plpd_soft_mean = 0.0
        self.last_probe_reliability = 0.0
        self.last_bn_entropy_loss = 0.0
        self.last_bn_consistency_loss = 0.0
        self.source_anchor_agreement = float("nan")
        self.group_reward_std = float("nan")
        self.sos_mean_penalty = float("nan")
        return True

    def _adapt_on_batch_ln_vit_natural_shift(
        self,
        *,
        raw_logits: torch.Tensor,
        raw_probs: torch.Tensor,
        raw_entropy: torch.Tensor,
        source_probs: torch.Tensor,
        source_entropy: torch.Tensor,
        plpd_scores: torch.Tensor,
        probe_used: bool = True,
    ) -> bool:
        entropy_mask = raw_entropy < self.ln_natural_entropy_margin
        final_mask = (
            entropy_mask & (plpd_scores > self.ln_natural_plpd_threshold)
            if probe_used else entropy_mask
        )
        selected_ratio = float(final_mask.float().mean().item())
        entropy_keep_ratio = float(entropy_mask.float().mean().item())
        entropy_plpd_mask = (
            plpd_scores[entropy_mask] > self.ln_natural_plpd_threshold
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
        entropy_active_classes = (
            int(raw_probs[entropy_mask].detach().argmax(dim=1).unique(sorted=True).numel())
            if int(entropy_mask.sum().item()) > 0 else 0
        )
        self.ln_source_calib_steps += 1

        if int(final_mask.sum().item()) == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.last_selected_ratio = 0.0
            self.last_advantage_mean = 0.0
            self.last_ln_entropy_keep_ratio = entropy_keep_ratio
            self.last_ln_plpd_keep_ratio = plpd_keep_ratio
            self.last_ln_selected_ratio = selected_ratio
            self.last_ln_plpd_mean = plpd_mean
            self.last_ln_adadem_loss = 0.0
            self.last_ln_topology_loss = 0.0
            self.last_ln_topology_active_classes = 0
            self.last_ln_topology_weight = 0.0
            self.last_ln_loss_mode = "no_update"
            self.last_ln_active_classes = 0
            self.last_ln_selection_mode = "entropy_plpd_deyo_bridge"
            self.last_ln_entropy_active_classes = entropy_active_classes
            self.last_ln_source_anchor_loss = 0.0
            self.last_ln_param_scope = self.ln_param_scope
            self.last_ln_natural_shift_profile = "shared"
            self.last_ln_natural_shift_rescue_active = False
            self.last_ln_deyo_coeff_mean = 0.0
            self.last_ln_deyo_coeff_max = 0.0
            self.last_ln_source_entropy_mean = float(source_entropy.mean().item())
            self.last_ln_entropy_calibration_active = False
            self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
            self.last_ln_calibrated_entropy_keep_ratio = entropy_keep_ratio
            self.last_ln_entropy_abs_override_ratio = 0.0
            self.last_ln_base_selected_ratio = selected_ratio
            self.last_ln_lowtemp_trigger_reason = "none"
            self.last_ln_lowtemp_plpd_soft_mean = 0.0
            return False

        selected_logits = raw_logits[final_mask]
        selected_probs = raw_probs[final_mask]
        selected_source_probs = source_probs[final_mask]
        selected_entropy = raw_entropy[final_mask]
        selected_plpd_scores = plpd_scores[final_mask]
        active_classes = int(
            selected_probs.detach().argmax(dim=1).unique(sorted=True).numel()
        )

        coeff = self._compute_ln_deyo_coeff(selected_entropy, selected_plpd_scores)
        if not probe_used and self.ln_natural_reweight_ent != 0.0:
            coeff = self.ln_natural_reweight_ent * torch.exp(
                self.ln_natural_entropy_weight_margin - selected_entropy.detach()
            )
        weighted_entropy = selected_entropy * coeff
        deyo_loss = weighted_entropy.mean(0)

        if self.ln_natural_topology_weight != 0.0:
            topology_loss, topology_active_classes = self._compute_ln_topology_loss(
                selected_probs,
                selected_source_probs,
                sample_weights=coeff.detach(),
            )
        else:
            topology_loss = selected_logits.new_zeros(())
            topology_active_classes = 0

        if self.ln_natural_source_anchor_weight != 0.0:
            source_anchor_loss = F.kl_div(
                F.log_softmax(selected_logits, dim=1),
                selected_source_probs.detach(),
                reduction="batchmean",
            )
        else:
            source_anchor_loss = selected_logits.new_zeros(())

        self.optimizer.zero_grad(set_to_none=True)
        loss = (
            deyo_loss
            + self.ln_natural_topology_weight * topology_loss
            + self.ln_natural_source_anchor_weight * source_anchor_loss
        )
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._update_ln_class_template(selected_probs.detach(), sample_weights=coeff.detach())

        self.update_count += 1
        selected_count = int(final_mask.sum().item())
        self.dig_kept_groups += selected_count
        self.last_selected_ratio = selected_ratio
        self.last_advantage_mean = 0.0
        self.last_ln_entropy_keep_ratio = entropy_keep_ratio
        self.last_ln_plpd_keep_ratio = plpd_keep_ratio
        self.last_ln_selected_ratio = selected_ratio
        self.last_ln_plpd_mean = plpd_mean
        self.last_ln_adadem_loss = float(deyo_loss.detach().item())
        self.last_ln_topology_loss = float(topology_loss.detach().item())
        self.last_ln_topology_active_classes = topology_active_classes
        self.last_ln_topology_weight = self.ln_natural_topology_weight
        loss_mode_parts = ["natural_shift_deyo"]
        if self.ln_natural_topology_weight != 0.0:
            loss_mode_parts.append("topology")
        if self.ln_natural_source_anchor_weight != 0.0:
            loss_mode_parts.append("source_anchor")
        self.last_ln_loss_mode = "_".join(loss_mode_parts)
        self.last_ln_active_classes = active_classes
        self.last_ln_selection_mode = "entropy_plpd_deyo_bridge"
        self.last_ln_entropy_active_classes = entropy_active_classes
        self.last_ln_source_anchor_loss = float(source_anchor_loss.detach().item())
        self.last_ln_param_scope = self.ln_param_scope
        self.last_ln_natural_shift_profile = "shared"
        self.last_ln_natural_shift_rescue_active = False
        self.last_ln_deyo_coeff_mean = float(coeff.mean().detach().item())
        self.last_ln_deyo_coeff_max = float(coeff.max().detach().item())
        self.last_ln_source_entropy_mean = float(source_entropy.mean().item())
        self.last_ln_entropy_calibration_active = False
        self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
        self.last_ln_calibrated_entropy_keep_ratio = entropy_keep_ratio
        self.last_ln_entropy_abs_override_ratio = 0.0
        self.last_ln_base_selected_ratio = selected_ratio
        self.last_ln_lowtemp_trigger_reason = "none"
        self.last_ln_lowtemp_plpd_soft_mean = 0.0
        self.last_probe_reliability = 0.0
        self.last_bn_entropy_loss = 0.0
        self.last_bn_consistency_loss = 0.0
        self.source_anchor_agreement = float("nan")
        self.group_reward_std = float("nan")
        self.sos_mean_penalty = float("nan")
        return True

    def _adapt_on_batch_ln_vit(self, x: torch.Tensor) -> bool:
        if (
            self._ln_continual_mode
            and not self.ln_natural_shift_enabled
            and not self.ln_bs1_loss_enabled
        ):
            return self._adapt_on_batch_ln_vit_legacy_continual(x)

        raw_logits = self.model(x)
        raw_probs = raw_logits.softmax(dim=1)
        raw_entropy = softmax_entropy(raw_logits)
        with torch.no_grad():
            source_logits = self.source_model(x)
            source_probs = source_logits.softmax(dim=1)
            source_entropy = softmax_entropy(source_logits)

        corruption_key = str(self.current_corruption).lower()
        snow_tuned_standard = (
            self.use_branch_refinements
            and self._ln_standard_mode
            and corruption_key == "snow"
        )
        effective_entropy_margin = self.ln_snow_entropy_margin if snow_tuned_standard else self.ln_entropy_margin
        effective_plpd_threshold = self.ln_snow_plpd_threshold if snow_tuned_standard else self.ln_plpd_threshold
        effective_patch_len = (
            self.ln_bs1_patch_len
            if self.ln_bs1_loss_enabled
            else self.ln_natural_patch_len
            if self.ln_natural_shift_enabled
            else (self.ln_snow_patch_len if snow_tuned_standard else self.ln_patch_len)
        )
        with torch.no_grad():
            probe_used = self._should_run_ln_probe()
            if probe_used:
                plpd_logits = self.model(build_vit_plpd_view(x, patch_len=effective_patch_len))
                plpd_probs = plpd_logits.softmax(dim=1)

        top1 = raw_probs.argmax(dim=1, keepdim=True)
        if probe_used:
            plpd_scores = (
                raw_probs.gather(1, top1) - plpd_probs.gather(1, top1)
            ).reshape(-1)
        else:
            plpd_scores = raw_entropy.new_zeros(raw_entropy.shape)
        if self.ln_natural_shift_enabled:
            return self._adapt_on_batch_ln_vit_natural_shift(
                raw_logits=raw_logits,
                raw_probs=raw_probs,
                raw_entropy=raw_entropy,
                source_probs=source_probs,
                source_entropy=source_entropy,
                plpd_scores=plpd_scores,
                probe_used=probe_used,
            )
        if self.ln_bs1_loss_enabled:
            return self._adapt_on_batch_ln_vit_bs1(
                raw_logits=raw_logits,
                raw_probs=raw_probs,
                raw_entropy=raw_entropy,
                source_probs=source_probs,
                source_entropy=source_entropy,
                plpd_scores=plpd_scores,
                probe_used=probe_used,
            )
        base_entropy_mask = raw_entropy < effective_entropy_margin
        base_final_mask = base_entropy_mask & (plpd_scores > effective_plpd_threshold)
        base_selected_ratio = float(base_final_mask.float().mean().item())

        calibration_active = (
            not self.ln_shared_role_only
            and self.ln_source_calib_steps < self.ln_source_calib_max_steps
            and base_selected_ratio < 0.5
        )
        calibrated_entropy = raw_entropy
        entropy_abs_override = raw_entropy < (effective_entropy_margin * self.ln_source_entropy_abs_scale)
        if calibration_active:
            calibrated_entropy = raw_entropy - self.ln_source_entropy_alpha * source_entropy.detach()
            entropy_mask = (calibrated_entropy < effective_entropy_margin) | entropy_abs_override
        else:
            entropy_mask = base_entropy_mask
        entropy_keep_ratio = float(entropy_mask.float().mean().item())
        calibrated_entropy_keep_ratio = (
            float((calibrated_entropy < effective_entropy_margin).float().mean().item())
            if calibration_active else entropy_keep_ratio
        )
        entropy_abs_override_ratio = (
            float(entropy_abs_override.float().mean().item())
            if calibration_active else 0.0
        )
        entropy_plpd_mask = plpd_scores[entropy_mask] > effective_plpd_threshold
        plpd_keep_ratio = (
            float(entropy_plpd_mask.float().mean().item())
            if entropy_plpd_mask.numel() > 0 else 0.0
        )
        plpd_mean = (
            float(plpd_scores[entropy_mask].mean().item())
            if int(entropy_mask.sum().item()) > 0 else 0.0
        )

        entropy_active_classes = (
            int(raw_probs[entropy_mask].detach().argmax(dim=1).unique(sorted=True).numel())
            if int(entropy_mask.sum().item()) > 0 else 0
        )
        if self._ln_standard_mode and corruption_key != "snow":
            return self._adapt_on_batch_ln_vit_legacy_standard(
                raw_logits=raw_logits,
                raw_probs=raw_probs,
                source_probs=source_probs,
                source_entropy=source_entropy,
                entropy_mask=entropy_mask,
                entropy_keep_ratio=entropy_keep_ratio,
                calibrated_entropy_keep_ratio=calibrated_entropy_keep_ratio,
                entropy_abs_override_ratio=entropy_abs_override_ratio,
                plpd_scores=plpd_scores,
                plpd_keep_ratio=plpd_keep_ratio,
                plpd_mean=plpd_mean,
                base_selected_ratio=base_selected_ratio,
                calibration_active=calibration_active,
            )
        lowtemp_eligible = (
            self.use_branch_refinements
            and self._ln_standard_mode
            and corruption_key == "snow"
            and "snow" in self.ln_lowtemp_corruptions
        )
        if self.ln_lowtemp_branch_active:
            lowtemp_active = True
        elif lowtemp_eligible and not self.ln_lowtemp_decision_locked:
            collapse_reasons = []
            if corruption_key == "snow" and self.ln_lowtemp_snow_always:
                collapse_reasons.append("snow_always")
            elif self.ln_lowtemp_probe_updates < self.ln_lowtemp_collapse_probe_steps:
                if base_selected_ratio < self.ln_lowtemp_collapse_base_selected_ratio:
                    collapse_reasons.append("low_base_selected_ratio")
                if entropy_active_classes < self.ln_lowtemp_collapse_entropy_active_classes:
                    collapse_reasons.append("low_entropy_active_classes")
            if collapse_reasons:
                self.ln_lowtemp_branch_active = True
                self.ln_lowtemp_phase = "phase1"
                self.ln_lowtemp_decision_locked = True
                self.ln_lowtemp_trigger_reason = "+".join(collapse_reasons)
            else:
                self.ln_lowtemp_probe_updates += 1
                if self.ln_lowtemp_probe_updates >= self.ln_lowtemp_collapse_probe_steps:
                    self.ln_lowtemp_decision_locked = True
                    self.ln_lowtemp_trigger_reason = "probe_clear"
            lowtemp_active = self.ln_lowtemp_branch_active
        else:
            lowtemp_active = self.ln_lowtemp_branch_active

        if lowtemp_active:
            with torch.no_grad():
                lowtemp_plpd_logits = self.model(
                    build_vit_plpd_view(x, patch_len=self.ln_lowtemp_patch_len)
                )
                lowtemp_plpd_probs = lowtemp_plpd_logits.softmax(dim=1)

            lowtemp_phase = self.ln_lowtemp_phase
            lowtemp_plpd_scores = (
                raw_probs.gather(1, top1) - lowtemp_plpd_probs.gather(1, top1)
            ).reshape(-1)
            phase_entropy_mask = raw_entropy < self.ln_lowtemp_phase1_entropy_margin
            selected_ratio = float(phase_entropy_mask.float().mean().item())
            plpd_keep_ratio = (
                float((lowtemp_plpd_scores[phase_entropy_mask] > effective_plpd_threshold).float().mean().item())
                if int(phase_entropy_mask.sum().item()) > 0 else 0.0
            )
            plpd_mean = (
                float(lowtemp_plpd_scores[phase_entropy_mask].mean().item())
                if int(phase_entropy_mask.sum().item()) > 0 else 0.0
            )
            entropy_active_classes = (
                int(raw_probs[phase_entropy_mask].detach().argmax(dim=1).unique(sorted=True).numel())
                if int(phase_entropy_mask.sum().item()) > 0 else 0
            )
            selection_mode = (
                "entropy_only"
                if lowtemp_phase == "phase1"
                else "entropy_soft_plpd"
            )
            self.ln_source_calib_steps += 1

            if int(phase_entropy_mask.sum().item()) == 0:
                self.optimizer.zero_grad(set_to_none=True)
                self.last_selected_ratio = 0.0
                self.last_advantage_mean = 0.0
                self.last_ln_entropy_keep_ratio = selected_ratio
                self.last_ln_plpd_keep_ratio = plpd_keep_ratio
                self.last_ln_selected_ratio = selected_ratio
                self.last_ln_plpd_mean = plpd_mean
                self.last_ln_adadem_loss = 0.0
                self.last_ln_topology_loss = 0.0
                self.last_ln_topology_active_classes = 0
                self.last_ln_topology_weight = 0.0
                self.last_ln_loss_mode = "no_update"
                self.last_ln_active_classes = 0
                self.last_ln_selection_mode = selection_mode
                self.last_ln_entropy_active_classes = entropy_active_classes
                self.last_ln_source_anchor_loss = 0.0
                self.last_ln_source_entropy_mean = float(source_entropy.mean().item())
                self.last_ln_entropy_calibration_active = False
                self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
                self.last_ln_calibrated_entropy_keep_ratio = selected_ratio
                self.last_ln_entropy_abs_override_ratio = 0.0
                self.last_ln_base_selected_ratio = base_selected_ratio
                self.last_ln_lowtemp_trigger_reason = self.ln_lowtemp_trigger_reason
                self.last_ln_lowtemp_plpd_soft_mean = 0.0
                return False

            selected_logits = raw_logits[phase_entropy_mask]
            selected_probs = raw_probs[phase_entropy_mask]
            selected_source_probs = source_probs[phase_entropy_mask]
            selected_lowtemp_plpd_scores = lowtemp_plpd_scores[phase_entropy_mask]
            active_classes = int(
                selected_probs.detach().argmax(dim=1).unique(sorted=True).numel()
            )
            self.optimizer.zero_grad(set_to_none=True)

            if lowtemp_phase == "phase1":
                adadem_loss = selected_logits.new_zeros(())
                topology_loss = selected_logits.new_zeros(())
                topology_active_classes = 0
                plpd_soft_mean = 0.0
                source_anchor_loss = F.kl_div(
                    F.log_softmax(selected_logits, dim=1),
                    selected_source_probs.detach(),
                    reduction="batchmean",
                )
                loss = softmax_entropy(selected_logits).mean(0) + (
                    self.ln_lowtemp_phase1_source_anchor_weight * source_anchor_loss
                )
                loss_mode = "entropy_source_anchor"
                template_weights = None
                self.ln_lowtemp_phase1_updates += 1
                if (
                    selected_ratio >= self.ln_lowtemp_transition_selected_ratio
                    and entropy_active_classes >= self.ln_lowtemp_transition_active_classes
                ):
                    self.ln_lowtemp_transition_streak += 1
                else:
                    self.ln_lowtemp_transition_streak = 0
                if (
                    self.ln_lowtemp_phase1_updates >= self.ln_lowtemp_phase1_min_steps
                    and self.ln_lowtemp_transition_streak >= self.ln_lowtemp_transition_patience
                ):
                    self.ln_lowtemp_phase = "phase2"
            else:
                plpd_soft = torch.sigmoid(
                    (selected_lowtemp_plpd_scores - effective_plpd_threshold)
                    / max(self.ln_lowtemp_phase2_plpd_temperature, 1e-6)
                ).detach()
                template_weights = plpd_soft
                per_sample_loss = self._compute_ln_adadem_loss(selected_logits, selected_probs)
                adadem_loss = (
                    (per_sample_loss * plpd_soft).sum()
                    / plpd_soft.sum().clamp_min(1e-6)
                )
                topology_loss, topology_active_classes = self._compute_ln_topology_loss(
                    selected_probs,
                    selected_source_probs,
                )
                source_anchor_loss = F.kl_div(
                    F.log_softmax(selected_logits, dim=1),
                    selected_source_probs.detach(),
                    reduction="batchmean",
                )
                loss = (
                    adadem_loss
                    + 0.05 * topology_loss
                    + self.ln_lowtemp_phase2_source_anchor_weight * source_anchor_loss
                )
                loss_mode = "adadem_topology_source_anchor"
                plpd_soft_mean = float(plpd_soft.mean().item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if template_weights is not None:
                self._update_ln_class_template(
                    selected_probs.detach(),
                    sample_weights=template_weights,
                )

            self.update_count += 1
            selected_count = int(phase_entropy_mask.sum().item())
            self.dig_kept_groups += selected_count
            self.last_selected_ratio = selected_ratio
            self.last_advantage_mean = 0.0
            self.last_ln_entropy_keep_ratio = selected_ratio
            self.last_ln_plpd_keep_ratio = plpd_keep_ratio
            self.last_ln_selected_ratio = selected_ratio
            self.last_ln_plpd_mean = plpd_mean
            self.last_ln_adadem_loss = float(adadem_loss.detach().item())
            self.last_ln_topology_loss = float(topology_loss.detach().item())
            self.last_ln_topology_active_classes = topology_active_classes
            self.last_ln_topology_weight = 0.0 if lowtemp_phase == "phase1" else 0.05
            self.last_ln_loss_mode = loss_mode
            self.last_ln_active_classes = active_classes
            self.last_ln_selection_mode = selection_mode
            self.last_ln_entropy_active_classes = entropy_active_classes
            self.last_ln_source_anchor_loss = float(source_anchor_loss.detach().item())
            self.last_ln_source_entropy_mean = float(source_entropy.mean().item())
            self.last_ln_entropy_calibration_active = False
            self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
            self.last_ln_calibrated_entropy_keep_ratio = selected_ratio
            self.last_ln_entropy_abs_override_ratio = 0.0
            self.last_ln_base_selected_ratio = base_selected_ratio
            self.last_ln_lowtemp_trigger_reason = self.ln_lowtemp_trigger_reason
            self.last_ln_lowtemp_plpd_soft_mean = plpd_soft_mean
            self.last_probe_reliability = 0.0
            self.last_bn_entropy_loss = 0.0
            self.last_bn_consistency_loss = 0.0
            self.source_anchor_agreement = float("nan")
            self.group_reward_std = float("nan")
            self.sos_mean_penalty = float("nan")
            return True

        warmup_entropy_mask = raw_entropy < self.ln_standard_warmup_entropy_margin
        warmup_entropy_active_classes = (
            int(raw_probs[warmup_entropy_mask].detach().argmax(dim=1).unique(sorted=True).numel())
            if int(warmup_entropy_mask.sum().item()) > 0 else 0
        )
        warmup_ready = (
            self.ln_standard_warmup_updates >= self.ln_standard_plpd_warmup_steps
            and self.ln_standard_recovery_streak >= self.ln_standard_recover_patience
        )
        use_entropy_only_selection = (
            self._ln_standard_mode
            and (
                not warmup_ready
                or base_selected_ratio < self.ln_standard_base_selected_floor
            )
        )

        if use_entropy_only_selection:
            final_mask = warmup_entropy_mask
            selection_mode = "entropy_only"
            entropy_active_classes = warmup_entropy_active_classes
        else:
            final_mask = entropy_mask & (plpd_scores > effective_plpd_threshold)
            selection_mode = "entropy_plpd"
        selected_ratio = float(final_mask.float().mean().item())
        self.ln_source_calib_steps += 1
        if self._ln_standard_mode and use_entropy_only_selection:
            self.ln_standard_warmup_updates += 1
            if (
                selected_ratio >= self.ln_standard_recover_selected_ratio
                and entropy_active_classes >= self.ln_standard_recover_active_classes
            ):
                self.ln_standard_recovery_streak += 1
            else:
                self.ln_standard_recovery_streak = 0

        if int(final_mask.sum().item()) == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.last_selected_ratio = 0.0
            self.last_advantage_mean = 0.0
            self.last_ln_entropy_keep_ratio = entropy_keep_ratio
            self.last_ln_plpd_keep_ratio = plpd_keep_ratio
            self.last_ln_selected_ratio = selected_ratio
            self.last_ln_plpd_mean = plpd_mean
            self.last_ln_adadem_loss = 0.0
            self.last_ln_topology_loss = 0.0
            self.last_ln_topology_active_classes = 0
            self.last_ln_topology_weight = 0.0
            self.last_ln_loss_mode = "no_update"
            self.last_ln_active_classes = 0
            self.last_ln_selection_mode = selection_mode
            self.last_ln_entropy_active_classes = entropy_active_classes
            self.last_ln_source_anchor_loss = 0.0
            self.last_ln_source_entropy_mean = float(source_entropy.mean().item())
            self.last_ln_entropy_calibration_active = calibration_active
            self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
            self.last_ln_calibrated_entropy_keep_ratio = calibrated_entropy_keep_ratio
            self.last_ln_entropy_abs_override_ratio = entropy_abs_override_ratio
            self.last_ln_base_selected_ratio = base_selected_ratio
            self.last_ln_lowtemp_trigger_reason = self.ln_lowtemp_trigger_reason
            self.last_ln_lowtemp_plpd_soft_mean = 0.0
            return False

        selected_logits = raw_logits[final_mask]
        selected_probs = raw_probs[final_mask]
        selected_source_probs = source_probs[final_mask]
        active_classes = int(
            selected_probs.detach().argmax(dim=1).unique(sorted=True).numel()
        )
        self.optimizer.zero_grad(set_to_none=True)
        if use_entropy_only_selection:
            adadem_loss = selected_logits.new_zeros(())
            topology_loss = selected_logits.new_zeros(())
            topology_active_classes = 0
            source_anchor_loss = F.kl_div(
                F.log_softmax(selected_logits, dim=1),
                selected_source_probs.detach(),
                reduction="batchmean",
            )
            loss = softmax_entropy(selected_logits).mean(0) + (
                self.ln_standard_source_anchor_weight * source_anchor_loss
            )
            loss_mode = "entropy_source_anchor"
        else:
            per_sample_loss = self._compute_ln_adadem_loss(selected_logits, selected_probs)
            adadem_loss = per_sample_loss.mean(0)
            topology_loss, topology_active_classes = self._compute_ln_topology_loss(
                selected_probs,
                selected_source_probs,
            )
            source_anchor_loss = selected_logits.new_zeros(())
            loss = adadem_loss + 0.05 * topology_loss
            loss_mode = "adadem_topology"
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if not use_entropy_only_selection:
            self._update_ln_class_template(selected_probs.detach())

        self.update_count += 1
        selected_count = int(final_mask.sum().item())
        self.dig_kept_groups += selected_count
        self.last_selected_ratio = selected_ratio
        self.last_advantage_mean = 0.0
        self.last_ln_entropy_keep_ratio = entropy_keep_ratio
        self.last_ln_plpd_keep_ratio = plpd_keep_ratio
        self.last_ln_selected_ratio = selected_ratio
        self.last_ln_plpd_mean = plpd_mean
        self.last_ln_adadem_loss = float(adadem_loss.detach().item())
        self.last_ln_topology_loss = float(topology_loss.detach().item())
        self.last_ln_topology_active_classes = topology_active_classes
        self.last_ln_topology_weight = 0.0 if use_entropy_only_selection else 0.05
        self.last_ln_loss_mode = loss_mode
        self.last_ln_active_classes = active_classes
        self.last_ln_selection_mode = selection_mode
        self.last_ln_entropy_active_classes = entropy_active_classes
        self.last_ln_source_anchor_loss = float(source_anchor_loss.detach().item())
        self.last_ln_source_entropy_mean = float(source_entropy.mean().item())
        self.last_ln_entropy_calibration_active = calibration_active
        self.last_ln_entropy_calibration_alpha = self.ln_source_entropy_alpha
        self.last_ln_calibrated_entropy_keep_ratio = calibrated_entropy_keep_ratio
        self.last_ln_entropy_abs_override_ratio = entropy_abs_override_ratio
        self.last_ln_base_selected_ratio = base_selected_ratio
        self.last_ln_lowtemp_trigger_reason = self.ln_lowtemp_trigger_reason
        self.last_ln_lowtemp_plpd_soft_mean = 0.0
        self.last_probe_reliability = 0.0
        self.last_bn_entropy_loss = 0.0
        self.last_bn_consistency_loss = 0.0
        self.source_anchor_agreement = float("nan")
        self.group_reward_std = float("nan")
        self.sos_mean_penalty = float("nan")
        return True

    def _adapt_on_batch_bn_continual(
        self,
        x: torch.Tensor,
        raw_logits: torch.Tensor,
        group_std: torch.Tensor,
        sos_weight: torch.Tensor,
        pseudo_targets: torch.Tensor,
        source_logits: torch.Tensor,
        raw_advantage: torch.Tensor,
        aux_info: Dict[str, torch.Tensor],
    ) -> bool:
        self._ensure_bn_continual_state(raw_logits.device, raw_logits.dtype)

        raw_probs = aux_info["raw_probs"]
        source_probs = aux_info["source_probs"]
        raw_pred = aux_info["raw_pred"]
        source_pred = aux_info["source_pred"]
        destroy_sensitivity = aux_info["destroy_sensitivity"]
        raw_entropy = aux_info["raw_entropy"]
        entropy_score = aux_info["entropy_score"]

        informative_mask = (
            group_std > self.dig_config["std_threshold"]
            if self.dig_enabled else torch.ones_like(group_std, dtype=torch.bool)
        )
        confidence_mask = entropy_score < self.bn_continual_selection_margin
        plpd_mask = destroy_sensitivity > self.bn_continual_plpd_margin
        final_mask = informative_mask & (sos_weight > 0.0) & confidence_mask & plpd_mask

        if final_mask.sum() == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.last_bn_continual_plpd = 0.0
            self.last_bn_continual_harmful = 0.0
            self.last_bn_continual_veto = 1.0
            self.last_bn_continual_anchor = 0.0
            self.last_bn_continual_batch_alignment = 0.0
            self.last_bn_continual_balance = 0.0
            self.last_bn_continual_arc_guard = 1.0
            self.last_bn_continual_debias = 1.0
            self.last_bn_continual_d3lr_alpha = 1.0
            self.last_bn_continual_d3lr_driver = "none"
            self.last_bn_continual_rescue = 0.0
            self.last_bn_continual_rescue_plpd_edge = 0.0
            self.last_bn_continual_rescue_anchor_edge = 0.0
            return False

        selected_logits = raw_logits[final_mask]
        selected_source_logits = source_logits[final_mask]
        selected_targets = pseudo_targets[final_mask]
        selected_advantage = raw_advantage[final_mask].clamp_min(0.0)
        selected_weights = sos_weight[final_mask]
        selected_probs = raw_probs[final_mask]
        selected_source_probs = source_probs[final_mask]
        selected_entropy = entropy_score[final_mask]
        selected_raw_entropy = raw_entropy[final_mask]
        selected_plpd = destroy_sensitivity[final_mask]
        selected_anchor = (raw_pred[final_mask] == source_pred[final_mask]).float()
        selected_source_support = selected_source_probs.gather(
            1,
            selected_targets.unsqueeze(1),
        ).squeeze(1).clamp_min(1e-6)

        prior = self.bn_continual_class_prior[selected_targets].clamp_min(1e-6)
        balance_weight = torch.pow(
            1.0 / (prior * float(self.num_classes)),
            self.bn_continual_prior_power,
        ).clamp_(self.bn_continual_balance_floor, 2.25)
        class_template = self.bn_continual_class_template[selected_targets]
        template_match = F.cosine_similarity(
            selected_probs.detach(),
            class_template,
            dim=1,
        ).clamp(-1.0, 1.0)
        harmful_score = (
            0.35 * selected_entropy
            + 0.35 * (1.0 - selected_plpd)
            + 0.20 * (1.0 - selected_source_support)
            + 0.10 * (1.0 - selected_anchor)
        ).clamp(0.0, 1.0)
        harmful_gate = (
            (harmful_score.detach() - self.bn_continual_arc_safe_margin)
            / max(1e-6, 1.0 - self.bn_continual_arc_safe_margin)
        ).clamp(0.0, 1.0)
        trusted_repetitive = (
            harmful_gate
            * template_match.clamp_min(0.0)
            * selected_source_support
            * (0.5 + 0.5 * selected_anchor)
        )
        trusted_hard = (
            harmful_gate
            * selected_source_support
            * (0.5 + 0.5 * selected_anchor)
            * (0.5 + 0.5 * selected_plpd.detach())
            * (0.35 + 0.65 * (1.0 - template_match.clamp(0.0, 1.0)))
        )
        debias_weight = (
            1.15
            - 0.50 * template_match * selected_source_support * (0.5 + 0.5 * selected_anchor)
            - self.bn_continual_trusted_template_debias * trusted_repetitive
        ).clamp_(0.55, 1.15)
        diversity_weight, batch_alignment = self._compute_bn_continual_diversity(selected_probs)

        entropy_weight = torch.exp(
            (
                self.bn_continual_reweight_margin * math.log(max(self.num_classes, 2))
                - selected_raw_entropy.detach()
            ).clamp(min=-1.0, max=1.0)
        )
        plpd_drive = (
            (selected_plpd.detach() - self.bn_continual_plpd_margin)
            / max(1e-6, 1.0 - self.bn_continual_plpd_margin)
        ).clamp(min=0.0, max=1.0)
        plpd_focus = (
            plpd_drive
            + self.bn_continual_plpd_sharpness
            * plpd_drive
            * (plpd_drive - 0.5).clamp_min(0.0)
        ).clamp(min=0.0, max=1.15)
        plpd_weight = torch.exp(
            plpd_focus
        )
        core_reweight = (
            self.bn_continual_core_entropy_mix * entropy_weight
            + self.bn_continual_core_plpd_mix * plpd_weight
        )
        anchor_weight = 1.0 + self.bn_continual_anchor_weight * selected_anchor
        support_weight = (0.75 + 0.25 * selected_source_support).clamp_(0.75, 1.0)
        trusted_boost = 1.0 + self.bn_continual_trusted_hard_boost * trusted_hard

        selected_weights = (
            selected_weights
            * core_reweight
            * balance_weight
            * debias_weight
            * diversity_weight
            * anchor_weight
            * support_weight
            * trusted_boost
        )
        selected_weights = selected_weights / selected_weights.mean().detach().clamp_min(1e-4)
        selected_weights = selected_weights.clamp_(0.15, 3.0)

        harmful_mean = harmful_score.mean()
        anchor_mean = selected_anchor.mean()
        plpd_mean = selected_plpd.mean()
        rescue_drive = (
            (selected_plpd.detach() - self.bn_continual_rescue_plpd_margin)
            / max(1e-6, 1.0 - self.bn_continual_rescue_plpd_margin)
        ).clamp(0.0, 1.0)
        support_gap = (
            (self.bn_continual_rescue_support_ceiling - selected_source_support)
            / max(1e-6, self.bn_continual_rescue_support_ceiling)
        ).clamp(0.0, 1.0)
        anchor_gap = (1.0 - selected_anchor).clamp(0.0, 1.0)
        entropy_ok = (
            (self.bn_continual_rescue_entropy_ceiling - selected_entropy)
            / max(1e-6, self.bn_continual_rescue_entropy_ceiling)
        ).clamp(0.0, 1.0)
        rescue_score = rescue_drive * entropy_ok * (0.65 * support_gap + 0.35 * anchor_gap)
        rescue_batch = rescue_score.mean()
        selected_ratio = float(final_mask.sum().item()) / max(1, x.shape[0])
        batch_alignment_value = float(batch_alignment.detach().item())
        harmful_value_raw = harmful_mean.detach().item()
        drifted_batch_raw = batch_alignment_value > self.bn_continual_batch_alignment_margin
        weak_plpd_raw = plpd_mean.detach().item() < self.bn_continual_plpd_veto_margin
        weak_anchor_raw = anchor_mean.detach().item() < self.bn_continual_anchor_margin
        very_weak_anchor_raw = anchor_mean.detach().item() < self.bn_continual_hard_anchor_margin
        harmful_edge = (
            (harmful_mean - self.bn_continual_veto_margin) / 0.06
        ).clamp(0.0, 1.0)
        plpd_edge = (
            (self.bn_continual_plpd_veto_margin + 0.05 - plpd_mean) / 0.07
        ).clamp(0.0, 1.0)
        hard_anchor_edge = (
            (self.bn_continual_hard_anchor_margin + 0.04 - anchor_mean) / 0.08
        ).clamp(0.0, 1.0)
        soft_anchor_edge = (
            (self.bn_continual_anchor_margin + 0.04 - anchor_mean) / 0.10
        ).clamp(0.0, 1.0)
        plpd_edge_batch = rescue_batch * float(
            harmful_value_raw > self.bn_continual_veto_margin and drifted_batch_raw and weak_plpd_raw
        ) * (0.70 * plpd_edge + 0.30 * harmful_edge)
        hard_anchor_edge_batch = rescue_batch * float(
            harmful_value_raw > self.bn_continual_veto_margin and drifted_batch_raw and very_weak_anchor_raw
        ) * (0.70 * hard_anchor_edge + 0.30 * harmful_edge)
        soft_anchor_edge_batch = rescue_batch * float(
            harmful_value_raw > (self.bn_continual_veto_margin + 0.07) and weak_anchor_raw
        ) * (0.60 * soft_anchor_edge + 0.40 * harmful_edge)
        effective_plpd_mean = plpd_mean + 0.18 * plpd_edge_batch
        effective_hard_anchor_mean = (anchor_mean + 0.14 * hard_anchor_edge_batch).clamp(max=1.0)
        effective_soft_anchor_mean = (anchor_mean + 0.10 * soft_anchor_edge_batch).clamp(max=1.0)
        weak_plpd = effective_plpd_mean.detach().item() < self.bn_continual_plpd_veto_margin
        weak_anchor = effective_soft_anchor_mean.detach().item() < self.bn_continual_anchor_margin
        very_weak_anchor = (
            effective_hard_anchor_mean.detach().item() < self.bn_continual_hard_anchor_margin
        )
        drifted_batch = drifted_batch_raw
        harmful_value = harmful_value_raw

        arc_guard = torch.ones_like(harmful_score)
        if self.bn_continual_arc_hard_margin > self.bn_continual_arc_safe_margin:
            scaled_guard = 1.0 - (
                (harmful_score - self.bn_continual_arc_safe_margin)
                / (self.bn_continual_arc_hard_margin - self.bn_continual_arc_safe_margin)
            ).clamp(0.0, 1.0) * (1.0 - self.bn_continual_arc_floor)
            arc_guard = torch.minimum(arc_guard, scaled_guard)
        arc_guard = arc_guard.clamp_(self.bn_continual_arc_floor, 1.0)

        veto_update = (
            selected_ratio < self.bn_continual_min_fraction
            or (
                harmful_value_raw > self.bn_continual_veto_margin
                and (
                    (weak_plpd and drifted_batch_raw)
                    or (very_weak_anchor and drifted_batch_raw)
                    or (harmful_value_raw > (self.bn_continual_veto_margin + 0.07) and weak_anchor)
                )
            )
        )

        self.last_bn_continual_plpd = float(plpd_mean.detach().item())
        self.last_bn_continual_harmful = float(harmful_mean.detach().item())
        self.last_bn_continual_veto = 1.0 if veto_update else 0.0
        self.last_bn_continual_anchor = float(anchor_mean.detach().item())
        self.last_bn_continual_batch_alignment = batch_alignment_value
        self.last_bn_continual_balance = float(balance_weight.mean().detach().item())
        self.last_bn_continual_arc_guard = float(arc_guard.mean().detach().item())
        self.last_bn_continual_debias = float(debias_weight.mean().detach().item())
        self.last_bn_continual_rescue = float(rescue_batch.detach().item())
        self.last_bn_continual_rescue_plpd_edge = float(plpd_edge_batch.detach().item())
        self.last_bn_continual_rescue_anchor_edge = float(
            torch.maximum(hard_anchor_edge_batch, soft_anchor_edge_batch).detach().item()
        )

        if veto_update:
            self.optimizer.zero_grad(set_to_none=True)
            self._update_bn_continual_history(selected_probs.detach())
            self.last_bn_continual_d3lr_alpha = 1.0
            self.last_bn_continual_d3lr_driver = "none"
            return False
        self.last_bn_continual_d3lr_alpha = 1.0
        self.last_bn_continual_d3lr_driver = "none"

        target_prob = selected_probs.gather(1, selected_targets.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        ref_prob = selected_source_probs.gather(1, selected_targets.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        ratio = target_prob / ref_prob
        ratio = 1.0 + (ratio - 1.0) * arc_guard
        selected_advantage = selected_advantage * arc_guard
        selected_weights = selected_weights * (0.75 + 0.25 * arc_guard)

        surrogate = arc_surrogate(
            ratio=ratio,
            advantage=selected_advantage,
            eps_low=self.arc_config["eps_low"],
            eps_high=self.arc_config["eps_high"],
            enabled=self.arc_enabled,
        )
        loss = -(selected_weights * surrogate).sum()
        loss = loss / float(
            active_unit_count(
                batch_size=int(final_mask.sum().item()),
                mode=self.uan_mode,
                token_count=self.token_count,
            )
        )

        anchor_loss = consistency_cross_entropy(selected_logits, selected_source_logits.detach())
        loss = loss + 0.02 * self.num_classes * harmful_mean.detach() * anchor_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self._update_bn_continual_history(selected_probs.detach())

        self.update_count += 1
        selected_count = int(final_mask.sum().item())
        self.dig_kept_groups += selected_count
        self.last_selected_ratio = selected_ratio
        self.last_advantage_mean = float(selected_advantage.mean().detach().item())
        self.last_probe_reliability = 0.0
        self.last_bn_entropy_loss = 0.0
        self.last_bn_consistency_loss = float(anchor_loss.detach().item())
        source_pred_selected = selected_source_logits.argmax(dim=1)
        raw_pred_selected = selected_logits.detach().argmax(dim=1)
        anchor_agree = float((source_pred_selected == raw_pred_selected).float().mean().item())
        reward_std_mean = float(group_std[final_mask].mean().detach().item())
        sos_penalty_mean = float((1.0 - sos_weight[final_mask]).mean().detach().item())
        self.source_anchor_agreement = merge_metric(self.source_anchor_agreement, anchor_agree)
        self.group_reward_std = merge_metric(self.group_reward_std, reward_std_mean)
        self.sos_mean_penalty = merge_metric(self.sos_mean_penalty, sos_penalty_mean)
        return True

    def _adapt_on_batch_gn(
        self,
        x: torch.Tensor,
        raw_logits: torch.Tensor,
        group_std: torch.Tensor,
        sos_weight: torch.Tensor,
        pseudo_targets: torch.Tensor,
        source_logits: torch.Tensor,
        raw_advantage: torch.Tensor,
        aux_info: Dict[str, torch.Tensor],
    ) -> bool:
        self._ensure_gn_state(raw_logits.device, raw_logits.dtype)

        raw_probs = aux_info["raw_probs"]
        source_probs = aux_info["source_probs"]
        structural_probs = aux_info["structural_probs"]
        raw_pred = aux_info["raw_pred"]
        source_pred = aux_info["source_pred"]
        destroy_sensitivity = aux_info["destroy_sensitivity"]
        entropy_score = aux_info["entropy_score"]
        view_inconsistency = aux_info["view_inconsistency"]

        batch_size = max(1, x.shape[0])

        def _reset_gn_failure(
            batch_alignment: float = 0.0,
            redundancy_removed: float = 0.0,
            stage1_ratio: float = 0.0,
            nonredundant_ratio: float = 0.0,
            distill_factor: float = 0.0,
            structural_factor: float = 0.0,
            redundancy_active: float = 0.0,
        ) -> bool:
            self.optimizer.zero_grad(set_to_none=True)
            self.last_gn_plpd = 0.0
            self.last_gn_anchor = 0.0
            self.last_gn_keep_ratio = 0.0
            self.last_gn_redundancy_skip = redundancy_removed
            self.last_gn_arc_guard = 1.0
            self.last_gn_batch_alignment = batch_alignment
            self.last_gn_stage1_ratio = stage1_ratio
            self.last_gn_nonredundant_ratio = nonredundant_ratio
            self.last_gn_entropy_loss = 0.0
            self.last_gn_distill_loss = 0.0
            self.last_gn_distill_factor = distill_factor
            self.last_gn_structural_factor = structural_factor
            self.last_gn_redundancy_active = redundancy_active
            self.gn_d3lr_current_alpha = 1.0
            self.last_gn_d3lr_alpha = 1.0
            self.last_gn_raw_entropy = 0.0
            self.last_selected_ratio = 0.0
            self.last_advantage_mean = 0.0
            self.last_probe_reliability = 0.0
            self.last_bn_entropy_loss = 0.0
            self.last_bn_consistency_loss = 0.0
            return False

        if self._gn_standard_mode and not self.use_gn_unified_67b:
            informative_mask = (
                group_std > self.dig_config["std_threshold"]
                if self.dig_enabled else torch.ones_like(group_std, dtype=torch.bool)
            )
            candidate_mask = (
                informative_mask
                & (sos_weight > 0.0)
                & (entropy_score < self.gn_entropy_margin)
            )
            candidate_indices = torch.where(candidate_mask)[0]
            self.last_gn_stage1_ratio = float(candidate_indices.numel()) / float(batch_size)
            self.last_gn_nonredundant_ratio = self.last_gn_stage1_ratio
            self.last_gn_entropy_loss = 0.0
            self.last_gn_distill_loss = 0.0
            self.last_gn_distill_factor = 0.0
            self.last_gn_structural_factor = 0.0
            self.last_gn_redundancy_active = 0.0
            self.gn_d3lr_current_alpha = 1.0
            self.last_gn_d3lr_alpha = 1.0
            self.last_gn_raw_entropy = 0.0
            if candidate_indices.numel() == 0:
                return _reset_gn_failure()

            candidate_targets = pseudo_targets[candidate_indices]
            candidate_source_support = source_probs[candidate_indices].gather(
                1,
                candidate_targets.unsqueeze(1),
            ).squeeze(1)
            candidate_source_support_core = (
                candidate_source_support
                if self.source_view_active else self._neutral_support(candidate_targets, source_probs)
            )
            candidate_struct_support = structural_probs[candidate_indices].gather(
                1,
                candidate_targets.unsqueeze(1),
            ).squeeze(1)
            candidate_struct_support_core = (
                candidate_struct_support
                if self.structural_view_active else self._neutral_support(candidate_targets, structural_probs)
            )
            candidate_anchor = (
                raw_pred[candidate_indices] == source_pred[candidate_indices]
            ).float()
            candidate_anchor_core = (
                candidate_anchor if self.source_view_active else self._neutral_anchor(candidate_targets)
            )
            candidate_score = (
                0.35 * (
                    destroy_sensitivity[candidate_indices]
                    if self.destroy_view_active else torch.zeros_like(candidate_targets, dtype=raw_probs.dtype)
                )
                + 0.25 * candidate_source_support_core
                + 0.20 * candidate_struct_support_core
                + 0.10 * (1.0 - entropy_score[candidate_indices])
                + 0.10 * candidate_anchor_core
            ).clamp(0.0, 1.0)
            min_keep = min(
                candidate_indices.numel(),
                max(1, int(math.ceil(batch_size * self.gn_min_fraction))),
            )
            keep_mask = self._select_top_fraction_mask(
                scores=candidate_score,
                target_keep=self.gn_keep_target,
                min_keep=min_keep,
                floor=self.gn_score_floor,
            )
            selected_indices = candidate_indices[keep_mask]
            if selected_indices.numel() == 0:
                return _reset_gn_failure()

            selected_logits = raw_logits[selected_indices]
            selected_source_logits = source_logits[selected_indices]
            selected_targets = pseudo_targets[selected_indices]
            selected_advantage = raw_advantage[selected_indices].clamp_min(0.0)
            selected_base_weights = sos_weight[selected_indices]
            selected_weights = selected_base_weights.clone()
            selected_group_std = group_std[selected_indices]
            selected_probs = raw_probs[selected_indices]
            selected_source_probs = source_probs[selected_indices]
            selected_struct_probs = structural_probs[selected_indices]
            selected_entropy = entropy_score[selected_indices]
            selected_plpd = destroy_sensitivity[selected_indices]
            selected_anchor = (raw_pred[selected_indices] == source_pred[selected_indices]).float()
            selected_view_inconsistency = view_inconsistency[selected_indices]
            selected_source_support = selected_source_probs.gather(
                1,
                selected_targets.unsqueeze(1),
            ).squeeze(1).clamp_min(1e-6)
            selected_source_support_core = (
                selected_source_support
                if self.source_view_active else self._neutral_support(selected_targets, selected_source_probs)
            )
            selected_struct_support = selected_struct_probs.gather(
                1,
                selected_targets.unsqueeze(1),
            ).squeeze(1).clamp_min(1e-6)
            selected_struct_support_core = (
                selected_struct_support
                if self.structural_view_active else self._neutral_support(selected_targets, selected_struct_probs)
            )
            selected_anchor_core = (
                selected_anchor if self.source_view_active else self._neutral_anchor(selected_targets)
            )
            selected_plpd_core = (
                selected_plpd if self.destroy_view_active else torch.zeros_like(selected_plpd)
            )

            harmful_score = (
                0.28 * selected_entropy
                + 0.28 * (1.0 - selected_plpd_core)
                + 0.18 * selected_view_inconsistency
                + 0.14 * (1.0 - selected_source_support_core)
                + 0.12 * (1.0 - selected_struct_support_core)
            ).clamp(0.0, 1.0)
            harmful_mean = harmful_score.mean()
            plpd_mean = selected_plpd.mean()
            anchor_mean = selected_anchor.mean()
            selected_ratio = float(selected_logits.shape[0]) / float(batch_size)

            plpd_norm = ((selected_plpd_core.detach() - 0.05) / 0.95).clamp(0.0, 1.0)
            source_norm = selected_source_support_core.detach().clamp(0.0, 1.0)
            struct_norm = selected_struct_support_core.detach().clamp(0.0, 1.0)
            trust_guard = torch.minimum(plpd_norm, 0.5 * (source_norm + struct_norm))
            arc_guard = (
                self.gn_arc_floor + (1.0 - self.gn_arc_floor) * trust_guard
            ).clamp_(self.gn_arc_floor, 1.0)

            self.last_gn_plpd = float(plpd_mean.detach().item())
            self.last_gn_anchor = float(anchor_mean.detach().item())
            self.last_gn_keep_ratio = selected_ratio
            self.last_gn_redundancy_skip = 0.0
            self.last_gn_arc_guard = float(arc_guard.mean().detach().item())
            self.last_gn_batch_alignment = 0.0
            self.last_gn_nonredundant_ratio = selected_ratio

            entropy_weight = torch.exp(
                (self.gn_entropy_margin - selected_entropy.detach()).clamp(min=-1.0, max=1.0)
            )
            plpd_weight = 1.0 + self.gn_plpd_weight * plpd_norm
            anchor_weight = 1.0 + self.gn_anchor_weight * selected_anchor_core
            struct_weight = 0.80 + self.gn_struct_weight * selected_struct_support_core
            support_weight = 0.75 + 0.25 * selected_source_support_core
            selected_weights = (
                selected_weights
                * entropy_weight
                * plpd_weight
                * anchor_weight
                * struct_weight
                * support_weight
            )
            selected_weights = selected_weights / selected_weights.mean().detach().clamp_min(1e-4)
            selected_weights = selected_weights.clamp_(0.25, 3.0)

            target_prob = selected_probs.gather(1, selected_targets.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
            ref_prob = selected_source_probs.gather(1, selected_targets.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
            ratio = target_prob / ref_prob
            ratio = 1.0 + (ratio - 1.0) * arc_guard
            selected_advantage = selected_advantage * arc_guard
            selected_weights = selected_weights * (0.80 + 0.20 * arc_guard)

            surrogate = arc_surrogate(
                ratio=ratio,
                advantage=selected_advantage,
                eps_low=self.arc_config["eps_low"],
                eps_high=self.arc_config["eps_high"],
                enabled=self.arc_enabled,
            )
            loss = -(selected_weights * surrogate).sum()
            loss = loss / float(
                active_unit_count(
                    batch_size=int(selected_logits.shape[0]),
                    mode=self.uan_mode,
                    token_count=self.token_count,
                )
            )

            anchor_loss = consistency_cross_entropy(selected_logits, selected_source_logits.detach())
            structural_loss = -(
                selected_logits.softmax(1) * selected_struct_probs.clamp_min(1e-6).log()
            ).sum(1).mean()
            loss = loss + 0.001 * self.num_classes * (0.6 * anchor_loss + 0.4 * structural_loss)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self._update_gn_history(selected_probs.detach())

            self.update_count += 1
            self.dig_kept_groups += int(selected_logits.shape[0])
            self.gn_d3lr_current_alpha = 1.0
            self.last_gn_d3lr_alpha = 1.0
            self.last_gn_raw_entropy = 0.0
            self.last_selected_ratio = selected_ratio
            self.last_advantage_mean = float(selected_advantage.mean().detach().item())
            self.last_probe_reliability = 0.0
            self.last_bn_entropy_loss = 0.0
            self.last_bn_consistency_loss = 0.0
            anchor_agree = float(selected_anchor.mean().detach().item())
            reward_std_mean = float(selected_group_std.mean().detach().item())
            sos_penalty_mean = float((1.0 - selected_base_weights).mean().detach().item())
            self.source_anchor_agreement = merge_metric(self.source_anchor_agreement, anchor_agree)
            self.group_reward_std = merge_metric(self.group_reward_std, reward_std_mean)
            self.sos_mean_penalty = merge_metric(self.sos_mean_penalty, sos_penalty_mean)
            return True

        stage1_plpd_mask = destroy_sensitivity > self.gn_plpd_threshold
        if not self.destroy_view_active:
            stage1_plpd_mask = torch.ones_like(stage1_plpd_mask, dtype=torch.bool)
        stage1_mask = (
            (entropy_score < self.gn_entropy_margin)
            & stage1_plpd_mask
        )
        distill_factor, structural_factor, redundancy_active = self._get_gn_continual_factors()
        stage1_indices = torch.where(stage1_mask)[0]
        stage1_ratio = float(stage1_indices.numel()) / float(batch_size)
        self.last_gn_stage1_ratio = stage1_ratio
        self.last_gn_nonredundant_ratio = 0.0
        self.last_gn_entropy_loss = 0.0
        self.last_gn_distill_loss = 0.0
        self.last_gn_distill_factor = distill_factor
        self.last_gn_structural_factor = structural_factor
        self.last_gn_redundancy_active = float(redundancy_active)
        if stage1_indices.numel() == 0:
            return _reset_gn_failure(
                distill_factor=distill_factor,
                structural_factor=structural_factor,
                redundancy_active=float(redundancy_active),
            )

        selected_logits = raw_logits[stage1_indices]
        selected_source_logits = source_logits[stage1_indices]
        selected_targets = pseudo_targets[stage1_indices]
        selected_advantage = raw_advantage[stage1_indices].clamp_min(0.0)
        selected_base_weights = sos_weight[stage1_indices]
        selected_group_std = group_std[stage1_indices]
        selected_probs = raw_probs[stage1_indices]
        selected_source_probs = source_probs[stage1_indices]
        selected_struct_probs = structural_probs[stage1_indices]
        selected_plpd = destroy_sensitivity[stage1_indices]
        selected_anchor = (raw_pred[stage1_indices] == source_pred[stage1_indices]).float()
        selected_source_support = selected_source_probs.gather(
            1,
            selected_targets.unsqueeze(1),
        ).squeeze(1).clamp_min(1e-6)
        selected_source_support_core = (
            selected_source_support
            if self.source_view_active else self._neutral_support(selected_targets, selected_source_probs)
        )
        selected_struct_support = selected_struct_probs.gather(
            1,
            selected_targets.unsqueeze(1),
        ).squeeze(1).clamp_min(1e-6)
        selected_struct_support_core = (
            selected_struct_support
            if self.structural_view_active else self._neutral_support(selected_targets, selected_struct_probs)
        )
        selected_plpd_core = (
            selected_plpd if self.destroy_view_active else torch.zeros_like(selected_plpd)
        )

        redundancy_removed = 0.0
        batch_alignment_value = 0.0
        if (
            redundancy_active
            and self.gn_current_model_probs is not None
            and selected_probs.shape[0] > 0
        ):
            history = self.gn_current_model_probs.to(
                device=selected_probs.device,
                dtype=selected_probs.dtype,
            )
            cosine_similarities = F.cosine_similarity(
                history.unsqueeze(0),
                selected_probs,
                dim=1,
            ).clamp(-1.0, 1.0)
            anti_mask = torch.abs(cosine_similarities) < self.gn_d_margin
            if int(anti_mask.sum().item()) == 0:
                redundancy_removed = 1.0
                self.last_gn_redundancy_skip = redundancy_removed
                self.last_gn_batch_alignment = 0.0
                return _reset_gn_failure(
                    batch_alignment=0.0,
                    redundancy_removed=redundancy_removed,
                    stage1_ratio=stage1_ratio,
                    nonredundant_ratio=0.0,
                    distill_factor=distill_factor,
                    structural_factor=structural_factor,
                    redundancy_active=float(redundancy_active),
                )

            redundancy_removed = float((~anti_mask).float().mean().item())
            selected_logits = selected_logits[anti_mask]
            selected_source_logits = selected_source_logits[anti_mask]
            selected_targets = selected_targets[anti_mask]
            selected_advantage = selected_advantage[anti_mask]
            selected_base_weights = selected_base_weights[anti_mask]
            selected_group_std = selected_group_std[anti_mask]
            selected_probs = selected_probs[anti_mask]
            selected_source_probs = selected_source_probs[anti_mask]
            selected_struct_probs = selected_struct_probs[anti_mask]
            selected_plpd = selected_plpd[anti_mask]
            selected_anchor = selected_anchor[anti_mask]
            selected_source_support = selected_source_support[anti_mask]
            selected_source_support_core = selected_source_support_core[anti_mask]
            selected_struct_support = selected_struct_support[anti_mask]
            selected_struct_support_core = selected_struct_support_core[anti_mask]
            selected_plpd_core = selected_plpd_core[anti_mask]
            batch_alignment = F.cosine_similarity(
                selected_probs.mean(dim=0, keepdim=True),
                history.unsqueeze(0),
                dim=1,
            ).clamp(-1.0, 1.0)
            batch_alignment_value = float(batch_alignment.detach().item())

        if selected_logits.shape[0] == 0:
            return _reset_gn_failure(
                batch_alignment=batch_alignment_value,
                redundancy_removed=redundancy_removed,
                stage1_ratio=stage1_ratio,
                nonredundant_ratio=0.0,
                distill_factor=distill_factor,
                structural_factor=structural_factor,
                redundancy_active=float(redundancy_active),
            )

        selected_ratio = float(selected_logits.shape[0]) / float(batch_size)
        entropys_filtered = softmax_entropy(selected_logits)
        coeff_entropy = 1.0 / torch.exp(entropys_filtered.detach() - self.gn_entropy_margin)
        coeff_plpd = torch.exp(selected_plpd_core.detach())
        coeff = self.gn_entropy_weight * coeff_entropy + self.gn_plpd_weight * coeff_plpd
        coeff = coeff * (0.80 + 0.20 * selected_base_weights.detach())
        coeff = coeff * (0.85 + 0.15 * selected_source_support_core.detach())
        loss_main = (entropys_filtered * coeff).mean()

        distill_log_probs = F.log_softmax(
            selected_logits / self.gn_distill_temp,
            dim=1,
        )
        distill_targets = F.softmax(
            selected_source_logits.detach() / self.gn_distill_temp,
            dim=1,
        )
        distill_loss = (self.gn_distill_temp ** 2) * F.kl_div(
            distill_log_probs,
            distill_targets,
            reduction="batchmean",
        )

        structural_loss = -(
            selected_logits.softmax(1) * selected_struct_probs.detach().clamp_min(1e-6).log()
        ).sum(1).mean()

        loss = loss_main
        loss = loss + (distill_factor * self.gn_distill_weight) * distill_loss
        loss = loss + (structural_factor * self.gn_structural_weight) * structural_loss

        batch_entropy = float(entropys_filtered.mean().detach().item())
        self._update_gn_continual_d3lr(batch_entropy)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self._update_gn_history(selected_probs.detach())

        self.update_count += 1
        self.dig_kept_groups += int(selected_logits.shape[0])
        self.last_gn_plpd = float(selected_plpd.mean().detach().item())
        self.last_gn_anchor = float(selected_anchor.mean().detach().item())
        self.last_gn_keep_ratio = selected_ratio
        self.last_gn_redundancy_skip = redundancy_removed
        self.last_gn_arc_guard = 1.0
        self.last_gn_batch_alignment = batch_alignment_value
        self.last_gn_nonredundant_ratio = selected_ratio
        self.last_gn_entropy_loss = float(loss_main.detach().item())
        self.last_gn_distill_loss = float(distill_loss.detach().item())
        self.last_gn_distill_factor = distill_factor
        self.last_gn_structural_factor = structural_factor
        self.last_gn_redundancy_active = float(redundancy_active)
        self.last_gn_raw_entropy = batch_entropy
        self.last_selected_ratio = selected_ratio
        self.last_advantage_mean = float(selected_advantage.mean().detach().item())
        self.last_probe_reliability = 0.0
        self.last_bn_entropy_loss = 0.0
        self.last_bn_consistency_loss = 0.0
        anchor_agree = float(selected_anchor.mean().detach().item())
        reward_std_mean = float(selected_group_std.mean().detach().item())
        sos_penalty_mean = float((1.0 - selected_base_weights).mean().detach().item())
        self.source_anchor_agreement = merge_metric(self.source_anchor_agreement, anchor_agree)
        self.group_reward_std = merge_metric(self.group_reward_std, reward_std_mean)
        self.sos_mean_penalty = merge_metric(self.sos_mean_penalty, sos_penalty_mean)
        return True

    def _adapt_on_batch(self, x: torch.Tensor) -> bool:
        if self._ln_mode:
            return self._adapt_on_batch_ln_vit(x)

        raw_logits = self.model(x)
        (
            group_std,
            sos_weight,
            pseudo_targets,
            source_logits,
            raw_advantage,
            raw_reward_stack,
            aux_info,
        ) = self._compute_group_signals(x, raw_logits)

        if self.use_branch_refinements and self._gn_mode:
            return self._adapt_on_batch_gn(
                x=x,
                raw_logits=raw_logits,
                group_std=group_std,
                sos_weight=sos_weight,
                pseudo_targets=pseudo_targets,
                source_logits=source_logits,
                raw_advantage=raw_advantage,
                aux_info=aux_info,
            )

        if self.use_branch_refinements and self._bn_continual_mode:
            return self._adapt_on_batch_bn_continual(
                x=x,
                raw_logits=raw_logits,
                group_std=group_std,
                sos_weight=sos_weight,
                pseudo_targets=pseudo_targets,
                source_logits=source_logits,
                raw_advantage=raw_advantage,
                aux_info=aux_info,
            )

        informative_mask = group_std > self.dig_config["std_threshold"] if self.dig_enabled else torch.ones_like(group_std, dtype=torch.bool)
        certainty_mask = torch.ones_like(informative_mask, dtype=torch.bool)
        certainty_weight = torch.ones_like(group_std)
        probe_reliability = 1.0
        bn_standard_refinements = self._bn_standard_mode and self.use_branch_refinements
        if bn_standard_refinements:
            raw_entropies = softmax_entropy(raw_logits.detach())
            probe_count = min(self.bn_standard_probe_count, raw_logits.shape[0])
            probe_indices = torch.randperm(raw_logits.shape[0], device=raw_logits.device)[:probe_count]
            probe_entropy = float(raw_entropies[probe_indices].mean().item()) if probe_count > 0 else 0.0
            probe_entropy_ema = self._update_bn_standard_probe_entropy(probe_entropy)
            probe_reliability = float(
                (1.0 - normalized_entropy(raw_logits.detach()[probe_indices])).mean().item()
            )
            adaptive_margin = 0.40 + 0.08 * max(0.0, 1.0 - probe_reliability)
            normalized_entropies = normalized_entropy(raw_logits.detach())
            certainty_weight = torch.sigmoid(
                (adaptive_margin - normalized_entropies) / 0.10
            )
            certainty_mask = certainty_weight > 0.15
            self.last_bn_standard_probe_entropy = probe_entropy_ema
        else:
            self.last_bn_standard_probe_entropy = 0.0

        if not self.use_branch_refinements:
            informative_mask = quantile_selection_mask(
                aux_info["reward_mean"], self.selection_quantile
            )
            sos_weight = torch.ones_like(sos_weight)
            certainty_mask = torch.ones_like(informative_mask, dtype=torch.bool)
        final_mask = informative_mask & (sos_weight > 0.0) & certainty_mask
        if final_mask.sum() == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.last_probe_reliability = probe_reliability if bn_standard_refinements else 0.0
            self.last_bn_entropy_loss = 0.0
            self.last_bn_consistency_loss = 0.0
            self.last_bn_standard_debias = 1.0
            self.last_bn_standard_trusted_boost = 1.0
            self.last_bn_standard_entropy_core = 0.0
            self.last_bn_standard_entropy_keep_ratio = 0.0
            self.last_bn_standard_arc_reg = 0.0
            self.last_bn_standard_aug_mode = "none"
            self.last_bn_standard_teacher_mode = "none"
            self.last_bn_standard_consistency_factor = 0.0
            self.last_bn_standard_consistency_mix = 0.0
            self.last_bn_standard_aux_mix = 0.0
            self.last_bn_standard_probe_entropy = 0.0
            self.last_bn_standard_ais_soft_weight = 0.0
            self.last_bn_standard_accepted_entropy = 0.0
            self.last_bn_standard_d3lr_driver = "none"
            return False

        selected_logits = raw_logits[final_mask]
        selected_source_logits = source_logits[final_mask]
        selected_targets = pseudo_targets[final_mask]
        selected_advantage = raw_advantage[final_mask]
        selected_weights = sos_weight[final_mask]
        if bn_standard_refinements:
            self._ensure_bn_standard_state(selected_logits.device, selected_logits.dtype)
            selected_advantage = selected_advantage.clamp_min(0.0)
            selected_weights = selected_weights * certainty_weight[final_mask]

        probs = selected_logits.softmax(dim=1)
        source_probs = selected_source_logits.softmax(dim=1)
        if bn_standard_refinements:
            standard_template = self.bn_standard_class_template[selected_targets]
            standard_template_match = F.cosine_similarity(
                probs.detach(),
                standard_template,
                dim=1,
            ).clamp(-1.0, 1.0)
            standard_anchor = (
                selected_logits.detach().argmax(dim=1)
                == selected_source_logits.argmax(dim=1)
            ).float()
            standard_support = source_probs.gather(
                1,
                selected_targets.unsqueeze(1),
            ).squeeze(1).clamp_min(1e-6)
            mean_standard_support = float(standard_support.mean().detach().item())
            standard_debias_weight = (
                1.08
                - 0.28
                * standard_template_match
                * standard_support
                * (0.5 + 0.5 * standard_anchor)
            ).clamp_(0.72, 1.08)
            standard_trusted_hard = (
                standard_support
                * (0.5 + 0.5 * standard_anchor)
                * (0.35 + 0.65 * (1.0 - standard_template_match.clamp(0.0, 1.0)))
            )
            standard_trusted_boost = (
                1.0
                + self.bn_standard_trusted_hard_boost
                * probe_reliability
                * standard_trusted_hard
            )
            selected_weights = selected_weights * standard_debias_weight
            selected_weights = selected_weights * standard_trusted_boost
            self.last_bn_standard_debias = float(standard_debias_weight.mean().detach().item())
            self.last_bn_standard_trusted_boost = float(standard_trusted_boost.mean().detach().item())
        else:
            self.last_bn_standard_debias = 1.0
            self.last_bn_standard_trusted_boost = 1.0
        target_prob = probs.gather(1, selected_targets.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        ref_prob = source_probs.gather(1, selected_targets.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        ratio = target_prob / ref_prob

        surrogate = arc_surrogate(
            ratio=ratio,
            advantage=selected_advantage,
            eps_low=self.arc_config["eps_low"],
            eps_high=self.arc_config["eps_high"],
            enabled=self.arc_enabled,
        )
        arc_reg = -(selected_weights * surrogate).sum()
        arc_reg = arc_reg / float(
            active_unit_count(
                batch_size=int(final_mask.sum().item()),
                mode=self.uan_mode,
                token_count=self.token_count,
            )
        )

        bn_entropy_loss = raw_logits.new_zeros(())
        bn_consistency_loss = raw_logits.new_zeros(())
        if bn_standard_refinements:
            selected_entropies = softmax_entropy(selected_logits)
            entropy_margin = (
                self.bn_standard_reliable_margin
                + 0.06 * max(0.0, 1.0 - probe_reliability)
                + 0.04 * max(0.0, 1.0 - mean_standard_support)
            )
            entropy_core_mask = selected_entropies.detach() < entropy_margin
            min_core = max(2, int(math.ceil(0.10 * float(selected_logits.shape[0]))))
            if int(entropy_core_mask.sum().item()) < min_core:
                entropy_core_mask = torch.ones_like(entropy_core_mask, dtype=torch.bool)

            core_entropies = selected_entropies[entropy_core_mask]
            core_weights = selected_weights[entropy_core_mask].detach()
            entropy_coeff = torch.exp(
                (entropy_margin - core_entropies.detach()).clamp(min=-1.0, max=2.0)
            )
            weighted_entropy = core_entropies * entropy_coeff
            bn_entropy_loss = (
                (core_weights * weighted_entropy).sum()
                / core_weights.sum().clamp_min(1e-6)
            )
            accepted_entropy_mean = float(selected_entropies.detach().mean().item())
            self.last_bn_standard_accepted_entropy = accepted_entropy_mean
            self._update_bn_standard_d3lr(accepted_entropy_mean)
            self.last_bn_standard_d3lr_driver = "accepted_entropy"

            weak_stability_logits = self.model(self._build_bn_standard_view(x))
            strong_stability_logits = self.model(self._build_bn_standard_surgeon_view(x))
            aux_stability_logits = self.model(self._build_bn_standard_aux_view(x))
            weak_stability_selected = weak_stability_logits[final_mask]
            strong_stability_selected = strong_stability_logits[final_mask]
            aux_stability_selected = aux_stability_logits[final_mask]
            hardness = min(max(1.0 - probe_reliability, 0.0), 1.0)
            strong_mix = 0.35 + 0.45 * hardness
            weak_mix = 1.0 - strong_mix
            aux_mix = 0.0
            if probe_reliability < 0.74:
                aux_mix = 0.05
            if probe_reliability < 0.70:
                aux_mix = 0.08
            if probe_reliability < 0.66:
                aux_mix = 0.10
            weak_consistency = -(
                selected_logits.softmax(dim=1)
                * weak_stability_selected.log_softmax(dim=1)
            ).sum(dim=1)
            strong_consistency = -(
                selected_logits.softmax(dim=1)
                * strong_stability_selected.log_softmax(dim=1)
            ).sum(dim=1)
            base_consistency = weak_mix * weak_consistency + strong_mix * strong_consistency
            if aux_mix > 0.0:
                aux_consistency = -(
                    selected_logits.softmax(dim=1)
                    * aux_stability_selected.log_softmax(dim=1)
                ).sum(dim=1)
                base_consistency = (1.0 - aux_mix) * base_consistency + aux_mix * aux_consistency
            with torch.no_grad():
                flip_logits = self.model(torch.flip(x, dims=[3]))
                flip_preds = flip_logits[final_mask].argmax(dim=1)
                selected_preds = selected_logits.detach().argmax(dim=1)
                aux_preds = aux_stability_selected.detach().argmax(dim=1)
                flip_agreement = (selected_preds == flip_preds).float()
                aux_agreement = (selected_preds == aux_preds).float()
                ais_soft_weight = 0.85 + 0.15 * flip_agreement + 0.10 * aux_agreement
            bn_consistency_loss = (ais_soft_weight * base_consistency).mean()
            lambda_arc = 0.12 + 0.08 * probe_reliability
            consistency_warmup = min(1.0, float(self.step_count + 1) / 4.0)
            consistency_severity = min(
                1.25,
                max(0.75, 0.75 + 1.2 * hardness),
            )
            probe_severity = min(
                1.15,
                max(
                    0.90,
                    0.90
                    + 0.25
                    * (
                        self.last_bn_standard_probe_entropy
                        / max(self.bn_standard_reliable_margin, 1e-6)
                    ),
                ),
            )
            lambda_cons = (
                consistency_warmup
                * consistency_severity
                * probe_severity
                * 0.01
                * self.num_classes
            )
            loss = bn_entropy_loss + lambda_arc * arc_reg + lambda_cons * bn_consistency_loss
            self.last_bn_standard_entropy_core = float(bn_entropy_loss.detach().item())
            self.last_bn_standard_entropy_keep_ratio = float(entropy_core_mask.float().mean().item())
            self.last_bn_standard_arc_reg = float(arc_reg.detach().item())
            self.last_bn_standard_aug_mode = "surgeon_dual_aux"
            self.last_bn_standard_teacher_mode = "none"
            self.last_bn_standard_consistency_factor = float(
                consistency_warmup * consistency_severity * probe_severity
            )
            self.last_bn_standard_consistency_mix = float(strong_mix)
            self.last_bn_standard_aux_mix = float(aux_mix)
            self.last_bn_standard_ais_soft_weight = float(ais_soft_weight.mean().item())
        else:
            loss = arc_reg
            self.last_bn_standard_entropy_core = 0.0
            self.last_bn_standard_entropy_keep_ratio = 0.0
            self.last_bn_standard_arc_reg = 0.0
            self.last_bn_standard_aug_mode = "none"
            self.last_bn_standard_teacher_mode = "none"
            self.last_bn_standard_consistency_factor = 0.0
            self.last_bn_standard_consistency_mix = 0.0
            self.last_bn_standard_aux_mix = 0.0
            self.last_bn_standard_probe_entropy = 0.0
            self.last_bn_standard_ais_soft_weight = 0.0
            self.last_bn_standard_accepted_entropy = 0.0
            self.last_bn_standard_d3lr_driver = "none"

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if bn_standard_refinements:
            self._update_bn_standard_history(probs.detach())

        self.update_count += 1
        selected_count = int(final_mask.sum().item())
        self.dig_kept_groups += selected_count
        self.last_selected_ratio = selected_count / max(1, x.shape[0])
        self.last_advantage_mean = float(selected_advantage.mean().detach().item())
        self.last_probe_reliability = probe_reliability if bn_standard_refinements else 0.0
        self.last_bn_entropy_loss = float(bn_entropy_loss.detach().item()) if bn_standard_refinements else 0.0
        self.last_bn_consistency_loss = float(bn_consistency_loss.detach().item()) if bn_standard_refinements else 0.0
        source_pred = selected_source_logits.argmax(dim=1)
        raw_pred = selected_logits.detach().argmax(dim=1)
        anchor_agree = float((source_pred == raw_pred).float().mean().item())
        reward_std_mean = float(group_std[final_mask].mean().detach().item())
        sos_penalty_mean = float((1.0 - selected_weights).mean().detach().item())
        self.source_anchor_agreement = merge_metric(self.source_anchor_agreement, anchor_agree)
        self.group_reward_std = merge_metric(self.group_reward_std, reward_std_mean)
        self.sos_mean_penalty = merge_metric(self.sos_mean_penalty, sos_penalty_mean)
        return True

    def _compute_group_signals(
        self,
        x: torch.Tensor,
        raw_logits: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        with torch.no_grad():
            source_logits = self.source_model(x)
            weak_view = build_photometric_view(x)
            structural_view = build_structural_view(x)
            destroy_view = build_destroy_view(x)
            weak_logits = self.model(weak_view)
            structural_logits = self.model(structural_view)
            destroy_logits = self.model(destroy_view)

        raw_probs = raw_logits.softmax(dim=1)
        source_probs = source_logits.softmax(dim=1)
        weak_probs = weak_logits.softmax(dim=1)
        structural_probs = structural_logits.softmax(dim=1)
        destroy_probs = destroy_logits.softmax(dim=1)

        raw_pred = raw_probs.argmax(dim=1)
        source_pred = source_probs.argmax(dim=1)
        weak_pred = weak_probs.argmax(dim=1)
        structural_pred = structural_probs.argmax(dim=1)

        destroy_pred = destroy_probs.argmax(dim=1)
        vote_predictions = {
            "source": source_pred,
            "raw": raw_pred,
            "photometric": weak_pred,
            "structural": structural_pred,
            "destroy": destroy_pred,
        }
        vote_weights = {
            "source": source_probs.max(dim=1).values + 0.25,
            "raw": raw_probs.max(dim=1).values,
            "photometric": weak_probs.max(dim=1).values,
            "structural": structural_probs.max(dim=1).values,
            "destroy": destroy_probs.max(dim=1).values,
        }
        if self._explicit_active_views:
            vote_view_names = list(self.active_views)
            reward_view_names = ["raw"]
            for name in ("photometric", "structural", "destroy"):
                if name in self.active_views:
                    reward_view_names.append(name)
            anchor_enabled = self.source_view_active
            inconsistency_view_names = ["raw"]
            for name in ("photometric", "structural"):
                if name in self.active_views:
                    inconsistency_view_names.append(name)
        else:
            vote_view_names = ["source", "raw", "photometric", "structural"]
            reward_view_names = ["raw", "photometric", "structural", "destroy"]
            anchor_enabled = True
            inconsistency_view_names = ["raw", "photometric", "structural"]

        pseudo_targets = weighted_majority_vote(
            predictions=[vote_predictions[name] for name in vote_view_names],
            weights=[vote_weights[name] for name in vote_view_names],
            num_classes=self.num_classes,
        )

        def _target_prob(probs: torch.Tensor) -> torch.Tensor:
            return probs.gather(1, pseudo_targets.unsqueeze(1)).squeeze(1)

        view_probs_map = {
            "raw": raw_probs,
            "photometric": weak_probs,
            "structural": structural_probs,
            "destroy": destroy_probs,
        }
        view_logits_map = {
            "raw": raw_logits.detach(),
            "photometric": weak_logits,
            "structural": structural_logits,
            "destroy": destroy_logits,
        }
        view_preds_map = {
            "raw": raw_pred,
            "photometric": weak_pred,
            "structural": structural_pred,
            "destroy": destroy_pred,
        }
        view_probs = [view_probs_map[name] for name in reward_view_names]
        view_logits = [view_logits_map[name] for name in reward_view_names]
        target_probs = torch.stack([_target_prob(probs) for probs in view_probs], dim=1)
        ent_scores = torch.stack(
            [1.0 - normalized_entropy(logits) for logits in view_logits],
            dim=1,
        )
        preds = [view_preds_map[name] for name in reward_view_names]
        if anchor_enabled:
            anchor_scores = torch.stack([(pred == source_pred).float() for pred in preds], dim=1)
            anchor_disagree = 1.0 - (raw_pred == source_pred).float()
        else:
            anchor_scores = torch.zeros_like(target_probs)
            anchor_disagree = torch.zeros_like(raw_probs.max(dim=1).values)
        consensus_scores = torch.stack([(pred == pseudo_targets).float() for pred in preds], dim=1)
        evidence = torch.stack(
            (target_probs, ent_scores, anchor_scores, consensus_scores),
            dim=-1,
        )
        legacy_weight_tensor = None
        if self.legacy_reward_weights is not None:
            legacy_weight_tensor = evidence.new_tensor(
                [
                    self.legacy_reward_weights[name]
                    for name in ("target", "entropy", "source", "consensus")
                ]
            )
        rewards = aggregate_evidence(
            evidence,
            mode=self.reward_aggregation,
            legacy_weights=legacy_weight_tensor,
        )

        reward_mean = rewards.mean(dim=1)
        reward_std = rewards.std(dim=1, unbiased=False)
        raw_view_index = reward_view_names.index("raw")
        raw_advantage = (rewards[:, raw_view_index] - reward_mean) / reward_std.clamp_min(1e-4)

        entropy_score = normalized_entropy(raw_logits.detach())
        inconsistency_pairs = []
        for idx, first_name in enumerate(inconsistency_view_names):
            for second_name in inconsistency_view_names[idx + 1:]:
                inconsistency_pairs.append(
                    (view_preds_map[first_name] == view_preds_map[second_name]).float()
                )
        if inconsistency_pairs:
            view_inconsistency = 1.0 - torch.stack(inconsistency_pairs, dim=1).mean(dim=1)
        else:
            view_inconsistency = torch.zeros_like(entropy_score)

        if "destroy" in reward_view_names:
            destroy_view_index = reward_view_names.index("destroy")
            destroy_sensitivity_score = destroy_sensitivity(
                target_probs[:, raw_view_index], target_probs[:, destroy_view_index]
            )
        else:
            destroy_sensitivity_score = torch.zeros_like(entropy_score)

        destroy_risk = insufficient_destroy_penalty(
            destroy_sensitivity_score,
            minimum_drop=self.sos_config["destroy_support_margin"],
        )

        shift_score = (
            self.sos_config["entropy_weight"] * entropy_score
            + self.sos_config["anchor_weight"] * anchor_disagree
            + self.sos_config["view_weight"] * view_inconsistency
            + self.sos_config["destroy_weight"] * destroy_risk
        ).clamp(0.0, 1.0)
        sos_weight = (
            sos_weight_from_score(
                shift_score,
                safe_margin=self.sos_config["safe_margin"],
                hard_margin=self.sos_config["hard_margin"],
            )
            if self.sos_enabled else torch.ones_like(shift_score)
        )

        aux_info = {
            "raw_view": x.detach(),
            "photometric_view": weak_view,
            "structural_view": structural_view,
            "destroy_view": destroy_view,
            "reward_mean": reward_mean,
            "raw_probs": raw_probs,
            "source_probs": source_probs,
            "photometric_probs": weak_probs,
            "structural_probs": structural_probs,
            "destroy_probs": destroy_probs,
            "raw_pred": raw_pred,
            "source_pred": source_pred,
            "photometric_pred": weak_pred,
            "structural_pred": structural_pred,
            "destroy_pred": destroy_pred,
            "pseudo_target": pseudo_targets,
            "target_probs": target_probs,
            "target_prob_rewards": target_probs,
            "entropy_rewards": ent_scores,
            "source_agreement_rewards": anchor_scores,
            "consensus_rewards": consensus_scores,
            "entropy_score": entropy_score,
            "raw_entropy": softmax_entropy(raw_logits.detach()),
            "destroy_sensitivity": destroy_sensitivity_score,
            "destroy_risk": destroy_risk,
            "view_inconsistency": view_inconsistency,
            "source_disagreement": anchor_disagree,
            "safety_score": shift_score,
            "safety_weight": sos_weight,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "raw_advantage": raw_advantage,
            "selected": (
                (reward_std > self.dig_config["std_threshold"])
                & (sos_weight > 0.0)
            ),
            "reward_view_names": reward_view_names,
        }

        return reward_std, sos_weight, pseudo_targets, source_logits, raw_advantage, rewards, aux_info

    @torch.no_grad()
    def diagnose_group_signals(self, x: torch.Tensor) -> Dict[str, object]:
        """Return per-sample Eq.7/CORE diagnostics without adapting model state."""
        if self._ln_mode:
            raise RuntimeError(
                "Eq.7 diagnostics are defined for the generic BN/GN multi-view path, "
                "not the branch-specific ViT/LN path."
            )
        raw_logits = self.model(x)
        reward_std, sos_weight, pseudo_targets, _, raw_advantage, rewards, aux = (
            self._compute_group_signals(x, raw_logits)
        )
        result = dict(aux)
        result.update(
            {
                "rewards": rewards,
                "reward_std": reward_std,
                "safety_weight": sos_weight,
                "pseudo_target": pseudo_targets,
                "raw_advantage": raw_advantage,
                "tau_delta": float(self.dig_config["std_threshold"]),
                "safe_margin": float(self.sos_config["safe_margin"]),
                "hard_margin": float(self.sos_config["hard_margin"]),
                "epsilon_high": float(self.arc_config["eps_high"]),
            }
        )
        return _clone_to_cpu(result)


def _load_legacy_69c_create_atlas():
    backup_dir = os.path.join(os.path.dirname(__file__), "backups", "69C_best")
    package_name = "_atlas_legacy_69c"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [backup_dir]
        sys.modules[package_name] = package

    for module_name in ("common", "classification"):
        full_name = f"{package_name}.{module_name}"
        if full_name in sys.modules:
            continue
        module_path = os.path.join(backup_dir, f"{module_name}.py")
        spec = importlib.util.spec_from_file_location(full_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load legacy ATLAS module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}.classification"].create_atlas


def create_atlas(model: nn.Module, config: Optional[Dict] = None, **kwargs) -> ATLAS:
    config = config or {}
    if config.get("legacy_69c_exact", False):
        if str(config.get("method_variant", "full")).lower() != "full":
            raise ValueError("legacy_69c_exact is available only with method_variant='full'")
        return _load_legacy_69c_create_atlas()(model=model, config=config, **kwargs)
    return ATLAS(
        model=model,
        arc_config=config.get("arc", {}),
        dig_config=config.get("dig", {}),
        uan_config=config.get("uan", {}),
        sos_config=config.get("sos", {}),
        view_config=config.get("views", {}),
        reward_config=config.get("reward", {}),
        selection_config=config.get("selection", {}),
        vit_ln_config=config.get("vit_ln", {}),
        adapt_type=config.get("adapt_type", kwargs.pop("adapt_type", "bn")),
        adaptation_mode=config.get("adaptation_mode", kwargs.pop("adaptation_mode", "continual")),
        scenario=config.get("scenario", kwargs.pop("scenario", "normal")),
        lr=config.get("lr", kwargs.pop("lr", 2.5e-4)),
        optimizer_name=config.get("optimizer_name", kwargs.pop("optimizer_name", "sgd")),
        optim_momentum=config.get("optim_momentum", kwargs.pop("optim_momentum", 0.9)),
        optim_weight_decay=config.get("optim_weight_decay", kwargs.pop("optim_weight_decay", 0.0)),
        num_classes=config.get("num_classes", kwargs.pop("num_classes", 1000)),
        no_arc=config.get("no_arc", kwargs.pop("no_arc", False)),
        no_dig=config.get("no_dig", kwargs.pop("no_dig", False)),
        no_uan=config.get("no_uan", kwargs.pop("no_uan", False)),
        no_sos=config.get("no_sos", kwargs.pop("no_sos", False)),
        ablation_row=config.get("ablation_row", kwargs.pop("ablation_row", None)),
        method_variant=config.get("method_variant", kwargs.pop("method_variant", "full")),
    )
