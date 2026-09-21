"""TRIAD: TRi-space Invariant ADaptation

Architecture: EATA core + 3 complementary innovations (C1–C3).

Core Insight:
    Image corruptions primarily disrupt low-level statistics (shallow layers)
    while high-level semantics (deep layers) should be preserved. TRIAD
    instantiates this single principle through three complementary mechanisms
    that operate in orthogonal spaces: optimization (C1), function (C2),
    and sample selection (C3).

Paper Contributions:
    C1: Dynamic Depth-Decay Learning Rate (D³LR) —
        Entropy-aware dynamic layer-wise learning rates:
        α_t = 1 + (α_max - 1) · σ((H_t - H̄) / κ)
        lr_{l,t} = lr_base × α_t^((L-l)/L)
        Shallow layers get dynamically higher lr when shift is severe;
        deep layers always get lower lr to preserve semantics.
        Zero overhead (scalar computation per batch).
        Related: LANTON (ICLR 2026) noise-adaptive layer-wise scaling;
        LAW (WACV 2024), PALM (AAAI 2025) data-driven layer weighting.

    C2: Source-Anchored Output Distillation (SAOD) —
        Anchors adapted model to source behavior in function space via
        KL divergence: L_KD = T² · KL(p_src || p_adapt).
        Depth coupling is implicit through D³LR layer-wise lr.
        Related: SANTA (2023) source anchoring + contrastive alignment.

    C3: AIS (Anchored Intrinsic Safety) —
        Sample-safety filtering in selection space. Key insight: not all
        low-entropy samples are truly reliable; some are "fragile confident"
        predictions that sit off the semantic manifold and produce harmful
        gradients. AIS scores a sample with three cheap tests:
        1) augmentation stability, 2) low-frequency manifold consistency, and
        3) source-anchor agreement. Only intrinsically safe samples are used
        for adaptation. This keeps the module stateless while making it
        substantially stronger than a pure flip-consistency rule.
        Orthogonal to C1 (optimization) and C2 (function space).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple
from copy import deepcopy
import math
import logging

logger = logging.getLogger(__name__)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def entropy_filtered_loss(
    logits: torch.Tensor,
    e_margin: Optional[float] = None,
    return_mask: bool = False
) -> torch.Tensor:
    """EATA-style entropy filtering with weighting."""
    if e_margin is None:
        num_classes = logits.shape[1]
        e_margin = math.log(num_classes) / 2.0 - 1.0
    entropys = softmax_entropy(logits)
    reliable_mask = entropys < e_margin
    if reliable_mask.sum() == 0:
        if return_mask:
            return torch.tensor(0.0, device=logits.device), reliable_mask
        return torch.tensor(0.0, device=logits.device)
    reliable_entropys = entropys[reliable_mask]
    coeff = 1.0 / torch.exp(reliable_entropys - e_margin)
    loss = (reliable_entropys * coeff).mean(0)
    if return_mask:
        return loss, reliable_mask
    return loss


def update_model_probs(current_model_probs, new_probs, momentum: float = 0.9):
    """Update moving average of model prediction probabilities (EATA Eqn.4)."""
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return momentum * current_model_probs + (1 - momentum) * new_probs.mean(0)


def update_ema(ema, new_data):
    """Update exponential moving average."""
    if ema is None:
        return new_data
    else:
        return 0.9 * ema + (1.0 - 0.9) * new_data


@torch.jit.script
def consistency_cross_entropy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SURGEON-style consistency regularization between two predictions."""
    return -(x.softmax(1) * y.log_softmax(1)).sum(1).mean()


def normalized_logit_hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sample normalized linear HSIC/CKA proxy between two logit vectors."""
    x_centered = x.float() - x.float().mean(dim=1, keepdim=True)
    y_centered = y.float() - y.float().mean(dim=1, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=1).pow(2)
    denominator = (
        x_centered.pow(2).sum(dim=1)
        * y_centered.pow(2).sum(dim=1)
    ).clamp_min(1e-6)
    return (numerator / denominator).clamp_(0.0, 1.0)


class ClassConditionalEMA:
    """AdaDEM-style class-conditional average prediction tracker."""
    def __init__(self, num_classes: int, momentum: float = 0.1):
        self.num_classes = num_classes
        self.momentum = momentum
        self.avg_pred = None

    def reset(self):
        self.avg_pred = None

    def update(self, probs: torch.Tensor):
        pseudo_labels = probs.argmax(dim=1)
        if self.avg_pred is None:
            self.avg_pred = torch.full(
                (self.num_classes, self.num_classes),
                1.0 / float(self.num_classes),
                device=probs.device,
                dtype=probs.dtype,
            )

        self.avg_pred = self.avg_pred.detach()
        for label in torch.unique(pseudo_labels):
            self.avg_pred[label] = (
                (1.0 - self.momentum) * self.avg_pred[label]
                + self.momentum * probs[pseudo_labels == label].mean(0).detach()
            )
        return pseudo_labels


class SAMOptimizer:
    """SAM optimizer wrapper. Kept for ablation only, NOT used by default."""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.state = {}
        for p in self.params:
            self.state[p] = {}

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for p in self.params:
            if p.grad is None:
                continue
            eps = scale * p.grad * (torch.abs(p) + 1e-12) if self.adaptive else scale * p.grad
            self.state[p]['eps'] = eps.clone()
            p.add_(eps)

    @torch.no_grad()
    def second_step(self):
        for p in self.params:
            if 'eps' in self.state[p]:
                p.sub_(self.state[p]['eps'])
        self.base_optimizer.step()

    def _grad_norm(self):
        shared_device = self.params[0].device
        return torch.norm(
            torch.stack([p.grad.norm(p=2).to(shared_device)
                         for p in self.params if p.grad is not None]),
            p=2
        )

    def zero_grad(self):
        self.base_optimizer.zero_grad()


def configure_model(model, adapt_type='gn'):
    """
    Configure model for TTA — matches EATA's configure_model exactly.
    Key: use model.train() NOT model.eval(), to match EATA behavior.
    """
    model.train()
    model.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, nn.GroupNorm):
            m.requires_grad_(True)
        elif isinstance(m, nn.LayerNorm):
            if adapt_type in ('ln', 'gn+ln', 'bn+ln', 'all'):
                m.requires_grad_(True)

    return model


def _parse_vit_block_index(module_name: str) -> Optional[int]:
    if not module_name.startswith('blocks.'):
        return None
    parts = module_name.split('.')
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _get_vit_adaptation_cutoff(model) -> Optional[int]:
    blocks = getattr(model, 'blocks', None)
    if blocks is None:
        return None
    total_blocks = len(blocks)
    if total_blocks <= 0:
        return None
    return max(1, math.ceil(total_blocks * 2.0 / 3.0))


def _get_vit_lr_scale(param_name: str) -> float:
    """Default no-op scale; reserved for future clean C1-only ablations."""
    return 1.0


def _looks_like_vit_backbone(model) -> bool:
    """Best-effort structural detection for ViT-like backbones."""
    child_names = {name for name, _ in model.named_children()}
    if 'blocks' in child_names:
        return True

    has_patch_embed = hasattr(model, 'patch_embed')
    has_tokens = hasattr(model, 'cls_token') or hasattr(model, 'pos_embed')
    attn_modules = False
    ln_count = 0
    bn_count = 0
    gn_count = 0
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            ln_count += 1
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_count += 1
        elif isinstance(module, nn.GroupNorm):
            gn_count += 1
        if module_name.endswith('attn') or '.attn' in module_name:
            attn_modules = True

    if has_patch_embed and (has_tokens or attn_modules):
        return True
    if has_patch_embed and ln_count > 0 and ln_count >= (bn_count + gn_count):
        return True
    return False


def collect_all_norm_params(
    model,
    adapt_type='gn',
    arch_type: Optional[str] = None,
    vit_full_depth: bool = False,
):
    """
    Collect ALL normalization layer parameters including layer4.
    Matches EATA's collect_params (which includes all layers).
    """
    params = []
    param_names = []

    if adapt_type == 'gn':
        norm_types = (nn.GroupNorm,)
    elif adapt_type == 'bn':
        norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    elif adapt_type == 'ln':
        norm_types = (nn.LayerNorm,)
    elif adapt_type in ('gn+ln', 'bn+ln'):
        if adapt_type == 'gn+ln':
            norm_types = (nn.GroupNorm, nn.LayerNorm)
        else:
            norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)
    else:
        norm_types = (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                      nn.BatchNorm3d, nn.LayerNorm)

    vit_cutoff = (
        None if vit_full_depth
        else (_get_vit_adaptation_cutoff(model) if arch_type == 'vit' else None)
    )

    for nm, m in model.named_modules():
        if arch_type == 'vit' and isinstance(m, nn.LayerNorm):
            if (nm == 'norm' or nm.startswith('norm.')) and not vit_full_depth:
                continue
            block_idx = _parse_vit_block_index(nm)
            if block_idx is not None and vit_cutoff is not None and block_idx >= vit_cutoff:
                continue

        if isinstance(m, norm_types):
            for np_name, p in m.named_parameters():
                if np_name in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    param_names.append(f"{nm}.{np_name}")

    return params, param_names


class DELTA(nn.Module):
    """
    DELTA: Dynamic Entropy-Led Test-time Adaptation.

    Architecture matches EATA exactly in the base case:
    - Single forward pass
    - Entropy filtering (e_margin = ln(K)/2-1)
    - Anti-redundancy filtering (cosine similarity < d_margin)
    - Entropy-weighted loss (coeff = 1/exp(H-E0))
    - SGD optimizer, no gradient clipping, no perturbation

    Innovations:
    - C1: D³LR — Dynamic entropy-aware depth-decay learning rate.
      α_t = 1 + (α_max-1)·σ((H_t-H̄)/κ); lr_{l,t} = lr_base × α_t^((L-l)/L).
      Shallow layers adapt aggressively under severe shift, gently under mild.
    - C2: SAOD — Source-Anchored Output Distillation.
      KL divergence to frozen source model for function-space anchoring.
    - C3: AIS — Anchored Intrinsic Safety.
      Stateless sample safety filter via augmentation stability, low-frequency
      manifold consistency, and source-anchor agreement.
      Operates in sample selection space.
    """

    def __init__(
        self,
        model,
        optimizer=None,
        gate_config=None,
        distill_config=None,
        vit_schedule_config=None,
        sam_config=None,
        adapt_type='gn',
        lr=0.00025,
        optimizer_name='sgd',
        optim_momentum=0.9,
        optim_weight_decay=0.0,
        use_sam=False,
        use_sample_filtering=True,
        num_classes=1000,
        d_margin=0.05,
        reset_constant=0.2,
        # GGES params (legacy, disabled by default)
        grad_clip_norm=1.0,
        perturbation_scale=0.0,  # DEFAULT OFF
        # Ablation switches
        no_gate=False,
        no_proximal=False,     # C2 SAOD on/off
        no_recovery=False,     # C3 AIS on/off (legacy name kept for CLI compat)
        no_anti_redundancy=False,
        no_rslr=False,  # Legacy, ignored
        # C1: D³LR parameters
        d3lr_alpha_max=75.0,   # Maximum depth-decay factor (Std=75, Ctn=50 via config)
        d3lr_kappa=0.5,        # Sigmoid temperature (optimized from 1.0)
        no_ddlr=False,         # Disable D³LR entirely
        # C3: AIS — no hyperparameters needed (binary flip-consistency filter)
        # Legacy DGFR params kept for CLI compat but ignored
        dgfr_momentum=0.1,    # IGNORED (legacy)
        dgfr_gamma=2,         # IGNORED (legacy)
        no_gges=True,    # DEFAULT OFF — GGES disabled
        # Legacy compatibility (ignored)
        fishers=None,
        fisher_alpha=2000.0,
        skip_layer4=False,
        no_dual_gate=False,
        no_fisher=False,
        no_psga=False,
        # Legacy DDSR/DDLR (ignored, kept for CLI compat)
        ddlr_alpha=None,
        ddsr_interval=None,
        ddsr_rho_base=None,
        adaptation_mode='continual',
    ):
        super().__init__()

        self.adapt_type = adapt_type
        self.adaptation_mode = adaptation_mode
        self.use_sam = use_sam
        self.use_sample_filtering = use_sample_filtering
        self.num_classes = num_classes

        self.no_gate = no_gate
        self.no_proximal = no_proximal
        self.no_recovery = no_recovery
        self.no_anti_redundancy = no_anti_redundancy
        self.no_gges = no_gges

        # Stage 1 target: BN backbone + Standard TTA. Keep C1/C2/C3 intact,
        # and add only engineering stabilizers in this regime.
        self._bn_standard_mode = (
            adapt_type == 'bn' and adaptation_mode == 'standard'
        )
        self._bn_continual_mode = (
            adapt_type == 'bn' and adaptation_mode != 'standard'
        )

        # ===== C1: D³LR (Dynamic Depth-Decay Learning Rate) =====
        self.use_d3lr = not no_ddlr
        self.d3lr_alpha_max = d3lr_alpha_max
        self.d3lr_kappa = d3lr_kappa
        if self._bn_standard_mode or self._bn_continual_mode:
            self.d3lr_h_bar = 0.4 * math.log(num_classes)
        else:
            self.d3lr_h_bar = 0.2 * math.log(num_classes)
        self.base_lr = min(lr, 0.00025) if self._bn_continual_mode else lr
        self._current_alpha = d3lr_alpha_max  # initial alpha (will be updated dynamically)
        if self._bn_standard_mode:
            self._effective_d3lr_alpha_max = min(d3lr_alpha_max, 6.0)
        elif self._bn_continual_mode:
            self._effective_d3lr_alpha_max = min(d3lr_alpha_max, 4.0)
        else:
            self._effective_d3lr_alpha_max = d3lr_alpha_max

        # Configure model: use train() mode like EATA
        self.model = configure_model(model, adapt_type)

        # Detect architecture before parameter collection so ViT can use a
        # dedicated depth map and safer transformer-specific fallback logic.
        self._arch_type = self._detect_architecture()
        self._max_depth = self._infer_max_depth()
        self._vit_mode = self._arch_type == 'vit'
        self._vit_continual_mode = (
            self._vit_mode and adaptation_mode != 'standard'
        )
        if self._vit_continual_mode:
            self.d3lr_h_bar = 0.4 * math.log(num_classes)
            self.base_lr = 2e-4
            self._effective_d3lr_alpha_max = min(d3lr_alpha_max, 3.0)
            self._vit_lr_decay_gamma = 0.9998

        # ViT uses the stronger full-LN path only in continual mode; standard
        # ViT falls back to the earlier conservative subset.
        self.params, self.param_names = collect_all_norm_params(
            self.model,
            adapt_type,
            arch_type=self._arch_type,
            vit_full_depth=self._vit_continual_mode,
        )

        if len(self.params) == 0:
            raise ValueError(
                f"No adaptable parameters found for adapt_type='{adapt_type}'."
            )

        logger.info(
            "D³LR: %s",
            (
                "ON (α_max="
                + str(self.d3lr_alpha_max)
                + ", effective="
                + str(self._effective_d3lr_alpha_max)
                + ", κ="
                + str(self.d3lr_kappa)
                + ")"
            ) if self.use_d3lr else "OFF"
        )

        # Batch-level gate (legacy, still useful as pre-check)
        gate_cfg = gate_config or {}
        self.gate_fraction = gate_cfg.get('gate_fraction', 1.0)
        self.use_aux_consistency = (
            gate_cfg.get('use_aux_consistency', True)
            and (not self._bn_continual_mode)
            and (not self._vit_continual_mode)
        )
        self.consistency_patch_ratio = gate_cfg.get('consistency_patch_ratio', 0.125)
        self.model_probs_momentum = gate_cfg.get('model_probs_momentum', 0.9)

        # ===== C2: SAOD (Source-Anchored Output Distillation) =====
        distill_cfg = distill_config or {}
        self.use_source_distill = not no_proximal
        self.distill_temp = distill_cfg.get('temperature', 5.0)
        self.distill_weight = distill_cfg.get('weight', 0.8)
        if self._vit_continual_mode:
            # ViT continual + DEM 主损失：关闭 SAOD。此前在 `distill_cfg` 之后执行
            # `max(weight, 2.0)`，会覆盖构造函数前段对 `distill_weight` 的赋值，
            # 导致实际仍以高权重跑蒸馏，与 Attempt 44 设计不一致。
            self.distill_weight = 0.0

        # ===== C3: AIS (Anchored Intrinsic Safety) =====
        # Uses augmentation stability for cheap fragility detection, then
        # upgrades ViT with a low-frequency manifold projection and
        # source-anchor screening.
        self.use_ais = not no_recovery

        vit_sched_cfg = vit_schedule_config or {}
        self.use_vit_schedule = self._arch_type == 'vit' and vit_sched_cfg.get('enabled', True)
        default_distill_warmup = 16 if self.adaptation_mode == 'standard' else 64
        default_ais_warmup = 4 if self.adaptation_mode == 'standard' else 16
        self.vit_distill_warmup_steps = vit_sched_cfg.get('distill_warmup_steps', default_distill_warmup)
        self.vit_ais_warmup_steps = vit_sched_cfg.get('ais_warmup_steps', default_ais_warmup)
        self.vit_filter_margin = 0.5 * math.log(num_classes) if self._vit_continual_mode else 0.0
        self.vit_weight_margin = 0.4 * math.log(num_classes) if self._vit_continual_mode else 0.0
        self.vit_disable_redundancy = False
        self.vit_adadem = (
            ClassConditionalEMA(num_classes, momentum=0.1)
            if self._vit_continual_mode else None
        )
        self.vit_adadem_weight = 0.0
        self.vit_main_loss = 'dem' if self._vit_continual_mode else (
            'entropy' if self._vit_mode else None
        )
        self.vit_low_rank_ratio = 0.25 if self._vit_mode else 1.0
        self.vit_recovery_threshold = 0.35 if self._vit_continual_mode else 0.0
        self.vit_last_safety_mean = 0.0
        self.vit_last_safety_threshold = 0.0
        self.vit_last_anchor_rate = 0.0
        self.vit_last_high_freq_mean = 0.0
        self.vit_last_plpd_mean = 0.0
        self.vit_last_keep_ratio = 0.0
        self.vit_last_source_conf = 0.0
        self.vit_last_source_disagree = 0.0
        self.bn_distill_warmup_steps = 8 if self._bn_standard_mode else 0
        self.bn_ais_warmup_steps = 2 if self._bn_standard_mode else 0
        self.bn_redundancy_warmup_steps = 8 if self._bn_standard_mode else 0
        self.bn_consistency_warmup_steps = 4 if self._bn_standard_mode else 0
        self.bn_consistency_weight = 0.01 if self._bn_standard_mode else 0.0
        self.bn_probe_size = 16 if self._bn_standard_mode else 0
        self.bn_probe_entropy_ema = None
        self.bn_continual_distill_warmup_steps = 16 if self._bn_continual_mode else 0
        self.bn_continual_ais_warmup_steps = 64 if self._bn_continual_mode else 0
        self.bn_continual_filter_margin = 0.5 * math.log(num_classes) if self._bn_continual_mode else 0.0
        self.bn_continual_weight_margin = 0.4 * math.log(num_classes) if self._bn_continual_mode else 0.0
        self.bn_continual_plpd_threshold = 0.3 if self._bn_continual_mode else 0.0
        self.bn_continual_entropy_weight = 1.0 if self._bn_continual_mode else 0.0
        self.bn_continual_plpd_weight = 1.0 if self._bn_continual_mode else 0.0
        self.bn_continual_disable_redundancy = self._bn_continual_mode
        self.bn_continual_adadem = None
        self.bn_continual_subspace_rank = 0
        self.bn_continual_grad_history = None

        self.optimizer_name = optimizer_name.lower()
        if self._vit_continual_mode:
            self.optimizer_name = 'sgd'
            self.vit_distill_warmup_steps = 4
            self.vit_ais_warmup_steps = 4
        self.optim_momentum = optim_momentum
        self.optim_weight_decay = optim_weight_decay

        # Create frozen source model (for SAOD)
        if self.use_source_distill:
            self.source_model = deepcopy(self.model)
            self.source_model.eval()
            for p in self.source_model.parameters():
                p.requires_grad_(False)
            logger.info(f"SAOD: ON (T={self.distill_temp}, μ={self.distill_weight})")
        else:
            self.source_model = None
            logger.info("SAOD: OFF")

        if self.use_ais:
            if self._vit_mode:
                logger.info("AIS: ON (Anchored Intrinsic Safety for ViT)")
            else:
                logger.info("AIS: ON (flip/augmentation safety filter)")
        else:
            logger.info("AIS: OFF")

        # Store initial params for hard reset
        self.initial_params = {}
        for name, param in self.model.named_parameters():
            if name in self.param_names:
                self.initial_params[name] = param.data.clone()

        # Build depth-grouped param info
        self._param_depth = {}
        self.depth_param_info = {}
        for name in self.param_names:
            d = self._get_layer_depth(name)
            self._param_depth[name] = d
            if d not in self.depth_param_info:
                self.depth_param_info[d] = {'names': [], 'count': 0}
            self.depth_param_info[d]['names'].append(name)
            self.depth_param_info[d]['count'] += self.initial_params[name].numel()

        # BN + standard works better with the broader SAR/SURGEON-style margin.
        self.e_margin = (
            0.4 * math.log(num_classes)
            if self._bn_standard_mode
            else (math.log(num_classes) / 2.0 - 1.0)
        )
        self.selection_margin = (
            self.vit_filter_margin if self._vit_continual_mode else (
                self.bn_continual_filter_margin if self._bn_continual_mode else self.e_margin
            )
        )
        self.weight_margin = (
            self.vit_weight_margin if self._vit_continual_mode else (
                self.bn_continual_weight_margin if self._bn_continual_mode else self.e_margin
            )
        )

        # Anti-redundancy
        self.current_model_probs = None
        if self._vit_continual_mode:
            self.d_margin = max(d_margin, 0.20)
        elif self._bn_standard_mode:
            self.d_margin = max(d_margin, 0.15)
        else:
            self.d_margin = d_margin

        # Save source state for recovery
        self.model_state = deepcopy(model.state_dict())

        # Setup optimizer: SGD like EATA, with layer-wise lr for D³LR
        if optimizer is None:
            optimizer_name_l = self.optimizer_name
            if self.use_d3lr:
                param_groups = self._build_d3lr_param_groups(self.base_lr)
                optimizer_params = param_groups
            else:
                optimizer_params = self.params

            if optimizer_name_l == 'adam':
                base_optimizer = optim.Adam(
                    optimizer_params, lr=self.base_lr, weight_decay=self.optim_weight_decay
                )
            elif optimizer_name_l == 'adamw':
                base_optimizer = optim.AdamW(
                    optimizer_params, lr=self.base_lr, weight_decay=self.optim_weight_decay
                )
            else:
                base_optimizer = optim.SGD(
                    optimizer_params, lr=self.base_lr, momentum=self.optim_momentum,
                    weight_decay=self.optim_weight_decay
                )
        else:
            base_optimizer = optimizer

        sam_cfg = sam_config or {}
        if self.use_sam:
            self.sam_optimizer = SAMOptimizer(
                params=self.params,
                base_optimizer=base_optimizer,
                rho=sam_cfg.get('rho', 0.05),
            )
            self.optimizer = base_optimizer
        else:
            self.sam_optimizer = None
            self.optimizer = base_optimizer

        # Save optimizer state for recovery
        self.optimizer_state = deepcopy(
            (self.sam_optimizer.base_optimizer if self.sam_optimizer else self.optimizer
            ).state_dict()
        )

        # Stats
        self.step_count = 0
        self.update_count = 0
        self.skip_count = 0
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0
        self.recovery_count = 0

    # ========================
    # Architecture Detection
    # ========================

    def _detect_architecture(self):
        """Detect whether the model is ResNet-style or WideResNet-style."""
        child_names = [name for name, _ in self.model.named_children()]
        if 'layer1' in child_names:
            return 'resnet'
        elif 'block1' in child_names:
            return 'wideresnet'
        elif _looks_like_vit_backbone(self.model):
            return 'vit'
        else:
            return 'resnet'  # default fallback

    def _infer_max_depth(self):
        if self._arch_type == 'wideresnet':
            return 3
        if self._arch_type == 'vit':
            blocks = getattr(self.model, 'blocks', None)
            if blocks is not None and len(blocks) > 0:
                return len(blocks)
            return 12
        return 4

    def _get_layer_depth(self, param_name):
        """Map a parameter name to its depth index."""
        if self._arch_type == 'wideresnet':
            # WRN: block1(shallow) → block2(mid) → block3(deep) → bn1(deepest)
            if param_name.startswith('block1'):
                return 1
            elif param_name.startswith('block2'):
                return 2
            elif param_name.startswith('block3'):
                return 3
            elif param_name.startswith('bn1') or param_name.startswith('gn1'):
                # In WRN, bn1 is AFTER block3, so it's the deepest
                return 3  # same depth as block3 (deepest stage)
            else:
                return 2  # default: middle depth
        elif self._arch_type == 'vit':
            if param_name.startswith('patch_embed') or param_name.startswith('norm_pre'):
                return 0
            block_idx = _parse_vit_block_index(param_name)
            if block_idx is not None:
                return min(block_idx + 1, self._max_depth)
            if param_name == 'norm.weight' or param_name == 'norm.bias' or param_name.startswith('norm.'):
                return self._max_depth
            return max(1, self._max_depth // 2)
        else:
            # ResNet: stem(0) → layer1(1) → ... → layer4(4)
            if param_name.startswith('bn1') or param_name.startswith('gn1'):
                return 0  # stem
            elif param_name.startswith('layer1'):
                return 1
            elif param_name.startswith('layer2'):
                return 2
            elif param_name.startswith('layer3'):
                return 3
            elif param_name.startswith('layer4'):
                return 4
            else:
                return 2  # default: middle depth

    def _get_module_depth(self, module_name):
        """Map a top-level module name to depth index (for hook registration)."""
        if self._arch_type == 'wideresnet':
            mapping = {'block1': 1, 'block2': 2, 'block3': 3}
        elif self._arch_type == 'vit':
            block_idx = _parse_vit_block_index(module_name)
            if block_idx is not None:
                return min(block_idx + 1, self._max_depth)
            if module_name.startswith('patch_embed') or module_name.startswith('norm_pre'):
                return 0
            if module_name == 'norm' or module_name.startswith('norm.'):
                return self._max_depth
            return None
        else:
            mapping = {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4': 4}
        return mapping.get(module_name, None)

    def _get_vit_schedule_factors(self):
        if not self.use_vit_schedule:
            return 1.0, self.use_ais

        distill_factor = min(
            1.0,
            float(self.step_count + 1) / max(float(self.vit_distill_warmup_steps), 1.0)
        )
        ais_active = self.use_ais and self.step_count >= self.vit_ais_warmup_steps
        return distill_factor, ais_active

    def _get_adaptation_factors(self):
        """Return schedule factors for distillation / AIS / consistency."""
        if self._bn_standard_mode:
            distill_factor = min(
                1.0,
                float(self.step_count + 1) / max(float(self.bn_distill_warmup_steps), 1.0)
            )
            ais_active = self.use_ais and self.step_count >= self.bn_ais_warmup_steps
            consistency_factor = min(
                1.0,
                float(self.step_count + 1) / max(float(self.bn_consistency_warmup_steps), 1.0)
            )
            return distill_factor, ais_active, consistency_factor

        if self._bn_continual_mode:
            distill_factor = min(
                1.0,
                float(self.step_count + 1) / max(float(self.bn_continual_distill_warmup_steps), 1.0)
            )
            ais_active = self.use_ais and self.step_count >= self.bn_continual_ais_warmup_steps
            return distill_factor, ais_active, 0.0

        distill_factor, ais_active = self._get_vit_schedule_factors()
        return distill_factor, ais_active, 0.0

    def _build_aux_consistency_view(self, x):
        if x.ndim != 4:
            return x
        _, _, h, w = x.shape
        patch_size = max(1, int(min(h, w) * self.consistency_patch_ratio))
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

    def _build_bn_stability_view(self, x):
        """Weak tensor-only augmentation used for Stage 1 consistency regularization."""
        if x.ndim != 4:
            return x

        x_aug = x.clone()
        _, _, h, w = x_aug.shape

        if torch.rand(1, device=x.device).item() > 0.5:
            x_aug = torch.flip(x_aug, dims=[3])

        max_shift = 2 if min(h, w) <= 64 else 4
        if max_shift > 0:
            shift_h = int(torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item())
            shift_w = int(torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item())
            x_aug = torch.roll(x_aug, shifts=(shift_h, shift_w), dims=(2, 3))

        noise_scale = 0.02 if min(h, w) <= 64 else 0.01
        x_aug = x_aug + torch.randn_like(x_aug) * noise_scale
        return x_aug

    def _build_low_rank_view(self, x):
        """Low-frequency projection used as a cheap manifold-preserving view."""
        if x.ndim != 4:
            return x

        _, _, h, w = x.shape
        min_side = min(h, w)
        target_side = max(28, int(min_side * self.vit_low_rank_ratio))
        target_side = min(target_side, min_side)
        if target_side >= min_side:
            return x

        x_low = F.interpolate(
            x,
            size=(target_side, target_side),
            mode='bilinear',
            align_corners=False,
        )
        return F.interpolate(
            x_low,
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        )

    def _build_mid_rank_view(self, x):
        """Mid-frequency projection for blur/zoom-sensitive manifold scoring."""
        if x.ndim != 4:
            return x

        _, _, h, w = x.shape
        min_side = min(h, w)
        target_side = max(56, int(min_side * 0.5))
        target_side = min(target_side, min_side)
        if target_side >= min_side:
            return x

        x_mid = F.interpolate(
            x,
            size=(target_side, target_side),
            mode='bilinear',
            align_corners=False,
        )
        return F.interpolate(
            x_mid,
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        )

    def _estimate_probe_entropy(self, x):
        """Small probe pass to gauge current shift severity."""
        if not self._bn_standard_mode:
            return None

        probe_count = min(self.bn_probe_size, x.shape[0])
        if probe_count <= 0:
            return None

        probe_indices = torch.randperm(x.shape[0], device=x.device)[:probe_count]
        with torch.no_grad():
            probe_logits = self.model(x[probe_indices])
            probe_entropy = softmax_entropy(probe_logits).mean().detach()
        self.bn_probe_entropy_ema = update_ema(self.bn_probe_entropy_ema, probe_entropy)
        return self.bn_probe_entropy_ema

    def _build_plpd_view(self, x):
        """DeYO-style object-destructive patch shuffle view."""
        if x.ndim != 4:
            return x

        b, c, h, w = x.shape
        patch_len = 4 if min(h, w) <= 64 else 8
        usable_h = (h // patch_len) * patch_len
        usable_w = (w // patch_len) * patch_len
        x_crop = x[:, :, :usable_h, :usable_w]

        num_h = usable_h // patch_len
        num_w = usable_w // patch_len
        patches = x_crop.view(b, c, num_h, patch_len, num_w, patch_len)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            b, num_h * num_w, c, patch_len, patch_len
        )

        perm = torch.stack(
            [torch.randperm(num_h * num_w, device=x.device) for _ in range(b)],
            dim=0,
        )
        patches = patches[torch.arange(b, device=x.device).unsqueeze(1), perm]
        x_perm = patches.reshape(b, num_h, num_w, c, patch_len, patch_len)
        x_perm = x_perm.permute(0, 3, 1, 4, 2, 5).reshape(b, c, usable_h, usable_w)

        if usable_h != h or usable_w != w:
            x_perm = F.interpolate(
                x_perm, size=(h, w), mode='bilinear', align_corners=False
            )
        return x_perm

    def _compute_adadem_loss(self, logits, tracker, reduction: str = 'mean', skip_update: bool = False):
        """
        @param skip_update: when True, skip tracker.update() (use when
            tracker was already updated with a wider sample pool).
        """
        if tracker is None or logits.numel() == 0:
            return None

        probs = F.softmax(logits, dim=1)
        if skip_update:
            pseudo_labels = probs.argmax(dim=1)
        else:
            pseudo_labels = tracker.update(probs)
        avg_pred = tracker.avg_pred
        if avg_pred is None:
            return None

        with torch.no_grad():
            entropy_term = -(probs * logits).sum(1, keepdim=True)
            grad_weight = (logits + entropy_term + 1.0) * probs
            grad_weight = grad_weight.abs().sum(1, keepdim=True).clamp_min(1e-6)

        corrected = (
            probs - avg_pred[pseudo_labels].detach()
        ) / grad_weight.detach()
        loss = -(corrected * logits).sum(1)
        if reduction == 'none':
            return loss
        return loss.mean(0)

    def _compute_safety_threshold(self, scores: torch.Tensor) -> Optional[torch.Tensor]:
        if scores.numel() == 0:
            return None
        mean_score = scores.mean()
        median_score = scores.median()
        threshold = 0.5 * (mean_score + median_score)
        return threshold.clamp_(0.15, 0.85)

    def _forward_aux_view_logits(self, x_view: torch.Tensor) -> torch.Tensor:
        """Run auxiliary no-grad ViT views with lower peak memory when CUDA is used."""
        if x_view.is_cuda:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = self.model(x_view)
            return logits.float()
        return self.model(x_view)

    def _estimate_high_frequency_ratio(
        self,
        x: torch.Tensor,
        low_rank_view: torch.Tensor,
    ) -> torch.Tensor:
        residual = (x - low_rank_view).abs().mean(dim=(1, 2, 3))
        base = x.abs().mean(dim=(1, 2, 3)).clamp_min(1e-6)
        return (residual / base).clamp_(0.0, 2.0)

    def _apply_vit_intrinsic_safety(
        self,
        x: torch.Tensor,
        outputs: torch.Tensor,
        source_outputs: Optional[torch.Tensor],
        reliable_indices: torch.Tensor,
        ais_active: bool,
        selection_margin: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """ViT C3: ranked manifold + anchor safety filter."""
        if (
            (not self._vit_mode)
            or (not self.use_ais)
            or reliable_indices.numel() == 0
        ):
            return reliable_indices, None, None

        with torch.no_grad():
            orig_logits = outputs.detach()
            orig_preds = orig_logits.argmax(dim=1)
            rel_x = x[reliable_indices]
            rel_logits = orig_logits[reliable_indices]
            rel_preds = orig_preds[reliable_indices]
            orig_probs = F.softmax(orig_logits, dim=1)

            low_rank_view = self._build_low_rank_view(x)
            low_rank_logits = self._forward_aux_view_logits(low_rank_view)
            coarse_scores = normalized_logit_hsic(orig_logits, low_rank_logits)
            low_rank_agree = (low_rank_logits.argmax(dim=1) == orig_preds)
            low_rank_high_freq = self._estimate_high_frequency_ratio(x, low_rank_view)
            del low_rank_view, low_rank_logits

            mid_rank_view = self._build_mid_rank_view(x)
            mid_rank_logits = self._forward_aux_view_logits(mid_rank_view)
            mid_scores = normalized_logit_hsic(orig_logits, mid_rank_logits)
            mid_rank_agree = (mid_rank_logits.argmax(dim=1) == orig_preds)
            mid_rank_high_freq = self._estimate_high_frequency_ratio(x, mid_rank_view)
            del mid_rank_view, mid_rank_logits

            low_rank_scores = 0.4 * coarse_scores + 0.6 * mid_scores
            low_rank_agree = low_rank_agree | mid_rank_agree
            high_freq_ratio = 0.5 * (low_rank_high_freq + mid_rank_high_freq)
            high_freq_weight = torch.sigmoid(
                6.0 * (high_freq_ratio - high_freq_ratio.median())
            )
            safety_scores = low_rank_scores
            support_scores = low_rank_agree.float()

            destroy_view = self._build_plpd_view(x)
            destroy_logits = self._forward_aux_view_logits(destroy_view)
            destroy_probs = F.softmax(destroy_logits, dim=1)
            del destroy_view, destroy_logits
            top1 = orig_probs.argmax(dim=1, keepdim=True)
            plpd_scores = (
                orig_probs.gather(1, top1) - destroy_probs.gather(1, top1)
            ).reshape(-1)
            plpd_threshold = self._compute_safety_threshold(plpd_scores[reliable_indices])
            if plpd_threshold is not None:
                plpd_support = (plpd_scores >= plpd_threshold).float()
                plpd_scores = torch.sigmoid(10.0 * (plpd_scores - plpd_threshold))
                safety_scores = (
                    (1.0 - high_freq_weight) * safety_scores
                    + high_freq_weight * plpd_scores
                )
                support_scores = (
                    (1.0 - high_freq_weight) * support_scores
                    + high_freq_weight * plpd_support
                )

            if ais_active:
                flip_logits = self._forward_aux_view_logits(torch.flip(x, dims=[3]))
                flip_scores = normalized_logit_hsic(orig_logits, flip_logits)
                safety_scores = 0.5 * safety_scores + 0.5 * flip_scores
                support_scores = torch.maximum(
                    support_scores,
                    (flip_logits.argmax(dim=1) == orig_preds).float(),
                )
                del flip_logits

            anchor_mask = torch.ones_like(low_rank_agree, dtype=torch.bool)
            if source_outputs is not None:
                source_logits = source_outputs.detach()
                source_scores = normalized_logit_hsic(orig_logits, source_logits)
                safety_scores = 0.7 * safety_scores + 0.3 * source_scores
                source_preds = source_logits.argmax(dim=1)
                source_entropy_raw = softmax_entropy(source_logits)
                anchor_threshold = self._compute_safety_threshold(
                    source_scores[reliable_indices]
                )
                if anchor_threshold is not None:
                    anchor_mask = (
                        (source_entropy_raw > selection_margin)
                        | (source_preds == orig_preds)
                        | (source_scores >= anchor_threshold)
                    )

            candidate_scores = safety_scores[reliable_indices]
            safety_threshold = self._compute_safety_threshold(candidate_scores)
            if safety_threshold is None:
                return reliable_indices, safety_scores, safety_scores

            candidate_support_scores = support_scores[reliable_indices].clamp_(0.0, 1.0)
            candidate_anchor_mask = anchor_mask[reliable_indices]
            rank_scores = candidate_scores * (0.55 + 0.45 * candidate_support_scores)
            rank_scores = rank_scores * (0.8 + 0.2 * candidate_anchor_mask.float())
            rank_threshold = self._compute_safety_threshold(rank_scores)
            if rank_threshold is None:
                selection_scores = safety_scores.clone()
                selection_scores[reliable_indices] = rank_scores
                return reliable_indices, safety_scores, selection_scores
            safe_mask = (
                (rank_scores >= rank_threshold)
                & ((candidate_support_scores >= 0.35) | candidate_anchor_mask)
            )

            rel_entropy = softmax_entropy(rel_logits).mean().detach()
            entropy_pressure = (
                rel_entropy / max(selection_margin, 1e-6)
            ).clamp(0.0, 1.0)
            high_freq_mean = float(
                high_freq_ratio[reliable_indices].mean().clamp(0.0, 1.0).item()
            )
            anchor_rate = float(candidate_anchor_mask.float().mean().item())
            low_freq_keep = 1.0 - float(
                high_freq_mean
            )
            difficulty_score = min(
                1.0,
                0.45 * high_freq_mean
                + 0.35 * float(entropy_pressure.item())
                + 0.20 * (1.0 - anchor_rate),
            )
            target_keep_ratio = min(
                0.55,
                max(
                    0.25,
                    0.22
                    + 0.10 * max(0.0, low_freq_keep)
                    + 0.18 * difficulty_score,
                ),
            )
            target_keep = min(
                reliable_indices.numel(),
                max(1, int(math.ceil(reliable_indices.numel() * target_keep_ratio))),
            )

            if safe_mask.sum() < target_keep:
                chosen_local_ids = torch.topk(rank_scores, k=target_keep, largest=True).indices
            else:
                chosen_local_ids = torch.where(safe_mask)[0]

            reliable_indices = reliable_indices[chosen_local_ids]
            selection_scores = safety_scores.clone()
            selection_scores[reliable_indices] = rank_scores[chosen_local_ids]

            self.vit_last_safety_mean = float(candidate_scores.mean().item())
            self.vit_last_safety_threshold = float(safety_threshold.item())
            self.vit_last_anchor_rate = float(candidate_anchor_mask.float().mean().item())
            self.vit_last_high_freq_mean = float(high_freq_ratio[reliable_indices].mean().item())
            self.vit_last_plpd_mean = float(plpd_scores[reliable_indices].mean().item())
            self.vit_last_keep_ratio = float(target_keep_ratio)

        return reliable_indices, safety_scores, selection_scores

    def _flatten_adapt_grads(self):
        flat_grads = []
        for p in self.params:
            if p.grad is None:
                flat_grads.append(torch.zeros_like(p).reshape(-1))
            else:
                flat_grads.append(p.grad.detach().reshape(-1))
        if not flat_grads:
            return None
        return torch.cat(flat_grads, dim=0)

    def _assign_flat_grads(self, flat_grad):
        offset = 0
        for p in self.params:
            numel = p.numel()
            grad_slice = flat_grad[offset:offset + numel].view_as(p)
            if p.grad is None:
                p.grad = grad_slice.clone()
            else:
                p.grad.copy_(grad_slice)
            offset += numel

    def _project_grad_to_history(self, flat_grad):
        if (
            not self._bn_continual_mode
            or self.bn_continual_grad_history is None
            or len(self.bn_continual_grad_history) < self.bn_continual_subspace_rank
        ):
            return None

        hist = torch.stack(list(self.bn_continual_grad_history), dim=0)
        hist = hist - hist.mean(dim=0, keepdim=True)
        if torch.allclose(hist, torch.zeros_like(hist)):
            return None

        try:
            _, _, vh = torch.linalg.svd(hist, full_matrices=False)
        except RuntimeError:
            return None

        k = min(int(self.bn_continual_subspace_rank), vh.shape[0])
        if k <= 0:
            return None

        proj_basis = vh[:k, :].to(device=flat_grad.device, dtype=flat_grad.dtype)
        g = flat_grad.reshape(-1, 1)
        return (proj_basis.t() @ (proj_basis @ g)).reshape(-1)

    # ========================
    # C1: D³LR
    # ========================

    def _build_d3lr_param_groups(self, base_lr):
        """
        C1: Build per-layer param groups with depth-decaying learning rates.

        Initial lr uses α_max (will be dynamically modulated per-batch).
        lr_l = base_lr * α^((L-l)/L)
        """
        L = self._max_depth
        alpha = self._effective_d3lr_alpha_max  # initial alpha

        # Group params by depth. ViT keeps a small extra lr scale to favor the
        # attention path (`norm1`) over `norm2/final norm`, which is a cleaner
        # C1 refinement than increasing the whole ViT step size.
        max_depth_idx = L + 1 if self._arch_type == 'resnet' else L + 1
        depth_params = {}
        for param, name in zip(self.params, self.param_names):
            d = self._get_layer_depth(name)
            lr_scale = _get_vit_lr_scale(name) if self._arch_type == 'vit' else 1.0
            key = (d, lr_scale)
            depth_params.setdefault(key, []).append(param)

        param_groups = []
        for (d, lr_scale) in sorted(depth_params.keys()):
            if depth_params[(d, lr_scale)]:
                lr_d = base_lr * lr_scale * (alpha ** ((L - d) / L))
                param_groups.append({
                    'params': depth_params[(d, lr_scale)],
                    'lr': lr_d,
                    'depth': d,  # stored for dynamic D³LR updates
                    'lr_scale': lr_scale,
                })
                logger.info(
                    f"  D³LR depth={d}, scale={lr_scale:.2f}: "
                    f"{len(depth_params[(d, lr_scale)])} params, "
                    f"lr={lr_d:.6f} ({lr_d/base_lr:.2f}x)"
                )

        return param_groups

    def _update_d3lr(self, batch_entropy):
        """
        C1: Update optimizer learning rates based on batch entropy.

        α_t = 1.0 + (α_max - 1.0) · σ((H_t - H̄) / κ)
        lr_{l,t} = lr_base × α_t^((L-l)/L)
        """
        L = self._max_depth
        H_t = batch_entropy
        H_bar = self.d3lr_h_bar
        kappa = self.d3lr_kappa

        # Sigmoid modulation: high entropy → α closer to α_max
        sigmoid_val = 1.0 / (1.0 + math.exp(-(H_t - H_bar) / max(kappa, 1e-8)))
        alpha_t = 1.0 + (self._effective_d3lr_alpha_max - 1.0) * sigmoid_val
        self._current_alpha = alpha_t

        # Update each param group's lr
        opt = self.sam_optimizer.base_optimizer if self.sam_optimizer else self.optimizer
        for group in opt.param_groups:
            depth = group.get('depth', None)
            if depth is not None:
                lr_scale = group.get('lr_scale', 1.0)
                group['lr'] = self.base_lr * lr_scale * (alpha_t ** ((L - depth) / L))

    # ========================
    # C3: AIS (Anchored Intrinsic Safety)
    # ========================
    # AIS operates inline in _forward_and_adapt:
    #   BN/WRN: flip/augmentation stability.
    #   ViT continual: low-frequency manifold projection + source anchor
    #   agreement + optional flip stability.
    # No state is maintained beyond scalar logging stats.

    # ========================
    # Forward + Adapt
    # ========================

    def forward(self, x):
        return self._forward_and_adapt(x)

    @torch.enable_grad()
    def _forward_and_adapt(self, x):
        """
        Core adaptation loop.

        Pipeline:
        1. Source model forward (for SAOD)
        2. Adapted model forward
        2b. C3 AIS: safety views (flip and/or low-frequency manifold projection)
        3. Batch-level gate (skip if all unreliable)
        4. Filter 1: entropy threshold (H < E₀)
        4b. Filter 1b: C3 AIS stability filter (argmax(p) == argmax(p_flip))
        5. Filter 2: anti-redundancy (cosine sim < d_margin)
        6. C1: D³LR — update optimizer lr based on batch entropy
        7. Entropy-weighted loss (EATA Eqn.3) + C2 SAOD loss
        8. Backward + step
        """
        distill_factor, ais_active, consistency_factor = self._get_adaptation_factors()
        probe_entropy = self._estimate_probe_entropy(x)
        if probe_entropy is not None:
            severity = float(
                torch.clamp(
                    probe_entropy / max(self.e_margin, 1e-6),
                    min=0.5,
                    max=1.5,
                ).item()
            )
            distill_factor *= severity
            consistency_factor *= severity

        # === Step 1: Source model forward (SAOD anchoring) ===
        source_outputs = None
        if self.use_source_distill:
            with torch.no_grad():
                if x.is_cuda:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        source_outputs = self.source_model(x).float()
                else:
                    source_outputs = self.source_model(x)

        # === Step 2: Adapted model forward ===
        outputs = self.model(x)
        entropys = softmax_entropy(outputs)
        aug_outputs = None
        if self._bn_standard_mode:
            aug_outputs = self.model(self._build_bn_stability_view(x))
        selection_margin = self.selection_margin
        weight_margin = self.weight_margin
        plpd_scores = None
        if self._bn_continual_mode:
            with torch.no_grad():
                destroy_outputs = self.model(self._build_plpd_view(x))
                probs = outputs.softmax(1)
                destroy_probs = destroy_outputs.softmax(1)
                top1 = probs.argmax(dim=1, keepdim=True)
                plpd_scores = (
                    probs.gather(1, top1) - destroy_probs.gather(1, top1)
                ).reshape(-1)

        # === Step 2b: C3 AIS — auxiliary safety views ===
        # BN branches use a light flip-stability check.
        flip_preds = None
        if ais_active and (not self._vit_mode):
            with torch.no_grad():
                x_flip = torch.flip(x, dims=[3])  # horizontal flip (W dim)
                flip_preds = self.model(x_flip).argmax(dim=1)  # [B]

        perturb_preds = None
        if self.use_aux_consistency:
            with torch.no_grad():
                x_aux = self._build_aux_consistency_view(x)
                perturb_preds = self.model(x_aux).argmax(dim=1)

        # === Step 3: Batch-level entropy gating (pre-check) ===
        if not self.no_gate:
            reliable_ratio = (entropys < selection_margin).float().mean().item()
            if reliable_ratio <= (1.0 - self.gate_fraction):
                self.skip_count += 1
                self.step_count += 1
                return outputs.detach()

        # === Step 4: Filter 1 — entropy reliability (EATA Eqn.2) ===
        reliable_indices = torch.where(entropys < selection_margin)[0]

        # === Step 4b: C3 AIS — safety filter ===
        if ais_active and flip_preds is not None and reliable_indices.numel() > 0:
            orig_preds = outputs[reliable_indices].argmax(dim=1)
            flip_preds_filtered = flip_preds[reliable_indices]
            stable_mask = (orig_preds == flip_preds_filtered)
            reliable_indices = reliable_indices[stable_mask]

        vit_safety_scores = None
        vit_selection_scores = None
        pre_c3_reliable_indices = reliable_indices.clone() if (
            self._vit_continual_mode and reliable_indices.numel() > 0
        ) else None
        if self._vit_mode and reliable_indices.numel() > 0:
            reliable_indices, vit_safety_scores, vit_selection_scores = self._apply_vit_intrinsic_safety(
                x=x,
                outputs=outputs,
                source_outputs=source_outputs,
                reliable_indices=reliable_indices,
                ais_active=ais_active,
                selection_margin=selection_margin,
            )

        if perturb_preds is not None and reliable_indices.numel() > 0:
            orig_preds = outputs[reliable_indices].argmax(dim=1)
            perturb_preds_filtered = perturb_preds[reliable_indices]
            aux_stable_mask = (orig_preds == perturb_preds_filtered)
            reliable_indices = reliable_indices[aux_stable_mask]

        if self._bn_continual_mode and plpd_scores is not None and reliable_indices.numel() > 0:
            plpd_mask = plpd_scores[reliable_indices] > self.bn_continual_plpd_threshold
            reliable_indices = reliable_indices[plpd_mask]

        selected_indices = reliable_indices
        entropys_filtered = entropys[selected_indices]
        outputs_filtered = outputs[selected_indices]
        plpd_filtered = plpd_scores[selected_indices] if plpd_scores is not None and selected_indices.numel() > 0 else None
        self.num_samples_update_1 += selected_indices.size(0)

        # === Step 5: Filter 2 — anti-redundancy (EATA Eqn.4) ===
        use_anti_redundancy = (
            (not self.no_anti_redundancy)
            and (not self.bn_continual_disable_redundancy)
            and (not self.vit_disable_redundancy)
            and self.current_model_probs is not None
            and (
                (not self._bn_standard_mode)
                or self.step_count >= self.bn_redundancy_warmup_steps
            )
        )
        if use_anti_redundancy:
            cosine_similarities = F.cosine_similarity(
                self.current_model_probs.unsqueeze(dim=0),
                outputs_filtered.softmax(1),
                dim=1
            )
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)[0]
            selected_indices = selected_indices[filter_ids_2]
            entropys_filtered = entropys[selected_indices]
            outputs_filtered = outputs[selected_indices]
            if plpd_filtered is not None:
                plpd_filtered = plpd_filtered[filter_ids_2]
            updated_probs = update_model_probs(
                self.current_model_probs,
                outputs_filtered.softmax(1),
                momentum=self.model_probs_momentum,
            )
        else:
            updated_probs = update_model_probs(
                self.current_model_probs,
                outputs_filtered.softmax(1),
                momentum=self.model_probs_momentum,
            )

        self.current_model_probs = updated_probs
        self.num_samples_update_2 += entropys_filtered.size(0)

        n_filtered = entropys_filtered.size(0)
        if n_filtered == 0:
            self.optimizer.zero_grad()
            self.step_count += 1
            return outputs.detach()

        vit_entropy_scale = 1.0
        vit_distill_scale = 1.0
        vit_lr_bonus = 1.0
        vit_trigger_recovery = False
        source_conf_mean = 0.0
        source_disagree_rate = 0.0
        selected_score_weights = None
        if self._vit_continual_mode and vit_safety_scores is not None:
            safety_mean = vit_safety_scores[selected_indices].mean().detach().clamp(0.0, 1.0)
            safety_mean_f = float(safety_mean.item())
            low_freq_bonus = max(0.0, 1.0 - self.vit_last_high_freq_mean)
            vit_entropy_scale = min(
                1.15,
                0.45 + 0.45 * safety_mean_f + 0.25 * low_freq_bonus,
            )
            vit_distill_scale = max(
                0.8,
                1.55 - 0.45 * safety_mean_f - 0.25 * low_freq_bonus,
            )
            vit_lr_bonus = min(
                1.15,
                1.0 + 0.15 * low_freq_bonus * max(0.0, safety_mean_f - 0.4),
            )
            effective_recovery_threshold = max(
                0.2,
                self.vit_recovery_threshold - 0.15 * self.vit_last_high_freq_mean,
            )
            vit_trigger_recovery = (
                safety_mean_f < effective_recovery_threshold
                and self.vit_last_anchor_rate < 0.5
            )

        if (
            self._vit_continual_mode
            and self.use_source_distill
            and source_outputs is not None
            and selected_indices.numel() > 0
        ):
            src_logits_sel = source_outputs[selected_indices].detach()
            src_entropy_sel = softmax_entropy(src_logits_sel)
            max_entropy = max(math.log(src_logits_sel.shape[1]), 1e-6)
            source_conf_mean = float(
                (1.0 - src_entropy_sel / max_entropy).clamp(0.0, 1.0).mean().item()
            )
            source_disagree_rate = float(
                (
                    src_logits_sel.argmax(dim=1)
                    != outputs_filtered.detach().argmax(dim=1)
                ).float().mean().item()
            )
            self.vit_last_source_conf = source_conf_mean
            self.vit_last_source_disagree = source_disagree_rate
            structured_bonus = max(0.0, 1.0 - self.vit_last_high_freq_mean)
            structured_disagree = structured_bonus * source_disagree_rate
            vit_distill_scale = min(
                1.9,
                vit_distill_scale
                + 0.15 * structured_bonus * source_conf_mean
                + 0.10 * structured_disagree,
            )

        # === Step 6: C1 D³LR — update lr based on batch entropy ===
        if self.use_d3lr:
            batch_entropy = entropys_filtered.mean().item()
            self._update_d3lr(batch_entropy)
            if self._vit_continual_mode and vit_lr_bonus != 1.0:
                opt = self.sam_optimizer.base_optimizer if self.sam_optimizer else self.optimizer
                for group in opt.param_groups:
                    group['lr'] *= vit_lr_bonus

        # === Step 7: Main loss + C2 SAOD ===
        coeff = 1.0 / (torch.exp(entropys_filtered.clone().detach() - weight_margin))
        if self._vit_continual_mode and vit_safety_scores is not None:
            selected_score_weights = (
                vit_selection_scores if vit_selection_scores is not None
                else vit_safety_scores
            )[selected_indices].detach()
            coeff = coeff * (0.5 + selected_score_weights)
        if self._bn_continual_mode and plpd_filtered is not None:
            coeff = (
                self.bn_continual_entropy_weight * coeff
                + self.bn_continual_plpd_weight * torch.exp(plpd_filtered.clone().detach())
            )
        if self._vit_continual_mode and self.vit_main_loss == 'dem':
            if (
                pre_c3_reliable_indices is not None
                and pre_c3_reliable_indices.numel() > 0
                and self.vit_adadem is not None
            ):
                with torch.no_grad():
                    wide_probs = F.softmax(
                        outputs[pre_c3_reliable_indices].detach(), dim=1
                    )
                    self.vit_adadem.update(wide_probs)
            dem_per_sample = self._compute_adadem_loss(
                outputs_filtered, self.vit_adadem,
                reduction='none', skip_update=True,
            )
            if dem_per_sample is not None:
                loss = vit_entropy_scale * (dem_per_sample * coeff).mean(0)
            else:
                loss = vit_entropy_scale * (entropys_filtered * coeff).mean(0)
        else:
            loss = vit_entropy_scale * (entropys_filtered * coeff).mean(0)

        # C2: SAOD loss
        if self.use_source_distill and source_outputs is not None:
            T = self.distill_temp
            source_logits_sel = source_outputs[selected_indices]
            source_probs = F.softmax(source_logits_sel / T, dim=1)
            adapted_log_probs = F.log_softmax(outputs_filtered / T, dim=1)
            if self._vit_continual_mode:
                source_entropy_sel = softmax_entropy(source_logits_sel.detach())
                max_entropy = max(math.log(source_logits_sel.shape[1]), 1e-6)
                source_conf = (1.0 - source_entropy_sel / max_entropy).clamp(0.0, 1.0)
                source_disagree = (
                    source_logits_sel.detach().argmax(dim=1)
                    != outputs_filtered.detach().argmax(dim=1)
                ).float()
                structured_bonus = max(0.0, 1.0 - self.vit_last_high_freq_mean)
                structured_disagree = structured_bonus * source_disagree
                anchor_weights = (
                    0.90
                    + 0.15 * source_conf
                    + 0.10 * structured_disagree
                    + 0.05 * structured_bonus
                ).detach()
                per_sample_kl = F.kl_div(
                    adapted_log_probs,
                    source_probs,
                    reduction='none',
                ).sum(dim=1)
                distill_loss = T * T * (anchor_weights * per_sample_kl).mean(0)
            else:
                distill_loss = T * T * F.kl_div(
                    adapted_log_probs, source_probs,
                    reduction='batchmean'
                )
            loss = loss + (
                vit_distill_scale
                * (distill_factor * self.distill_weight)
                * distill_loss
            )

        if self._vit_continual_mode and self.vit_main_loss != 'dem':
            if (
                pre_c3_reliable_indices is not None
                and pre_c3_reliable_indices.numel() > 0
                and self.vit_adadem is not None
            ):
                with torch.no_grad():
                    wide_probs = F.softmax(
                        outputs[pre_c3_reliable_indices].detach(), dim=1
                    )
                    self.vit_adadem.update(wide_probs)
            vit_adadem_loss = self._compute_adadem_loss(
                outputs_filtered,
                self.vit_adadem,
                reduction='mean',
                skip_update=True,
            )
            if vit_adadem_loss is not None:
                loss = loss + self.vit_adadem_weight * vit_adadem_loss

        if (
            self._bn_standard_mode
            and aug_outputs is not None
            and selected_indices.numel() > 0
            and consistency_factor > 0.0
        ):
            aug_outputs_filtered = aug_outputs[selected_indices]
            consistency_loss = consistency_cross_entropy(outputs_filtered, aug_outputs_filtered)
            loss = loss + (
                consistency_factor
                * self.bn_consistency_weight
                * outputs.shape[1]
                * consistency_loss
            )

        # === Step 8: Backward + Step ===
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if hasattr(self, '_vit_lr_decay_gamma'):
            for pg in self.optimizer.param_groups:
                pg['lr'] *= self._vit_lr_decay_gamma

        if vit_trigger_recovery:
            self.current_model_probs = None
            if self.vit_adadem is not None:
                self.vit_adadem.reset()
            self.recovery_count += 1

        self.update_count += 1
        self.step_count += 1
        return outputs.detach()

    # ========================
    # Utility Methods
    # ========================

    def _get_layer_depth_legacy(self, param_name):
        """Legacy depth mapping (ResNet only)."""
        if param_name.startswith('bn1') or param_name.startswith('gn1'):
            return 0
        elif param_name.startswith('layer1'):
            return 1
        elif param_name.startswith('layer2'):
            return 2
        elif param_name.startswith('layer3'):
            return 3
        elif param_name.startswith('layer4'):
            return 4
        else:
            return 2

    def _reset_model(self):
        """Hard reset model and optimizer to source state."""
        self.model.load_state_dict(self.model_state, strict=True)
        opt = self.sam_optimizer.base_optimizer if self.sam_optimizer else self.optimizer
        opt.load_state_dict(self.optimizer_state)

        # Re-register initial params
        self.initial_params = {}
        for name, param in self.model.named_parameters():
            if name in self.param_names:
                self.initial_params[name] = param.data.clone()

        # AIS has no state to reset beyond logging scalars.
        if self.vit_adadem is not None:
            self.vit_adadem.reset()

    def reset(self):
        """Reset model to source state (Standard TTA: between corruptions)."""
        self._reset_model()
        self.step_count = 0
        self.update_count = 0
        self.skip_count = 0
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0
        self.recovery_count = 0
        self.current_model_probs = None
        self.bn_probe_entropy_ema = None
        self.vit_last_safety_mean = 0.0
        self.vit_last_safety_threshold = 0.0
        self.vit_last_anchor_rate = 0.0
        self.vit_last_high_freq_mean = 0.0
        self.vit_last_plpd_mean = 0.0
        self.vit_last_keep_ratio = 0.0
        self.vit_last_source_conf = 0.0
        self.vit_last_source_disagree = 0.0

    def get_stats(self):
        """Get adaptation statistics."""
        stats = {
            'step_count': self.step_count,
            'update_count': self.update_count,
            'skip_count': self.skip_count,
            'update_rate': self.update_count / max(self.step_count, 1),
            'num_reliable': self.num_samples_update_1,
            'num_non_redundant': self.num_samples_update_2,
            'recovery_count': self.recovery_count,
            'd3lr_enabled': self.use_d3lr,
            'd3lr_alpha_max': self.d3lr_alpha_max,
            'd3lr_kappa': self.d3lr_kappa,
            'd3lr_current_alpha': self._current_alpha,
            'saod_enabled': self.use_source_distill,
            'saod_temp': self.distill_temp,
            'saod_weight': self.distill_weight,
            'ais_enabled': self.use_ais,
            'optimizer_name': self.optimizer_name,
            'adaptation_mode': self.adaptation_mode,
            'bn_standard_mode': self._bn_standard_mode,
            'bn_continual_mode': self._bn_continual_mode,
            'aux_consistency_enabled': getattr(self, 'use_aux_consistency', False),
            'vit_schedule_enabled': self.use_vit_schedule,
            'vit_distill_warmup_steps': self.vit_distill_warmup_steps,
            'vit_ais_warmup_steps': self.vit_ais_warmup_steps,
        }
        if self._bn_continual_mode:
            stats['bn_continual_plpd_threshold'] = self.bn_continual_plpd_threshold
            stats['bn_continual_filter_margin'] = self.bn_continual_filter_margin
            stats['bn_continual_weight_margin'] = self.bn_continual_weight_margin
            stats['bn_continual_disable_redundancy'] = self.bn_continual_disable_redundancy
        if self._vit_mode:
            stats['vit_filter_margin'] = self.vit_filter_margin
            stats['vit_weight_margin'] = self.vit_weight_margin
            stats['vit_disable_redundancy'] = self.vit_disable_redundancy
            stats['vit_main_loss'] = self.vit_main_loss
            stats['vit_adadem_weight'] = self.vit_adadem_weight
            stats['vit_last_safety_mean'] = self.vit_last_safety_mean
            stats['vit_last_safety_threshold'] = self.vit_last_safety_threshold
            stats['vit_last_anchor_rate'] = self.vit_last_anchor_rate
            stats['vit_last_high_freq_mean'] = self.vit_last_high_freq_mean
            stats['vit_last_plpd_mean'] = self.vit_last_plpd_mean
            stats['vit_last_keep_ratio'] = self.vit_last_keep_ratio
            stats['vit_last_source_conf'] = self.vit_last_source_conf
            stats['vit_last_source_disagree'] = self.vit_last_source_disagree
        if self.initial_params:
            drift_sq = 0.0
            for name, param in self.model.named_parameters():
                if name in self.initial_params:
                    drift_sq += torch.sum(
                        (param - self.initial_params[name]) ** 2
                    ).item()
            stats['current_drift'] = drift_sq ** 0.5
        return stats


# Aliases for backward compatibility
TRIAD = DELTA  # Primary name: TRIAD (TRi-space Invariant ADaptation)


def create_delta(model, config=None, scenario='standard',
                 fishers=None, num_classes=1000):
    """Factory function to create DELTA with scenario-specific defaults."""
    config = config or {}

    scenario_defaults = {
        'standard': {
            'gate': {'gate_fraction': 1.0},
            'distill': {'temperature': 2.0, 'weight': 1.0},
            'lr': 0.00025,
            'use_sam': False,
            'd_margin': 0.05,
            'd3lr_alpha_max': 3.0,
            'd3lr_kappa': 1.0,
            'no_recovery': False,  # C3: AIS enabled
        },
        'continual': {
            'gate': {'gate_fraction': 1.0},
            'distill': {'temperature': 2.0, 'weight': 2.0},
            'lr': 0.00025,
            'use_sam': False,
            'd_margin': 0.05,
            'd3lr_alpha_max': 3.0,
            'd3lr_kappa': 1.0,
            'no_recovery': False,  # C3: AIS enabled
        },
        'wild_bs1': {
            'gate': {'gate_fraction': 1.0},
            'distill': {'temperature': 2.0, 'weight': 2.0},
            'lr': 0.00025,
            'use_sam': False,
            'd_margin': 0.05,
            'd3lr_alpha_max': 3.0,
            'd3lr_kappa': 1.0,
            'no_recovery': False,  # C3: AIS enabled
        },
    }

    defaults = scenario_defaults.get(scenario, scenario_defaults['standard'])

    def merge_dict(base, override):
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = merge_dict(result[k], v)
            else:
                result[k] = v
        return result

    fc = merge_dict(defaults, config)

    return DELTA(
        model=model,
        gate_config=fc.get('gate', {}),
        distill_config=fc.get('distill', {}),
        vit_schedule_config=fc.get('vit_schedule', {}),
        sam_config=fc.get('sam', {}),
        adapt_type=fc.get('adapt_type', 'gn'),
        lr=fc.get('lr', 0.00025),
        use_sam=fc.get('use_sam', False),
        use_sample_filtering=fc.get('use_sample_filtering', True),
        num_classes=num_classes,
        d_margin=fc.get('d_margin', 0.05),
        no_gate=fc.get('no_gate', False),
        no_proximal=fc.get('no_proximal', False),
        no_recovery=fc.get('no_recovery', False),  # C3: AIS
        no_anti_redundancy=fc.get('no_anti_redundancy', False),
        d3lr_alpha_max=fc.get('d3lr_alpha_max', 3.0),
        d3lr_kappa=fc.get('d3lr_kappa', 1.0),
        no_ddlr=fc.get('no_ddlr', False),
    )


# Primary factory alias
create_triad = create_delta
