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

    C3: Augmentation-Invariant Selection (AIS) —
        Hyperparameter-free sample reliability filter in sample selection
        space. Key insight: not all low-entropy samples are truly reliable;
        some are "fragile confident" predictions that sit on decision
        boundaries and produce harmful gradients.
        Detection: for each sample x, create x_flip = horizontal_flip(x)
        and forward through the ADAPTED model (not source). A sample is
        deemed reliable iff argmax(f(x)) == argmax(f(x_flip)). Unreliable
        samples (prediction changes under flip) are excluded from adaptation.
        One extra no_grad forward pass through the adapted model.
        Zero hyperparameters (binary filter). Zero state to maintain.
        Orthogonal to C1 (optimization) and C2 (function space).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple, Dict, List
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
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


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


def collect_all_norm_params(model, adapt_type='gn', arch_type: Optional[str] = None):
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

    vit_cutoff = _get_vit_adaptation_cutoff(model) if arch_type == 'vit' else None

    for nm, m in model.named_modules():
        if arch_type == 'vit' and isinstance(m, nn.LayerNorm):
            if nm == 'norm' or nm.startswith('norm.'):
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
    - C3: AIS — Augmentation-Invariant Selection.
      Hyperparameter-free sample reliability filter via flip-consistency.
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
        # dedicated depth map and conservative norm selection.
        self._arch_type = self._detect_architecture()
        self._max_depth = self._infer_max_depth()

        # Collect norm params. ViT uses a conservative LN subset (early/mid blocks).
        self.params, self.param_names = collect_all_norm_params(
            self.model, adapt_type, arch_type=self._arch_type
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
        self.use_aux_consistency = gate_cfg.get('use_aux_consistency', True) and (not self._bn_continual_mode)
        self.consistency_patch_ratio = gate_cfg.get('consistency_patch_ratio', 0.125)

        # ===== C2: SAOD (Source-Anchored Output Distillation) =====
        distill_cfg = distill_config or {}
        self.use_source_distill = not no_proximal
        self.distill_temp = distill_cfg.get('temperature', 5.0)
        self.distill_weight = distill_cfg.get('weight', 0.8)

        # ===== C3: AIS (Augmentation-Invariant Selection) =====
        # Hyperparameter-free sample reliability filter.
        # Uses horizontal flip consistency to detect fragile confident predictions.
        # Only adapts on samples whose predicted class is stable under flip.
        self.use_ais = not no_recovery

        vit_sched_cfg = vit_schedule_config or {}
        self.use_vit_schedule = self._arch_type == 'vit' and vit_sched_cfg.get('enabled', True)
        default_distill_warmup = 16 if self.adaptation_mode == 'standard' else 64
        default_ais_warmup = 4 if self.adaptation_mode == 'standard' else 16
        self.vit_distill_warmup_steps = vit_sched_cfg.get('distill_warmup_steps', default_distill_warmup)
        self.vit_ais_warmup_steps = vit_sched_cfg.get('ais_warmup_steps', default_ais_warmup)
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
            logger.info("AIS: ON (zero hyperparameters)")
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
            self.bn_continual_filter_margin if self._bn_continual_mode else self.e_margin
        )
        self.weight_margin = (
            self.bn_continual_weight_margin if self._bn_continual_mode else self.e_margin
        )

        # Anti-redundancy
        self.current_model_probs = None
        self.d_margin = max(d_margin, 0.15) if self._bn_standard_mode else d_margin

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
        elif 'blocks' in child_names:
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

    def _compute_adadem_aux_loss(self, logits):
        if self.bn_continual_adadem is None or logits.numel() == 0:
            return None

        probs = F.softmax(logits, dim=1)
        pseudo_labels = self.bn_continual_adadem.update(probs)
        avg_pred = self.bn_continual_adadem.avg_pred
        if avg_pred is None:
            return None

        with torch.no_grad():
            entropy_term = -(probs * logits).sum(1, keepdim=True)
            grad_weight = (logits + entropy_term + 1.0) * probs
            grad_weight = grad_weight.abs().sum(1, keepdim=True).clamp_min(1e-6)

        corrected = (
            probs - avg_pred[pseudo_labels].detach()
        ) / grad_weight.detach()
        return -(corrected * logits).sum(1).mean(0)

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

        # Group params by depth
        max_depth_idx = L + 1 if self._arch_type == 'resnet' else L + 1
        depth_params = {d: [] for d in range(max_depth_idx)}
        for param, name in zip(self.params, self.param_names):
            d = self._get_layer_depth(name)
            depth_params[d].append(param)

        param_groups = []
        for d in sorted(depth_params.keys()):
            if depth_params[d]:
                lr_d = base_lr * (alpha ** ((L - d) / L))
                param_groups.append({
                    'params': depth_params[d],
                    'lr': lr_d,
                    'depth': d,  # stored for dynamic D³LR updates
                })
                logger.info(f"  D³LR depth={d}: {len(depth_params[d])} params, "
                           f"lr={lr_d:.6f} ({lr_d/base_lr:.2f}x)")

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
                group['lr'] = self.base_lr * (alpha_t ** ((L - depth) / L))

    # ========================
    # C3: AIS (Augmentation-Invariant Selection)
    # ========================
    # AIS operates inline in _forward_and_adapt:
    #   x_flip = horizontal_flip(x)
    #   stable = argmax(f(x)) == argmax(f(x_flip))
    #   Only stable samples pass to entropy minimization.
    # No separate method needed — zero state, zero hyperparameters.

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
        2b. C3 AIS: flip-consistency check (no_grad forward on flipped input)
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

        # === Step 2b: C3 AIS — flip-consistency check ===
        # Create horizontally-flipped input and forward through adapted model.
        # Used to identify "fragile confident" predictions that sit on decision
        # boundaries and would produce harmful adaptation gradients.
        flip_preds = None
        if ais_active:
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

        # === Step 4b: C3 AIS — stability filter ===
        # Among entropy-filtered samples, keep only those whose prediction
        # is invariant under horizontal flip (argmax(p) == argmax(p_flip)).
        if ais_active and flip_preds is not None and reliable_indices.numel() > 0:
            orig_preds = outputs[reliable_indices].argmax(dim=1)
            flip_preds_filtered = flip_preds[reliable_indices]
            stable_mask = (orig_preds == flip_preds_filtered)
            reliable_indices = reliable_indices[stable_mask]

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
            )
        else:
            updated_probs = update_model_probs(
                self.current_model_probs,
                outputs_filtered.softmax(1),
            )

        self.current_model_probs = updated_probs
        self.num_samples_update_2 += entropys_filtered.size(0)

        n_filtered = entropys_filtered.size(0)
        if n_filtered == 0:
            self.optimizer.zero_grad()
            self.step_count += 1
            return outputs.detach()

        # === Step 6: C1 D³LR — update lr based on batch entropy ===
        if self.use_d3lr:
            batch_entropy = entropys_filtered.mean().item()
            self._update_d3lr(batch_entropy)

        # === Step 7: Entropy-weighted loss + C2 SAOD ===
        # (C3 AIS already filtered unreliable samples in Step 4b)
        coeff = 1.0 / (torch.exp(entropys_filtered.clone().detach() - weight_margin))
        if self._bn_continual_mode and plpd_filtered is not None:
            coeff = (
                self.bn_continual_entropy_weight * coeff
                + self.bn_continual_plpd_weight * torch.exp(plpd_filtered.clone().detach())
            )
        loss = (entropys_filtered * coeff).mean(0)

        # C2: SAOD loss
        if self.use_source_distill and source_outputs is not None:
            T = self.distill_temp
            source_probs = F.softmax(source_outputs[selected_indices] / T, dim=1)
            adapted_log_probs = F.log_softmax(outputs_filtered / T, dim=1)
            distill_loss = T * T * F.kl_div(
                adapted_log_probs, source_probs,
                reduction='batchmean'
            )
            loss = loss + (distill_factor * self.distill_weight) * distill_loss

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

        # AIS has no state to reset (stateless flip-consistency filter)
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
