"""Public entry point for the released test-time adaptation method."""

import sys
# Fix robustbench autoattack import issue
try:
    import pyautoattack
    sys.modules['autoattack'] = pyautoattack
except ImportError:
    pass

import argparse
import inspect
import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import math
from copy import deepcopy

try:
    from triad import (
        TRIAD,
        TRIAD_F1,
        TRIAD_F2,
        TRIAD_F3,
        TRIAD_F4,
        create_triad,
    )
except ImportError:
    # Comparison-method code is intentionally absent from this release.  The
    # symbols are resolved only if a caller explicitly requests that method.
    TRIAD = TRIAD_F1 = TRIAD_F2 = TRIAD_F3 = TRIAD_F4 = create_triad = None
from atlas import ATLAS, create_atlas
from atlas.common import looks_like_vit_family
try:
    from triad.utils import (
        set_seed, save_results, AverageMeter, accuracy,
        IMAGENET_C_CORRUPTIONS, CIFAR_C_CORRUPTIONS, get_timestamp
    )
except ImportError:
    from utils.runtime import (
        set_seed, save_results, AverageMeter, accuracy,
        IMAGENET_C_CORRUPTIONS, CIFAR_C_CORRUPTIONS, get_timestamp
    )
from datasets import (
    build_cifar_transform,
    get_corruption_loader,
    get_dataloader,
    get_label_shift_indices,
    get_natural_shift_loader,
)
from models import get_model

# Comparison implementations are optional external checkouts.  The released
# method and its smoke path must remain importable without them.
baselines_path = os.path.join(os.path.dirname(__file__), 'baselines')
if baselines_path not in sys.path:
    sys.path.append(baselines_path)
try:
    from tent import tent
    from EATA import eata
    from SAR import sar
    from SAR.sam import SAM
    import adadem as adadem_method
    import foa as foa_method
    import lcotta as lcotta_method
    import surgeon as surgeon_method
except ImportError:
    tent = eata = sar = SAM = None
    adadem_method = foa_method = lcotta_method = surgeon_method = None

# Import new baseline methods (CoTTA, DeYO, RoTTA)
# Note: CoTTA path selected dynamically based on dataset (cifar vs imagenet)
deyo_path = os.path.join(os.path.dirname(__file__), 'baselines/DeYO')

# Add paths for dependencies
if deyo_path not in sys.path:
    sys.path.append(deyo_path)

# Dynamic import to avoid path conflicts
import importlib.util
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_with_supported_kwargs(func, *args, **kwargs):
    """Call a function while dropping kwargs it doesn't declare.

    Several vendored baselines ship dataset-specific variants with slightly
    different helper signatures (e.g. CIFAR vs ImageNet CoTTA). This keeps the
    integration path stable without special-casing every fork.
    """
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return func(*args, **kwargs)
    filtered_kwargs = {
        key: value for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(*args, **filtered_kwargs)

def get_cotta_module(dataset_name: str, transform_family: str = None):
    """Dynamically load CoTTA module based on the active transform family."""
    family = (transform_family or '').lower()
    if family not in {'cifar', 'imagenet'}:
        family = 'imagenet' if 'imagenet' in dataset_name.lower() else 'cifar'

    if family == 'imagenet':
        cotta_path = os.path.join(os.path.dirname(__file__), 'baselines/cotta/imagenet')
    else:
        cotta_path = os.path.join(os.path.dirname(__file__), 'baselines/cotta/cifar')
    # Add path for my_transforms dependency
    if cotta_path not in sys.path:
        sys.path.append(cotta_path)
    return import_from_path('cotta', os.path.join(cotta_path, 'cotta.py'))

try:
    deyo_method = import_from_path('deyo', os.path.join(deyo_path, 'methods/deyo.py'))
except (ImportError, FileNotFoundError):
    deyo_method = None


# =============================================================================
# RoTTA Implementation (based on official CVPR 2023 paper)
# Key components:
# 1. RobustBN - momentum-based batch normalization
# 2. CSTU memory bank - class-balanced, timeliness-aware sampling
# 3. Timeliness reweighting - age-based instance weighting
# =============================================================================

class RobustBN2d(nn.Module):
    """Robust Batch Normalization with momentum-based statistics"""
    def __init__(self, bn_layer, momentum=0.05):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        self.eps = bn_layer.eps

        # Source statistics
        if bn_layer.track_running_stats and bn_layer.running_var is not None:
            self.register_buffer("source_mean", bn_layer.running_mean.clone())
            self.register_buffer("source_var", bn_layer.running_var.clone())
        else:
            self.register_buffer("source_mean", torch.zeros(self.num_features))
            self.register_buffer("source_var", torch.ones(self.num_features))

        # Learnable parameters
        self.weight = nn.Parameter(bn_layer.weight.clone())
        self.bias = nn.Parameter(bn_layer.bias.clone())

    def forward(self, x):
        if self.training:
            # Compute batch statistics
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)
            # Update with momentum
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            # Update buffers
            self.source_mean.copy_(mean.detach())
            self.source_var.copy_(var.detach())
            mean = mean.view(1, -1, 1, 1)
            var = var.view(1, -1, 1, 1)
        else:
            mean = self.source_mean.view(1, -1, 1, 1)
            var = self.source_var.view(1, -1, 1, 1)

        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class RobustGN(nn.Module):
    """
    Robust Group Normalization for GroupNorm models.

    Unlike BN which has running mean/var to smooth, GN computes statistics
    per-sample per-group (no running stats). However, RoTTA's key insight is
    that the affine parameters (weight/bias) also need stabilization during
    adaptation. We apply momentum-smoothed affine parameters:

    weight_used = (1-alpha) * source_weight + alpha * current_weight
    bias_used = (1-alpha) * source_bias + alpha * current_bias

    This prevents abrupt parameter changes while still allowing adaptation.
    The learnable weight/bias are optimized by the optimizer, but at forward
    time we blend them with the frozen source values for robustness.
    """
    def __init__(self, gn_layer, momentum=0.05):
        super().__init__()
        self.num_groups = gn_layer.num_groups
        self.num_channels = gn_layer.num_channels
        self.eps = gn_layer.eps
        self.affine = gn_layer.affine
        self.momentum = momentum

        if self.affine:
            # Learnable affine parameters (will be updated by optimizer)
            self.weight = nn.Parameter(gn_layer.weight.clone())
            self.bias = nn.Parameter(gn_layer.bias.clone())
            # Frozen source affine parameters (for momentum blending)
            self.register_buffer('source_weight', gn_layer.weight.clone())
            self.register_buffer('source_bias', gn_layer.bias.clone())
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.affine and self.training:
            # Momentum-smoothed affine: blend source and current learned parameters
            # This prevents catastrophic parameter drift during online adaptation
            weight = (1 - self.momentum) * self.source_weight + self.momentum * self.weight
            bias = (1 - self.momentum) * self.source_bias + self.momentum * self.bias
        elif self.affine:
            weight = self.source_weight
            bias = self.source_bias
        else:
            weight = None
            bias = None
        return nn.functional.group_norm(x, self.num_groups, weight, bias, self.eps)


class RobustLN(nn.Module):
    """Robust LayerNorm for ViT-style backbones.

    RoTTA official code is BN-only. This wrapper is a project-side extension for
    LayerNorm models that mirrors the RobustGN idea: keep the source affine
    parameters as anchors and blend them with the current online-updated affine
    parameters during training.
    """
    def __init__(self, ln_layer, momentum=0.05):
        super().__init__()
        self.normalized_shape = ln_layer.normalized_shape
        self.eps = ln_layer.eps
        self.elementwise_affine = ln_layer.elementwise_affine
        self.momentum = momentum

        if self.elementwise_affine:
            self.weight = nn.Parameter(ln_layer.weight.clone())
            self.bias = nn.Parameter(ln_layer.bias.clone())
            self.register_buffer('source_weight', ln_layer.weight.clone())
            self.register_buffer('source_bias', ln_layer.bias.clone())
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.elementwise_affine and self.training:
            weight = (1 - self.momentum) * self.source_weight + self.momentum * self.weight
            bias = (1 - self.momentum) * self.source_bias + self.momentum * self.bias
        elif self.elementwise_affine:
            weight = self.source_weight
            bias = self.source_bias
        else:
            weight = None
            bias = None
        return nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


def get_backbone_flags(backbone_name: str):
    """Return normalized backbone-type flags used by baseline integrations."""
    name = backbone_name.lower()
    return {
        'is_vit': 'vit' in name or 'swin' in name,
        'is_gn': 'gn' in name or 'groupnorm' in name,
        'is_bn': 'gn' not in name and 'groupnorm' not in name and 'vit' not in name and 'swin' not in name,
    }


def get_config_batch_size(config: dict) -> int:
    """Resolve dataset batch size with backbone-aware defaults."""
    backbone_name = str(config.get('model', {}).get('backbone', '')).lower()
    flags = get_backbone_flags(backbone_name)
    default_bs = 48 if flags['is_vit'] else 64
    return config.get('dataset', {}).get('batch_size', default_bs)


def get_dataset_preprocess(config: dict) -> str:
    """Return the configured dataset preprocessing mode with safe legacy defaults."""
    dataset_cfg = config.get('dataset', {})
    dataset_name = str(dataset_cfg.get('name', '')).lower()
    default = 'imagenet_224' if 'imagenet' in dataset_name else 'cifar_default'
    return str(dataset_cfg.get('preprocess', default)).lower()


def get_effective_input_size(config: dict) -> int:
    """Resolve the runtime input size used by data loading and backbone branches."""
    dataset_cfg = config.get('dataset', {})
    input_size = dataset_cfg.get('input_size')
    if input_size is not None:
        return int(input_size)
    return 224 if get_dataset_preprocess(config) == 'imagenet_224' else 32


def get_tta_transform_family(config: dict) -> str:
    """Choose the augmentation family (CIFAR vs ImageNet) from the effective input path."""
    dataset_name = str(config.get('dataset', {}).get('name', '')).lower()
    if 'imagenet' in dataset_name:
        return 'imagenet'
    return 'imagenet' if get_effective_input_size(config) >= 224 else 'cifar'


def get_sar_family_lr(config: dict, dataset_name: str, backbone_name: str, batch_size: int,
                      method: str = None, scenario: str = None) -> float:
    """Match the official SAR-family ImageNet-C lr rules for Tent/EATA/SAR/DeYO-like methods."""
    flags = get_backbone_flags(backbone_name)

    if 'imagenet' in dataset_name.lower():
        if flags['is_vit']:
            lr = (0.001 / 64.0) * batch_size
        else:
            lr = (0.00025 / 64.0) * batch_size * 2 if batch_size < 32 else 0.00025
    else:
        default_lr = 0.001 if flags['is_vit'] else config['optim'].get('lr', 0.001)
        lr = config['optim'].get('lr', default_lr)

    if method == 'sar' and scenario == 'bs1':
        lr *= 2.0

    return lr


def get_lcotta_hparams(config: dict, dataset_name: str, backbone_name: str) -> dict:
    """Resolve LCoTTA hyperparameters with config overrides and backbone-aware defaults."""
    flags = get_backbone_flags(backbone_name)
    lcotta_cfg = config.get('lcotta', {})

    if 'imagenet' in dataset_name.lower():
        defaults = {
            'lr': 0.0005 if flags['is_vit'] else 0.00015,
            'window_length': 100,
            'n_components': 50 if flags['is_vit'] else 25,
            'batch_step': 100 if flags['is_vit'] else 50,
            'cosine_margin': 0.05,
            'prob_momentum': 0.9,
        }
    else:
        defaults = {
            'lr': config['optim'].get('lr', 0.001),
            'window_length': 100,
            'n_components': 20,
            'batch_step': 50,
            'cosine_margin': 0.05,
            'prob_momentum': 0.9,
        }

    return {
        'lr': lcotta_cfg.get('lr', defaults['lr']),
        'window_length': lcotta_cfg.get('window_length', lcotta_cfg.get('w_num', defaults['window_length'])),
        'n_components': lcotta_cfg.get('n_components', defaults['n_components']),
        'batch_step': lcotta_cfg.get('batch_step', defaults['batch_step']),
        'cosine_margin': lcotta_cfg.get('cosine_margin', defaults['cosine_margin']),
        'prob_momentum': lcotta_cfg.get('prob_momentum', defaults['prob_momentum']),
    }


def get_adadem_hparams(config: dict, dataset_name: str, backbone_name: str) -> dict:
    """Resolve AdaDEM hyperparameters with scenario-aware official defaults."""
    flags = get_backbone_flags(backbone_name)
    adadem_cfg = config.get('adadem', {})
    standard_setting = config['tta'].get('reset_each_corruption', False)

    if 'imagenet' in dataset_name.lower():
        if flags['is_vit']:
            default_lr = 0.1 if standard_setting else 0.05
        else:
            default_lr = 0.005 if standard_setting else 0.0005
    else:
        default_lr = config['optim'].get('lr', 0.001)

    return {
        'lr': adadem_cfg.get('lr', default_lr),
        'pi': adadem_cfg.get('pi', 0.1),
    }


def infer_adapt_type(backbone_name: str, model, config: dict) -> str:
    """Infer normalization family robustly for public backbone variants."""
    requested = config.get('model', {}).get('adapt_params')
    name = backbone_name.lower()

    ln_count = 0
    bn_count = 0
    gn_count = 0
    attn_named = False
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            ln_count += 1
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_count += 1
        elif isinstance(module, nn.GroupNorm):
            gn_count += 1
        if module_name.endswith('attn') or '.attn' in module_name:
            attn_named = True

    has_patch_embed = hasattr(model, 'patch_embed')
    has_tokens = hasattr(model, 'cls_token') or hasattr(model, 'pos_embed')

    # Respect explicit request ONLY when compatible with model structure.
    if requested in {'bn', 'gn', 'ln'}:
        if requested == 'bn' and bn_count > 0:
            return 'bn'
        if requested == 'gn' and gn_count > 0:
            return 'gn'
        if requested == 'ln' and ln_count > 0:
            return 'ln'
        print(
            f"⚠️  adapt_params='{requested}' is incompatible with backbone='{backbone_name}' "
            f"(BN={bn_count}, GN={gn_count}, LN={ln_count}); fallback to auto-infer."
        )

    if 'gn' in name or 'groupnorm' in name:
        return 'gn'
    if any(tag in name for tag in ('vit', 'swin', 'deit', 'beit', 'mixer')):
        return 'ln'

    if has_patch_embed and (has_tokens or attn_named or ln_count > 0):
        return 'ln'
    if gn_count > 0 and bn_count == 0:
        return 'gn'
    if bn_count > 0:
        return 'bn'
    if ln_count > 0:
        return 'ln'
    return 'bn'


def get_surgeon_hparams(config: dict, dataset_name: str, backbone_name: str) -> dict:
    """Resolve SURGEON hyperparameters, preserving official BN-only defaults where applicable."""
    flags = get_backbone_flags(backbone_name)
    surgeon_cfg = config.get('surgeon', {})
    default_bn_only = surgeon_cfg.get('bn_only')
    if default_bn_only is None:
        bn_only = flags['is_bn'] and not flags['is_vit'] and not flags['is_gn']
    else:
        bn_only = bool(default_bn_only)

    name = dataset_name.lower()
    if flags['is_vit']:
        default_lr = config['optim'].get('lr', 1e-5)
    elif 'cifar10' in name:
        default_lr = 1e-3 if bn_only else 22e-6
    elif 'cifar100' in name:
        default_lr = 3e-4 if bn_only else 1e-5
    elif 'imagenet' in name and flags['is_bn']:
        default_lr = 2e-4 if bn_only else 5e-6
    else:
        default_lr = config['optim'].get('lr', 2.5e-4)

    return {
        'lr': surgeon_cfg.get('lr', default_lr),
        'bn_only': bn_only,
        'probe_samples': surgeon_cfg.get('probe_samples', 10),
        'css_weight': surgeon_cfg.get('css_weight', 0.01),
    }


def get_foa_hparams(config: dict, dataset_name: str, backbone_name: str) -> dict:
    """Resolve FOA hyperparameters, matching the official ViT defaults where possible."""
    flags = get_backbone_flags(backbone_name)
    if not flags['is_vit']:
        raise ValueError("FOA official release is ViT-focused; this integration only supports ViT backbones")

    foa_cfg = config.get('foa', {})
    dataset_key = dataset_name.lower().replace('-', '_')
    default_fitness_lambda = 0.2 if dataset_key == 'imagenet_r' else 0.4

    return {
        'num_prompts': foa_cfg.get('num_prompts', 3),
        'fitness_lambda': foa_cfg.get('fitness_lambda', default_fitness_lambda),
        'sigma': foa_cfg.get('sigma', 1.0),
        'popsize': foa_cfg.get('popsize', 27),
        'hist_momentum': foa_cfg.get('hist_momentum', 0.9),
        'reference_batch_size': foa_cfg.get('reference_batch_size', 64),
        'source_samples': foa_cfg.get('source_samples', None),  # None = use all source data (matches official)
    }


def timeliness_reweighting(ages):
    """Age-based instance reweighting: newer samples get higher weight"""
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))


def get_tta_transforms(img_shape=(32, 32, 3), gaussian_std=0.005, soft=False):
    """
    Get TTA-specific transforms (ColorJitterPro + GaussianNoise)
    Based on official RoTTA implementation
    """
    import torchvision.transforms.functional as TF

    class ColorJitterPro:
        """Extended ColorJitter with gamma adjustment"""
        def __init__(self, brightness, contrast, saturation, hue, gamma):
            self.brightness = brightness
            self.contrast = contrast
            self.saturation = saturation
            self.hue = hue
            self.gamma = gamma

        def __call__(self, img):
            # Random order of transforms
            fn_idx = torch.randperm(5)
            for fn_id in fn_idx:
                if fn_id == 0 and self.brightness is not None:
                    factor = torch.empty(1).uniform_(self.brightness[0], self.brightness[1]).item()
                    img = TF.adjust_brightness(img, factor)
                elif fn_id == 1 and self.contrast is not None:
                    factor = torch.empty(1).uniform_(self.contrast[0], self.contrast[1]).item()
                    img = TF.adjust_contrast(img, factor)
                elif fn_id == 2 and self.saturation is not None:
                    factor = torch.empty(1).uniform_(self.saturation[0], self.saturation[1]).item()
                    img = TF.adjust_saturation(img, factor)
                elif fn_id == 3 and self.hue is not None:
                    factor = torch.empty(1).uniform_(self.hue[0], self.hue[1]).item()
                    img = TF.adjust_hue(img, factor)
                elif fn_id == 4 and self.gamma is not None:
                    factor = torch.empty(1).uniform_(self.gamma[0], self.gamma[1]).item()
                    img = img.clamp(1e-8, 1.0)
                    img = TF.adjust_gamma(img, factor)
            return img

    class GaussianNoise:
        def __init__(self, std=0.005):
            self.std = std
        def __call__(self, img):
            noise = torch.randn_like(img) * self.std
            return (img + noise).clamp(0, 1)

    class Clip:
        def __init__(self, min_val=0.0, max_val=1.0):
            self.min_val = min_val
            self.max_val = max_val
        def __call__(self, img):
            return img.clamp(self.min_val, self.max_val)

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms
        def __call__(self, img):
            for t in self.transforms:
                img = t(img)
            return img

    if soft:
        color_jitter = ColorJitterPro(
            brightness=[0.8, 1.2], contrast=[0.85, 1.15],
            saturation=[0.75, 1.25], hue=[-0.03, 0.03], gamma=[0.85, 1.15]
        )
    else:
        color_jitter = ColorJitterPro(
            brightness=[0.6, 1.4], contrast=[0.7, 1.3],
            saturation=[0.5, 1.5], hue=[-0.06, 0.06], gamma=[0.7, 1.3]
        )

    transforms = Compose([
        Clip(0.0, 1.0),
        color_jitter,
        GaussianNoise(std=gaussian_std),
        Clip(0.0, 1.0)
    ])

    return transforms


class CSTU:
    """
    Class-balanced, Timeliness-aware Sample Storage with Uncertainty (CSTU)
    Based on official RoTTA implementation
    """
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = self.capacity / self.num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        # Class-balanced storage: list of lists
        self.data = [[] for _ in range(self.num_class)]

    def get_occupancy(self):
        return sum(len(d) for d in self.data)

    def add_instance(self, instance):
        data, prediction, uncertainty = instance
        new_item = {'data': data, 'uncertainty': uncertainty, 'age': 0}
        new_score = self.heuristic_score(0, uncertainty)

        if self._remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self._add_age()

    def _remove_instance(self, cls, new_score):
        """Try to make room for new instance"""
        class_list = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()

        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            else:
                # Remove from majority classes
                return self._remove_from_majority(new_score)
        else:
            # Remove from same class
            return self._remove_from_class(cls, new_score)

    def _remove_from_majority(self, score_base):
        """Remove item with highest score from majority class"""
        per_class_count = [len(d) for d in self.data]
        max_count = max(per_class_count)
        majority_classes = [i for i, c in enumerate(per_class_count) if c == max_count]

        max_score = None
        max_class = None
        max_idx = None

        for cls in majority_classes:
            for idx, item in enumerate(self.data[cls]):
                score = self.heuristic_score(item['age'], item['uncertainty'])
                if max_score is None or score > max_score:
                    max_score = score
                    max_class = cls
                    max_idx = idx

        if max_class is not None and max_score > score_base:
            self.data[max_class].pop(max_idx)
            return True
        return False

    def _remove_from_class(self, cls, score_base):
        """Remove item with highest score from specific class"""
        class_list = self.data[cls]
        max_score = None
        max_idx = None

        for idx, item in enumerate(class_list):
            score = self.heuristic_score(item['age'], item['uncertainty'])
            if max_score is None or score > max_score:
                max_score = score
                max_idx = idx

        if max_idx is not None and max_score > score_base:
            class_list.pop(max_idx)
            return True
        return False

    def heuristic_score(self, age, uncertainty):
        """Higher score = more likely to be replaced"""
        return (self.lambda_t * 1 / (1 + math.exp(-age / self.capacity)) +
                self.lambda_u * uncertainty / math.log(max(self.num_class, 2)))

    def _add_age(self):
        for class_list in self.data:
            for item in class_list:
                item['age'] += 1

    def get_memory(self):
        """Get all stored samples and their ages"""
        tmp_data = []
        tmp_ages = []
        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item['data'])
                tmp_ages.append(item['age'] / self.capacity)  # Normalize age
        return tmp_data, tmp_ages

    def reset(self):
        self.data = [[] for _ in range(self.num_class)]


class RoTTAWrapper(nn.Module):
    """
    RoTTA (Robust Test-Time Adaptation) implementation
    Based on CVPR 2023 paper: "Robust Test-Time Adaptation in Dynamic Scenarios"

    Key features:
    - RobustBN: Momentum-based batch normalization for stability
    - CSTU memory: Class-balanced, timeliness-aware sample storage
    - Age reweighting: Newer samples get higher weight in adaptation
    - Strong augmentation: ColorJitterPro + GaussianNoise for student
    """
    def __init__(self, model, optimizer_cls, optimizer_kwargs, memory_size=64, nu=0.001, update_freq=64, alpha=0.05, num_classes=100):
        super().__init__()
        self.memory_size = memory_size
        self.nu = nu
        self.update_freq = update_freq
        self.alpha = alpha  # RobustBN/RobustGN momentum
        self.num_classes = num_classes

        # FIRST: Replace BN/GN layers with RobustBN/RobustGN
        # (must happen BEFORE optimizer creation so optimizer gets correct params)
        self.model = self._configure_model(model)

        # SECOND: Collect trainable params from configured model
        params, param_names = [], []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                params.append(p)
                param_names.append(n)

        # THIRD: Create optimizer with the CORRECT (new) parameters
        if len(params) > 0:
            self.optimizer = optimizer_cls(params, **optimizer_kwargs)
        else:
            print("Warning: RoTTA has no trainable parameters! Model will not adapt.")
            self.optimizer = None

        print(f"  RoTTA optimizer: {len(params)} trainable params: {param_names[:5]}{'...' if len(param_names) > 5 else ''}")

        # Create EMA teacher (from original model before RobustBN replacement)
        self.model_ema = deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.requires_grad = False

        # CSTU memory bank (class-balanced, timeliness-aware)
        self.mem = CSTU(capacity=memory_size, num_class=num_classes, lambda_t=1.0, lambda_u=1.0)
        self.current_instance = 0

        # Strong augmentation for student (critical for RoTTA!)
        self.transform = get_tta_transforms()

        # Save initial state for reset
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optimizer else None
        # Save optimizer factory for reset
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = optimizer_kwargs

    def _configure_model(self, model):
        """Configure model for RoTTA - support BatchNorm, GroupNorm, and LayerNorm."""
        model.requires_grad_(False)

        # Find all normalization layers (BN or GN)
        bn_layers = []
        gn_layers = []
        ln_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_layers.append(name)
            elif isinstance(module, nn.GroupNorm):
                gn_layers.append(name)
            elif isinstance(module, nn.LayerNorm):
                ln_layers.append(name)

        # Replace BN layers with RobustBN
        for name in bn_layers:
            # Get parent module and layer name
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, layer_name = parts
                parent = model
                for p in parent_name.split('.'):
                    parent = getattr(parent, p)
            else:
                parent = model
                layer_name = name

            bn_layer = getattr(parent, layer_name)
            robust_bn = RobustBN2d(bn_layer, self.alpha)
            robust_bn.requires_grad_(True)
            setattr(parent, layer_name, robust_bn)

        # Replace GN layers with RobustGN (similar to BN but adapted for GN)
        for name in gn_layers:
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, layer_name = parts
                parent = model
                for p in parent_name.split('.'):
                    parent = getattr(parent, p)
            else:
                parent = model
                layer_name = name

            gn_layer = getattr(parent, layer_name)
            robust_gn = RobustGN(gn_layer, self.alpha)
            robust_gn.requires_grad_(True)
            setattr(parent, layer_name, robust_gn)

        # Replace LN layers with RobustLN (ViT custom extension)
        for name in ln_layers:
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, layer_name = parts
                parent = model
                for p in parent_name.split('.'):
                    parent = getattr(parent, p)
            else:
                parent = model
                layer_name = name

            ln_layer = getattr(parent, layer_name)
            robust_ln = RobustLN(ln_layer, self.alpha)
            robust_ln.requires_grad_(True)
            setattr(parent, layer_name, robust_ln)

        if len(bn_layers) == 0 and len(gn_layers) == 0 and len(ln_layers) == 0:
            print("Warning: No normalization layers found for RoTTA, enabling all parameters")
            model.requires_grad_(True)

        return model

    def forward(self, x):
        # Get teacher predictions
        with torch.no_grad():
            self.model_ema.eval()
            ema_out = self.model_ema(x)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(-predict * torch.log(predict + 1e-6), dim=1)

        # Add samples to CSTU memory (class-balanced)
        for i in range(x.size(0)):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (x[i].detach(), p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            # Periodic update when memory has samples
            if self.current_instance % self.update_freq == 0:
                self._update_model()

        return ema_out

    def _update_model(self):
        """Update model using memory bank with strong augmentation"""
        if self.optimizer is None:
            return

        # Get memory data
        sup_data, ages = self.mem.get_memory()
        if len(sup_data) == 0:
            return

        self.model.train()
        self.model_ema.train()

        # Stack data
        sup_data = torch.stack(sup_data)

        # Apply strong augmentation to student input (critical for RoTTA!)
        strong_sup_aug = self.transform(sup_data)

        # Get predictions
        with torch.no_grad():
            ema_out = self.model_ema(sup_data)  # Teacher uses original data
        stu_out = self.model(strong_sup_aug)    # Student uses augmented data

        # Timeliness reweighting
        ages_tensor = torch.tensor(ages).float().cuda()
        instance_weight = timeliness_reweighting(ages_tensor)

        # Cross-entropy with soft labels (weighted)
        loss = -(torch.softmax(ema_out, dim=1) * torch.log_softmax(stu_out, dim=1)).sum(1)
        loss = (loss * instance_weight).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update EMA teacher
        for ema_param, param in zip(self.model_ema.parameters(), self.model.parameters()):
            ema_param.data[:] = (1 - self.nu) * ema_param.data[:] + self.nu * param.data[:]

        # Clean up to save memory
        del sup_data, strong_sup_aug, ema_out, stu_out, ages_tensor, instance_weight, loss
        torch.cuda.empty_cache()

    def reset(self):
        """Reset model to initial state"""
        self.model.load_state_dict(self.model_state, strict=True)
        # Recreate optimizer with correct parameter references after state_dict load
        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) > 0:
            self.optimizer = self._optimizer_cls(params, **self._optimizer_kwargs)
        else:
            self.optimizer = None
        # Reset EMA model
        self.model_ema = deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.requires_grad = False
        # Clear CSTU memory
        self.mem.reset()
        self.current_instance = 0


def parse_args():
    parser = argparse.ArgumentParser(description='TRIAD: TRi-space Invariant ADaptation')

    parser.add_argument('--config', '--configs', dest='config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--method', type=str, default='triad',
                        choices=['source', 'tent', 'eata', 'sar', 'cotta', 'lcotta', 'adadem', 'surgeon', 'foa', 'deyo', 'rotta', 'triad', 'triad_f1', 'triad_f2', 'triad_f3', 'triad_f4', 'atlas'],
                        help='TTA method to use')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Override backbone model')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override dataset batch size')

    # TRIAD specific
    parser.add_argument('--no_gate', action='store_true',
                        help='Disable batch-level entropy gate')
    parser.add_argument('--fixed_lambda', type=float, default=None,
                        help='Use fixed lambda (disable adaptive)')

    # TRIAD ablation switches (--no_c1, --no_c2, --no_c3)
    parser.add_argument('--no_c2', '--no_proximal', action='store_true', dest='no_c2',
                        help='Disable Source-Anchored Output Distillation (C2 SAOD)')
    parser.add_argument('--no_c3', '--no_recovery', action='store_true', dest='no_c3',
                        help='Disable Augmentation-Invariant Selection (C3 AIS)')
    parser.add_argument('--no_dual_gate', action='store_true',
                        help='(Legacy, ignored)')
    parser.add_argument('--no_sam', action='store_true',
                        help='Disable SAM optimization (default: off)')
    parser.add_argument('--no_anti_redundancy', action='store_true',
                        help='Disable anti-redundancy filtering (EATA-style)')
    parser.add_argument('--no_gges', action='store_true',
                        help='(Legacy, GGES disabled by default)')
    parser.add_argument('--no_psga', action='store_true',
                        help='Legacy flag, ignored')
    parser.add_argument('--no_diversity', action='store_true',
                        help='Legacy flag, ignored')
    parser.add_argument('--no_edcr', action='store_true',
                        help='Legacy flag, ignored')
    parser.add_argument('--no_gc', action='store_true',
                        help='Legacy flag, ignored')
    parser.add_argument('--no_rslr', action='store_true',
                        help='Legacy flag, ignored')
    # C1: D³LR (Dynamic Depth-Decay Learning Rate)
    parser.add_argument('--d3lr_alpha_max', type=float, default=75.0,
                        help='D³LR max depth-decay factor α_max (C1, Std=75, Ctn=50)')
    parser.add_argument('--d3lr_kappa', type=float, default=0.5,
                        help='D³LR sigmoid temperature κ (C1, default=0.5)')
    parser.add_argument('--no_c1', '--no_ddlr', action='store_true', dest='no_c1',
                        help='Disable Dynamic Depth-Decay LR (C1 D³LR)')
    # Legacy DDLR/DDSR args (kept for backward compat, ignored)
    parser.add_argument('--ddlr_alpha', type=float, default=2.0,
                        help='(Legacy, ignored. Use --d3lr_alpha_max)')
    parser.add_argument('--ddsr_interval', type=int, default=10,
                        help='(Legacy, ignored. DDSR replaced by DGFR)')
    parser.add_argument('--ddsr_rho_base', type=float, default=0.01,
                        help='(Legacy, ignored. DDSR replaced by DGFR)')
    # C2: SAOD
    parser.add_argument('--distill_weight', type=float, default=None,
                        help='SAOD distillation weight μ override (C2, default=0.8)')
    parser.add_argument('--distill_temp', type=float, default=None,
                        help='SAOD distillation temperature T override (C2, default=5.0)')
    # C3: AIS (Augmentation-Invariant Selection) — no hyperparameters
    # Legacy DGFR args kept for backward compat
    parser.add_argument('--dgfr_momentum', type=float, default=0.1,
                        help='(Legacy, ignored)')
    parser.add_argument('--dgfr_gamma', type=float, default=2,
                        help='(Legacy, ignored)')

    # ATLAS specific
    parser.add_argument('--atlas-method-variant', choices=['core', 'full'], default=None,
                        help='ATLAS variant: core disables branch refinements; full preserves the canonical legacy recipe')
    parser.add_argument('--atlas-selection-quantile', type=float, default=None,
                        help='Unlabeled ATLAS Core reliability quantile (default: 0.5)')
    parser.add_argument('--no_arc', action='store_true',
                        help='Disable ATLAS asymmetric relative clipping')
    parser.add_argument('--no_dig', action='store_true',
                        help='Disable ATLAS dynamic informative grouping')
    parser.add_argument('--no_uan', action='store_true',
                        help='Disable ATLAS unit-aware normalization')
    parser.add_argument('--no_sos', action='store_true',
                        help='Disable ATLAS soft over-shift shaping')
    parser.add_argument('--atlas-dig-std-threshold', type=float, default=None,
                        help='Override ATLAS dig.std_threshold for sensitivity sweeps')
    parser.add_argument('--atlas-sos-safe-margin', type=float, default=None,
                        help='Override ATLAS sos.safe_margin for sensitivity sweeps')
    parser.add_argument('--atlas-sos-hard-margin', type=float, default=None,
                        help='Override ATLAS sos.hard_margin for sensitivity sweeps')
    parser.add_argument('--atlas-arc-eps-low', type=float, default=None,
                        help='Override ATLAS arc.eps_low for sensitivity sweeps')
    parser.add_argument('--atlas-arc-eps-high', type=float, default=None,
                        help='Override ATLAS arc.eps_high for sensitivity sweeps')
    parser.add_argument('--atlas-vit-ln-pi', type=float, default=None,
                        help='Experiment-only override for active ViT/LN pi')
    parser.add_argument('--atlas-vit-ln-entropy-margin-scale', type=float, default=None,
                        help='Experiment-only override for active ViT/LN entropy margin scale')
    parser.add_argument('--atlas-vit-ln-plpd-threshold', type=float, default=None,
                        help='Experiment-only override for active ViT/LN PLPD threshold')
    parser.add_argument('--atlas-vit-ln-patch-len', type=int, default=None,
                        help='Experiment-only override for active ViT/LN patch length')
    parser.add_argument('--atlas-probe-policy', choices=['always', 'alternate', 'off'], default=None,
                        help='Experiment-only ViT probe policy; skipped steps do not cache predictions')
    parser.add_argument('--atlas-shared-role-only', action='store_true',
                        help='Use the shared CORE/CARE/SANE role ablation on ViT/LN')
    parser.add_argument('--atlas-active-views', type=str, default=None,
                        help='Comma-separated ATLAS view set override for GN ablations '
                             '(e.g., "source,raw,photometric,structural,destroy")')
    parser.add_argument('--atlas-reward-weights', type=str, default=None,
                        help='DEPRECATED: legacy reproduction only; four Eq. (7) weights as '
                             'target,entropy,source,consensus')
    parser.add_argument('--atlas-ablation-row', type=str, default=None,
                        choices=['A0', 'A1', 'A2', 'B1', 'B2'],
                        help='ATLAS Table 7 ViT continual ablation row override')
    parser.add_argument('--run-suffix', type=str, default=None,
                        help='Optional suffix appended to saved result filenames/method tags')

    # Continual TTA with multiple corruption orders (following CoTTA)
    parser.add_argument('--corruption-order', type=str, default=None,
                        help='Comma-separated list of corruption types (e.g., "snow,fog,brightness")')
    parser.add_argument('--order-idx', type=int, default=0,
                        help='Index of corruption order (for logging/saving)')
    parser.add_argument('--max-batches', type=int, default=0,
                        help='Optional cap on evaluated batches for smoke tests and efficiency probes')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Override logging.results_dir for isolated experiment outputs')

    # Long-term continual evaluation protocol override
    parser.add_argument('--continual-repeats', '--lcotta-repeats', dest='continual_repeats',
                        type=int, default=None,
                        help='Override long-term continual repeats (e.g., 50 in the official '
                            'LCoTTA/ImageNet-C protocol). Default is single-pass (1) unless this '
                            'argument is explicitly set. --lcotta-repeats is kept as a legacy alias.')

    # Wild scenarios (following SAR/DeYO)
    parser.add_argument('--scenario', type=str, default=None,
                        choices=['normal', 'label_shifts', 'mix_shifts', 'bs1'],
                        help='Wild scenario type')
    parser.add_argument('--imbalance-ratio', type=float, default=500000,
                        help='Imbalance ratio for label_shifts (1=uniform, 500000=extreme)')

    # EATA Fisher computation
    parser.add_argument('--no-fisher', action='store_true',
                        help='Disable Fisher regularization for EATA (faster but less effective)')
    parser.add_argument('--force-fisher-on-vit', action='store_true',
                        help='Force Fisher regularization for ViT/LN EATA. By default, ViT falls back to no-Fisher because official EATA only supports BN/ResNet and Fisher hurts our ViT reproduction.')

    return parser.parse_args()


def parse_atlas_active_views(raw_value: str | None):
    if raw_value is None:
        return None

    normalized = []
    seen = set()
    for item in raw_value.split(','):
        candidate = item.strip().lower()
        if not candidate:
            continue
        if candidate == 'weak':
            candidate = 'photometric'
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)

    if not normalized:
        raise ValueError('--atlas-active-views cannot be empty')

    return normalized


def parse_atlas_reward_weights(raw_value: str | None):
    if raw_value is None:
        return None

    parts = [item.strip() for item in raw_value.split(',')]
    if len(parts) != 4 or any(not item for item in parts):
        raise ValueError(
            '--atlas-reward-weights requires exactly four comma-separated values: '
            'target,entropy,source,consensus'
        )
    try:
        values = [float(item) for item in parts]
    except ValueError as exc:
        raise ValueError('--atlas-reward-weights values must be numeric') from exc
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError('--atlas-reward-weights values must be finite and non-negative')
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError('--atlas-reward-weights values must sum to 1')

    return dict(zip(("target", "entropy", "source", "consensus"), values))


def collect_atlas_hparam_overrides(args):
    nested = {}
    flat = {}

    if getattr(args, 'atlas_selection_quantile', None) is not None:
        value = float(args.atlas_selection_quantile)
        if not 0.0 <= value <= 1.0:
            raise ValueError('--atlas-selection-quantile must be in [0, 1]')
        nested.setdefault('selection', {})['quantile'] = value
        flat['selection.quantile'] = value

    if getattr(args, 'atlas_dig_std_threshold', None) is not None:
        value = float(args.atlas_dig_std_threshold)
        nested.setdefault('dig', {})['std_threshold'] = value
        flat['dig.std_threshold'] = value

    if getattr(args, 'atlas_sos_safe_margin', None) is not None:
        value = float(args.atlas_sos_safe_margin)
        nested.setdefault('sos', {})['safe_margin'] = value
        flat['sos.safe_margin'] = value

    if getattr(args, 'atlas_sos_hard_margin', None) is not None:
        value = float(args.atlas_sos_hard_margin)
        nested.setdefault('sos', {})['hard_margin'] = value
        flat['sos.hard_margin'] = value

    if getattr(args, 'atlas_arc_eps_low', None) is not None:
        value = float(args.atlas_arc_eps_low)
        nested.setdefault('arc', {})['eps_low'] = value
        flat['arc.eps_low'] = value

    if getattr(args, 'atlas_arc_eps_high', None) is not None:
        value = float(args.atlas_arc_eps_high)
        nested.setdefault('arc', {})['eps_high'] = value
        flat['arc.eps_high'] = value

    reward_weights = parse_atlas_reward_weights(
        getattr(args, 'atlas_reward_weights', None)
    )
    if reward_weights is not None:
        nested['reward'] = {
            'aggregation': 'legacy_weighted',
            'legacy_weights': reward_weights,
        }
        flat['reward.aggregation'] = 'legacy_weighted'
        for name, value in reward_weights.items():
            flat[f'reward.legacy_weights.{name}'] = value

    vit_ln_fields = {
        'pi': getattr(args, 'atlas_vit_ln_pi', None),
        'entropy_margin_scale': getattr(args, 'atlas_vit_ln_entropy_margin_scale', None),
        'plpd_threshold': getattr(args, 'atlas_vit_ln_plpd_threshold', None),
        'patch_len': getattr(args, 'atlas_vit_ln_patch_len', None),
        'probe_policy': getattr(args, 'atlas_probe_policy', None),
    }
    for field_name, raw_value in vit_ln_fields.items():
        if raw_value is None:
            continue
        value = raw_value
        if field_name in {'pi', 'entropy_margin_scale', 'plpd_threshold'}:
            value = float(raw_value)
        elif field_name == 'patch_len':
            value = int(raw_value)
        nested.setdefault('vit_ln', {})[field_name] = value
        flat[f'vit_ln.{field_name}'] = value
    if getattr(args, 'atlas_shared_role_only', False):
        nested.setdefault('vit_ln', {})['shared_role_only'] = True
        flat['vit_ln.shared_role_only'] = True

    return nested, flat


def load_config(config_path: str) -> dict:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    def deep_update(base: dict, override: dict) -> None:
        for key, value in override.items():
            if key == '_BASE_':
                continue
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                deep_update(base[key], value)
            else:
                base[key] = value

    # Handle inheritance
    if '_BASE_' in config:
        base_path = os.path.join(os.path.dirname(config_path), config['_BASE_'])
        base_config = load_config(base_path)
        # Merge recursively (config overrides base)
        deep_update(base_config, config)
        config = base_config

    return config


def get_result_method_name(args) -> str:
    if getattr(args, 'run_suffix', None):
        return f"{args.method}_{args.run_suffix}"
    if args.method == 'atlas' and getattr(args, 'atlas_method_variant', None):
        return f"atlas_{args.atlas_method_variant}"
    return args.method


def evaluate_corruption(
    model: nn.Module,
    dataloader,
    device: torch.device,
    method: str = 'source',
    max_batches: int = 0,
) -> dict:
    """
    Evaluate model on a corruption dataset

    Returns:
        Dictionary with accuracy and other metrics
    """
    top1_meter = AverageMeter()

    # Set model mode based on method
    if method == 'source':
        model.eval()

    iter_count = 0
    evaluated_batches = 0
    evaluated_samples = 0
    evaluated_classes = set()
    batch_pbar = tqdm(dataloader, leave=False, desc='  Batches',
                     ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

    for images, labels in batch_pbar:
        if max_batches and evaluated_batches >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass with/without gradients based on method
        if method == 'source':
            with torch.no_grad():
                outputs = model(images)
        elif method == 'deyo':
            # DeYO requires iter_ parameter; when flag=True it adapts and returns (outputs, backward, final_backward)
            with torch.enable_grad():
                out = model(images, iter_=iter_count, flag=True)
                outputs = out[0] if isinstance(out, (tuple, list)) else out
            iter_count += 1
        else:
            # TTA methods (tent, eata, sar, cotta, rotta, triad*, atlas) need gradients
            with torch.enable_grad():
                outputs = model(images)

        # Compute accuracy
        acc1, = accuracy(outputs, labels, topk=(1,))
        top1_meter.update(acc1.item(), images.size(0))
        evaluated_batches += 1
        evaluated_samples += int(images.size(0))
        evaluated_classes.update(int(label) for label in labels.detach().cpu().tolist())

    results = {
        'accuracy': top1_meter.avg,
        'evaluated_batches': evaluated_batches,
        'evaluated_samples': evaluated_samples,
        'evaluated_class_count': len(evaluated_classes),
    }

    # Add TRIAD-specific stats
    if hasattr(model, 'get_stats'):
        stats = model.get_stats()
        results.update(stats)

    return results


def run_experiment(config: dict, args):
    """Run the main experiment"""

    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # Get dataset settings early (needed for method configuration and loaders)
    dataset_name = config['dataset']['name']
    dataset_preprocess = get_dataset_preprocess(config)
    input_size = get_effective_input_size(config)
    transform_family = get_tta_transform_family(config)
    result_method_name = get_result_method_name(args)

    print(f"Running {result_method_name} on {config['dataset']['name']}")
    print(f"Backbone: {config['model']['backbone']}")
    print(f"Input Path: {dataset_preprocess} ({input_size}px, family={transform_family})")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # Get model (pass dataset name for correct pretrained weights)
    model = get_model(
        config['model']['backbone'],
        pretrained=config['model'].get('pretrained', True),
        dataset=dataset_name
    )
    model = model.to(device)

    # Record backbone provenance if loaded from RobustBench
    backbone_provenance = None
    if hasattr(model, 'rb_model_name') and hasattr(model, 'rb_dataset'):
        backbone_provenance = {
            'source': 'robustbench',
            'rb_dataset': getattr(model, 'rb_dataset'),
            'rb_model_name': getattr(model, 'rb_model_name'),
        }
        print(f"Backbone provenance: RobustBench {backbone_provenance['rb_model_name']} ({backbone_provenance['rb_dataset']})")

    # Wrap model based on method
    if args.method == 'source':
        model.eval()

    elif args.method == 'atlas':
        atlas_cfg = config.get('atlas', {})
        atlas_method_variant = (
            getattr(args, 'atlas_method_variant', None)
            or atlas_cfg.get('method_variant', 'full')
        )
        vit_ln_cfg = atlas_cfg.get('vit_ln', {})
        backbone_name = config['model']['backbone'].lower()
        adapt_type = infer_adapt_type(backbone_name, model, config)

        if 'cifar10' in dataset_name and 'cifar100' not in dataset_name:
            num_classes = 10
        elif 'cifar100' in dataset_name:
            num_classes = 100
        else:
            num_classes = 1000

        scenario = config['tta'].get('scenario', 'normal')
        if hasattr(args, 'scenario') and args.scenario:
            scenario = args.scenario
        adaptation_mode = 'standard' if config['tta'].get('reset_each_corruption', False) else 'continual'
        use_vit_ln_core = adapt_type == 'ln' and looks_like_vit_family(model)
        generic_override_sections = {'arc', 'dig', 'sos', 'reward'}
        requested_sections = set(collect_atlas_hparam_overrides(args)[0])
        if (
            use_vit_ln_core
            and atlas_method_variant == 'full'
            and requested_sections.intersection(generic_override_sections)
        ):
            raise ValueError(
                'GN/BN-only ATLAS overrides (--atlas-dig/--atlas-sos/--atlas-arc/'
                '--atlas-reward-weights) are inactive on the ViT/LN path; '
                'use --atlas-vit-ln-* overrides instead.'
            )
        use_vit_ln_refinements = use_vit_ln_core and atlas_method_variant == 'full'
        atlas_lr = float(vit_ln_cfg.get('lr', 0.05)) if use_vit_ln_refinements else config['optim'].get('lr', 0.00025)
        atlas_optimizer = str(vit_ln_cfg.get('optimizer', 'sgd')).lower() if use_vit_ln_refinements else config['optim'].get('optimizer', 'sgd')
        atlas_momentum = float(vit_ln_cfg.get('momentum', 0.0)) if use_vit_ln_refinements else config['optim'].get('momentum', 0.9)
        atlas_weight_decay = float(vit_ln_cfg.get('weight_decay', 0.0)) if use_vit_ln_refinements else config['optim'].get('weight_decay', 0.0)

        atlas_create_cfg = dict(atlas_cfg)
        atlas_create_cfg.update({
            'adapt_type': adapt_type,
            'adaptation_mode': adaptation_mode,
            'scenario': scenario,
            'lr': atlas_lr,
            'optimizer_name': atlas_optimizer,
            'optim_momentum': atlas_momentum,
            'optim_weight_decay': atlas_weight_decay,
            'num_classes': num_classes,
            'method_variant': atlas_method_variant,
            'no_arc': getattr(args, 'no_arc', False),
            'no_dig': getattr(args, 'no_dig', False),
            'no_uan': getattr(args, 'no_uan', False),
            'no_sos': getattr(args, 'no_sos', False),
            'ablation_row': getattr(args, 'atlas_ablation_row', None),
        })
        atlas_hparam_nested, _ = collect_atlas_hparam_overrides(args)
        for section_name, section_updates in atlas_hparam_nested.items():
            atlas_create_cfg[section_name] = dict(atlas_create_cfg.get(section_name, {}))
            atlas_create_cfg[section_name].update(section_updates)
        atlas_active_views = parse_atlas_active_views(getattr(args, 'atlas_active_views', None))
        if atlas_active_views is not None:
            atlas_create_cfg['views'] = dict(atlas_create_cfg.get('views', {}))
            atlas_create_cfg['views']['active'] = atlas_active_views
        model = create_atlas(model, config=atlas_create_cfg)
        print(
            f"ATLAS[{model.method_variant}]: adapting {len(model.params)} params, "
            f"ARC={'ON' if model.arc_enabled else 'OFF'}, "
            f"DIG={'ON' if model.dig_enabled else 'OFF'}, "
            f"UAN={model.uan_mode}, "
            f"SOS={'ON' if model.sos_enabled else 'OFF'}, "
            f"vit_ctta_core={model.vit_ctta_core}, "
            f"ablation={model.atlas_ablation_row or 'default'}, "
            f"views={'/'.join(model.active_views) if model._explicit_active_views else 'legacy'}, "
            f"reward_aggregation={model.reward_aggregation}"
        )

    elif args.method in ('triad', 'triad_f1', 'triad_f2', 'triad_f3', 'triad_f4'):
        # Configure TRIAD
        triad_cfg = config.get('triad', {})
        gate_config = triad_cfg.get('gate', {})
        distill_config = triad_cfg.get('distill', {})
        d3lr_config = triad_cfg.get('d3lr', {})
        vit_schedule_config = triad_cfg.get('vit_schedule', {})

        gate_config['image_size'] = input_size

        # NOTE: TRIAD uses explicit ablation flags (--no_c1/--no_c2/--no_c3),
        # do not rely on gate_config['enabled'].
        if args.fixed_lambda is not None:
            distill_config['weight'] = args.fixed_lambda
        if getattr(args, 'distill_weight', None) is not None:
            distill_config['weight'] = args.distill_weight
        if getattr(args, 'distill_temp', None) is not None:
            distill_config['temperature'] = args.distill_temp

        # C1: D³LR — resolve α_max and κ (config YAML < CLI override)
        # Priority: CLI --d3lr_alpha_max > config triad.d3lr.alpha_max > code default 75.0
        resolved_alpha_max = d3lr_config.get('alpha_max', 75.0)
        resolved_kappa = d3lr_config.get('kappa', 0.5)
        if getattr(args, 'd3lr_alpha_max', 75.0) != 75.0:
            resolved_alpha_max = args.d3lr_alpha_max  # CLI override
        if getattr(args, 'd3lr_kappa', 0.5) != 0.5:
            resolved_kappa = args.d3lr_kappa  # CLI override

        # Determine adapt_type from explicit config first, then backbone structure.
        backbone_name = config['model']['backbone'].lower()
        adapt_type = infer_adapt_type(backbone_name, model, config)

        # Determine number of classes
        if 'cifar10' in dataset_name and 'cifar100' not in dataset_name:
            num_classes = 10
        elif 'cifar100' in dataset_name:
            num_classes = 100
        else:
            num_classes = 1000

        adaptation_mode = 'standard' if config['tta'].get('reset_each_corruption', False) else 'continual'

        triad_class_map = {
            'triad': TRIAD,
            'triad_f1': TRIAD_F1,
            'triad_f2': TRIAD_F2,
            'triad_f3': TRIAD_F3,
            'triad_f4': TRIAD_F4,
        }
        triad_cls = triad_class_map[args.method]

        # TRIAD: D³LR (C1) + SAOD (C2) + AIS (C3)
        model = triad_cls(
            model,
            gate_config=gate_config,
            distill_config=distill_config,
            vit_schedule_config=vit_schedule_config,
            adapt_type=adapt_type,
            adaptation_mode=adaptation_mode,
            lr=config['optim'].get('lr', 0.00025),
            optimizer_name=config['optim'].get('optimizer', 'sgd'),
            optim_momentum=config['optim'].get('momentum', 0.9),
            optim_weight_decay=config['optim'].get('weight_decay', 0.0),
            # Number of classes
            num_classes=num_classes,
            # GGES parameters (legacy)
            grad_clip_norm=triad_cfg.get('grad_clip_norm', 1.0),
            perturbation_scale=triad_cfg.get('perturbation_scale', 0.01),
            # TRIAD switches
            use_sam=not args.no_sam,
            no_gate=args.no_gate,
            no_proximal=args.no_c2,
            no_recovery=args.no_c3,
            no_anti_redundancy=getattr(args, 'no_anti_redundancy', False),
            no_gges=getattr(args, 'no_gges', False),
            # C1: D³LR
            d3lr_alpha_max=resolved_alpha_max,
            d3lr_kappa=resolved_kappa,
            no_ddlr=getattr(args, 'no_c1', False),
            # C3: AIS (no hyperparameters — binary flip-consistency filter)
            # Legacy DGFR params accepted but ignored
            dgfr_momentum=getattr(args, 'dgfr_momentum', 0.1),
            dgfr_gamma=getattr(args, 'dgfr_gamma', 2),
        )
        d3lr_str = f"ON (α_max={model.d3lr_alpha_max}, κ={model.d3lr_kappa})" if not getattr(args, 'no_c1', False) else 'OFF'
        saod_str = f"ON (T={model.distill_temp}, μ={model.distill_weight})" if model.use_source_distill else 'OFF'
        ais_str = 'ON (zero hyperparameters)' if model.use_ais else 'OFF'
        print(f"TRIAD: adapting {len(model.params)} params, e_margin={model.e_margin:.2f}, "
              f"D³LR={d3lr_str}, SAOD={saod_str}, AIS={ais_str}, "
              f"lr={config['optim'].get('lr', 0.00025)}")

    elif args.method == 'tent':
        # Configure Tent
        model = tent.configure_model(model)
        params, param_names = tent.collect_params(model)
        backbone_name = config['model']['backbone'].lower()
        batch_size = get_config_batch_size(config)
        tent_lr = get_sar_family_lr(
            config, dataset_name, backbone_name, batch_size,
            method='tent', scenario=args.scenario
        )
        optimizer = torch.optim.SGD(params, lr=tent_lr, momentum=0.9)
        model = tent.Tent(model, optimizer, steps=1, episodic=False)
        print(f"Tent: adapting {len(params)} parameters, lr={tent_lr:.6f} (official SAR-family protocol)")

    elif args.method == 'eata':
        # Configure EATA with Fisher regularization (TODO 2: Fixed)
        backbone_name = config['model']['backbone'].lower()
        backbone_flags = get_backbone_flags(backbone_name)
        batch_size = get_config_batch_size(config)

        disable_fisher_for_vit = backbone_flags['is_vit'] and not args.force_fisher_on_vit
        use_fisher = (not args.no_fisher) and (not disable_fisher_for_vit)

        if disable_fisher_for_vit:
            print("⚠️  ViT/LN EATA defaults to WITHOUT Fisher: official EATA supports BN/ResNet only, and Fisher regularization on our LayerNorm extension hurts ViT reproduction. Use --force-fisher-on-vit to override.")

        # ✅ Step 1: Compute Fisher information for anti-forgetting
        # EATA paper uses 2000 ID samples to compute diagonal Fisher Information
        # This applies to ALL datasets (ImageNet, CIFAR-10, CIFAR-100)
        fishers = None
        if use_fisher:
            # Import fisher module using absolute path (handles any working directory)
            import importlib.util as _ilu
            _fisher_path = os.path.join(os.path.dirname(__file__), 'utils', 'fisher.py')
            _spec = _ilu.spec_from_file_location('fisher', _fisher_path)
            _fisher_mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_fisher_mod)
            compute_fishers = _fisher_mod.compute_fishers

            fisher_loader = None
            if 'imagenet' in dataset_name.lower():
                from datasets import get_imagenet_val_loader
                imagenet_val_root = config['dataset'].get('imagenet_val_root', './data/imagenet/val')
                if not os.path.exists(imagenet_val_root):
                    print(f"⚠️  Warning: ImageNet val path not found: {imagenet_val_root}")
                    print(f"   EATA will run without Fisher regularization")
                else:
                    print(f"Computing Fisher information from {imagenet_val_root}...")
                    fisher_loader = get_imagenet_val_loader(
                        val_root=imagenet_val_root,
                        batch_size=get_config_batch_size(config),
                        num_workers=config['dataset'].get('num_workers', 4)
                    )
            else:
                # CIFAR-10/100: use training set for Fisher computation
                from datasets import get_cifar_train_loader
                print(f"Computing Fisher information from {dataset_name} training set...")
                fisher_loader = get_cifar_train_loader(
                    dataset_name=dataset_name,
                    data_root=config['dataset'].get('cifar_data_root', './data'),
                    batch_size=get_config_batch_size(config),
                    num_workers=config['dataset'].get('num_workers', 4),
                    preprocess=dataset_preprocess,
                    input_size=input_size,
                )

            if fisher_loader is not None:
                # Temporarily configure model for Fisher computation
                tmp_model = deepcopy(model)
                tmp_model = eata.configure_model(tmp_model)
                fishers = compute_fishers(tmp_model, fisher_loader, num_samples=2000, device=device)
                del tmp_model
                torch.cuda.empty_cache()
                print(f"✅ Fisher information computed successfully")

        # Step 2: Configure EATA normally
        model = eata.configure_model(model)
        params, param_names = eata.collect_params(model)
        eata_lr = get_sar_family_lr(
            config, dataset_name, backbone_name, batch_size,
            method='eata', scenario=args.scenario
        )
        optimizer = torch.optim.SGD(params, lr=eata_lr, momentum=0.9)

        # Get number of classes from config
        if 'cifar10' in dataset_name:
            num_classes = 10
        elif 'cifar100' in dataset_name:
            num_classes = 100
        else:
            num_classes = 1000

        e_margin = 0.4 * math.log(num_classes)
        d_margin = 0.05
        model = eata.EATA(model, optimizer, fishers=fishers, fisher_alpha=2000.0,
                         e_margin=e_margin, d_margin=d_margin, steps=1, episodic=False)
        if fishers is not None:
            fisher_status = "WITH Fisher"
        elif disable_fisher_for_vit:
            fisher_status = "WITHOUT Fisher (ViT default)"
        else:
            fisher_status = "WITHOUT Fisher"
        print(f"EATA: adapting {len(params)} parameters, e_margin={e_margin:.2f}, lr={eata_lr:.6f}, {fisher_status}")


    elif args.method == 'sar':
        # Configure SAR with SAM optimizer
        model = sar.configure_model(model)
        params, param_names = sar.collect_params(model)
        base_optimizer = torch.optim.SGD
        backbone_name = config['model']['backbone'].lower()
        batch_size = get_config_batch_size(config)
        sar_lr = get_sar_family_lr(
            config, dataset_name, backbone_name, batch_size,
            method='sar', scenario=args.scenario
        )
        optimizer = SAM(params, base_optimizer, lr=sar_lr, momentum=0.9)

        # Get number of classes from config
        if 'cifar10' in dataset_name:
            num_classes = 10
        elif 'cifar100' in dataset_name:
            num_classes = 100
        else:
            num_classes = 1000

        margin_e0 = 0.4 * math.log(num_classes)
        model = sar.SAR(model, optimizer, steps=1, episodic=False,
                       margin_e0=margin_e0, reset_constant_em=0.2)
        print(f"SAR: adapting {len(params)} parameters, margin={margin_e0:.2f}, lr={sar_lr:.6f}")

    elif args.method == 'cotta':
        # Dynamically load CoTTA based on dataset (different image sizes!)
        # CIFAR uses 32x32, ImageNet uses 224x224
        cotta = get_cotta_module(dataset_name, transform_family=transform_family)
        backbone_name = config['model']['backbone'].lower()
        backbone_flags = get_backbone_flags(backbone_name)
        use_imagenet_family = transform_family == 'imagenet'

        if backbone_flags['is_vit']:
            cotta_adapt_type = 'ln'
            cotta_arch_type = 'vit'
        elif backbone_flags['is_gn']:
            cotta_adapt_type = 'gn'
            cotta_arch_type = 'resnet'
        else:
            cotta_adapt_type = 'bn'
            cotta_arch_type = 'resnet'

        # Backbone-aware CoTTA adaptation: BN for ResNet, GN for ResNet-GN, LN for ViT
        model = call_with_supported_kwargs(
            cotta.configure_model,
            model,
            adapt_type=cotta_adapt_type,
        )
        params, param_names = call_with_supported_kwargs(
            cotta.collect_params,
            model,
            adapt_type=cotta_adapt_type,
            arch_type=cotta_arch_type,
        )

        # CoTTA optimizer: backbone-dependent (matching official repo)
        # BN ImageNet: SGD(lr=0.01, momentum=0.9, nesterov=True) per cotta0.yaml
        # GN/CIFAR: Adam(lr=0.001) per cifar default
        is_bn_backbone = backbone_flags['is_bn']

        if use_imagenet_family and is_bn_backbone:
            # Official CoTTA ImageNet config: SGD lr=0.01, momentum=0.9, nesterov=True
            cotta_lr = 0.01
            optimizer = torch.optim.SGD(params, lr=cotta_lr, momentum=0.9,
                                        dampening=0.0, nesterov=True)
            print(f"CoTTA: using SGD(lr={cotta_lr}, nesterov=True) for BN backbone (per official repo)")
        else:
            optimizer = torch.optim.Adam(params, lr=config['optim'].get('lr', 0.001))
            print(f"CoTTA: using Adam optimizer")

        # CIFAR version has mt_alpha, rst_m, ap parameters; ImageNet version doesn't
        if use_imagenet_family:
            model = cotta.CoTTA(model, optimizer, steps=1, episodic=False)
        else:
            model = cotta.CoTTA(model, optimizer, steps=1, episodic=False,
                               mt_alpha=0.999, rst_m=0.01, ap=0.92)
        print(
            f"CoTTA: adapting {len(params)} parameters "
            f"(adapt_type={cotta_adapt_type}, backbone={backbone_name}, "
            f"using {transform_family} transforms)"
        )

    elif args.method == 'lcotta':
        backbone_name = config['model']['backbone'].lower()
        backbone_flags = get_backbone_flags(backbone_name)

        if backbone_flags['is_vit']:
            lcotta_adapt_type = 'ln'
            lcotta_arch_note = 'vit-ln'
        elif backbone_flags['is_gn']:
            lcotta_adapt_type = 'gn'
            lcotta_arch_note = 'resnet-gn'
        else:
            lcotta_adapt_type = 'bn'
            lcotta_arch_note = 'resnet-bn'

        model = lcotta_method.configure_model(model, adapt_type=lcotta_adapt_type)
        params, param_names = lcotta_method.collect_params(model, adapt_type=lcotta_adapt_type)

        if 'cifar10' in dataset_name:
            num_classes = 10
        elif 'cifar100' in dataset_name:
            num_classes = 100
        else:
            num_classes = 1000

        lcotta_hparams = get_lcotta_hparams(config, dataset_name, backbone_name)
        # Match official LCoTTA (subspace_plus) defaults: SGD with Nesterov
        # (see code/baselines/LCoTTA-main/classification/conf.py)
        optimizer = torch.optim.SGD(
            params,
            lr=lcotta_hparams['lr'],
            momentum=0.9,
            dampening=0.0,
            nesterov=True,
        )
        model = lcotta_method.LCoTTA(
            model,
            optimizer,
            steps=1,
            episodic=False,
            window_length=lcotta_hparams['window_length'],
            n_components=lcotta_hparams['n_components'],
            batch_step=lcotta_hparams['batch_step'],
            e_margin=0.4 * math.log(num_classes),
            cosine_margin=lcotta_hparams['cosine_margin'],
            prob_momentum=lcotta_hparams['prob_momentum'],
        )
        print(
            f"LCoTTA: adapting {len(params)} parameters "
            f"(backbone={backbone_name}, norm={lcotta_arch_note}, lr={lcotta_hparams['lr']}, "
            f"window={lcotta_hparams['window_length']}, rank={lcotta_hparams['n_components']}, "
            f"batch_step={lcotta_hparams['batch_step']})"
        )
        if 'imagenet' not in dataset_name.lower():
            print("LCoTTA: CIFAR/natural-shift support is a project-side extension beyond the official ImageNet-C release")

    elif args.method == 'adadem':
        backbone_name = config['model']['backbone'].lower()
        adadem_hparams = get_adadem_hparams(config, dataset_name, backbone_name)

        model = adadem_method.configure_model(model)
        params, param_names = adadem_method.collect_params(model)
        # Official AdaDEM uses plain SGD without momentum: torch.optim.SGD(params, lr)
        optimizer = torch.optim.SGD(params, lr=adadem_hparams['lr'])
        model = adadem_method.AdaDEM(
            model,
            optimizer,
            steps=1,
            episodic=False,
            pi=adadem_hparams['pi'],
        )
        print(
            f"AdaDEM: adapting {len(params)} parameters "
            f"(backbone={backbone_name}, lr={adadem_hparams['lr']}, pi={adadem_hparams['pi']})"
        )
        if 'imagenet' not in dataset_name.lower():
            print("AdaDEM: CIFAR support is a project-side extension beyond the official ImageNet release")

    elif args.method == 'surgeon':
        backbone_name = config['model']['backbone'].lower()
        backbone_flags = get_backbone_flags(backbone_name)
        surgeon_hparams = get_surgeon_hparams(config, dataset_name, backbone_name)

        model = surgeon_method.configure_model(
            model,
            bn_only=surgeon_hparams['bn_only'],
            backbone_flags=backbone_flags,
        )
        model = model.to(device)
        params, param_names = surgeon_method.collect_params(model, bn_only=surgeon_hparams['bn_only'])
        optimizer = torch.optim.Adam(params, lr=surgeon_hparams['lr'])

        if 'cifar10' in dataset_name:
            num_classes = 10
        elif 'cifar100' in dataset_name:
            num_classes = 100
        else:
            num_classes = 1000

        model = surgeon_method.SURGEON(
            model,
            optimizer,
            dataset_name=dataset_name,
            num_classes=num_classes,
            steps=1,
            episodic=False,
            bn_only=surgeon_hparams['bn_only'],
            probe_samples=surgeon_hparams['probe_samples'],
            css_weight=surgeon_hparams['css_weight'],
            backbone_flags=backbone_flags,
            transform_family=transform_family,
        )
        print(
            f"SURGEON: adapting {len(params)} parameters "
            f"(backbone={backbone_name}, bn_only={surgeon_hparams['bn_only']}, "
            f"lr={surgeon_hparams['lr']}, probe_samples={surgeon_hparams['probe_samples']})"
        )
        if backbone_flags['is_vit']:
            print("SURGEON: ViT path is a project-side extension; official released classification code does not provide ViT configs.")
        elif backbone_flags['is_gn']:
            print("SURGEON: GroupNorm path is a project-side extension; official released classification code focuses on BN ResNet backbones.")

    elif args.method == 'foa':
        backbone_name = config['model']['backbone'].lower()
        backbone_flags = get_backbone_flags(backbone_name)
        foa_hparams = get_foa_hparams(config, dataset_name, backbone_name)

        if not backbone_flags['is_vit']:
            raise ValueError("FOA currently supports ViT backbones only")
        if 'imagenet' not in dataset_name.lower():
            raise ValueError("FOA currently supports ImageNet-family datasets only")

        from datasets import get_imagenet_val_loader
        imagenet_val_root = config['dataset'].get('imagenet_val_root', './data/imagenet/val')
        if not os.path.exists(imagenet_val_root):
            raise ValueError(f"FOA requires ImageNet source data at {imagenet_val_root} to compute source statistics")

        model = foa_method.configure_model(model, num_prompts=foa_hparams['num_prompts'])
        model = model.to(device)

        # Set imagenet_mask for ImageNet-R (200-class subset)
        dataset_key = dataset_name.lower().replace('-', '_')
        if dataset_key == 'imagenet_r':
            imagenet_mask = foa_method.get_imagenet_r_mask()
        else:
            imagenet_mask = None

        model = foa_method.FOA(
            model,
            fitness_lambda=foa_hparams['fitness_lambda'],
            sigma=foa_hparams['sigma'],
            popsize=foa_hparams['popsize'],
            hist_momentum=foa_hparams['hist_momentum'],
            reference_batch_size=foa_hparams['reference_batch_size'],
            imagenet_mask=imagenet_mask,
        )
        source_loader = get_imagenet_val_loader(
            val_root=imagenet_val_root,
            batch_size=get_config_batch_size(config),
            num_workers=config['dataset'].get('num_workers', 4),
        )
        model.obtain_origin_stat(source_loader, max_samples=foa_hparams['source_samples'])
        print(
            f"FOA: ViT-only forward adaptation initialized "
            f"(prompts={foa_hparams['num_prompts']}, lambda={foa_hparams['fitness_lambda']}, "
            f"popsize={foa_hparams['popsize']}, source_samples={foa_hparams['source_samples']})"
        )
        print("FOA: this integration follows the official ViT/ImageNet path; CNN/GN backbones remain unsupported by design.")
        if config.get('tta', {}).get('scenario') == 'bs1':
            print(
                "FOA: BS=1 uses a project-side singleton-batch extension with population variance; "
                "the variance-alignment term becomes non-informative, so adaptation is driven by mean alignment and entropy."
            )

    elif args.method == 'deyo':
        # Configure DeYO
        model = deyo_method.configure_model(model)
        params, param_names = deyo_method.collect_params(model)

        # Get number of classes from config
        if 'cifar10' in dataset_name:
            num_classes = 10
        elif 'cifar100' in dataset_name:
            num_classes = 100
        else:
            num_classes = 1000

        backbone_name = config['model']['backbone'].lower()
        batch_size = get_config_batch_size(config)
        is_vit_backbone = 'vit' in backbone_name or 'swin' in backbone_name
        is_gn_backbone = 'gn' in backbone_name or 'groupnorm' in backbone_name

        # Match the official DeYO implementation more closely:
        # - Optimizer: plain SGD (not SAM)
        # - ViT uses lr = (0.001 / 64) * batch_size
        # - ResNet uses lr = 0.00025 for batch_size >= 32, scaled for smaller batches
        if is_vit_backbone:
            deyo_lr = (0.001 / 64.0) * batch_size
            plpd_threshold = 0.2
        else:
            deyo_lr = (0.00025 / 64.0) * batch_size * 2 if batch_size < 32 else 0.00025
            plpd_threshold = 0.2 if is_gn_backbone else 0.3

        optimizer = torch.optim.SGD(params, lr=deyo_lr, momentum=0.9)

        class DeYOArgs:
            def __init__(self):
                self.wandb_log = False
                self.filter_ent = True
                self.filter_plpd = True
                self.plpd_threshold = plpd_threshold
                self.reweight_ent = 1
                self.reweight_plpd = 1
                self.aug_type = 'patch'
                self.patch_len = 4
                if input_size >= 224:
                    self.occlusion_size = 112
                    self.row_start = 56
                    self.column_start = 56
                else:
                    self.occlusion_size = 16
                    self.row_start = 8
                    self.column_start = 8
                self.counts = [1e-6, 1e-6, 1e-6, 1e-6]
                self.correct_counts = [0, 0, 0, 0]

        deyo_args = DeYOArgs()
        deyo_margin = 0.5 * math.log(num_classes)
        margin_e0 = 0.4 * math.log(num_classes)
        model = deyo_method.DeYO(model, deyo_args, optimizer, steps=1, episodic=False,
                                 deyo_margin=deyo_margin, margin_e0=margin_e0)
        print(
            f"DeYO: adapting {len(params)} parameters, deyo_margin={deyo_margin:.2f}, "
            f"plpd_threshold={plpd_threshold}, aug_type=patch, lr={deyo_lr:.6f}"
        )

    elif args.method == 'rotta':
        # Configure RoTTA with RobustBN/RobustGN (momentum-based normalization)
        # RoTTA uses RobustBN for BatchNorm models, RobustGN for GroupNorm models,
        # and RobustLN as a project-side extension for ViT LayerNorm backbones.
        # Both provide momentum-smoothed statistics/parameters for stability during adaptation
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
        has_gn = any(isinstance(m, nn.GroupNorm) for m in model.modules())
        has_ln = any(isinstance(m, nn.LayerNorm) for m in model.modules())

        if has_bn:
            print("RoTTA: BatchNorm model detected, will replace BN with RobustBN")
        elif has_gn:
            print("RoTTA: GroupNorm model detected, will replace GN with RobustGN")
            print("       (RobustGN uses momentum-smoothed affine parameters for stability)")
        elif has_ln:
            print("RoTTA: LayerNorm model detected, will replace LN with RobustLN")
            print("       (custom project extension for ViT; not part of official BN-only protocol)")

        # RoTTA paper: Adam lr=0.001 - ALWAYS use paper lr, ignore config
        # (config lr=0.00025 is for EATA/Tent, not RoTTA)
        rotta_lr = 0.001
        # Pass optimizer CLASS and kwargs (NOT instance) to RoTTAWrapper
        # Official RoTTA creates optimizer AFTER configure_model, so it gets the correct
        # RobustBN/RobustGN parameters. We follow the same pattern.
        optimizer_cls = torch.optim.Adam
        optimizer_kwargs = {'lr': rotta_lr}

        # Determine num_classes and memory_size for CSTU memory
        # Memory size should scale with num_classes to ensure class balance
        # For ImageNet with 224x224 images, use smaller memory to avoid OOM
        if 'cifar10' in dataset_name and 'cifar100' not in dataset_name:
            num_classes = 10
            memory_size = 64  # ~6.4 samples per class
        elif 'cifar100' in dataset_name:
            num_classes = 100
            memory_size = 64  # ~2.56 samples per class
        else:
            num_classes = 1000
            memory_size = 32  # Reduced to avoid OOM (ImageNet: 224x224 images)

        # RoTTA wrapper with memory bank, EMA teacher, and RobustBN/RobustGN
        # Following official paper hyperparameters:
        # - memory_size: Size of class-balanced memory bank (scaled by num_classes)
        # - nu=0.001: EMA update rate for teacher model
        # - update_freq: Update frequency (same as memory size)
        # - alpha=0.05: Momentum for RobustBN/RobustGN statistics/parameters
        # - num_classes: For CSTU class-balanced memory
        model = RoTTAWrapper(model, optimizer_cls, optimizer_kwargs,
                            memory_size=memory_size,
                            nu=0.001,
                            update_freq=memory_size,  # Match memory_size
                            alpha=0.05,
                            num_classes=num_classes)  # RobustBN/RobustGN momentum + class-balanced CSTU
        norm_type = "RobustBN" if has_bn else ("RobustGN" if has_gn else "RobustLN")
        print(f"RoTTA: using {norm_type}, alpha=0.05, memory_size={memory_size}, nu=0.001, lr={rotta_lr}")
        print(f"RoTTA: CSTU class-balanced memory + strong augmentation (ColorJitterPro)")

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Determine dataset type
    dataset_type = config['dataset'].get('dataset_type', 'corruption')

    # Natural distribution shift datasets (ImageNet-R, A, V2, Sketch)
    # These are single-domain datasets, not corruption-based
    NATURAL_SHIFT_DATASETS = ('imagenet_r', 'imagenet_a', 'imagenet_v2', 'imagenet_sketch',
                               'imagenetr', 'imagineta', 'imagenetv2', 'imagenetsketch')

    if dataset_type == 'natural_shift' or dataset_name.lower().replace('-', '_') in NATURAL_SHIFT_DATASETS:
        print(f"Natural shift dataset: {dataset_name}")

        shuffle = config['dataset'].get('shuffle', False)

        dataloader = get_natural_shift_loader(
            dataset_name=dataset_name,
            data_root=config['dataset']['data_root'],
            batch_size=get_config_batch_size(config),
            num_workers=config['dataset'].get('num_workers', 4),
            shuffle=shuffle,
            seed=args.seed
        )

        # Evaluate on the entire dataset as one domain
        results = evaluate_corruption(
            model,
            dataloader,
            device,
            method=args.method,
            max_batches=getattr(args, 'max_batches', 0),
        )
        all_results = {dataset_name: results}

        summary = {
            'method': result_method_name,
            'base_method': args.method,
            'backbone': config['model']['backbone'],
            'backbone_provenance': backbone_provenance,
            'dataset': dataset_name,
            'seed': args.seed,
            'scenario': 'natural_shift',
            'max_batches': getattr(args, 'max_batches', 0),
            'mean_accuracy': results['accuracy'],
            'per_corruption': all_results,
            'timestamp': get_timestamp(),
        }
        if getattr(args, 'run_suffix', None):
            summary['run_suffix'] = args.run_suffix
        if getattr(args, 'atlas_ablation_row', None):
            summary['atlas_ablation_row'] = args.atlas_ablation_row
        if args.method == 'atlas':
            summary['atlas_method_variant'] = getattr(model, 'method_variant', atlas_method_variant)
        atlas_hparam_nested, atlas_hparam_flat = collect_atlas_hparam_overrides(args)
        if atlas_hparam_flat:
            summary['atlas_hparam_overrides'] = atlas_hparam_flat
            for section_name, section_values in atlas_hparam_nested.items():
                for field_name, field_value in section_values.items():
                    summary[f'atlas_{section_name}_{field_name}'] = field_value

        print(f"\n{dataset_name} Accuracy: {summary['mean_accuracy']:.2f}%")

        # Save results
        if config['logging'].get('save_results', True):
            results_dir = os.path.join(
                config['logging'].get('results_dir', './results'),
                f"{dataset_name}_{config['model']['backbone']}"
            )
            filename = f"{result_method_name}_seed{args.seed}.json"
            save_results(summary, results_dir, filename)

        return summary

    # Get corruption types (for corruption-based datasets)
    if 'imagenet' in dataset_name:
        corruptions = IMAGENET_C_CORRUPTIONS
    else:
        corruptions = CIFAR_C_CORRUPTIONS

    # Check if specific corruptions requested
    corruption_config = config['dataset'].get('corruption_types', 'all')
    if corruption_config != 'all':
        corruptions = corruption_config

    # Support custom corruption order (for Continual TTA evaluation with multiple orders)
    if hasattr(args, 'corruption_order') and args.corruption_order:
        corruptions = [c.strip() for c in args.corruption_order.split(',')]
        print(f"Using custom corruption order (order {args.order_idx}): {corruptions}")

    severity = config['dataset'].get('severity', 5)

    # Optional long-term evaluation repeats (used by LCoTTA official protocol)
    lcotta_repeats = 1

    # Get scenario from config or args
    scenario = config['tta'].get('scenario', 'normal')
    if hasattr(args, 'scenario') and args.scenario:
        scenario = args.scenario

    label_shift_indices = None
    if scenario == 'label_shifts':
        imbalance_ratio = getattr(args, 'imbalance_ratio', 500000)
        label_shift_indices = get_label_shift_indices(
            dataset_name=dataset_name,
            imbalance_ratio=imbalance_ratio,
            seed=args.seed,
            cache_dir=os.path.join('results', '_label_shift_indices'),
        )
        print(
            "Prepared label_shifts indices: "
            f"steps={len(label_shift_indices)}, imbalance_ratio={imbalance_ratio}"
        )

    continual_repeats = None
    cycle_mean_accuracies = None
    cycle_summaries = None

    # Handle wild scenarios
    if scenario == 'mix_shifts':
        # Mix all corruptions into one dataset
        print("Wild scenario: mix_shifts - mixing all corruptions")
        from torch.utils.data import ConcatDataset

        all_datasets = []
        for corruption in corruptions:
            if 'cifar' in dataset_name.lower():
                from datasets import CIFAR_C_Dataset
                ds = CIFAR_C_Dataset(
                    data_root=config['dataset']['data_root'],
                    corruption=corruption,
                    severity=severity,
                    transform=build_cifar_transform(
                        preprocess=dataset_preprocess,
                        input_size=input_size,
                    ),
                )
            else:
                from datasets import ImageNetC_Dataset
                ds = ImageNetC_Dataset(
                    data_root=config['dataset']['data_root'],
                    corruption=corruption,
                    severity=severity
                )
            all_datasets.append(ds)

        mixed_dataset = ConcatDataset(all_datasets)
        shuffle = config['dataset'].get('shuffle', True)  # Mix shifts typically shuffled

        generator = None
        if shuffle and args.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(args.seed)

        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            mixed_dataset,
            batch_size=get_config_batch_size(config),
            shuffle=shuffle,
            num_workers=config['dataset'].get('num_workers', 4),
            pin_memory=True,
            generator=generator if shuffle else None
        )

        # Evaluate on mixed dataset
        results = evaluate_corruption(
            model,
            dataloader,
            device,
            method=args.method,
            max_batches=getattr(args, 'max_batches', 0),
        )
        all_results = {'mix_shifts': results}

        print(f"  mix_shifts: {results['accuracy']:.2f}%")

    elif scenario == 'label_shifts':
        # Non-IID label distribution
        print(f"Wild scenario: label_shifts - imbalance_ratio={args.imbalance_ratio if hasattr(args, 'imbalance_ratio') else 500000}")

        all_results = {}
        shuffle = False  # Label shifts uses pre-defined order

        corruption_pbar = tqdm(corruptions, desc=f'{args.method.upper()} - Label Shifts',
                              ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for corruption in corruption_pbar:
            corruption_pbar.set_postfix_str(f'current: {corruption}')

            dataloader = get_corruption_loader(
                dataset_name=dataset_name,
                corruption=corruption,
                severity=severity,
                data_root=config['dataset']['data_root'],
                batch_size=get_config_batch_size(config),
                num_workers=config['dataset'].get('num_workers', 4),
                shuffle=shuffle,
                seed=args.seed,
                preprocess=dataset_preprocess,
                input_size=input_size,
                subset_indices=label_shift_indices,
            )

            # Reset model if configured
            if config['tta'].get('reset_each_corruption', False):
                if hasattr(model, 'reset'):
                    model.reset()
            if hasattr(model, 'set_current_corruption'):
                model.set_current_corruption(corruption)

            results = evaluate_corruption(
                model,
                dataloader,
                device,
                method=args.method,
                max_batches=getattr(args, 'max_batches', 0),
            )
            all_results[corruption] = results
            corruption_pbar.set_postfix_str(f'{corruption}: {results["accuracy"]:.2f}%')
            print(f"  ✓ {corruption}: {results['accuracy']:.2f}%")

    elif scenario == 'bs1':
        # Single sample adaptation
        print("Wild scenario: bs1 - single sample adaptation")
        config['dataset']['batch_size'] = 1  # Override batch size

        all_results = {}
        shuffle = config['dataset'].get('shuffle', True)  # BS1 typically uses shuffle

        corruption_pbar = tqdm(corruptions, desc=f'{args.method.upper()} - BS1 (slow!)',
                              ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for corruption in corruption_pbar:
            corruption_pbar.set_postfix_str(f'current: {corruption}')

            dataloader = get_corruption_loader(
                dataset_name=dataset_name,
                corruption=corruption,
                severity=severity,
                data_root=config['dataset']['data_root'],
                batch_size=1,  # Force batch size 1
                num_workers=config['dataset'].get('num_workers', 4),
                shuffle=shuffle,
                seed=args.seed,
                preprocess=dataset_preprocess,
                input_size=input_size,
            )

            # Reset model if configured
            if config['tta'].get('reset_each_corruption', False):
                if hasattr(model, 'reset'):
                    model.reset()
            if hasattr(model, 'set_current_corruption'):
                model.set_current_corruption(corruption)

            results = evaluate_corruption(
                model,
                dataloader,
                device,
                method=args.method,
                max_batches=getattr(args, 'max_batches', 0),
            )
            all_results[corruption] = results
            corruption_pbar.set_postfix_str(f'{corruption}: {results["accuracy"]:.2f}%')
            print(f"  ✓ {corruption}: {results['accuracy']:.2f}%")

        # BS=1 is a complete evaluation branch and must not fall through to
        # the normal continual-cycle summary closure below.
        accuracies = [float(result["accuracy"]) for result in all_results.values()]
        bs1_summary = {
            "method": result_method_name,
            "base_method": args.method,
            "backbone": config["model"]["backbone"],
            "backbone_provenance": backbone_provenance,
            "dataset": dataset_name,
            "severity": severity,
            "seed": args.seed,
            "scenario": scenario,
            "max_batches": getattr(args, "max_batches", 0),
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
            "per_corruption": all_results,
            "timestamp": get_timestamp(),
        }
        if getattr(args, "run_suffix", None):
            bs1_summary["run_suffix"] = args.run_suffix
        if args.method == 'atlas':
            bs1_summary['atlas_method_variant'] = getattr(model, 'method_variant', atlas_method_variant)
        atlas_hparam_nested, atlas_hparam_flat = collect_atlas_hparam_overrides(args)
        if atlas_hparam_flat:
            bs1_summary["atlas_hparam_overrides"] = atlas_hparam_flat
            for section_name, section_values in atlas_hparam_nested.items():
                for field_name, field_value in section_values.items():
                    bs1_summary[f"atlas_{section_name}_{field_name}"] = field_value
        if getattr(args, "corruption_order", None):
            bs1_summary["corruption_order"] = corruptions
            bs1_summary["order_idx"] = getattr(args, "order_idx", 0)
        bs1_results_dir = os.path.join(
            config["logging"].get("results_dir", "./results"),
            f"{dataset_name}_{config['model']['backbone']}_{scenario}",
        )
        bs1_filename = f"{result_method_name}_seed{args.seed}.json"
        if config["logging"].get("save_results", True):
            save_results(bs1_summary, bs1_results_dir, bs1_filename)
        return bs1_summary

    else:
        # Normal evaluation (Standard or Continual TTA)
        all_results = {}

        # LCoTTA official protocol is long-term continual adaptation (e.g. 50 cycles on ImageNet-C),
        # but the repeat control is protocol-level rather than method-specific.
        continual_repeats = getattr(args, 'continual_repeats', None)
        if continual_repeats is None:
            continual_repeats = 1
        continual_repeats = max(1, int(continual_repeats))

        # Get shuffle setting from config (default: False for backward compatibility)
        shuffle = config['dataset'].get('shuffle', False)

        # Enhanced progress bar with method name and order info
        order_suffix = f" [Order {args.order_idx}]" if hasattr(args, 'order_idx') and args.order_idx is not None else ""
        desc = f'{args.method.upper()}{order_suffix}'

        save_results_enabled = config['logging'].get('save_results', True)
        reset_each_corruption = config['tta'].get('reset_each_corruption', False)
        results_dir_suffix = ""
        if shuffle:
            results_dir_suffix += "_shuffle"
        if not reset_each_corruption:
            results_dir_suffix += "_continual"
        if scenario != 'normal':
            results_dir_suffix += f"_{scenario}"

        cycle_results_dir = os.path.join(
            config['logging'].get('results_dir', './results'),
            f"{dataset_name}_{config['model']['backbone']}{results_dir_suffix}"
        )
        if hasattr(args, 'order_idx') and args.corruption_order:
            cycle_results_filename = f"{result_method_name}_seed{args.seed}_order{args.order_idx}.json"
        else:
            cycle_results_filename = f"{result_method_name}_seed{args.seed}.json"

        cycle_mean_accuracies = []
        cycle_summaries = []

        def build_corruption_summary(current_results, completed_cycles=None):
            accuracies = [r['accuracy'] for r in current_results.values()]
            summary = {
                'method': result_method_name,
                'base_method': args.method,
                'backbone': config['model']['backbone'],
                'backbone_provenance': backbone_provenance,
                'dataset': dataset_name,
                'severity': severity,
                'seed': args.seed,
                'scenario': scenario,
                'max_batches': getattr(args, 'max_batches', 0),
                'reset_each_corruption': reset_each_corruption,
                'shuffle': shuffle,
                'mean_accuracy': sum(accuracies) / len(accuracies) if accuracies else None,
                'per_corruption': current_results,
                'timestamp': get_timestamp(),
            }

            if getattr(args, 'run_suffix', None):
                summary['run_suffix'] = args.run_suffix
            if getattr(args, 'atlas_ablation_row', None):
                summary['atlas_ablation_row'] = args.atlas_ablation_row
            if args.method == 'atlas':
                summary['atlas_method_variant'] = getattr(model, 'method_variant', atlas_method_variant)

            atlas_hparam_nested, atlas_hparam_flat = collect_atlas_hparam_overrides(args)
            if atlas_hparam_flat:
                summary['atlas_hparam_overrides'] = atlas_hparam_flat
                for section_name, section_values in atlas_hparam_nested.items():
                    for field_name, field_value in section_values.items():
                        summary[f'atlas_{section_name}_{field_name}'] = field_value

            if continual_repeats is not None:
                completed = int(completed_cycles if completed_cycles is not None else len(cycle_mean_accuracies or []))
                summary['continual_repeats'] = int(continual_repeats)
                summary['completed_cycles'] = completed
                summary['cycle_mean_accuracies'] = cycle_mean_accuracies or []
                summary['cycle_summaries'] = cycle_summaries or []
                summary['is_partial_cycle_dump'] = completed < int(continual_repeats)
                if cycle_mean_accuracies:
                    summary['latest_cycle_mean_accuracy'] = cycle_mean_accuracies[-1]
                    summary['final_cycle_mean_accuracy'] = cycle_mean_accuracies[-1]
                if args.method == 'lcotta':
                    summary['lcotta_repeats'] = int(continual_repeats)

            if hasattr(args, 'order_idx') and args.corruption_order:
                summary['order_idx'] = args.order_idx
                summary['corruption_order'] = corruptions

            return summary

        def save_cycle_checkpoint(current_results, completed_cycles):
            if not save_results_enabled:
                return
            checkpoint_summary = build_corruption_summary(current_results, completed_cycles=completed_cycles)
            save_results(checkpoint_summary, cycle_results_dir, cycle_results_filename)

        if continual_repeats == 1:
            cycle_per_corruption = {}
            corruption_pbar = tqdm(corruptions, desc=desc,
                                  ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

            for corruption in corruption_pbar:
                corruption_pbar.set_postfix_str(f'processing: {corruption}')

                dataloader = get_corruption_loader(
                    dataset_name=dataset_name,
                    corruption=corruption,
                    severity=severity,
                    data_root=config['dataset']['data_root'],
                    batch_size=get_config_batch_size(config),
                    num_workers=config['dataset'].get('num_workers', 4),
                    shuffle=shuffle,
                    seed=args.seed,
                    preprocess=dataset_preprocess,
                    input_size=input_size,
                )

                # Reset model if configured (for each_shift_reset mode, like SAR paper)
                # This prevents error accumulation across corruption types
                if config['tta'].get('reset_each_corruption', False):
                    if hasattr(model, 'reset'):
                        model.reset()
                        if args.method != 'source':
                            corruption_pbar.write(f"    [Reset] Model reset before {corruption}")
                if hasattr(model, 'set_current_corruption'):
                    model.set_current_corruption(corruption)

                # Evaluate
                results = evaluate_corruption(
                    model,
                    dataloader,
                    device,
                    method=args.method,
                    max_batches=getattr(args, 'max_batches', 0),
                )
                all_results[corruption] = results
                cycle_per_corruption[corruption] = float(results['accuracy'])

                corruption_pbar.set_postfix_str(f'{corruption}: {results["accuracy"]:.2f}%')
                corruption_pbar.write(f"  ✓ {corruption}: {results['accuracy']:.2f}%")

            single_cycle_mean = sum(cycle_per_corruption.values()) / max(1, len(cycle_per_corruption))
            cycle_mean_accuracies.append(single_cycle_mean)
            cycle_summaries.append({
                'cycle_index': 1,
                'mean_accuracy': single_cycle_mean,
                'per_corruption': cycle_per_corruption,
            })
            for corruption, results in all_results.items():
                results['repeat_accuracies'] = [float(results['accuracy'])]
                results['final_cycle_accuracy'] = float(results['accuracy'])

        else:
            # Long-term continual evaluation: repeat the whole corruption sequence multiple times.
            # We aggregate per-corruption accuracy across repeats and retain cycle-level summaries.
            per_corruption_accs = {c: [] for c in corruptions}
            total_steps = continual_repeats * len(corruptions)
            long_pbar = tqdm(range(total_steps), desc=f'{desc} (repeats={continual_repeats})',
                             ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

            # Cache dataloaders to avoid rebuilding datasets each repeat.
            dataloaders = {
                corruption: get_corruption_loader(
                    dataset_name=dataset_name,
                    corruption=corruption,
                    severity=severity,
                    data_root=config['dataset']['data_root'],
                    batch_size=get_config_batch_size(config),
                    num_workers=config['dataset'].get('num_workers', 4),
                    shuffle=shuffle,
                    seed=args.seed,
                    persistent_workers=True,
                    preprocess=dataset_preprocess,
                    input_size=input_size,
                )
                for corruption in corruptions
            }

            for repeat_idx in range(continual_repeats):
                cycle_per_corruption = {}
                cycle_accuracies = []
                for corruption in corruptions:
                    long_pbar.set_postfix_str(f'r{repeat_idx + 1}/{continual_repeats}: {corruption}')

                    dataloader = dataloaders[corruption]

                    if config['tta'].get('reset_each_corruption', False):
                        if hasattr(model, 'reset'):
                            model.reset()
                    if hasattr(model, 'set_current_corruption'):
                        model.set_current_corruption(corruption)

                    results = evaluate_corruption(
                        model,
                        dataloader,
                        device,
                        method=args.method,
                        max_batches=getattr(args, 'max_batches', 0),
                    )
                    accuracy = float(results['accuracy'])
                    per_corruption_accs[corruption].append(accuracy)
                    cycle_per_corruption[corruption] = accuracy
                    cycle_accuracies.append(accuracy)

                    long_pbar.update(1)

                cycle_mean = sum(cycle_accuracies) / max(1, len(cycle_accuracies))
                cycle_mean_accuracies.append(cycle_mean)
                cycle_summaries.append({
                    'cycle_index': repeat_idx + 1,
                    'mean_accuracy': cycle_mean,
                    'per_corruption': cycle_per_corruption,
                })

                current_all_results = {
                    corruption_name: {
                        'accuracy': sum(accs) / max(1, len(accs)),
                        'repeat_accuracies': list(accs),
                        'final_cycle_accuracy': accs[-1] if accs else None,
                    }
                    for corruption_name, accs in per_corruption_accs.items()
                    if accs
                }
                save_cycle_checkpoint(current_all_results, repeat_idx + 1)

            long_pbar.close()

            for corruption, accs in per_corruption_accs.items():
                mean_acc = sum(accs) / max(1, len(accs))
                all_results[corruption] = {
                    'accuracy': mean_acc,
                    'repeat_accuracies': accs,
                    'final_cycle_accuracy': accs[-1] if accs else None,
                }
                print(f"  ✓ {corruption}: {mean_acc:.2f}% (avg over {len(accs)} repeats)")

    # Compute summary statistics
    summary = build_corruption_summary(all_results, completed_cycles=continual_repeats)

    print(f"\nMean Accuracy: {summary['mean_accuracy']:.2f}%")

    # Save results
    if save_results_enabled:
        save_results(summary, cycle_results_dir, cycle_results_filename)

    return summary


def main():
    args = parse_args()

    external_requirements = {
        'tent': tent,
        'eata': eata,
        'sar': sar,
        'adadem': adadem_method,
        'foa': foa_method,
        'lcotta': lcotta_method,
        'surgeon': surgeon_method,
        'deyo': deyo_method,
        'triad': TRIAD,
        'triad_f1': TRIAD_F1,
        'triad_f2': TRIAD_F2,
        'triad_f3': TRIAD_F3,
        'triad_f4': TRIAD_F4,
    }
    if args.method in external_requirements and external_requirements[args.method] is None:
        raise RuntimeError(
            f"Method '{args.method}' requires its official external checkout. "
            "The public release intentionally ships only the released TTA method; "
            "see code/baselines/README.md for provenance."
        )

    _, atlas_hparam_flat = collect_atlas_hparam_overrides(args)

    if args.atlas_ablation_row and args.method != 'atlas':
        raise ValueError('--atlas-ablation-row only supports --method atlas')
    if args.atlas_active_views and args.method != 'atlas':
        raise ValueError('--atlas-active-views only supports --method atlas')
    if atlas_hparam_flat and args.method != 'atlas':
        raise ValueError('ATLAS hyperparameter overrides only support --method atlas')

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.backbone:
        config['model']['backbone'] = args.backbone
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.batch_size is not None:
        config.setdefault('dataset', {})
        config['dataset']['batch_size'] = args.batch_size
    if args.results_dir is not None:
        config.setdefault('logging', {})
        config['logging']['results_dir'] = args.results_dir

    # Run experiment
    results = run_experiment(config, args)

    return results


if __name__ == '__main__':
    main()
