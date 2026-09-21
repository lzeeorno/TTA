"""
Model definitions and loading utilities
"""

import importlib
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


def _import_installed_robustbench_load_model():
    """Import RobustBench from the active Python environment, not vendored baseline copies."""
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    original_sys_path = list(sys.path)
    removed_modules = {}

    def _is_vendored_robustbench_path(path: str) -> bool:
        if not path:
            return False
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(workspace_root):
            return False
        if '/baselines/' not in abs_path.replace('\\', '/'):
            return False
        return os.path.isdir(os.path.join(abs_path, 'robustbench'))

    try:
        sys.path[:] = [path for path in sys.path if not _is_vendored_robustbench_path(path)]

        for module_name, module in list(sys.modules.items()):
            if not (module_name == 'robustbench' or module_name.startswith('robustbench.')):
                continue
            module_file = getattr(module, '__file__', None)
            if module_file and _is_vendored_robustbench_path(os.path.dirname(module_file)):
                removed_modules[module_name] = module
                sys.modules.pop(module_name, None)

        return importlib.import_module('robustbench.utils').load_model
    finally:
        sys.path[:] = original_sys_path
        for module_name, module in removed_modules.items():
            sys.modules.setdefault(module_name, module)


def get_model(
    name: str,
    pretrained: bool = True,
    num_classes: int = 1000,
    dataset: str = 'cifar10c'
) -> nn.Module:
    """
    Get a pre-trained model

    Args:
        name: Model name (resnet50, vit_base_patch16_224, wideresnet28, etc.)
        pretrained: Whether to load pre-trained weights
        num_classes: Number of output classes
        dataset: Dataset name to determine which pre-trained weights to load

    Returns:
        PyTorch model
    """
    name = name.lower()

    if name == 'resnet50':
        # Standard ResNet-50 with BatchNorm (for RoTTA compatibility)
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            print(f"Loaded BatchNorm ResNet-50 from torchvision (for RoTTA)")
        else:
            model = models.resnet50(num_classes=num_classes)

    elif name == 'resnet50_gn':
        # GroupNorm ResNet-50 from timm - required for TTA methods on ImageNet-C
        # SAR paper shows that BatchNorm ResNet-50 causes TTA methods to collapse
        try:
            import timm
            # Try different model names for different timm versions
            try:
                model = timm.create_model('resnet50_gn.a1h_in1k', pretrained=pretrained, num_classes=num_classes)
            except RuntimeError:
                model = timm.create_model('resnet50_gn', pretrained=pretrained, num_classes=num_classes)
            print(f"Loaded GroupNorm ResNet-50 from timm")
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

    elif name == 'resnet101':
        if pretrained:
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet101(num_classes=num_classes)

    elif name.startswith('vit'):
        try:
            import timm
            model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

    elif 'wideresnet' in name or 'wrn' in name or 'resnext29' in name or 'resnext' in name:
        try:
            rb_load = _import_installed_robustbench_load_model()
            # Determine RobustBench dataset based on input dataset
            if 'cifar100' in dataset.lower():
                # CIFAR-100-C: fixed to ResNeXt-29 via RobustBench.
                rb_dataset = 'cifar100'
                model_name = 'Hendrycks2020AugMix_ResNeXt'
            else:
                # CIFAR-10-C: fixed to WideResNet-28-10 via RobustBench.
                rb_dataset = 'cifar10'
                model_name = 'Standard'

            model = rb_load(
                model_name=model_name,
                dataset=rb_dataset,
                threat_model='corruptions'
            )
            # Attach provenance for logging / result bookkeeping
            setattr(model, 'rb_model_name', model_name)
            setattr(model, 'rb_dataset', rb_dataset)
            print(f"Loaded RobustBench model: {model_name} for {rb_dataset}")
        except ImportError as exc:
            raise ImportError(
                "Failed to import installed robustbench. If the sata environment already has it, "
                "this usually means a vendored baseline copy shadowed site-packages. "
                "Please verify `python -c \"import robustbench; print(robustbench.__file__)\"` points to the environment package."
            ) from exc

    else:
        raise ValueError(f"Unknown model: {name}")

    return model


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about a model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count BN parameters
    bn_params = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_params += sum(p.numel() for p in module.parameters())

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'bn_params': bn_params,
    }
