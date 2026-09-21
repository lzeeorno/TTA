"""
Utility functions for TRIAD (TRi-space Invariant ADaptation)
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from typing import Dict, List, Optional
from datetime import datetime


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_results(
    results: Dict,
    save_dir: str,
    filename: str
):
    """Save results to JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# ImageNet-C corruption types (standard 15 corruptions)
# NOTE: Full dataset has 19 corruptions, but standard benchmark uses these 15
# (same as CIFAR-C to ensure fair comparison)
# Standard 15: noise(3) + blur(4) + weather(4) + digital(4)
IMAGENET_C_CORRUPTIONS = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_noise', 'glass_blur',
    'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
    'shot_noise', 'snow', 'zoom_blur'
]

# CIFAR-C corruption types (standard 15 corruptions)
# NOTE: Full dataset has 19 corruptions, but standard benchmark uses these 15
CIFAR_C_CORRUPTIONS = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_noise', 'glass_blur',
    'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
    'shot_noise', 'snow', 'zoom_blur'
]


def compute_mce(errors: Dict[str, float], baseline_errors: Dict[str, float]) -> float:
    """
    Compute mean Corruption Error (mCE)

    mCE = (1/n) * Σ (error_method / error_baseline)

    Args:
        errors: Dictionary of corruption_type -> error
        baseline_errors: Dictionary of corruption_type -> baseline error

    Returns:
        mCE value
    """
    ce_values = []
    for corruption in errors:
        if corruption in baseline_errors:
            ce = errors[corruption] / baseline_errors[corruption]
            ce_values.append(ce)

    return np.mean(ce_values) * 100  # as percentage


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
