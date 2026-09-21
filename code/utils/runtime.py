"""Small runtime helpers used by the public TTA entry point.

The historical research workspace provided these helpers through an internal
package that also contained comparison methods.  Keeping the utility subset
here lets the public release run its own method without redistributing those
comparison implementations.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_results(results: Dict, save_dir: str, filename: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Results saved to {filepath}")


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        values = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            values.append(correct_k.mul_(100.0 / batch_size))
        return values


IMAGENET_C_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "shot_noise", "snow",
    "zoom_blur",
]
CIFAR_C_CORRUPTIONS = list(IMAGENET_C_CORRUPTIONS)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
