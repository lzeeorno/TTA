"""Public entry point for the released test-time adaptation method.

The public runner intentionally contains only the released method and a
source-model evaluation path. Comparison implementations are external
dependencies documented in ``THIRD_PARTY_PROVENANCE.md``.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from atlas import create_atlas
from atlas.common import looks_like_vit_family
from datasets import (
    CIFAR_C_Dataset,
    ImageNetC_Dataset,
    build_cifar_transform,
    get_corruption_loader,
    get_label_shift_indices,
    get_natural_shift_loader,
)
from models import get_model
from utils.runtime import CIFAR_C_CORRUPTIONS, IMAGENET_C_CORRUPTIONS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_results(results: Dict[str, Any], directory: str, filename: str) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Results saved to {path}")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base_path = config.get("_BASE_")
    if not base_path:
        return config
    parent = load_config(os.path.join(os.path.dirname(config_path), base_path))

    def merge(base: dict, override: dict) -> None:
        for key, value in override.items():
            if key == "_BASE_":
                continue
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value

    merge(parent, config)
    return parent


def infer_adapt_type(backbone_name: str, model: nn.Module, config: dict) -> str:
    requested = config.get("model", {}).get("adapt_params")
    counts = {"bn": 0, "gn": 0, "ln": 0}
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            counts["bn"] += 1
        elif isinstance(module, nn.GroupNorm):
            counts["gn"] += 1
        elif isinstance(module, nn.LayerNorm):
            counts["ln"] += 1
    if requested in counts and counts[requested] > 0:
        return requested
    name = backbone_name.lower()
    if "gn" in name or "groupnorm" in name or (counts["gn"] and not counts["bn"]):
        return "gn"
    if looks_like_vit_family(model) or any(tag in name for tag in ("vit", "swin", "deit", "beit")):
        return "ln"
    if counts["bn"]:
        return "bn"
    if counts["ln"]:
        return "ln"
    return "bn"


def parse_views(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    aliases = {"weak": "photometric"}
    values: List[str] = []
    for item in raw.split(","):
        item = aliases.get(item.strip().lower(), item.strip().lower())
        if item and item not in values:
            values.append(item)
    if not values:
        raise ValueError("--atlas-active-views cannot be empty")
    return values


def parse_weights(raw: Optional[str]) -> Optional[dict]:
    if raw is None:
        return None
    values = [float(item.strip()) for item in raw.split(",")]
    if len(values) != 4 or any(not math.isfinite(item) or item < 0 for item in values):
        raise ValueError("--atlas-reward-weights requires four non-negative numbers")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("--atlas-reward-weights must sum to one")
    return dict(zip(("target", "entropy", "source", "consensus"), values))


def atlas_overrides(args: argparse.Namespace) -> dict:
    nested: Dict[str, dict] = {}

    def put(section: str, key: str, value: Any) -> None:
        if value is not None:
            nested.setdefault(section, {})[key] = value

    put("selection", "quantile", args.atlas_selection_quantile)
    put("dig", "std_threshold", args.atlas_dig_std_threshold)
    put("sos", "safe_margin", args.atlas_sos_safe_margin)
    put("sos", "hard_margin", args.atlas_sos_hard_margin)
    put("arc", "eps_low", args.atlas_arc_eps_low)
    put("arc", "eps_high", args.atlas_arc_eps_high)
    put("vit_ln", "pi", args.atlas_vit_ln_pi)
    put("vit_ln", "entropy_margin_scale", args.atlas_vit_ln_entropy_margin_scale)
    put("vit_ln", "plpd_threshold", args.atlas_vit_ln_plpd_threshold)
    put("vit_ln", "patch_len", args.atlas_vit_ln_patch_len)
    put("vit_ln", "probe_policy", args.atlas_probe_policy)
    if args.atlas_shared_role_only:
        put("vit_ln", "shared_role_only", True)
    weights = parse_weights(args.atlas_reward_weights)
    if weights is not None:
        nested["reward"] = {"weights": weights, "aggregation": "legacy_weighted"}
    return nested


def configure_method(model: nn.Module, config: dict, args: argparse.Namespace, dataset_name: str) -> nn.Module:
    if args.method == "source":
        model.eval()
        return model

    atlas_cfg = deepcopy(config.get("atlas", {}))
    backbone_name = config["model"]["backbone"]
    adapt_type = infer_adapt_type(backbone_name, model, config)
    num_classes = 100 if "cifar100" in dataset_name.lower() else 10 if "cifar10" in dataset_name.lower() else 1000
    tta_cfg = config.get("tta", {})
    variant = args.atlas_method_variant or atlas_cfg.get("method_variant", "full")
    is_vit_ln = adapt_type == "ln" and looks_like_vit_family(model)
    vit_cfg = atlas_cfg.get("vit_ln", {})
    optim_cfg = config.get("optim", {})
    use_vit_schedule = is_vit_ln and variant == "full"
    atlas_cfg.update({
        "adapt_type": adapt_type,
        "adaptation_mode": "standard" if tta_cfg.get("reset_each_corruption", False) else "continual",
        "scenario": args.scenario or tta_cfg.get("scenario", "normal"),
        "lr": float(vit_cfg.get("lr", 0.05) if use_vit_schedule else optim_cfg.get("lr", 2.5e-4)),
        "optimizer_name": str(vit_cfg.get("optimizer", "sgd") if use_vit_schedule else optim_cfg.get("optimizer", "sgd")),
        "optim_momentum": float(vit_cfg.get("momentum", 0.0) if use_vit_schedule else optim_cfg.get("momentum", 0.9)),
        "optim_weight_decay": float(vit_cfg.get("weight_decay", 0.0) if use_vit_schedule else optim_cfg.get("weight_decay", 0.0)),
        "num_classes": num_classes,
        "method_variant": variant,
        "no_arc": args.no_arc,
        "no_dig": args.no_dig,
        "no_uan": args.no_uan,
        "no_sos": args.no_sos,
        "ablation_row": args.atlas_ablation_row,
    })
    for section, values in atlas_overrides(args).items():
        atlas_cfg[section] = dict(atlas_cfg.get(section, {}))
        atlas_cfg[section].update(values)
    views = parse_views(args.atlas_active_views)
    if views is not None:
        atlas_cfg["views"] = dict(atlas_cfg.get("views", {}))
        atlas_cfg["views"]["active"] = views
    adapted = create_atlas(model, config=atlas_cfg)
    print(
        f"Method [{adapted.method_variant}]: adapting {len(adapted.params)} parameters; "
        f"ARC={adapted.arc_enabled}, DIG={adapted.dig_enabled}, "
        f"UAN={adapted.uan_mode}, SOS={adapted.sos_enabled}"
    )
    return adapted


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int) -> None:
        self.total += value * count
        self.count += count

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


def evaluate(model: nn.Module, loader: Iterable, device: torch.device, method: str, max_batches: int) -> dict:
    if method == "source":
        model.eval()
    meter = AverageMeter()
    classes = set()
    batches = 0
    samples = 0
    progress = tqdm(loader, leave=False, desc="batches", ncols=80)
    for images, labels in progress:
        if max_batches and batches >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        context = torch.no_grad() if method == "source" else torch.enable_grad()
        with context:
            outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correct = float((predictions == labels).sum().item())
        meter.update(correct / max(1, images.size(0)) * 100.0, int(images.size(0)))
        classes.update(int(label) for label in labels.detach().cpu().tolist())
        batches += 1
        samples += int(images.size(0))
    result = {
        "accuracy": meter.avg,
        "evaluated_batches": batches,
        "evaluated_samples": samples,
        "evaluated_class_count": len(classes),
    }
    if hasattr(model, "get_stats"):
        result.update(model.get_stats())
    return result


def batch_size(config: dict, args: argparse.Namespace) -> int:
    if args.batch_size is not None:
        return args.batch_size
    return int(config.get("dataset", {}).get("batch_size", 64))


def save_summary(config: dict, args: argparse.Namespace, summary: dict, suffix: str = "") -> None:
    if not config.get("logging", {}).get("save_results", True):
        return
    root = args.results_dir or config.get("logging", {}).get("results_dir", "./results")
    directory = os.path.join(root, f"{summary['dataset']}_{config['model']['backbone']}{suffix}")
    order = f"_order{args.order_idx}" if args.corruption_order else ""
    save_results(summary, directory, f"{summary['method']}_seed{args.seed}{order}.json")


def run_natural(model: nn.Module, config: dict, args: argparse.Namespace, device: torch.device, dataset_name: str) -> dict:
    loader = get_natural_shift_loader(
        dataset_name=dataset_name,
        data_root=config["dataset"]["data_root"],
        batch_size=batch_size(config, args),
        num_workers=config["dataset"].get("num_workers", 4),
        shuffle=config["dataset"].get("shuffle", False),
        seed=args.seed,
    )
    result = evaluate(model, loader, device, args.method, args.max_batches)
    summary = {
        "method": args.method,
        "base_method": args.method,
        "backbone": config["model"]["backbone"],
        "dataset": dataset_name,
        "seed": args.seed,
        "scenario": "natural_shift",
        "max_batches": args.max_batches,
        "mean_accuracy": result["accuracy"],
        "per_corruption": {dataset_name: result},
    }
    save_summary(config, args, summary)
    return summary


def run_corruptions(model: nn.Module, config: dict, args: argparse.Namespace, device: torch.device, dataset_name: str) -> dict:
    corruptions = list(IMAGENET_C_CORRUPTIONS if "imagenet" in dataset_name.lower() else CIFAR_C_CORRUPTIONS)
    configured = config["dataset"].get("corruption_types", "all")
    if configured != "all":
        corruptions = list(configured)
    if args.corruption_order:
        corruptions = [item.strip() for item in args.corruption_order.split(",") if item.strip()]
    severity = int(config["dataset"].get("severity", 5))
    scenario = args.scenario or config.get("tta", {}).get("scenario", "normal")
    loader_kwargs = {
        "dataset_name": dataset_name,
        "severity": severity,
        "data_root": config["dataset"]["data_root"],
        "num_workers": config["dataset"].get("num_workers", 4),
        "preprocess": config["dataset"].get("preprocess", "cifar_default"),
        "input_size": config["dataset"].get("input_size"),
    }
    results: Dict[str, dict] = {}

    if scenario == "mix_shifts":
        from torch.utils.data import ConcatDataset, DataLoader

        datasets = []
        for corruption in corruptions:
            if "cifar" in dataset_name.lower():
                datasets.append(CIFAR_C_Dataset(
                    data_root=loader_kwargs["data_root"], corruption=corruption,
                    severity=severity,
                    transform=build_cifar_transform(loader_kwargs["preprocess"], loader_kwargs["input_size"]),
                ))
            else:
                datasets.append(ImageNetC_Dataset(
                    data_root=loader_kwargs["data_root"], corruption=corruption, severity=severity
                ))
        loader = DataLoader(
            ConcatDataset(datasets), batch_size=batch_size(config, args), shuffle=True,
            num_workers=loader_kwargs["num_workers"], pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        results["mix_shifts"] = evaluate(model, loader, device, args.method, args.max_batches)
    else:
        indices = None
        if scenario == "label_shifts":
            indices = get_label_shift_indices(
                dataset_name=dataset_name, imbalance_ratio=args.imbalance_ratio,
                seed=args.seed,
                cache_dir=os.path.join(args.results_dir or "results", "_label_shift_indices"),
            )
        for corruption in corruptions:
            current_batch_size = 1 if scenario == "bs1" else batch_size(config, args)
            loader = get_corruption_loader(
                **loader_kwargs, corruption=corruption, batch_size=current_batch_size,
                shuffle=(scenario == "bs1") or bool(config["dataset"].get("shuffle", False)),
                seed=args.seed, subset_indices=indices,
            )
            if config.get("tta", {}).get("reset_each_corruption", False) and hasattr(model, "reset"):
                model.reset()
            if hasattr(model, "set_current_corruption"):
                model.set_current_corruption(corruption)
            results[corruption] = evaluate(model, loader, device, args.method, args.max_batches)

    accuracies = [float(item["accuracy"]) for item in results.values()]
    suffix = "_continual" if not config.get("tta", {}).get("reset_each_corruption", False) else ""
    if scenario != "normal":
        suffix += f"_{scenario}"
    summary = {
        "method": args.method,
        "base_method": args.method,
        "backbone": config["model"]["backbone"],
        "dataset": dataset_name,
        "severity": severity,
        "seed": args.seed,
        "scenario": scenario,
        "max_batches": args.max_batches,
        "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        "per_corruption": results,
    }
    if args.corruption_order:
        summary["corruption_order"] = corruptions
        summary["order_idx"] = args.order_idx
    save_summary(config, args, summary, suffix=suffix)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test-time adaptation runner")
    parser.add_argument("--config", "--configs", default="configs/default.yaml")
    parser.add_argument("--method", choices=["source", "atlas"], default="atlas")
    parser.add_argument("--backbone")
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--results-dir")
    parser.add_argument("--scenario", choices=["normal", "label_shifts", "mix_shifts", "bs1"])
    parser.add_argument("--imbalance-ratio", type=float, default=500000)
    parser.add_argument("--corruption-order")
    parser.add_argument("--order-idx", type=int, default=0)
    parser.add_argument("--atlas-method-variant", choices=["core", "full"])
    parser.add_argument("--atlas-selection-quantile", type=float)
    parser.add_argument("--no_arc", action="store_true")
    parser.add_argument("--no_dig", action="store_true")
    parser.add_argument("--no_uan", action="store_true")
    parser.add_argument("--no_sos", action="store_true")
    parser.add_argument("--atlas-dig-std-threshold", type=float)
    parser.add_argument("--atlas-sos-safe-margin", type=float)
    parser.add_argument("--atlas-sos-hard-margin", type=float)
    parser.add_argument("--atlas-arc-eps-low", type=float)
    parser.add_argument("--atlas-arc-eps-high", type=float)
    parser.add_argument("--atlas-vit-ln-pi", type=float)
    parser.add_argument("--atlas-vit-ln-entropy-margin-scale", type=float)
    parser.add_argument("--atlas-vit-ln-plpd-threshold", type=float)
    parser.add_argument("--atlas-vit-ln-patch-len", type=int)
    parser.add_argument("--atlas-probe-policy", choices=["always", "alternate", "off"])
    parser.add_argument("--atlas-shared-role-only", action="store_true")
    parser.add_argument("--atlas-active-views")
    parser.add_argument("--atlas-reward-weights")
    parser.add_argument("--atlas-ablation-row", choices=["A0", "A1", "A2", "B1", "B2"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("dataset", {})
    config.setdefault("model", {})
    config.setdefault("tta", {})
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.scenario:
        config["tta"]["scenario"] = args.scenario
    if args.results_dir:
        config.setdefault("logging", {})["results_dir"] = args.results_dir

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset_name = config["dataset"]["name"]
    model = get_model(
        config["model"]["backbone"],
        pretrained=config["model"].get("pretrained", True),
        dataset=dataset_name,
    ).to(device)
    model = configure_method(model, config, args, dataset_name)

    natural = config["dataset"].get("dataset_type") == "natural_shift"
    natural = natural or dataset_name.lower().replace("-", "_") in {
        "imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_sketch"
    }
    if natural:
        summary = run_natural(model, config, args, device, dataset_name)
    else:
        summary = run_corruptions(model, config, args, device, dataset_name)
    print(f"Mean accuracy: {summary['mean_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
