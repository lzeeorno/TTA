#!/usr/bin/env python3
"""Check the six Atlas ViT targets used by the Table 3 optimization loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_mean(path: Path) -> float:
    return float(read_json(path)["mean_accuracy"])


def read_corruption(path: Path, corruption: str) -> float:
    payload = read_json(path)
    return float(payload["per_corruption"][corruption]["accuracy"])


def main() -> int:
    imagenetc_partial = (
        PROJECT_ROOT
        / "results"
        / "imagenetc_vit_base_patch16_224_continual"
        / "atlas_seed1997_order75.json"
    )
    checks: Tuple[Tuple[str, float, Path, Callable[[Path], float]], ...] = (
        (
            "ImageNet-A",
            19.0,
            PROJECT_ROOT / "results" / "imagenet_a_vit_base_patch16_224" / "atlas_seed1997.json",
            read_mean,
        ),
        (
            "ImageNet-R",
            45.0,
            PROJECT_ROOT / "results" / "imagenet_r_vit_base_patch16_224" / "atlas_seed1997.json",
            read_mean,
        ),
        (
            "ImageNet-V2",
            69.99,
            PROJECT_ROOT / "results" / "imagenet_v2_vit_base_patch16_224" / "atlas_seed1997.json",
            read_mean,
        ),
        (
            "ImageNet-Sketch",
            42.0,
            PROJECT_ROOT / "results" / "imagenet_sketch_vit_base_patch16_224" / "atlas_seed1997.json",
            read_mean,
        ),
        (
            "ImageNet-C contrast",
            68.33600159133911,
            imagenetc_partial,
            lambda path: read_corruption(path, "contrast"),
        ),
        (
            "ImageNet-C defocus_blur",
            58.96200134590149,
            imagenetc_partial,
            lambda path: read_corruption(path, "defocus_blur"),
        ),
    )

    print("Atlas ViT six-target check")
    all_passed = True
    for name, target, path, reader in checks:
        try:
            value = reader(path)
        except Exception as exc:
            print(f"[FAIL] {name}: missing/unreadable ({path}) error={exc}")
            all_passed = False
            continue
        passed = value >= target
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {value:.6f} target={target:.6f} path={path}")
        all_passed = all_passed and passed

    if all_passed:
        print("All six Atlas ViT targets passed.")
        return 0
    print("At least one Atlas ViT target failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
