#!/usr/bin/env python3
"""
Table-5 semantic segmentation benchmark runner:
Cityscapes-pretrained SegFormer-B5 -> ACDC continual adaptation.

Implemented methods:
- source: no adaptation
- tent: entropy minimization on LayerNorm affine params
- atlas: ATLAS four-component TTA on LayerNorm affine params

This script is intentionally lightweight and self-contained for reproducible runs.
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from atlas import ATLASSegmentationAdapter


CONDITIONS = ["fog", "night", "rain", "snow"]
IGNORE_INDEX = 255
NUM_CLASSES = 19
PROTOCOL = "table5_surgeon_cotta_v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CITYSCAPES_COLORS = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Sample:
    image_path: str
    label_path: str
    condition: str
    sample_id: str = ""
    round_index: int = 0


class ACDCDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        input_width: int = 960,
        input_height: int = 540,
        normalize: bool = True,
    ):
        self.samples = samples
        self.input_width = input_width
        self.input_height = input_height
        self.normalize = normalize
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        label = Image.open(s.label_path)

        image = image.resize((self.input_width, self.input_height), Image.BILINEAR)
        label = label.resize((self.input_width, self.input_height), Image.NEAREST)

        image_vis = np.asarray(image, dtype=np.uint8).copy()
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = image_np.transpose(2, 0, 1)
        if self.normalize:
            image_np = (image_np - self.mean) / self.std
        label_np = np.asarray(label, dtype=np.int64)

        # Convert invalid IDs to ignore index.
        label_np[(label_np < 0) | (label_np >= NUM_CLASSES)] = IGNORE_INDEX

        return {
            "image": torch.from_numpy(image_np),
            "image_vis": torch.from_numpy(image_vis),
            "label": torch.from_numpy(label_np),
            "condition": s.condition,
            "sample_id": s.sample_id,
            "round_index": s.round_index,
        }


def collect_samples(acdc_root: str, split: str = "val") -> List[Sample]:
    rgb_root = os.path.join(acdc_root, "rgb_anon_trainvaltest", "rgb_anon")
    gt_root = os.path.join(acdc_root, "gt_trainval", "gt")

    samples: List[Sample] = []
    for condition in CONDITIONS:
        rgb_cond_root = os.path.join(rgb_root, condition, split)
        gt_cond_root = os.path.join(gt_root, condition, split)
        if not os.path.isdir(rgb_cond_root) or not os.path.isdir(gt_cond_root):
            continue

        for root, dirs, files in os.walk(rgb_cond_root):
            dirs.sort()
            files.sort()
            for fn in files:
                if not fn.endswith("_rgb_anon.png"):
                    continue
                rel_dir = os.path.relpath(root, rgb_cond_root)
                stem = fn.replace("_rgb_anon.png", "")
                label_name = f"{stem}_gt_labelTrainIds.png"
                label_path = os.path.join(gt_cond_root, rel_dir, label_name)
                image_path = os.path.join(root, fn)
                if os.path.isfile(label_path):
                    if rel_dir == ".":
                        sample_id = f"{condition}/{stem}"
                    else:
                        sample_id = os.path.join(condition, rel_dir, stem).replace(os.sep, "/")
                    samples.append(
                        Sample(
                            image_path=image_path,
                            label_path=label_path,
                            condition=condition,
                            sample_id=sample_id,
                        )
                    )

    return samples


def entropy_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    ent = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)
    return ent.mean()


def compute_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    mask = target != ignore_index
    pred = pred[mask].view(-1)
    target = target[mask].view(-1)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=pred.device)
    if pred.numel() == 0:
        return cm
    idx = target * num_classes + pred
    cm += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def miou_from_confusion_matrix(cm: torch.Tensor) -> float:
    cm = cm.float()
    tp = torch.diag(cm)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = tp / denom.clamp_min(1.0)
    valid = denom > 0
    if valid.sum() == 0:
        return 0.0
    return (iou[valid].mean().item() * 100.0)


def miou_or_none(cm: torch.Tensor) -> Optional[float]:
    if int(cm.sum().item()) == 0:
        return None
    return miou_from_confusion_matrix(cm)


def mean_present(values: List[Optional[float]]) -> Optional[float]:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return float(np.mean(present))


def error_from_miou(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return 100.0 - float(value)


def parse_timestamps(raw: str, rounds: int, ensure_round: bool = True) -> List[int]:
    timestamps: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        timestamp = int(part)
        if timestamp < 1:
            raise ValueError(f"Report timestamps must be >= 1, got {timestamp}")
        if timestamp <= rounds and timestamp not in timestamps:
            timestamps.append(timestamp)
    if ensure_round and rounds not in timestamps:
        timestamps.append(rounds)
    return sorted(timestamps)


def sample_visualization_ids(
    sequence: List[Sample],
    timestamps: List[int],
    count: int,
    seed: int,
) -> List[str]:
    if count <= 0 or not timestamps:
        return []

    rounds_by_id: Dict[str, Set[int]] = {}
    for sample in sequence:
        rounds_by_id.setdefault(sample.sample_id, set()).add(int(sample.round_index))

    candidate_ids = sorted(
        sample_id
        for sample_id, round_ids in rounds_by_id.items()
        if all(timestamp in round_ids for timestamp in timestamps)
    )
    if not candidate_ids:
        return []
    if count >= len(candidate_ids):
        return candidate_ids

    rng = random.Random(seed)
    return sorted(rng.sample(candidate_ids, count))


def slugify_sample_id(sample_id: str) -> str:
    safe = sample_id.replace("/", "__")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in safe)


def colorize_prediction(prediction: np.ndarray) -> np.ndarray:
    color = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for class_idx, rgb in enumerate(CITYSCAPES_COLORS):
        color[prediction == class_idx] = rgb
    return color


def colorize_entropy(entropy: np.ndarray) -> np.ndarray:
    entropy_norm = np.clip(entropy / np.log(NUM_CLASSES), 0.0, 1.0)
    finite_entropy = entropy_norm[np.isfinite(entropy_norm)]
    if finite_entropy.size == 0:
        scaled = np.zeros_like(entropy_norm, dtype=np.float32)
    else:
        vmin = float(np.percentile(finite_entropy, 1.0))
        vmax = float(np.percentile(finite_entropy, 99.0))
        if vmax <= vmin + 1e-8:
            vmin = float(finite_entropy.min())
            vmax = float(finite_entropy.max())
        if vmax <= vmin + 1e-8:
            scaled = np.zeros_like(entropy_norm, dtype=np.float32)
        else:
            scaled = np.clip((entropy_norm - vmin) / (vmax - vmin), 0.0, 1.0)
            scaled = np.power(scaled, 0.85, dtype=np.float32)

    entropy_rgb = colormaps["plasma"](scaled)[..., :3]
    return np.asarray(np.round(entropy_rgb * 255.0), dtype=np.uint8)


def render_triptych(
    original_image: np.ndarray,
    prediction_image: np.ndarray,
    entropy_image: np.ndarray,
    title: str,
) -> Image.Image:
    panels = [
        Image.fromarray(original_image),
        Image.fromarray(prediction_image),
        Image.fromarray(entropy_image),
    ]
    labels = ["Original Image", "Pixel Prediction", "Pixel Entropy"]
    panel_width, panel_height = panels[0].size
    gap = 20
    title_height = 20
    label_height = 18
    canvas_width = panel_width * len(panels) + gap * (len(panels) + 1)
    canvas_height = title_height + label_height + panel_height + gap * 2

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((gap, 6), title, fill=(242, 242, 242), font=font)
    for idx, (panel, label) in enumerate(zip(panels, labels)):
        offset_x = gap + idx * (panel_width + gap)
        offset_y = title_height + label_height
        draw.text((offset_x, title_height), label, fill=(220, 220, 220), font=font)
        canvas.paste(panel, (offset_x, offset_y))

    return canvas


def metrics_by_condition(cm_by_cond: Dict[str, torch.Tensor]) -> Dict[str, Optional[float]]:
    metrics = {condition: miou_or_none(cm_by_cond[condition]) for condition in CONDITIONS}
    metrics["mean"] = mean_present([metrics[condition] for condition in CONDITIONS])
    return metrics


def online_error_by_condition(miou: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    errors = {condition: error_from_miou(miou[condition]) for condition in CONDITIONS}
    errors["mean"] = mean_present([errors[condition] for condition in CONDITIONS])
    return errors


def configure_tent(model: SegformerForSemanticSegmentation) -> Tuple[List[torch.nn.Parameter], List[str]]:
    model.train()
    for p in model.parameters():
        p.requires_grad = False

    params, names = [], []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.LayerNorm):
            if m.weight is not None:
                m.weight.requires_grad = True
                params.append(m.weight)
                names.append(f"{n}.weight")
            if m.bias is not None:
                m.bias.requires_grad = True
                params.append(m.bias)
                names.append(f"{n}.bias")
    return params, names


def run(args: argparse.Namespace) -> Dict:
    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    samples = collect_samples(args.acdc_root, split=args.split)
    if not samples:
        raise RuntimeError(f"No ACDC samples found in {args.acdc_root}")

    if args.image_size is not None:
        input_width = int(args.image_size)
        input_height = int(args.image_size)
        input_shape_policy = "legacy_square_image_size"
    else:
        input_width = int(args.input_width)
        input_height = int(args.input_height)
        input_shape_policy = "surgeon_cotta_rectangular"
    report_timestamps = parse_timestamps(args.report_timestamps, args.rounds, ensure_round=True)

    # Continual order: fog -> night -> rain -> snow, repeated rounds.
    grouped = {}
    for condition in CONDITIONS:
        grouped[condition] = sorted(
            [s for s in samples if s.condition == condition],
            key=lambda sample: (sample.image_path, sample.label_path),
        )
        if args.max_samples_per_condition is not None and args.max_samples_per_condition > 0:
            grouped[condition] = grouped[condition][: args.max_samples_per_condition]
    sequence: List[Sample] = []
    for round_index in range(1, args.rounds + 1):
        for c in CONDITIONS:
            sequence.extend(
                Sample(
                    image_path=s.image_path,
                    label_path=s.label_path,
                    condition=s.condition,
                    sample_id=s.sample_id,
                    round_index=round_index,
                )
                for s in grouped[c]
            )

    if args.max_samples is not None and args.max_samples > 0:
        sequence = sequence[: args.max_samples]

    visualize_timestamps = (
        parse_timestamps(args.visualize_timestamps, args.rounds, ensure_round=False)
        if args.visualize else []
    )
    visualize_seed = args.seed if args.visualize_seed is None else args.visualize_seed
    selected_visual_ids = sample_visualization_ids(
        sequence=sequence,
        timestamps=visualize_timestamps,
        count=args.visualize_count,
        seed=visualize_seed,
    )
    selected_visual_id_set = set(selected_visual_ids)
    selected_visual_rank = {sample_id: idx + 1 for idx, sample_id in enumerate(selected_visual_ids)}
    visualize_root = None
    if args.visualize and selected_visual_ids:
        visualize_root = args.visualize_dir or os.path.join(
            args.output_dir,
            "visualizations",
            args.method,
            f"seed{args.seed}",
        )
        os.makedirs(visualize_root, exist_ok=True)

    dataset = ACDCDataset(
        sequence,
        input_width=input_width,
        input_height=input_height,
        normalize=not args.no_normalize,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_cuda,
    )

    samples_per_condition = {condition: len(grouped[condition]) for condition in CONDITIONS}
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "legacy_protocol": "table5_unified_repo_v1",
                "acdc_root": args.acdc_root,
                "split": args.split,
                "model_name_or_path": args.model_name_or_path,
                "device": str(device),
                "samples_per_condition": samples_per_condition,
                "rounds": args.rounds,
                "report_timestamps": report_timestamps,
                "visualize": args.visualize,
                "visualize_timestamps": visualize_timestamps,
                "visualize_count": len(selected_visual_ids),
                "input_width": input_width,
                "input_height": input_height,
                "normalization": "imagenet" if not args.no_normalize else "none",
                "total_sequence_length": len(sequence),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    model = SegformerForSemanticSegmentation.from_pretrained(args.model_name_or_path)
    model.to(device)

    optimizer = None
    atlas_adapter = None
    result_method = args.method
    if args.result_suffix:
        result_method = f"{result_method}_{args.result_suffix}"
    if args.method == "tent":
        params, _ = configure_tent(model)
        if len(params) == 0:
            raise RuntimeError("No trainable LayerNorm params found for TENT.")
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.method == "atlas":
        atlas_vit_ln_config = {
            "pi": args.atlas_vit_ln_pi,
            "entropy_margin_scale": args.atlas_vit_ln_entropy_margin_scale,
            "plpd_threshold": args.atlas_vit_ln_plpd_threshold,
            "patch_len": args.atlas_vit_ln_patch_len,
            "probe_policy": args.atlas_probe_policy,
            "shared_role_only": args.atlas_shared_role_only,
            "target_selected_ratio": args.atlas_target_selected_ratio,
            "selection_quantile": args.atlas_selection_quantile,
            "source_conf_threshold": args.atlas_source_conf_threshold,
            "source_conf_tolerance": args.atlas_source_conf_tolerance,
            "source_anchor_weight": args.atlas_source_anchor_weight,
            "predict_after_update": not args.atlas_no_predict_after_update,
            "output_hflip": not args.atlas_no_output_hflip,
            "output_hflip_weight": args.atlas_output_hflip_weight,
            "output_scale": args.atlas_output_scale,
            "output_scale_weight": args.atlas_output_scale_weight,
            "adapt_decode_bn": not args.atlas_no_decode_bn,
        }
        atlas_adapter = ATLASSegmentationAdapter(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            no_arc=args.no_arc,
            no_dig=args.no_dig,
            no_uan=args.no_uan,
            no_sos=args.no_sos,
            sane_normalizer=args.sane_normalizer,
            optimizer_name=args.atlas_optimizer,
            optim_momentum=args.atlas_momentum,
            method_variant=args.atlas_method_variant,
            vit_ln_config=atlas_vit_ln_config,
        )
    else:
        model.eval()

    cm_by_cond = {c: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device) for c in CONDITIONS}
    cm_by_timestamp = {
        t: {c: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device) for c in CONDITIONS}
        for t in report_timestamps
    }
    count_by_timestamp = {
        t: {c: 0 for c in CONDITIONS}
        for t in report_timestamps
    }
    generated_visualizations: List[Dict[str, object]] = []
    saved_visualization_keys: Set[Tuple[str, int]] = set()
    visualize_timestamp_set = set(visualize_timestamps)

    with tqdm(loader, desc=f"{args.method.upper()} ACDC", ncols=100) as pbar:
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            conds = batch["condition"]
            sample_ids = batch["sample_id"]
            round_indices = batch["round_index"]

            if args.method == "tent":
                model.train()
                optimizer.zero_grad(set_to_none=True)
                out = model(pixel_values=x)
                logits = out.logits
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = entropy_loss_from_logits(logits)
                loss.backward()
                optimizer.step()
            elif args.method == "atlas":
                logits = atlas_adapter.adapt_batch(x)
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            else:
                model.eval()
                with torch.no_grad():
                    out = model(pixel_values=x)
                    logits = out.logits
                    logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

            pred = torch.argmax(logits, dim=1)
            entropy_maps = None
            if selected_visual_id_set:
                probs = F.softmax(logits.detach(), dim=1)
                entropy_maps = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)
            for i in range(pred.shape[0]):
                c = conds[i]
                sample_id = sample_ids[i]
                round_index = int(round_indices[i].item())
                sample_cm = compute_confusion_matrix(pred[i], y[i], NUM_CLASSES, IGNORE_INDEX)
                cm_by_cond[c] += sample_cm
                if round_index in cm_by_timestamp:
                    cm_by_timestamp[round_index][c] += sample_cm
                    count_by_timestamp[round_index][c] += 1

                if (
                    visualize_root is not None
                    and entropy_maps is not None
                    and sample_id in selected_visual_id_set
                    and round_index in visualize_timestamp_set
                    and (sample_id, round_index) not in saved_visualization_keys
                ):
                    timestamp_dir = os.path.join(visualize_root, f"timestamp_{round_index}")
                    os.makedirs(timestamp_dir, exist_ok=True)

                    image_vis = batch["image_vis"][i].cpu().numpy().astype(np.uint8)
                    prediction_vis = colorize_prediction(pred[i].detach().cpu().numpy().astype(np.int64))
                    entropy_vis = colorize_entropy(entropy_maps[i].detach().cpu().numpy().astype(np.float32))
                    title = f"method={args.method} | timestamp={round_index} | sample={sample_id}"
                    triptych = render_triptych(image_vis, prediction_vis, entropy_vis, title)

                    file_name = f"{selected_visual_rank[sample_id]:02d}_{slugify_sample_id(sample_id)}.png"
                    file_path = os.path.join(timestamp_dir, file_name)
                    triptych.save(file_path)
                    generated_visualizations.append(
                        {
                            "sample_id": sample_id,
                            "timestamp": round_index,
                            "path": os.path.relpath(file_path, start=os.getcwd()),
                        }
                    )
                    saved_visualization_keys.add((sample_id, round_index))

    aggregate_miou = metrics_by_condition(cm_by_cond)
    aggregate_error = online_error_by_condition(aggregate_miou)
    miou_by_timestamp = {
        str(t): metrics_by_condition(cm_by_timestamp[t])
        for t in report_timestamps
    }
    online_error_by_timestamp = {
        str(t): online_error_by_condition(miou_by_timestamp[str(t)])
        for t in report_timestamps
    }
    timestamp_errors = [
        online_error_by_timestamp[str(t)][condition]
        for t in report_timestamps
        for condition in CONDITIONS
    ]
    mean_online_error = mean_present(timestamp_errors)
    mean_timestamp_miou = (
        100.0 - mean_online_error
        if mean_online_error is not None else None
    )
    primary_timestamp = str(args.rounds if args.rounds in report_timestamps else report_timestamps[-1])
    primary_miou = miou_by_timestamp.get(primary_timestamp, aggregate_miou)
    primary_error = online_error_by_timestamp.get(primary_timestamp, aggregate_error)
    if primary_miou.get("mean") is None:
        primary_miou = aggregate_miou
        primary_error = aggregate_error
    mean_primary_miou = primary_miou.get("mean")
    mean_primary_online_error = primary_error.get("mean")

    summary = {
        "method": result_method,
        "base_method": args.method,
        "backbone": "segformer-b5",
        "source_domain": "cityscapes_pretrained",
        "target_domain": "acdc",
        "setting": "continual",
        "protocol": PROTOCOL,
        "legacy_protocol": "table5_unified_repo_v1",
        "source_references": [
            "references/SURGEON_Memory-Adaptive_CVPR_2025_paper.pdf",
            "code/baselines/SURGEON-master/README.md",
            "code/baselines/SURGEON-master/test_time.py",
        ],
        "model_name_or_path": args.model_name_or_path,
        "device": str(device),
        "split": args.split,
        "rounds": args.rounds,
        "report_timestamps": report_timestamps,
        "primary_timestamp": int(primary_timestamp),
        "batch_size": args.batch_size,
        "input_width": input_width,
        "input_height": input_height,
        "input_size": [input_width, input_height],
        "image_size": args.image_size,
        "input_shape_policy": input_shape_policy,
        "normalization": {
            "type": "imagenet" if not args.no_normalize else "none",
            "mean": list(IMAGENET_MEAN) if not args.no_normalize else None,
            "std": list(IMAGENET_STD) if not args.no_normalize else None,
        },
        "total_sequence_length": len(sequence),
        "samples_per_condition": samples_per_condition,
        "observed_samples_per_timestamp": count_by_timestamp,
        "optimizer": "Adam" if args.method in {"tent", "atlas"} else "none",
        "lr": args.lr if args.method in {"tent", "atlas"} else None,
        "weight_decay": args.weight_decay if args.method in {"tent", "atlas"} else None,
        "seed": args.seed,
        "result_suffix": args.result_suffix,
        "sane_normalizer": args.sane_normalizer if args.method == "atlas" and not args.no_uan else None,
        "no_uan": args.no_uan if args.method == "atlas" else None,
        "miou": primary_miou,
        "online_error": primary_error,
        "miou_by_timestamp": miou_by_timestamp,
        "online_error_by_timestamp": online_error_by_timestamp,
        "aggregate_miou": aggregate_miou,
        "aggregate_online_error": aggregate_error,
        "mean_miou": mean_primary_miou,
        "mean_online_error": mean_online_error,
        "mean_primary_miou": mean_primary_miou,
        "mean_primary_online_error": mean_primary_online_error,
        "mean_timestamp_miou": mean_timestamp_miou,
        "visualizations": {
            "requested": args.visualize,
            "count": len(selected_visual_ids),
            "timestamps": visualize_timestamps,
            "seed": visualize_seed,
            "output_root": (
                os.path.relpath(visualize_root, start=os.getcwd())
                if visualize_root is not None else None
            ),
            "selected_sample_ids": selected_visual_ids,
            "generated_files": generated_visualizations,
        },
    }
    if atlas_adapter is not None:
        summary.update(atlas_adapter.get_stats())

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{result_method}_seed{args.seed}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_file}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--acdc-root", type=str, default="data/acdc")
    p.add_argument("--model-name-or-path", type=str, default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    p.add_argument("--method", type=str, choices=["source", "tent", "atlas"], required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--input-width", type=int, default=960)
    p.add_argument("--input-height", type=int, default=540)
    p.add_argument("--image-size", type=int, default=None, help="Legacy square input override.")
    p.add_argument("--report-timestamps", type=str, default="1,4,7,10")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-samples-per-condition", type=int, default=None,
                   help="Deterministically cap each ACDC condition before constructing the stream")
    p.add_argument("--seed", type=int, default=1997)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="results/table5_segformer_acdc_surgeon_cotta")
    p.add_argument("--result-suffix", type=str, default=None)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--visualize-count", type=int, default=12)
    p.add_argument("--visualize-timestamps", type=str, default="1,4,10")
    p.add_argument("--visualize-seed", type=int, default=None)
    p.add_argument("--visualize-dir", type=str, default=None)
    p.add_argument("--no_arc", action="store_true")
    p.add_argument("--no_dig", action="store_true")
    p.add_argument("--no_uan", action="store_true")
    p.add_argument("--no_sos", action="store_true")
    p.add_argument("--sane-normalizer", type=str, choices=["selected", "total"], default="selected")
    p.add_argument("--atlas-method-variant", type=str, choices=["core", "full"], default="full")
    p.add_argument("--atlas-selection-quantile", type=float, default=0.5)
    p.add_argument("--atlas-optimizer", type=str, choices=["sgd", "adam"], default="sgd")
    p.add_argument("--atlas-momentum", type=float, default=0.9)
    p.add_argument("--atlas-target-selected-ratio", type=float, default=0.15)
    p.add_argument("--atlas-vit-ln-pi", type=float, default=0.1)
    p.add_argument("--atlas-vit-ln-entropy-margin-scale", type=float, default=0.4)
    p.add_argument("--atlas-vit-ln-plpd-threshold", type=float, default=0.2)
    p.add_argument("--atlas-vit-ln-patch-len", type=int, default=4)
    p.add_argument("--atlas-probe-policy", choices=["always", "alternate", "off"], default="always")
    p.add_argument("--atlas-shared-role-only", action="store_true")
    p.add_argument("--atlas-source-conf-threshold", type=float, default=0.0)
    p.add_argument("--atlas-source-conf-tolerance", type=float, default=0.15)
    p.add_argument("--atlas-source-anchor-weight", type=float, default=0.03)
    p.add_argument("--atlas-no-predict-after-update", action="store_true")
    p.add_argument("--atlas-no-output-hflip", action="store_true")
    p.add_argument("--atlas-output-hflip-weight", type=float, default=1.0)
    p.add_argument("--atlas-output-scale", type=float, default=2.0)
    p.add_argument("--atlas-output-scale-weight", type=float, default=1.0)
    p.add_argument("--atlas-no-decode-bn", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
