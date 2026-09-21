#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


PREFERRED_CORRUPTION_ORDER = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur",
]

CORRUPTION_LABELS = {
    "brightness": "Bright.",
    "contrast": "Contr.",
    "defocus_blur": "Defoc.",
    "elastic_transform": "Elastic",
    "fog": "Fog",
    "frost": "Frost",
    "gaussian_noise": "Gauss.",
    "glass_blur": "Glass",
    "impulse_noise": "Impulse",
    "jpeg_compression": "JPEG",
    "motion_blur": "Motion",
    "pixelate": "Pixel.",
    "shot_noise": "Shot",
    "snow": "Snow",
    "zoom_blur": "Zoom",
}


FILENAME_PATTERNS = [
    re.compile(r"^(?P<method>.+?)_seed(?P<seed>\d+)_order(?P<order>\d+)$"),
    re.compile(r"^(?P<method>.+?)_seed(?P<seed>\d+)$"),
    re.compile(r"^(?P<method>.+?)_order(?P<order>\d+)$"),
    re.compile(r"^(?P<method>.+)$"),
]


def parse_filename(stem: str):
    for pattern in FILENAME_PATTERNS:
        match = pattern.match(stem)
        if match:
            return match.groupdict()
    return {"method": stem}


def compute_mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
    return mean, std


def format_mean_std(values):
    mean, std = compute_mean_std(values)
    if mean is None:
        return "-"
    return f"{mean:.4f} +- {std:.4f}"


def extract_per_corruption(payload):
    raw = payload.get("per_corruption")
    if not isinstance(raw, dict):
        return {}

    per_corruption = {}
    for corruption, stats in raw.items():
        value = stats.get("accuracy") if isinstance(stats, dict) else stats
        try:
            per_corruption[str(corruption)] = float(value)
        except Exception:
            continue
    return per_corruption


def get_corruption_order(rows):
    seen = []
    seen_set = set()
    for row in rows:
        for corruption in row.get("per_corruption", {}):
            if corruption not in seen_set:
                seen.append(corruption)
                seen_set.add(corruption)

    preferred = [name for name in PREFERRED_CORRUPTION_ORDER if name in seen_set]
    remaining = [name for name in seen if name not in set(preferred)]
    return preferred + remaining


def read_result(path: Path):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None

    meta = parse_filename(path.stem)
    method = payload.get("method") or meta.get("method") or path.stem
    seed = payload.get("seed", meta.get("seed"))
    order = payload.get("order_idx", meta.get("order"))
    acc = payload.get("mean_accuracy")
    if acc is None:
        acc = payload.get("avg_accuracy")
    if acc is None:
        acc = payload.get("accuracy")
    if acc is None:
        return None
    try:
        acc = float(acc)
    except Exception:
        return None
    return {
        "result_dir": str(path.parent),
        "file": path.name,
        "method": str(method),
        "seed": "" if seed is None else str(seed),
        "order": "" if order is None else str(order),
        "accuracy": acc,
        "per_corruption": extract_per_corruption(payload),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate unified CSV/Markdown summaries from result JSON files.")
    parser.add_argument("--results-dir", action="append", required=True, help="Result directory to scan; can be passed multiple times.")
    parser.add_argument("--output-dir", required=True, help="Directory to write summary files into.")
    parser.add_argument("--summary-name", default="summary", help="Base filename for generated summary files.")
    parser.add_argument("--methods", default="", help="Optional comma-separated method order/filter.")
    parser.add_argument("--glob", default="*.json", help="Glob used inside each results dir.")
    parser.add_argument(
        "--include-per-corruption-md",
        action="store_true",
        help="Include a per-corruption mean/std table in the Markdown summary.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    method_order = {name: idx for idx, name in enumerate(requested_methods)}

    rows = []
    for result_dir in args.results_dir:
        root = Path(result_dir)
        if not root.exists():
            continue
        for path in sorted(root.glob(args.glob)):
            row = read_result(path)
            if not row:
                continue
            if requested_methods and row["method"] not in method_order:
                continue
            rows.append(row)

    rows.sort(key=lambda row: (
        method_order.get(row["method"], math.inf),
        row["method"],
        row["seed"],
        row["order"],
        row["file"],
    ))

    csv_path = output_dir / f"{args.summary_name}.csv"
    md_path = output_dir / f"{args.summary_name}.md"

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "seed", "order", "accuracy", "file", "result_dir"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "method": row["method"],
                "seed": row["seed"],
                "order": row["order"],
                "accuracy": f"{row['accuracy']:.6f}",
                "file": row["file"],
                "result_dir": row["result_dir"],
            })

    by_method = defaultdict(list)
    by_method_per_corruption = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_method[row["method"]].append(row["accuracy"])
        for corruption, accuracy in row.get("per_corruption", {}).items():
            by_method_per_corruption[row["method"]][corruption].append(accuracy)

    corruption_order = get_corruption_order(rows)

    ordered_methods = requested_methods or sorted(by_method)
    md_lines = [
        f"# {args.summary_name}",
        "",
        "## Aggregate",
        "",
        "| Method | Runs | Mean Accuracy (%) | Std |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for method in ordered_methods:
        accs = by_method.get(method, [])
        if not accs:
            md_lines.append(f"| {method} | 0 | - | - |")
            continue
        mean, std = compute_mean_std(accs)
        md_lines.append(f"| {method} | {len(accs)} | {mean:.4f} | {std:.4f} |")

    if args.include_per_corruption_md and corruption_order:
        header_labels = [CORRUPTION_LABELS.get(name, name.replace("_", " ")) for name in corruption_order]
        md_lines.extend([
            "",
            "## Per-Corruption Aggregate",
            "",
            "Each cell is mean +- std across runs/orders.",
            "",
            "| Method | " + " | ".join(header_labels) + " | Avg |",
            "| :--- | " + " | ".join(["---:"] * (len(header_labels) + 1)) + " |",
        ])
        for method in ordered_methods:
            per_corruption = by_method_per_corruption.get(method, {})
            cells = [format_mean_std(per_corruption.get(corruption, [])) for corruption in corruption_order]
            cells.append(format_mean_std(by_method.get(method, [])))
            md_lines.append(f"| {method} | " + " | ".join(cells) + " |")

    md_lines.extend([
        "",
        "## Raw Results",
        "",
        "| Method | Seed | Order | Accuracy (%) | File |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ])
    for row in rows:
        order_val = row["order"] if row["order"] != "" else "-"
        seed_val = row["seed"] if row["seed"] != "" else "-"
        md_lines.append(
            f"| {row['method']} | {seed_val} | {order_val} | {row['accuracy']:.4f} | {row['file']} |"
        )

    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"Generated: {csv_path}")
    print(f"Generated: {md_path}")
    if rows:
        print("Top methods:")
        aggregates = []
        for method, accs in by_method.items():
            aggregates.append((sum(accs) / len(accs), method, len(accs)))
        for mean, method, count in sorted(aggregates, reverse=True)[:10]:
            print(f"  {method}: mean={mean:.4f}% over {count} run(s)")
    else:
        print("No result JSON files found.")


if __name__ == "__main__":
    main()
