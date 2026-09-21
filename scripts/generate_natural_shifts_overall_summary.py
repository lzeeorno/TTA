#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


DATASET_ORDER = [
    "imagenet_a",
    "imagenet_r",
    "imagenet_v2",
    "imagenet_sketch",
]

DATASET_LABELS = {
    "imagenet_a": "ImageNet-A",
    "imagenet_r": "ImageNet-R",
    "imagenet_v2": "ImageNet-V2",
    "imagenet_sketch": "ImageNet-Sketch",
}

BRANCH_ORDER = ["GN", "ViT-LN"]


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


def dataset_sort_key(dataset: str):
    try:
        return DATASET_ORDER.index(dataset)
    except ValueError:
        return len(DATASET_ORDER)


def branch_sort_key(branch: str):
    try:
        return BRANCH_ORDER.index(branch)
    except ValueError:
        return len(BRANCH_ORDER)


def infer_dataset_and_branch(result_dir: Path):
    name = result_dir.name
    prefixes = [
        "imagenet_a_",
        "imagenet_r_",
        "imagenet_v2_",
        "imagenet_sketch_",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            dataset = prefix[:-1]
            backbone = name[len(prefix):]
            break
    else:
        dataset = name
        backbone = name

    backbone_lower = backbone.lower()
    if "vit" in backbone_lower:
        branch = "ViT-LN"
    elif "gn" in backbone_lower:
        branch = "GN"
    else:
        branch = backbone
    return dataset, branch


def read_result(path: Path):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None

    meta = parse_filename(path.stem)
    method = payload.get("method") or meta.get("method") or path.stem
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

    dataset = payload.get("dataset")
    backbone = payload.get("backbone")
    if not dataset or not backbone:
        dataset, branch = infer_dataset_and_branch(path.parent)
    else:
        branch = "ViT-LN" if "vit" in str(backbone).lower() else ("GN" if "gn" in str(backbone).lower() else str(backbone))

    return {
        "dataset": str(dataset),
        "branch": branch,
        "method": str(method),
        "accuracy": acc,
        "result_dir": str(path.parent),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate an overall natural shifts markdown leaderboard.")
    parser.add_argument("--results-dir", action="append", required=True, help="Natural shifts result directory. Can be passed multiple times.")
    parser.add_argument("--output-dir", required=True, help="Directory to write output files into.")
    parser.add_argument("--summary-name", default="natural_shifts_overall_summary", help="Base filename for generated files.")
    parser.add_argument("--glob", default="*.json", help="Glob used inside each results dir.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    result_dir_lookup = {}
    for result_dir in args.results_dir:
        root = Path(result_dir)
        if not root.exists():
            continue
        for path in sorted(root.glob(args.glob)):
            row = read_result(path)
            if not row:
                continue
            key = (row["dataset"], row["branch"], row["method"])
            grouped[key].append(row["accuracy"])
            result_dir_lookup[key] = row["result_dir"]

    ranked_rows = []
    for (dataset, branch, method), accuracies in grouped.items():
        mean, std = compute_mean_std(accuracies)
        ranked_rows.append(
            {
                "dataset": dataset,
                "branch": branch,
                "method": method,
                "runs": len(accuracies),
                "mean_accuracy": mean,
                "std": std,
                "result_dir": result_dir_lookup[(dataset, branch, method)],
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            -(row["mean_accuracy"] if row["mean_accuracy"] is not None else -math.inf),
            row["dataset"],
            row["branch"],
            row["method"],
        )
    )

    grouped_ranked_rows = defaultdict(lambda: defaultdict(list))
    for row in ranked_rows:
        grouped_ranked_rows[row["dataset"]][row["branch"]].append(row)

    ordered_rows = []
    for dataset in sorted(grouped_ranked_rows, key=dataset_sort_key):
        for branch in sorted(grouped_ranked_rows[dataset], key=branch_sort_key):
            branch_rows = grouped_ranked_rows[dataset][branch]
            branch_rows.sort(
                key=lambda row: (
                    -(row["mean_accuracy"] if row["mean_accuracy"] is not None else -math.inf),
                    row["method"],
                )
            )
            for rank, row in enumerate(branch_rows, start=1):
                row["rank"] = rank
                ordered_rows.append(row)

    csv_path = output_dir / f"{args.summary_name}.csv"
    md_path = output_dir / f"{args.summary_name}.md"

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["rank", "dataset", "branch", "method", "runs", "mean_accuracy", "std", "result_dir"],
        )
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "dataset": row["dataset"],
                    "branch": row["branch"],
                    "method": row["method"],
                    "runs": row["runs"],
                    "mean_accuracy": f"{row['mean_accuracy']:.6f}",
                    "std": f"{row['std']:.6f}",
                    "result_dir": row["result_dir"],
                }
            )

    md_lines = [
        f"# {args.summary_name}",
        "",
        "> Aggregated across the 4 datasets x 2 backbones natural-shift result folders.",
        "> Methods are ranked only within the same dataset and backbone.",
        "",
    ]
    for dataset in sorted(grouped_ranked_rows, key=dataset_sort_key):
        md_lines.extend([
            f"## {DATASET_LABELS.get(dataset, dataset)}",
            "",
        ])
        for branch in sorted(grouped_ranked_rows[dataset], key=branch_sort_key):
            md_lines.extend([
                f"### {branch}",
                "",
                "| Rank | Method | Runs | Avg Accuracy (%) | Std | Result Folder |",
                "| ---: | :--- | ---: | ---: | ---: | :--- |",
            ])
            for row in grouped_ranked_rows[dataset][branch]:
                md_lines.append(
                    f"| {row['rank']} | {row['method']} | {row['runs']} | {row['mean_accuracy']:.4f} | {row['std']:.4f} | {row['result_dir']} |"
                )
            md_lines.append("")

    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"Generated: {csv_path}")
    print(f"Generated: {md_path}")
    if ranked_rows:
        top = ranked_rows[0]
        print(
            "Best overall: "
            f"{top['dataset']} / {top['branch']} / {top['method']} = {top['mean_accuracy']:.4f}%"
        )
    else:
        print("No result JSON files found.")


if __name__ == "__main__":
    main()