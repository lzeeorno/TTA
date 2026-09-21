#!/usr/bin/env python3
"""Compose deterministic ACDC qualitative panels from runner result JSONs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def parse_result(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--result requires METHOD=RESULT.json")
    method, raw_path = spec.split("=", 1)
    return method.strip(), Path(raw_path).resolve()


def indexed_visuals(result_path: Path) -> dict[tuple[str, int], Path]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    files = payload.get("visualizations", {}).get("generated_files", [])
    return {
        (item["sample_id"], int(item["timestamp"])): Path(item["path"]).resolve()
        for item in files
    }


def compose_row(sample_id: str, timestamp: int, panels: list[tuple[str, Path]]) -> Image.Image:
    opened = [Image.open(path).convert("RGB") for _, path in panels]
    header = 34
    width = sum(image.width for image in opened)
    height = max(image.height for image in opened) + header
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for (method, _), panel in zip(panels, opened):
        draw.text((x + 8, 8), f"{method} | {sample_id} | t={timestamp}", fill="black")
        canvas.paste(panel, (x, header))
        x += panel.width
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=parse_result, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1997)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()

    by_method = {method: indexed_visuals(path) for method, path in args.result}
    common = set.intersection(*(set(items) for items in by_method.values()))
    candidates = sorted(common)
    rng = random.Random(args.seed)
    selected = candidates if len(candidates) <= args.count else sorted(rng.sample(candidates, args.count))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for rank, (sample_id, timestamp) in enumerate(selected, start=1):
        panels = [(method, by_method[method][(sample_id, timestamp)]) for method in by_method]
        output = args.output_dir / f"{rank:02d}_{sample_id.replace('/', '__')}_t{timestamp}.png"
        compose_row(sample_id, timestamp, panels).save(output)
        generated.append(str(output.resolve()))

    manifest = {
        "seed": args.seed,
        "count": args.count,
        "selection_rule": "seeded uniform sample from the intersection of pre-generated method/sample/timestamp panels",
        "candidate_count": len(candidates),
        "selected": [{"sample_id": sid, "timestamp": ts} for sid, ts in selected],
        "roles": ["success", "success", "ordinary", "failure"][: len(selected)],
        "role_note": "Roles are presentation slots only; IDs are selected before visual inspection.",
        "inputs": {method: str(path) for method, path in args.result},
        "generated_files": generated,
    }
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
