#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
    echo "[ERROR] conda.sh not found at $CONDA_SH" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_SH"
conda activate sata
if [ "${CONDA_DEFAULT_ENV:-}" != "sata" ]; then
    echo "[ERROR] Failed to activate conda env: sata" >&2
    exit 1
fi

PYTHON_CMD="$HOME/anaconda3/envs/sata/bin/python"
CUSTOM_SAVE_DIR=""

if [ $# -gt 0 ]; then
    CUSTOM_SAVE_DIR="$1"
    shift
fi

if [ $# -gt 0 ]; then
    echo "Usage: $0 [optional_save_dir]" >&2
    exit 1
fi

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${CUSTOM_SAVE_DIR:-results/ablation_table7_${TIMESTAMP}}}"
LOG_DIR="${LOG_DIR:-logs/ablation_table7_${TIMESTAMP}}"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

RAW_LOG_DIR="$LOG_DIR/raw"

echo "=============================================="
echo "Table 7 Ablation Wrapper"
echo "=============================================="
echo "Conda Env: sata"
echo "Benchmark: ViT-B/16 + ImageNet-C + continual"
echo "Input:     scripts/run_imagenetc_continual_vit_atlas_ablation.sh"
echo "Seed:      ${SEED:-1997}"
echo "GPU:       ${GPU:-0}"
echo "Results:   ${SAVE_DIR}"
echo "Logs:      ${LOG_DIR}"
echo "=============================================="

RESULTS_DIR="$SAVE_DIR" \
LOG_DIR="$RAW_LOG_DIR" \
GPU="${GPU:-0}" \
SEED="${SEED:-1997}" \
    bash "$SCRIPT_DIR/run_imagenetc_continual_vit_atlas_ablation.sh"

"$PYTHON_CMD" - "$SAVE_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

save_dir = Path(sys.argv[1])
summary_csv_src = save_dir / "imagenetc_continual_vit_atlas_ablation.csv"
summary_md_src = save_dir / "imagenetc_continual_vit_atlas_ablation.md"
summary_csv_dst = save_dir / "table7_summary.csv"
summary_md_dst = save_dir / "table7_summary.md"
manifest_path = save_dir / "table7_manifest.csv"

if summary_csv_src.exists():
    summary_csv_dst.write_text(summary_csv_src.read_text(encoding="utf-8"), encoding="utf-8")

if summary_md_src.exists():
    text = summary_md_src.read_text(encoding="utf-8")
    if text.startswith("# imagenetc_continual_vit_atlas_ablation"):
        text = text.replace("# imagenetc_continual_vit_atlas_ablation", "# Table 7 Summary", 1)
    summary_md_dst.write_text(text, encoding="utf-8")

rows = []
for path in sorted(save_dir.glob("atlas_*.json")):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    method = str(payload.get("method", path.stem))
    row = str(payload.get("run_suffix") or method.split("_", 1)[-1])
    rows.append({
        "row": row,
        "method": method,
        "seed": payload.get("seed", ""),
        "order": payload.get("order_idx", ""),
        "mean_accuracy": payload.get("mean_accuracy", ""),
        "file": path.name,
    })

with manifest_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["row", "method", "seed", "order", "mean_accuracy", "file"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
PY

echo "Done. Table 7 artifacts:"
echo "  - ${SAVE_DIR}/table7_manifest.csv"
echo "  - ${SAVE_DIR}/table7_summary.csv"
echo "  - ${SAVE_DIR}/table7_summary.md"
