#!/bin/bash
# =============================================================================
# Continual TTA on ImageNet-C with ViT-B/16
# - Defaults to the single-pass 15-corruption stream used by the paper's
#   standard/continual table. Set CONTINUAL_REPEATS=50 for the separate
#   long-horizon protocol.
# - ORDER_IDX selects one of the three representative corruption orders.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/run_common.sh"
cd "$PROJECT_ROOT"

METHODS=("source" "atlas")
METHODS_CSV=$(IFS=,; echo "${METHODS[*]}")
SEED="${SEED:-1997}"
GPU="${GPU:-0}"
CONTINUAL_REPEATS="${CONTINUAL_REPEATS:-1}"
ORDER_IDX="${ORDER_IDX:-0}"
CONFIG="${CONFIG:-configs/vit_imagenetc_continual.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results/imagenetc_vit_base_patch16_224_continual}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${LOG_DIR:-logs/continual_vit_${TIMESTAMP}}"
mkdir -p "$LOG_DIR"

ORDER_0="brightness,contrast,defocus_blur,elastic_transform,fog,frost,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,shot_noise,snow,zoom_blur"
ORDER_1="brightness,jpeg_compression,motion_blur,gaussian_noise,shot_noise,snow,glass_blur,fog,contrast,pixelate,frost,defocus_blur,elastic_transform,impulse_noise,zoom_blur"
ORDER_2="jpeg_compression,pixelate,shot_noise,brightness,glass_blur,snow,elastic_transform,fog,gaussian_noise,impulse_noise,motion_blur,defocus_blur,frost,contrast,zoom_blur"
ORDERS=("$ORDER_0" "$ORDER_1" "$ORDER_2")

if ! [[ "$ORDER_IDX" =~ ^[0-2]$ ]]; then
    echo "[ERROR] ORDER_IDX must be 0, 1, or 2; got '$ORDER_IDX'" >&2
    exit 1
fi
ORDER="${ORDERS[$ORDER_IDX]}"

result_has_target_repeats() {
    local result_file="$1"
    local expected_repeats="$2"

    if [ ! -s "$result_file" ]; then
        return 1
    fi

    python - "$result_file" "$expected_repeats" <<'PY' >/dev/null 2>&1
import json
import sys

path = sys.argv[1]
expected_repeats = int(sys.argv[2])

try:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
except Exception:
    raise SystemExit(1)

if int(payload.get('continual_repeats', 1)) != expected_repeats:
    raise SystemExit(1)

completed_cycles = payload.get('completed_cycles')
if completed_cycles is None:
    cycle_mean_accuracies = payload.get('cycle_mean_accuracies') or []
    completed_cycles = len(cycle_mean_accuracies) if cycle_mean_accuracies else 1

if int(completed_cycles) != expected_repeats:
    raise SystemExit(1)

if bool(payload.get('is_partial_cycle_dump', False)):
    raise SystemExit(1)
PY
}

echo "=============================================="
echo "Continual TTA on ImageNet-C (ViT-B/16)"
echo "=============================================="
echo "Backbone:  ViT-B/16 (timm, pretrained)"
echo "Adapt:     LayerNorm parameters"
echo "Methods:   ${METHODS[*]}"
echo "Protocol:  order${ORDER_IDX}, ${CONTINUAL_REPEATS} cycle(s)"
echo "Seed:      ${SEED}"
echo "GPU:       ${GPU}"
echo "Config:    ${CONFIG}"
echo "Results:   ${RESULTS_DIR}"
echo "Logs:      ${LOG_DIR}"
echo "=============================================="

TOTAL_EXPERIMENTS=${#METHODS[@]}
CURRENT_EXP=0
COMPLETED_RESULTS=0

for METHOD in "${METHODS[@]}"; do
    RESULT_FILE="${RESULTS_DIR}/${METHOD}_seed${SEED}_order${ORDER_IDX}.json"
    if result_has_target_repeats "$RESULT_FILE" "$CONTINUAL_REPEATS"; then
        COMPLETED_RESULTS=$((COMPLETED_RESULTS + 1))
    fi
done

PENDING_EXPERIMENTS=$((TOTAL_EXPERIMENTS - COMPLETED_RESULTS))
echo "Existing Canonical Results: ${COMPLETED_RESULTS}/${TOTAL_EXPERIMENTS}"
echo "Pending Runs: ${PENDING_EXPERIMENTS}"

START_TIME=$(date +%s)

for METHOD in "${METHODS[@]}"; do
    echo ""
    echo -e "\033[1;34m=============================================="
    echo -e "Method: ${METHOD}"
    echo -e "==============================================\033[0m"

    METHOD_START_TIME=$(date +%s)
    CURRENT_EXP=$((CURRENT_EXP + 1))
    RESULT_FILE="${RESULTS_DIR}/${METHOD}_seed${SEED}_order${ORDER_IDX}.json"
    LOG_FILE="${LOG_DIR}/${METHOD}_order${ORDER_IDX}.log"
    FORCE_RUN=0

    echo -e "\033[1;36m  [Progress: ${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] order=${ORDER_IDX}, repeats=${CONTINUAL_REPEATS}\033[0m"
    echo "  Starting at: $(date '+%Y-%m-%d %H:%M:%S')"

    if result_has_target_repeats "$RESULT_FILE" "$CONTINUAL_REPEATS"; then
        echo -e "\033[1;33m  ↷ Skip continual run: found ${RESULT_FILE}\033[0m"
        continue
    fi

    if [ -s "$RESULT_FILE" ]; then
        echo -e "\033[1;33m  ! Stale order${ORDER_IDX} result found without continual_repeats=${CONTINUAL_REPEATS}; forcing rerun\033[0m"
        FORCE_RUN=1
    fi

    EXP_START_TIME=$(date +%s)

    RUN_FORCE="$FORCE_RUN" run_python_experiment "$LOG_FILE" "$METHOD order=${ORDER_IDX} continual-vit" \
        --config "$CONFIG" \
        --method "$METHOD" \
        --seed "$SEED" \
        --gpu "$GPU" \
        --corruption-order "$ORDER" \
        --order-idx "$ORDER_IDX" \
        --continual-repeats "$CONTINUAL_REPEATS"

    EXP_END_TIME=$(date +%s)
    EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))

    if result_has_target_repeats "$RESULT_FILE" "$CONTINUAL_REPEATS"; then
        ACC=$(python - <<PY
import json
from pathlib import Path

p = Path("$RESULT_FILE")
try:
    d = json.loads(p.read_text(encoding='utf-8'))
    v = d.get('final_cycle_mean_accuracy', d.get('mean_accuracy', d.get('avg_accuracy', d.get('accuracy'))))
    print('' if v is None else v)
except Exception:
    print('')
PY
)
        if [ -n "$ACC" ]; then
            echo -e "\033[1;32m  ✓ order=${ORDER_IDX}: final/mean accuracy = ${ACC}% (took ${EXP_DURATION}s)\033[0m"
        else
            echo -e "\033[1;31m  ✗ order=${ORDER_IDX}: failed to extract accuracy\033[0m"
        fi
    else
        echo -e "\033[1;31m  ✗ order=${ORDER_IDX}: missing continual metadata in ${RESULT_FILE}\033[0m"
        exit 1
    fi

    METHOD_END_TIME=$(date +%s)
    METHOD_DURATION=$((METHOD_END_TIME - METHOD_START_TIME))
    echo -e "\033[1;33m  Method ${METHOD} completed in ${METHOD_DURATION}s\033[0m"
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "Continual ViT experiments completed!"
echo "=============================================="
echo "Total time: ${TOTAL_DURATION}s ($((TOTAL_DURATION / 60))m $((TOTAL_DURATION % 60))s)"
echo "Results: ${RESULTS_DIR}/"
echo "Logs:    ${LOG_DIR}"
echo "=============================================="

echo ""
echo "Generating summary..."
generate_auto_summary "$RESULTS_DIR" "$LOG_DIR" "imagenetc_continual_vit_summary" "$METHODS_CSV" --include-per-corruption-md
