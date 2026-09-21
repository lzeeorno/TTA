#!/bin/bash
# =============================================================================
# Canonical 50-cycle continual TTA on ImageNet-C (ResNet-50 GN backbone)
# - Uses the official long-term protocol structure: 50 cycles over the canonical
#   15-corruption order at severity 5.
# - Drops the old 3-order shortcut so order0 can no longer be mistaken for a
#   50-cycle evaluation.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/run_common.sh"
cd "$PROJECT_ROOT"

METHODS=("source" "tent" "cotta" "rotta" "sar" "deyo" "adadem" "atlas")
METHODS_CSV=$(IFS=,; echo "${METHODS[*]}")
SEED="${SEED:-1997}"
GPU="${GPU:-0}"
CONTINUAL_REPEATS="${CONTINUAL_REPEATS:-50}"
ORDER_IDX=0
CONFIG="${CONFIG:-configs/imagenetc_continual.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results/imagenetc_resnet50_gn_continual}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${LOG_DIR:-logs/continual_${TIMESTAMP}}"
mkdir -p "$LOG_DIR"

ORDER_0="brightness,contrast,defocus_blur,elastic_transform,fog,frost,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,shot_noise,snow,zoom_blur"

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
echo "Canonical 50-cycle Continual TTA on ImageNet-C"
echo "=============================================="
echo "Methods:   ${METHODS[*]}"
echo "Protocol:  canonical order0, ${CONTINUAL_REPEATS} cycles"
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
echo "=============================================="

for METHOD in "${METHODS[@]}"; do
    echo ""
    echo -e "\033[1;34m=============================================="
    echo -e "Method: ${METHOD}"
    echo -e "==============================================\033[0m"

    CURRENT_EXP=$((CURRENT_EXP + 1))
    METHOD_START_TIME=$(date +%s)
    RESULT_FILE="${RESULTS_DIR}/${METHOD}_seed${SEED}_order${ORDER_IDX}.json"
    LOG_FILE="${LOG_DIR}/${METHOD}_order${ORDER_IDX}.log"
    FORCE_RUN=0

    echo -e "\033[1;36m  [Progress: ${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] order=${ORDER_IDX}, repeats=${CONTINUAL_REPEATS}\033[0m"
    echo "  Starting at: $(date '+%Y-%m-%d %H:%M:%S')"

    if result_has_target_repeats "$RESULT_FILE" "$CONTINUAL_REPEATS"; then
        echo -e "\033[1;33m  ↷ Skip canonical 50-cycle run: found ${RESULT_FILE}\033[0m"
        continue
    fi

    if [ -s "$RESULT_FILE" ]; then
        echo -e "\033[1;33m  ! Stale order0 result found without continual_repeats=${CONTINUAL_REPEATS}; forcing rerun\033[0m"
        FORCE_RUN=1
    fi

    EXP_START_TIME=$(date +%s)

    RUN_FORCE="$FORCE_RUN" run_python_experiment "$LOG_FILE" "$METHOD order=${ORDER_IDX} continual-50cycle" \
        --config "$CONFIG" \
        --method "$METHOD" \
        --seed "$SEED" \
        --gpu "$GPU" \
        --corruption-order "$ORDER_0" \
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
        echo -e "\033[1;31m  ✗ order=${ORDER_IDX}: missing canonical 50-cycle metadata in ${RESULT_FILE}\033[0m"
        exit 1
    fi

    METHOD_END_TIME=$(date +%s)
    METHOD_DURATION=$((METHOD_END_TIME - METHOD_START_TIME))
    echo -e "\033[1;33m  Method ${METHOD} completed in ${METHOD_DURATION}s ($(date -ud @${METHOD_DURATION} +'%H:%M:%S'))\033[0m"
done

echo ""
echo "=============================================="
echo "Canonical 50-cycle continual experiments completed!"
echo "Results saved to: ${RESULTS_DIR}/"
echo "Logs saved to: ${LOG_DIR}"
echo "=============================================="

echo ""
echo "Generating summary..."
generate_auto_summary "$RESULTS_DIR" "$LOG_DIR" "imagenetc_continual_summary" "$METHODS_CSV" --include-per-corruption-md
