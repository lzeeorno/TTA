#!/bin/bash
# =============================================================================
# Wild World Experiments on ImageNet-C with ViT-B/16
# Following SAR/DeYO evaluation protocol
# Three Wild Scenarios: BS=1, Label Shifts, Mixed Shifts
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/run_common.sh"
cd "$PROJECT_ROOT"

METHODS=("source" "tent" "eata" "sar" "cotta" "lcotta" "adadem" "surgeon" "foa" "deyo" "atlas")
# FOA BS=1 is enabled via a project-side singleton-batch extension in code/baselines/foa.py.
BS1_METHODS=("source" "tent" "eata" "sar" "cotta" "lcotta" "adadem" "surgeon" "foa" "deyo" "atlas")
DEFAULT_BS1_SEEDS=(1997)
DEFAULT_SHIFT_SEEDS=(1997 2048 2077)
REQUESTED_SEED=""
GPU=0
REQUESTED_METHOD=""
REQUESTED_SCENARIO=""

RESULTS_BS1_DIR="results/imagenetc_vit_base_patch16_224_shuffle_continual_bs1"
RESULTS_LABELSHIFT_DIR="results/imagenetc_vit_base_patch16_224_continual_label_shifts"
RESULTS_MIXSHIFTS_DIR="results/imagenetc_vit_base_patch16_224_shuffle_continual_mix_shifts"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_imagenetc_wild_vit.sh [--method METHOD] [--scenario SCENARIO] [--seed SEED] [--gpu GPU]

Options:
  --method METHOD      Run one method only, e.g. atlas.
  --scenario SCENARIO  Run one scenario only: bs1, label_shifts, or mix_shifts.
  --seed SEED          Run one seed only.
  --gpu GPU            GPU id to pass to code/main.py.
EOF
}

contains() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        if [ "$item" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

join_by_comma() {
    local IFS=","
    echo "$*"
}

while (( $# > 0 )); do
    case "$1" in
        --method)
            REQUESTED_METHOD="${2:?missing value for --method}"
            shift 2
            ;;
        --scenario)
            REQUESTED_SCENARIO="${2:?missing value for --scenario}"
            shift 2
            ;;
        --seed)
            REQUESTED_SEED="${2:?missing value for --seed}"
            shift 2
            ;;
        --gpu)
            GPU="${2:?missing value for --gpu}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -n "$REQUESTED_METHOD" ] && ! contains "$REQUESTED_METHOD" "${METHODS[@]}"; then
    echo "Unknown method: $REQUESTED_METHOD" >&2
    exit 2
fi

if [ -n "$REQUESTED_SCENARIO" ] && ! contains "$REQUESTED_SCENARIO" "bs1" "label_shifts" "mix_shifts"; then
    echo "Unknown scenario: $REQUESTED_SCENARIO" >&2
    exit 2
fi

SELECTED_METHODS=("${METHODS[@]}")
SELECTED_BS1_METHODS=("${BS1_METHODS[@]}")
SELECTED_BS1_SEEDS=("${DEFAULT_BS1_SEEDS[@]}")
SELECTED_SHIFT_SEEDS=("${DEFAULT_SHIFT_SEEDS[@]}")
if [ -n "$REQUESTED_METHOD" ]; then
    SELECTED_METHODS=("$REQUESTED_METHOD")
    if contains "$REQUESTED_METHOD" "${BS1_METHODS[@]}"; then
        SELECTED_BS1_METHODS=("$REQUESTED_METHOD")
    else
        SELECTED_BS1_METHODS=()
    fi
fi

if [ -n "$REQUESTED_SEED" ]; then
    SELECTED_BS1_SEEDS=("$REQUESTED_SEED")
    SELECTED_SHIFT_SEEDS=("$REQUESTED_SEED")
fi

RUN_BS1=0
RUN_LABELSHIFT=0
RUN_MIXSHIFTS=0
if [ -z "$REQUESTED_SCENARIO" ]; then
    RUN_BS1=1
    RUN_LABELSHIFT=1
    RUN_MIXSHIFTS=1
elif [ "$REQUESTED_SCENARIO" = "bs1" ]; then
    RUN_BS1=1
elif [ "$REQUESTED_SCENARIO" = "label_shifts" ]; then
    RUN_LABELSHIFT=1
elif [ "$REQUESTED_SCENARIO" = "mix_shifts" ]; then
    RUN_MIXSHIFTS=1
fi

# Targeted runs should overwrite stale JSONs from earlier failed attempts.
if [ -n "$REQUESTED_METHOD" ] || [ -n "$REQUESTED_SCENARIO" ]; then
    export RUN_FORCE="${RUN_FORCE:-1}"
fi

SUMMARY_METHODS="$(join_by_comma "${SELECTED_METHODS[@]}")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/wild_vit_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

TOTAL=0
if [ "$RUN_BS1" = "1" ]; then
    TOTAL=$((TOTAL + ${#SELECTED_BS1_METHODS[@]} * ${#SELECTED_BS1_SEEDS[@]}))
fi
if [ "$RUN_LABELSHIFT" = "1" ]; then
    TOTAL=$((TOTAL + ${#SELECTED_METHODS[@]} * ${#SELECTED_SHIFT_SEEDS[@]}))
fi
if [ "$RUN_MIXSHIFTS" = "1" ]; then
    TOTAL=$((TOTAL + ${#SELECTED_METHODS[@]} * ${#SELECTED_SHIFT_SEEDS[@]}))
fi

if [ "$TOTAL" -eq 0 ]; then
    echo "No jobs selected."
    exit 0
fi

echo "=============================================="
echo "Wild World Experiments (ViT-B/16) on ImageNet-C"
echo "=============================================="
echo "Methods: ${SELECTED_METHODS[*]}"
echo "BS=1 methods: ${SELECTED_BS1_METHODS[*]:-none}"
echo "BS=1 seeds: ${SELECTED_BS1_SEEDS[*]}"
echo "Label/Mix seeds: ${SELECTED_SHIFT_SEEDS[*]}"
echo "Scenarios: ${REQUESTED_SCENARIO:-bs1, label_shifts, mix_shifts}"
echo "Force overwrite existing targeted results: ${RUN_FORCE:-0}"
echo "=============================================="

CURRENT=0
SUMMARY_RESULT_DIRS=()

if [ "$RUN_BS1" = "1" ]; then
    SUMMARY_RESULT_DIRS+=("$RESULTS_BS1_DIR")
    echo ""
    echo -e "\033[1;35m=== Scenario 1: BS=1 (ViT-B/16) ===\033[0m"

    for method in "${SELECTED_BS1_METHODS[@]}"; do
        for seed in "${SELECTED_BS1_SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo -e "\033[1;36m  [$CURRENT/$TOTAL] BS=1: ${method} seed=${seed}\033[0m"

            run_python_experiment "${LOG_DIR}/bs1_${method}_seed${seed}.log" "bs1 ${method} seed=${seed}" \
                --config configs/vit_imagenetc_wild_bs1.yaml \
                --method "$method" \
                --scenario bs1 \
                --batch-size 1 \
                --seed "$seed" \
                --gpu "$GPU"
        done
    done
fi

if [ "$RUN_LABELSHIFT" = "1" ]; then
    SUMMARY_RESULT_DIRS+=("$RESULTS_LABELSHIFT_DIR")
    echo ""
    echo -e "\033[1;35m=== Scenario 2: Label Shifts (ViT-B/16) ===\033[0m"

    for method in "${SELECTED_METHODS[@]}"; do
        for seed in "${SELECTED_SHIFT_SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo -e "\033[1;36m  [$CURRENT/$TOTAL] Label Shift: ${method} seed=${seed}\033[0m"

            run_python_experiment "${LOG_DIR}/labelshift_${method}_seed${seed}.log" "labelshift ${method} seed=${seed}" \
                --config configs/vit_imagenetc_wild_labelshift.yaml \
                --method "$method" \
                --scenario label_shifts \
                --seed "$seed" \
                --gpu "$GPU"
        done
    done
fi

if [ "$RUN_MIXSHIFTS" = "1" ]; then
    SUMMARY_RESULT_DIRS+=("$RESULTS_MIXSHIFTS_DIR")
    echo ""
    echo -e "\033[1;35m=== Scenario 3: Mixed Shifts (ViT-B/16) ===\033[0m"

    for method in "${SELECTED_METHODS[@]}"; do
        for seed in "${SELECTED_SHIFT_SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo -e "\033[1;36m  [$CURRENT/$TOTAL] Mix Shifts: ${method} seed=${seed}\033[0m"

            run_python_experiment "${LOG_DIR}/mixshifts_${method}_seed${seed}.log" "mixshifts ${method} seed=${seed}" \
                --config configs/vit_imagenetc_wild_mixshifts.yaml \
                --method "$method" \
                --scenario mix_shifts \
                --seed "$seed" \
                --gpu "$GPU"
        done
    done
fi

echo ""
echo "=============================================="
echo "Selected ViT Wild experiments completed!"
echo "Logs: disabled"
echo "Results: results/imagenetc_vit_base_patch16_224_*/"
echo "=============================================="

generate_multi_dir_summary "imagenetc_vit_wild_summary" "$SUMMARY_METHODS" "${SUMMARY_RESULT_DIRS[@]}"
