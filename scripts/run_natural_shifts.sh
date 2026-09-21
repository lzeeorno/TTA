#!/bin/bash
# =============================================================================
# Natural Distribution Shift Experiments
# ImageNet-R / ImageNet-A / ImageNet-V2 / ImageNet-Sketch
# Standard TTA (no corruption types, single domain per dataset)
# Dual backbone: ResNet-50-GN + ViT-B/16, 3 seeds each
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/run_common.sh"
cd "$PROJECT_ROOT"

COMMON_METHODS=("source" "atlas")
VIT_EXTRA_METHODS=()
GN_METHODS=("${COMMON_METHODS[@]}")
VIT_METHODS=("${COMMON_METHODS[@]}" "${VIT_EXTRA_METHODS[@]}")
SEEDS=(1997 2048 2077)
GPU=0

DATASETS=("imagenet_r" "imagenet_a" "imagenet_v2" "imagenet_sketch")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/natural_shifts_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Natural Distribution Shift Experiments"
echo "=============================================="
echo "Datasets: ${DATASETS[*]}"
echo "GN Methods: ${GN_METHODS[*]}"
echo "ViT Methods: ${VIT_METHODS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Backbones: ResNet-50-GN + ViT-B/16"
echo "Total experiments: $((${#DATASETS[@]} * ${#SEEDS[@]} * (${#GN_METHODS[@]} + ${#VIT_METHODS[@]})))"
echo "=============================================="

TOTAL=$((${#DATASETS[@]} * ${#SEEDS[@]} * (${#GN_METHODS[@]} + ${#VIT_METHODS[@]})))
CURRENT=0

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo -e "\033[1;35m=============================================="
    echo -e "Dataset: ${dataset}"
    echo -e "==============================================\033[0m"

    for method in "${GN_METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # --- ResNet-50-GN ---
            CURRENT=$((CURRENT + 1))
            echo -e "\033[1;36m  [$CURRENT/$TOTAL] ${dataset} GN: ${method} seed=${seed}\033[0m"

            run_python_experiment "${LOG_DIR}/${dataset}_gn_${method}_seed${seed}.log" "${dataset} gn ${method} seed=${seed}" \
                --config "configs/${dataset}_gn.yaml" \
                --method "$method" \
                --seed "$seed" \
                --gpu "$GPU"

        done
    done

    for method in "${VIT_METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # --- ViT-B/16 ---
            CURRENT=$((CURRENT + 1))
            echo -e "\033[1;36m  [$CURRENT/$TOTAL] ${dataset} ViT: ${method} seed=${seed}\033[0m"

            run_python_experiment "${LOG_DIR}/${dataset}_vit_${method}_seed${seed}.log" "${dataset} vit ${method} seed=${seed}" \
                --config "configs/${dataset}_vit.yaml" \
                --method "$method" \
                --seed "$seed" \
                --gpu "$GPU"
        done
    done
done

echo ""
echo "=============================================="
echo "All Natural Shift experiments completed!"
echo "Logs: disabled"
echo "=============================================="

echo ""
echo "Generating summary..."

GN_RESULTS_DIRS=(
    "results/imagenet_a_resnet50_gn"
    "results/imagenet_r_resnet50_gn"
    "results/imagenet_v2_resnet50_gn"
    "results/imagenet_sketch_resnet50_gn"
)

VIT_RESULTS_DIRS=(
    "results/imagenet_a_vit_base_patch16_224"
    "results/imagenet_r_vit_base_patch16_224"
    "results/imagenet_v2_vit_base_patch16_224"
    "results/imagenet_sketch_vit_base_patch16_224"
)

GN_METHODS_CSV="source,atlas"
VIT_METHODS_CSV="source,atlas"

for results_dir in "${GN_RESULTS_DIRS[@]}"; do
    echo "[Summary] ${results_dir}/natural_shifts_summary.{csv,md}"
    python scripts/generate_run_summary.py \
        --results-dir "$results_dir" \
        --output-dir "$results_dir" \
        --summary-name natural_shifts_summary \
        --methods "$GN_METHODS_CSV"
done

for results_dir in "${VIT_RESULTS_DIRS[@]}"; do
    echo "[Summary] ${results_dir}/natural_shifts_summary.{csv,md}"
    python scripts/generate_run_summary.py \
        --results-dir "$results_dir" \
        --output-dir "$results_dir" \
        --summary-name natural_shifts_summary \
        --methods "$VIT_METHODS_CSV"
done

OVERALL_SUMMARY_ARGS=(
    --output-dir "results"
    --summary-name natural_shifts_overall_summary
)

for results_dir in "${GN_RESULTS_DIRS[@]}"; do
    OVERALL_SUMMARY_ARGS+=(--results-dir "$results_dir")
done

for results_dir in "${VIT_RESULTS_DIRS[@]}"; do
    OVERALL_SUMMARY_ARGS+=(--results-dir "$results_dir")
done

echo "[Summary] results/natural_shifts_overall_summary.{csv,md}"
python scripts/generate_natural_shifts_overall_summary.py "${OVERALL_SUMMARY_ARGS[@]}"

echo "Summary generation completed for 8 result folders."
