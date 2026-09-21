#!/bin/bash
# =============================================================================
# ATLAS Table 7 ablation rows on ImageNet-C continual ViT-B/16
# Reuses configs/vit_imagenetc_continual.yaml and stores each row separately.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/run_common.sh"
cd "$PROJECT_ROOT"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate sata
fi

CONFIG="${CONFIG:-configs/vit_imagenetc_continual.yaml}"
GPU="${GPU:-0}"
SEED="${SEED:-1997}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_BATCHES="${MAX_BATCHES:-0}"
RESULTS_DIR="${RESULTS_DIR:-results/imagenetc_vit_base_patch16_224_continual}"
ROWS_CSV="${ROWS_CSV:-A0,A1,A2,A3,B1,B2}"
IFS=',' read -r -a ROWS <<< "$ROWS_CSV"

ORDER_0="brightness,contrast,defocus_blur,elastic_transform,fog,frost,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,shot_noise,snow,zoom_blur"
ORDER_1="brightness,jpeg_compression,motion_blur,gaussian_noise,shot_noise,snow,glass_blur,fog,contrast,pixelate,frost,defocus_blur,elastic_transform,impulse_noise,zoom_blur"
ORDER_2="jpeg_compression,pixelate,shot_noise,brightness,glass_blur,snow,elastic_transform,fog,gaussian_noise,impulse_noise,motion_blur,defocus_blur,frost,contrast,zoom_blur"
ORDERS=("$ORDER_0" "$ORDER_1" "$ORDER_2")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${LOG_DIR:-logs/atlas_table7_vit_continual_${TIMESTAMP}}"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "ATLAS Table 7 Ablation on ImageNet-C Continual ViT-B/16"
echo "=============================================="
echo "Rows:     ${ROWS[*]}"
echo "Orders:   ${#ORDERS[@]}"
echo "Seed:     ${SEED}"
echo "GPU:      ${GPU}"
echo "Batch:    ${BATCH_SIZE}"
echo "MaxBatch: ${MAX_BATCHES}"
echo "Config:   ${CONFIG}"
echo "Results:  ${RESULTS_DIR}"
echo "Log Dir:  ${LOG_DIR}"
echo "=============================================="

TOTAL_EXPERIMENTS=$((${#ROWS[@]} * ${#ORDERS[@]}))
COMPLETED_RESULTS=0
CURRENT_EXP=0

for ROW in "${ROWS[@]}"; do
    for ORDER_IDX in "${!ORDERS[@]}"; do
        RESULT_FILE="${RESULTS_DIR}/atlas_${ROW}_seed${SEED}_order${ORDER_IDX}.json"
        if [ -s "$RESULT_FILE" ]; then
            COMPLETED_RESULTS=$((COMPLETED_RESULTS + 1))
        fi
    done
done

echo "Existing Results: ${COMPLETED_RESULTS}/${TOTAL_EXPERIMENTS}"
echo "Pending Runs: $((TOTAL_EXPERIMENTS - COMPLETED_RESULTS))"

START_TIME=$(date +%s)

for ROW in "${ROWS[@]}"; do
    echo ""
    echo "=============================================="
    echo "ATLAS row: ${ROW}"
    echo "=============================================="

    for ORDER_IDX in "${!ORDERS[@]}"; do
        ORDER="${ORDERS[$ORDER_IDX]}"
        RESULT_FILE="${RESULTS_DIR}/atlas_${ROW}_seed${SEED}_order${ORDER_IDX}.json"
        LOG_FILE="${LOG_DIR}/atlas_${ROW}_order${ORDER_IDX}.log"
        CURRENT_EXP=$((CURRENT_EXP + 1))

        echo "[Progress: ${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] ${ROW} order=${ORDER_IDX}"
        echo "Starting at: $(date '+%Y-%m-%d %H:%M:%S')"

        if [ -s "$RESULT_FILE" ]; then
            echo "Skip ${ROW} order=${ORDER_IDX}: found existing result ${RESULT_FILE}"
            continue
        fi

        EXTRA_ARGS=(
            --config "$CONFIG"
            --method atlas
            --seed "$SEED"
            --corruption-order "$ORDER"
            --order-idx "$ORDER_IDX"
            --gpu 0
            --batch-size "$BATCH_SIZE"
            --run-suffix "$ROW"
        )
        if [ "$ROW" != "A3" ]; then
            EXTRA_ARGS+=(--atlas-ablation-row "$ROW")
        fi
        if [ "$MAX_BATCHES" -gt 0 ]; then
            EXTRA_ARGS+=(--max-batches "$MAX_BATCHES")
        fi

        CUDA_VISIBLE_DEVICES="$GPU" \
        run_python_experiment "$LOG_FILE" "atlas ${ROW} order=${ORDER_IDX} continual-vit" \
            "${EXTRA_ARGS[@]}"

        if [ -s "$RESULT_FILE" ]; then
            echo "Saved ${RESULT_FILE}"
        else
            echo "Missing result after run: ${RESULT_FILE}"
            exit 1
        fi

        sleep_between_experiments
    done
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "ATLAS Table 7 continual ViT ablation finished"
echo "=============================================="
echo "Total time: ${TOTAL_DURATION}s"
echo "Results: ${RESULTS_DIR}/"
echo "Logs:    disabled"
echo "=============================================="

generate_auto_summary \
    "$RESULTS_DIR" \
    "$LOG_DIR" \
    "imagenetc_continual_vit_atlas_ablation" \
    "atlas_A0,atlas_A1,atlas_A2,atlas_A3,atlas_B1,atlas_B2" \
    --include-per-corruption-md