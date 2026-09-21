#!/bin/bash
# =============================================================================
# CIFAR-100-C Continual TTA
# 3 representative corruption orders, no reset between corruptions
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/run_common.sh"
cd "$PROJECT_ROOT"

METHODS=("source" "atlas")
NUM_ORDERS=3
SEED=1997
GPU=0
CONFIG="configs/cifar100c_continual.yaml"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/cifar100c_continual_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

ORDER_0="brightness,contrast,defocus_blur,elastic_transform,fog,frost,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,shot_noise,snow,zoom_blur"
ORDER_1="brightness,jpeg_compression,motion_blur,gaussian_noise,shot_noise,snow,glass_blur,fog,contrast,pixelate,frost,defocus_blur,elastic_transform,impulse_noise,zoom_blur"
ORDER_2="jpeg_compression,pixelate,shot_noise,brightness,glass_blur,snow,elastic_transform,fog,gaussian_noise,impulse_noise,motion_blur,defocus_blur,frost,contrast,zoom_blur"
ORDERS=("$ORDER_0" "$ORDER_1" "$ORDER_2")

echo "=========================================="
echo "CIFAR-100-C Continual TTA"
echo "=========================================="
echo "Methods: ${METHODS[*]}"
echo "Orders: ${NUM_ORDERS}"
echo "Seed: $SEED"
echo "Config: $CONFIG"
echo "Results: results/cifar100c_resnext29_continual/"
echo ""

for method in "${METHODS[@]}"; do
    for ((order_idx=0; order_idx<NUM_ORDERS; order_idx++)); do
        echo "[$(date)] Running CIFAR-100-C continual: ${method} order=${order_idx}"
        run_python_experiment "${LOG_DIR}/${method}_order${order_idx}.log" "$method order=${order_idx} cifar100-continual" \
            --config "$CONFIG" \
            --method "$method" \
            --seed "$SEED" \
            --gpu "$GPU" \
            --corruption-order "${ORDERS[$order_idx]}" \
            --order-idx "$order_idx"
        echo "[$(date)] Done CIFAR-100-C continual: ${method} order=${order_idx}"
    done
done

echo "All CIFAR-100-C continual experiments done. Logs: ${LOG_DIR}/"
generate_auto_summary "results/cifar100c_resnext29_continual" "$LOG_DIR" "cifar100c_continual_summary" "source,tent,eata,sar,cotta,lcotta,adadem,surgeon,deyo,rotta,triad,atlas" --include-per-corruption-md
