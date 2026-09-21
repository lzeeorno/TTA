#!/bin/bash
# ImageNet-C所有方法批量运行脚本
# 运行13个方法 (source, tent, eata, sar, cotta, lcotta, adadem, surgeon, deyo, rotta, triad, atlas, triad_f4) × 3个种子
# 注意：ImageNet-C实验耗时较长，每个方法约1-2小时

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/run_common.sh"
cd "$PROJECT_ROOT"


METHODS=("source" "tent" "eata" "sar" "cotta" "lcotta" "adadem" "surgeon" "deyo" "rotta" "triad" "atlas" "triad_f4")
SEEDS=(1997 2048 2077)
GPU=0
CONFIG="configs/imagenetc.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/imagenetc_${TIMESTAMP}"

echo "=========================================="
echo "ImageNet-C Baseline Reproduction"
echo "=========================================="
echo "Methods: ${METHODS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "GPU: $GPU"
echo "Config: $CONFIG"
echo "Log dir: $LOG_DIR"
echo "Total experiments: $((${#METHODS[@]} * ${#SEEDS[@]}))"
echo ""

START_TIME=$(date +%s)

# 创建logs目录
mkdir -p "$LOG_DIR"

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=========================================="
        echo "Running: $method (Seed $seed)"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="

        EXP_START=$(date +%s)

        run_python_experiment "$LOG_DIR/${method}_seed${seed}.log" "$method seed=${seed}" \
            --config "$CONFIG" \
            --method "$method" \
            --seed "$seed" \
            --gpu "$GPU"

        EXP_END=$(date +%s)
        EXP_TIME=$((EXP_END - EXP_START))

        echo ""
        echo "✅ Completed: $method (Seed $seed) - Time: ${EXP_TIME}s ($((EXP_TIME / 60))m $((EXP_TIME % 60))s)"
        echo ""

        # 短暂休息，避免GPU过热
        sleep 5
    done
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Total time: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s)"
echo "Results directory: results/imagenetc_resnet50_gn/"
echo "Logs directory: $LOG_DIR/"
echo ""
echo "View results with:"
echo "  ls -lh results/imagenetc_resnet50_gn/"
echo "  cat results/imagenetc_resnet50_gn/*.json"

generate_auto_summary "results/imagenetc_resnet50_gn" "$LOG_DIR" "imagenetc_summary" "source,tent,eata,sar,cotta,lcotta,adadem,surgeon,deyo,rotta,triad,atlas,triad_f4" --include-per-corruption-md
