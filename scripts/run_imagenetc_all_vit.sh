#!/bin/bash
# =============================================================================
# ImageNet-C Standard TTA with ViT-B/16 (LayerNorm backbone)
# =============================================================================
# Backbone: ViT-B/16 (timm, vit_base_patch16_224, pretrained=ImageNet-1K)
# Config:   configs/vit_imagenetc.yaml
# Params:   LayerNorm layers adapted, Transformer baselines use official/custom ViT settings
#
# Method notes:
#   - ViT uses LayerNorm (not BN), so adapt_params=ln
#   - Tent/EATA/SAR now follow SAR-family official ViT lr scaling in `code/main.py`
#   - batch_size=48 default for ViT experiments (24GB GPU target)
#   - CoTTA/RoTTA are included here as project-side ViT extensions for fair comparison
#     (they are not official BN-only reproduction settings)
#
# Results → results/imagenetc_vit_base_patch16_224/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/run_common.sh"
cd "$PROJECT_ROOT"

METHODS=("source" "tent" "eata" "sar" "cotta" "lcotta" "adadem" "surgeon" "foa" "deyo" "rotta" "triad" "atlas" "triad_f4")
SEEDS=(1997 2048 2077)
GPU=0
CONFIG="configs/vit_imagenetc.yaml"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/imagenetc_vit_standard_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "ImageNet-C Standard TTA (ViT-B/16)"
echo "=========================================="
echo "Backbone: ViT-B/16 (timm, pretrained)"
echo "Adapt:    LayerNorm parameters"
echo "Optim:    method-specific (see code/main.py)"
echo "BS:       48 (default for ViT configs; adjust in config if needed)"
echo "Methods:  ${METHODS[*]}"
echo "Seeds:    ${SEEDS[*]}"
echo "GPU:      $GPU"
echo "Config:   $CONFIG"
echo "Log Dir:  $LOG_DIR"
echo "Results → results/imagenetc_vit_base_patch16_224/"
echo "Total experiments: $((${#METHODS[@]} * ${#SEEDS[@]}))"
echo ""

START_TIME=$(date +%s)
CURRENT_EXP=0
TOTAL_EXPS=$((${#METHODS[@]} * ${#SEEDS[@]}))

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT_EXP=$((CURRENT_EXP + 1))
        echo "=========================================="
        echo "[${CURRENT_EXP}/${TOTAL_EXPS}] Running: $method (Seed $seed) [ViT-B/16]"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="

        EXP_START=$(date +%s)

        run_python_experiment "${LOG_DIR}/vit_${method}_seed${seed}.log" "$method seed=${seed} vit-standard" \
            --config "$CONFIG" \
            --method "$method" \
            --seed "$seed" \
            --gpu "$GPU"

        EXP_END=$(date +%s)
        EXP_TIME=$((EXP_END - EXP_START))

        echo ""
        echo "✅ Completed: $method (Seed $seed) — ${EXP_TIME}s ($((EXP_TIME / 60))m $((EXP_TIME % 60))s)"
        echo ""

        sleep 5
    done
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "=========================================="
echo "All ViT Standard TTA experiments completed!"
echo "=========================================="
echo "Total time: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s)"
echo "Results: results/imagenetc_vit_base_patch16_224/"
echo "Logs:    ${LOG_DIR}/"
echo ""

echo "Generating summary..."
generate_auto_summary "results/imagenetc_vit_base_patch16_224" "$LOG_DIR" "imagenetc_vit_standard_summary" "source,tent,eata,sar,cotta,lcotta,adadem,surgeon,foa,deyo,rotta,triad,atlas,triad_f4" --include-per-corruption-md
