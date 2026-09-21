#!/bin/bash
# =============================================================================
# Table-5 SANE ablation on ACDC SegFormer-B5
# Compares ATLAS w/ SANE against a true ViT-LN no_uan ablation.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/run_common.sh"
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
RUN_SAVE_LOGS="${RUN_SAVE_LOGS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

TIMESTAMP="${TIMESTAMP:-$(run_timestamp)}"
RESULTS_DIR="${RESULTS_DIR:-results/table5_sane_ablation}"
LOG_DIR="${LOG_DIR:-logs/table5_sane_ablation_${TIMESTAMP}}"
SUMMARY_JSON="$RESULTS_DIR/sane_ablation_summary.json"
SUMMARY_MD="$RESULTS_DIR/sane_ablation_summary.md"

SEED="${SEED:-1997}"
GPU="${GPU:-0}"
ATLAS_USE_REFERENCE_FULL="${ATLAS_USE_REFERENCE_FULL:-1}"
ATLAS_REFERENCE_JSON="${ATLAS_REFERENCE_JSON:-results/table5_segformer_acdc_surgeon_cotta/atlas_seed${SEED}.json}"
ACDC_ROOT="${ACDC_ROOT:-data/acdc}"
SEGFORMER_MODEL="${SEGFORMER_MODEL:-models/segformer/segformer-b5-cityscapes}"
SPLIT="${SPLIT:-train}"
INPUT_WIDTH="${INPUT_WIDTH:-960}"
INPUT_HEIGHT="${INPUT_HEIGHT:-540}"
IMAGE_SIZE="${IMAGE_SIZE:-}"
ROUNDS="${ROUNDS:-10}"
REPORT_TIMESTAMPS="${REPORT_TIMESTAMPS:-1,4,7,10}"
WORKERS="${WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
RUN_FORCE="${RUN_FORCE:-0}"
ATLAS_LR="${ATLAS_LR:-1e-5}"
ATLAS_OPTIMIZER="${ATLAS_OPTIMIZER:-sgd}"
ATLAS_MOMENTUM="${ATLAS_MOMENTUM:-0.9}"
ATLAS_TARGET_SELECTED_RATIO="${ATLAS_TARGET_SELECTED_RATIO:-0.15}"
ATLAS_SOURCE_CONF_THRESHOLD="${ATLAS_SOURCE_CONF_THRESHOLD:-0.0}"
ATLAS_SOURCE_CONF_TOLERANCE="${ATLAS_SOURCE_CONF_TOLERANCE:-0.15}"
ATLAS_SOURCE_ANCHOR_WEIGHT="${ATLAS_SOURCE_ANCHOR_WEIGHT:-0.03}"
ATLAS_PREDICT_AFTER_UPDATE="${ATLAS_PREDICT_AFTER_UPDATE:-1}"
ATLAS_OUTPUT_HFLIP="${ATLAS_OUTPUT_HFLIP:-1}"
ATLAS_OUTPUT_HFLIP_WEIGHT="${ATLAS_OUTPUT_HFLIP_WEIGHT:-0.5}"
ATLAS_DECODE_BN="${ATLAS_DECODE_BN:-1}"
VISUALIZE="${VISUALIZE:-0}"
VISUALIZE_COUNT="${VISUALIZE_COUNT:-12}"
VISUALIZE_TIMESTAMPS="${VISUALIZE_TIMESTAMPS:-1,4,10}"
VISUALIZE_SEED="${VISUALIZE_SEED:-$SEED}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

print_section "Table-5 SANE Ablation"
echo "Results:   $RESULTS_DIR"
echo "Logs:      $LOG_DIR"
echo "Seed:      $SEED"
echo "GPU:       $GPU"
echo "Rounds:    $ROUNDS"
echo "Input:     ${INPUT_WIDTH}x${INPUT_HEIGHT}"
echo "MaxSample: $MAX_SAMPLES"
echo "ATLAS LR:  $ATLAS_LR"
echo "Optimizer: $ATLAS_OPTIMIZER"
echo "SelRatio:  $ATLAS_TARGET_SELECTED_RATIO"
echo "PostPred:  $ATLAS_PREDICT_AFTER_UPDATE"
echo "HFlipOut:  $ATLAS_OUTPUT_HFLIP"
echo "DecodeBN:  $ATLAS_DECODE_BN"
echo "UseRef:    $ATLAS_USE_REFERENCE_FULL"
echo "RefJSON:   $ATLAS_REFERENCE_JSON"

validate_variant_result() {
    local result_json="$1"
    local expected_normalizer="$2"
    local expected_uan_enabled="$3"

    "$PYTHON_CMD" - "$result_json" "$expected_normalizer" "$expected_uan_enabled" <<'PY'
import json
import os
import sys

path, expected_normalizer, expected_uan_enabled = sys.argv[1:]
if not os.path.isfile(path):
    raise SystemExit(1)
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
required = [
    "mean_online_error",
    "mean_timestamp_miou",
    "mean_primary_miou",
    "mean_primary_online_error",
    "miou_by_timestamp",
    "online_error_by_timestamp",
    "optimizer",
    "last_ln_loss_normalizer",
    "last_ln_source_agreement",
    "last_ln_probe_drop",
    "ln_predict_after_update",
    "ln_output_hflip",
    "ln_adapt_decode_bn",
    "uan_enabled",
]
missing = [key for key in required if key not in data]
if missing:
    raise SystemExit(1)
if data.get("base_method") != "atlas":
    raise SystemExit(1)
expected_uan = expected_uan_enabled == "1"
if bool(data.get("uan_enabled")) != expected_uan:
    raise SystemExit(1)
actual_normalizer = data.get("sane_normalizer")
if expected_normalizer == "__NONE__":
    if actual_normalizer is not None:
        raise SystemExit(1)
else:
    if actual_normalizer != expected_normalizer:
        raise SystemExit(1)
PY
}

materialize_reference_full_result() {
    local result_json="$1"

    "$PYTHON_CMD" - "$ATLAS_REFERENCE_JSON" "$result_json" <<'PY'
import json
import os
import sys

reference_path, output_path = sys.argv[1:]
if not os.path.isfile(reference_path):
    raise SystemExit(f"missing reference result: {reference_path}")
with open(reference_path, "r", encoding="utf-8") as f:
    data = json.load(f)
data["method"] = "atlas_with_sane"
data["base_method"] = "atlas"
data["result_suffix"] = "with_sane"
data["sane_normalizer"] = "selected"
data["uan_enabled"] = True
data["no_uan"] = False
data["uan_mode"] = data.get("uan_mode") or "pixel"
data["virtual_result"] = True
data["reference_result_json"] = reference_path
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
PY
}

run_variant() {
    local suffix="$1"
    local label="$2"
    local expected_normalizer="$3"
    local expected_uan_enabled="$4"
    shift 4
    local result_json="$RESULTS_DIR/atlas_${suffix}_seed${SEED}.json"
    local log_file="$LOG_DIR/atlas_${suffix}.log"

    if [ "$RUN_FORCE" != "1" ] && [ -f "$result_json" ]; then
        if validate_variant_result "$result_json" "$expected_normalizer" "$expected_uan_enabled"; then
            echo "[$(run_now)] Skip ${label}: existing strict result ${result_json}"
            return 0
        fi
        echo "[$(run_now)] Existing ${label} result is stale for SANE diagnostics; rerunning ${result_json}"
    fi

    if [ "$suffix" = "with_sane" ] && [ "$ATLAS_USE_REFERENCE_FULL" = "1" ]; then
        if [ -f "$ATLAS_REFERENCE_JSON" ]; then
            materialize_reference_full_result "$result_json"
            validate_variant_result "$result_json" "$expected_normalizer" "$expected_uan_enabled"
            echo "[$(run_now)] Use reference full-ATLAS result for ${label}: ${ATLAS_REFERENCE_JSON} -> ${result_json}"
            return 0
        fi
        echo "[$(run_now)] Reference full-ATLAS result missing; falling back to running ${label}"
    fi

    local cmd=(
        "$PYTHON_CMD" code/segmentation/run_acdc_segformer_table5.py
        --acdc-root "$ACDC_ROOT"
        --model-name-or-path "$SEGFORMER_MODEL"
        --method atlas
        --split "$SPLIT"
        --rounds "$ROUNDS"
        --batch-size 1
        --lr "$ATLAS_LR"
        --input-width "$INPUT_WIDTH"
        --input-height "$INPUT_HEIGHT"
        --report-timestamps "$REPORT_TIMESTAMPS"
        --workers "$WORKERS"
        --seed "$SEED"
        --gpu "$GPU"
        --output-dir "$RESULTS_DIR"
        --result-suffix "$suffix"
        --sane-normalizer selected
        --atlas-optimizer "$ATLAS_OPTIMIZER"
        --atlas-momentum "$ATLAS_MOMENTUM"
        --atlas-target-selected-ratio "$ATLAS_TARGET_SELECTED_RATIO"
        --atlas-source-conf-threshold "$ATLAS_SOURCE_CONF_THRESHOLD"
        --atlas-source-conf-tolerance "$ATLAS_SOURCE_CONF_TOLERANCE"
        --atlas-source-anchor-weight "$ATLAS_SOURCE_ANCHOR_WEIGHT"
        --atlas-output-hflip-weight "$ATLAS_OUTPUT_HFLIP_WEIGHT"
    )
    while [ "$#" -gt 0 ]; do
        cmd+=("$1")
        shift
    done

    if [ "$ATLAS_PREDICT_AFTER_UPDATE" != "1" ]; then
        cmd+=(--atlas-no-predict-after-update)
    fi
    if [ "$ATLAS_OUTPUT_HFLIP" != "1" ]; then
        cmd+=(--atlas-no-output-hflip)
    fi
    if [ "$ATLAS_DECODE_BN" != "1" ]; then
        cmd+=(--atlas-no-decode-bn)
    fi
    if [ -n "$IMAGE_SIZE" ]; then
        cmd+=(--image-size "$IMAGE_SIZE")
    fi
    if [ "$MAX_SAMPLES" -gt 0 ]; then
        cmd+=(--max-samples "$MAX_SAMPLES")
    fi
    if [ "$VISUALIZE" = "1" ]; then
        cmd+=(
            --visualize
            --visualize-count "$VISUALIZE_COUNT"
            --visualize-timestamps "$VISUALIZE_TIMESTAMPS"
            --visualize-seed "$VISUALIZE_SEED"
            --visualize-dir "$RESULTS_DIR/visualizations/${suffix}"
        )
    fi

    run_with_retry "$log_file" "table5 sane ${label}" "${cmd[@]}"

    validate_variant_result "$result_json" "$expected_normalizer" "$expected_uan_enabled"
}

run_variant "with_sane" "ATLAS w/ SANE" "selected" "1"
run_variant "wo_sane" "ATLAS w/o SANE" "__NONE__" "0" --no_uan

"$PYTHON_CMD" - "$RESULTS_DIR" "$SUMMARY_JSON" "$SUMMARY_MD" "$SEED" <<'PY'
import json
import os
import sys

results_dir, summary_json, summary_md, seed = sys.argv[1:]
variants = [
    ("ATLAS w/ SANE", f"atlas_with_sane_seed{seed}.json"),
    ("ATLAS w/o SANE", f"atlas_wo_sane_seed{seed}.json"),
]

payload = {"results": []}
for label, filename in variants:
    path = os.path.join(results_dir, filename)
    if not os.path.isfile(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    payload["results"].append(
        {
            "label": label,
            "method": data.get("method"),
            "sane_state": "on" if data.get("uan_enabled", True) else "off",
            "sane_normalizer": data.get("sane_normalizer") or "-",
            "uan_mode": data.get("uan_mode") or "-",
            "optimizer": data.get("optimizer"),
            "mean_primary_miou": data.get("mean_primary_miou"),
            "mean_primary_online_error": data.get("mean_primary_online_error"),
            "mean_online_error": data.get("mean_online_error"),
            "mean_timestamp_miou": data.get("mean_timestamp_miou"),
            "primary_timestamp": data.get("primary_timestamp"),
            "mean_ln_selected_ratio": data.get("mean_ln_selected_ratio"),
            "last_ln_loss_normalizer": data.get("last_ln_loss_normalizer"),
            "mean_ln_source_agreement": data.get("mean_ln_source_agreement"),
            "mean_ln_probe_drop": data.get("mean_ln_probe_drop"),
            "ln_predict_after_update": data.get("ln_predict_after_update"),
            "ln_output_hflip": data.get("ln_output_hflip"),
            "ln_adapt_decode_bn": data.get("ln_adapt_decode_bn"),
            "virtual_result": data.get("virtual_result", False),
            "result_json": path,
        }
    )

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

lines = [
    "# Table-5 SANE Ablation Summary",
    "",
    "| Variant | SANE | Normalizer | UAN Mode | Optimizer | Mean Online Error | Mean Timestamp mIoU | Primary mIoU | Primary Online Error | Selected Ratio | Loss Normalizer | Source Agreement | Probe Drop | Post-Update Pred | HFlip Output | Decode BN | Virtual | Primary Timestamp | Result |",
    "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- |",
]
for row in payload["results"]:
    lines.append(
        "| {label} | {sane_state} | {sane_normalizer} | {uan_mode} | {optimizer} | {mean_online_error} | {mean_timestamp_miou} | {mean_primary_miou} | {mean_primary_online_error} | {mean_ln_selected_ratio} | {last_ln_loss_normalizer} | {mean_ln_source_agreement} | {mean_ln_probe_drop} | {ln_predict_after_update} | {ln_output_hflip} | {ln_adapt_decode_bn} | {virtual_result} | {primary_timestamp} | {result_json} |".format(**row)
    )

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
PY

echo "Done. Summary: $SUMMARY_MD"
