#!/bin/bash
# =============================================================================
# Table-5: Cityscapes-pretrained SegFormer-B5 -> ACDC continual segmentation TTA
# Active comparison set: source, tent, surgeon, dem, lcotta, cotta, deyo, eata, foa, sar, atlas
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
  echo "[ERROR] conda.sh not found at $CONDA_SH"
  exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_SH"
conda activate sata

if [ "${CONDA_DEFAULT_ENV:-}" != "sata" ]; then
  echo "[ERROR] Failed to activate conda env: sata"
  exit 1
fi

PYTHON_CMD="$HOME/anaconda3/envs/sata/bin/python"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=${RESULTS_DIR:-results/table5_segformer_acdc_surgeon_cotta}
mkdir -p "$RESULTS_DIR"

SEED=${SEED:-1997}
GPU=${GPU:-0}
ACDC_ROOT=${ACDC_ROOT:-data/acdc}
SEGFORMER_MODEL=${SEGFORMER_MODEL:-models/segformer/segformer-b5-cityscapes}
SPLIT=${SPLIT:-train}
INPUT_WIDTH=${INPUT_WIDTH:-960}
INPUT_HEIGHT=${INPUT_HEIGHT:-540}
IMAGE_SIZE=${IMAGE_SIZE:-}
ROUNDS=${ROUNDS:-10}
REPORT_TIMESTAMPS=${REPORT_TIMESTAMPS:-1,4,7,10}
WORKERS=${WORKERS:-4}
MAX_SAMPLES=${MAX_SAMPLES:-0}
VISUALIZE=${VISUALIZE:-0}
VISUALIZE_COUNT=${VISUALIZE_COUNT:-12}
VISUALIZE_TIMESTAMPS=${VISUALIZE_TIMESTAMPS:-1,4,10}
VISUALIZE_SEED=${VISUALIZE_SEED:-$SEED}
METHODS_CSV=${METHODS:-source,atlas}
ATLAS_METHOD_VARIANT=${ATLAS_METHOD_VARIANT:-full}

canonicalize_method() {
  local raw_method="${1//[[:space:]]/}"
  local lower_method="${raw_method,,}"

  case "$lower_method" in
    "")
      return 0
      ;;
    source)
      echo "source"
      ;;
    tent)
      echo "tent"
      ;;
    atlas)
      echo "atlas"
      ;;
    surgeon|surgeon-master)
      echo "surgeon"
      ;;
    dem|dem-main|adadem)
      echo "dem"
      ;;
    lcotta|lcotta-main)
      echo "lcotta"
      ;;
    cotta)
      echo "cotta"
      ;;
    deyo)
      echo "deyo"
      ;;
    eata)
      echo "eata"
      ;;
    foa|foa-main)
      echo "foa"
      ;;
    sar)
      echo "sar"
      ;;
    *)
      echo "[ERROR] Unsupported method alias: ${1}" >&2
      return 1
      ;;
  esac
}

IFS=',' read -r -a RAW_METHODS <<< "$METHODS_CSV"
REQUESTED_METHODS=()
declare -A SEEN_METHODS=()
for raw_method in "${RAW_METHODS[@]}"; do
  canonical_method=""
  if ! canonical_method="$(canonicalize_method "$raw_method")"; then
    exit 1
  fi
  if [ -z "$canonical_method" ] || [ -n "${SEEN_METHODS[$canonical_method]:-}" ]; then
    continue
  fi
  REQUESTED_METHODS+=("$canonical_method")
  SEEN_METHODS[$canonical_method]=1
done
if [ "${#REQUESTED_METHODS[@]}" -eq 0 ]; then
  echo "[ERROR] No valid methods requested."
  exit 1
fi
METHODS_CSV=$(IFS=','; echo "${REQUESTED_METHODS[*]}")

declare -A METHOD_STATUS=(
  [source]="local_supported"
  [tent]="local_supported"
  [atlas]="local_supported"
  [surgeon]="unavailable"
  [dem]="unavailable"
  [lcotta]="unavailable"
  [cotta]="external_adaptation"
  [deyo]="unavailable"
  [eata]="unavailable"
  [foa]="unavailable"
  [sar]="unavailable"
)

declare -A METHOD_REASON=(
  [source]="Repo-native unified Table-5 baseline."
  [tent]="Repo-native unified Table-5 baseline."
  [atlas]="Repo-native ATLAS Table-5 adapter with SegFormer LayerNorm adaptation."
  [surgeon]="Repo vendors the SURGEON classification baseline only; no verified ACDC SegFormer segmentation adapter is present."
  [dem]="Repo vendors the DEM/AdaDEM classification baseline only; no verified ACDC SegFormer segmentation adapter is present."
  [lcotta]="Repo vendors the LCoTTA classification baseline only; no verified ACDC SegFormer segmentation adapter is present."
  [cotta]="CoTTA README references the official Cityscapes-to-ACDC SegFormer release, but no local Table-5 adapter is vendored here."
  [deyo]="Repo vendors the DeYO classification baseline only; no verified ACDC SegFormer segmentation adapter is present."
  [eata]="Repo vendors the EATA classification baseline only; no verified ACDC SegFormer segmentation adapter is present."
  [foa]="Repo vendors the FOA classification baseline only; no verified ACDC SegFormer segmentation adapter is present."
  [sar]="Repo vendors the SAR classification baseline only; no verified ACDC SegFormer segmentation adapter is present."
)

declare -A METHOD_REFERENCE=(
  [source]="code/segmentation/run_acdc_segformer_table5.py"
  [tent]="code/segmentation/run_acdc_segformer_table5.py"
  [atlas]="code/segmentation/run_acdc_segformer_table5.py"
  [surgeon]="code/baselines/SURGEON-master/README.md"
  [dem]="code/baselines/DEM-main/README.md"
  [lcotta]="code/baselines/LCoTTA-main/README.md"
  [cotta]="code/baselines/cotta/README.md"
  [deyo]="code/baselines/DeYO/README.md"
  [eata]="code/baselines/EATA/README.md"
  [foa]="code/baselines/FOA-main/README.md"
  [sar]="code/baselines/SAR/README.md"
)

print_header() {
  echo "=============================================="
  echo "Table-5 Segmentation on ACDC (SegFormer-B5)"
  echo "=============================================="
  echo "Conda Env: sata"
  echo "ACDC Root: ${ACDC_ROOT}"
  echo "Model:     ${SEGFORMER_MODEL}"
  echo "Seed:      ${SEED}"
  echo "GPU:       ${GPU}"
  echo "BS:        1"
  echo "Split:     ${SPLIT}"
  echo "Rounds:    ${ROUNDS}"
  echo "Input:     ${INPUT_WIDTH}x${INPUT_HEIGHT}"
  if [ -n "$IMAGE_SIZE" ]; then
    echo "LegacySquareOverride: ${IMAGE_SIZE}"
  fi
  echo "Norm:      ImageNet mean/std"
  echo "Timestamps:${REPORT_TIMESTAMPS}"
  echo "Workers:   ${WORKERS}"
  echo "MaxSample: ${MAX_SAMPLES}"
  echo "Visualize: ${VISUALIZE}"
  if [ "$VISUALIZE" = "1" ]; then
    echo "VisCount:  ${VISUALIZE_COUNT}"
    echo "VisTS:     ${VISUALIZE_TIMESTAMPS}"
    echo "VisSeed:   ${VISUALIZE_SEED}"
  fi
  echo "Methods:   ${METHODS_CSV}"
  echo "Results:   ${RESULTS_DIR}"
  echo "Logs:      disabled"
  echo "=============================================="
}

write_method_status() {
  local method="$1"
  local status="$2"
  local reason="$3"
  local reference="$4"
  local status_file="${RESULTS_DIR}/${method}_seed${SEED}.status.json"

  $PYTHON_CMD - <<PY
import json
payload = {
    "method": "${method}",
    "status": "${status}",
    "reason": "${reason}",
    "reference": "${reference}",
    "seed": ${SEED},
    "protocol": "table5_surgeon_cotta_v1",
    "legacy_protocol": "table5_unified_repo_v1",
    "split": "${SPLIT}",
    "input_width": ${INPUT_WIDTH},
    "input_height": ${INPUT_HEIGHT},
    "normalization": "imagenet",
    "report_timestamps": "${REPORT_TIMESTAMPS}",
    "metric_contract": "online_error = 100 - miou for each reported condition/timestamp",
    "required_result_fields": [
        "protocol",
        "split",
        "rounds",
        "report_timestamps",
        "primary_timestamp",
        "input_width",
        "input_height",
        "normalization",
        "samples_per_condition",
        "miou",
        "online_error",
        "miou_by_timestamp",
        "online_error_by_timestamp",
        "mean_online_error"
    ],
    "expected_result_file": "${RESULTS_DIR}/${method}_seed${SEED}.json",
}
with open("${status_file}", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

validate_result_schema() {
  local method="$1"
  local result_file="$2"

  "$PYTHON_CMD" - "$method" "$result_file" "$SPLIT" "$ROUNDS" "$REPORT_TIMESTAMPS" "$INPUT_WIDTH" "$INPUT_HEIGHT" "$IMAGE_SIZE" <<'PY'
import json
import os
import sys

method, path, split, rounds_raw, timestamps_raw, input_width_raw, input_height_raw, image_size_raw = sys.argv[1:]
rounds = int(rounds_raw)
expected_width = int(image_size_raw) if image_size_raw else int(input_width_raw)
expected_height = int(image_size_raw) if image_size_raw else int(input_height_raw)

def parse_timestamps(raw, rounds):
    timestamps = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= rounds and value not in timestamps:
            timestamps.append(value)
    if rounds not in timestamps:
        timestamps.append(rounds)
    return sorted(timestamps)

expected_timestamps = parse_timestamps(timestamps_raw, rounds)
expected_primary_timestamp = rounds if rounds in expected_timestamps else expected_timestamps[-1]
metric_keys = ("fog", "night", "rain", "snow", "mean")

def fail(message):
    print(f"[WARN] {method}: invalid Table-5 result {path}: {message}", file=sys.stderr)
    sys.exit(1)

def is_number_or_none(value):
    return value is None or isinstance(value, (int, float))

def error_from_miou(miou):
    return {k: (None if v is None else 100.0 - float(v)) for k, v in miou.items()}

def validate_metric_dict(name, payload):
    if not isinstance(payload, dict):
        fail(f"{name} must be an object")
    missing = [k for k in metric_keys if k not in payload]
    if missing:
        fail(f"{name} missing keys {missing}")
    bad = [k for k in metric_keys if not is_number_or_none(payload.get(k))]
    if bad:
        fail(f"{name} has non-numeric values for {bad}")

def validate_error_relation(miou, error, name):
    for key in metric_keys:
        m = miou.get(key)
        e = error.get(key)
        if m is None or e is None:
            continue
        if abs((100.0 - float(m)) - float(e)) > 1e-3:
            fail(f"{name}.{key} is not 100 - miou")

if not os.path.isfile(path):
    fail("file does not exist")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

changed = False
if "online_error" not in data and isinstance(data.get("miou"), dict):
    data["online_error"] = error_from_miou(data["miou"])
    changed = True
if "online_error_by_timestamp" not in data and isinstance(data.get("miou_by_timestamp"), dict):
    data["online_error_by_timestamp"] = {
        str(t): error_from_miou(data["miou_by_timestamp"][str(t)])
        for t in expected_timestamps
        if isinstance(data["miou_by_timestamp"].get(str(t)), dict)
    }
    changed = True
if "aggregate_online_error" not in data and isinstance(data.get("aggregate_miou"), dict):
    data["aggregate_online_error"] = error_from_miou(data["aggregate_miou"])
    changed = True
if "mean_primary_online_error" not in data and isinstance(data.get("online_error"), dict):
    data["mean_primary_online_error"] = data["online_error"].get("mean")
    changed = True
if "mean_primary_miou" not in data and isinstance(data.get("miou"), dict):
    data["mean_primary_miou"] = data["miou"].get("mean")
    changed = True
if "mean_online_error" not in data:
    timestamp_errors = []
    for timestamp_payload in data.get("online_error_by_timestamp", {}).values():
        if isinstance(timestamp_payload, dict):
            timestamp_errors.extend(timestamp_payload.get(k) for k in ("fog", "night", "rain", "snow"))
    present = [float(v) for v in timestamp_errors if v is not None]
    data["mean_online_error"] = sum(present) / len(present) if present else data.get("online_error", {}).get("mean")
    changed = True
if "mean_timestamp_miou" not in data and data.get("mean_online_error") is not None:
    data["mean_timestamp_miou"] = 100.0 - float(data["mean_online_error"])
    changed = True
if "primary_timestamp" not in data:
    data["primary_timestamp"] = expected_primary_timestamp
    changed = True

if data.get("protocol") != "table5_surgeon_cotta_v1":
    fail(f"protocol={data.get('protocol')!r}, expected table5_surgeon_cotta_v1")
if data.get("split") != split:
    fail(f"split={data.get('split')!r}, expected {split!r}")
if int(data.get("rounds", -1)) != rounds:
    fail(f"rounds={data.get('rounds')!r}, expected {rounds}")
if int(data.get("input_width", -1)) != expected_width:
    fail(f"input_width={data.get('input_width')!r}, expected {expected_width}")
if int(data.get("input_height", -1)) != expected_height:
    fail(f"input_height={data.get('input_height')!r}, expected {expected_height}")
normalization = data.get("normalization")
normalization_type = normalization.get("type") if isinstance(normalization, dict) else normalization
if normalization_type != "imagenet":
    fail(f"normalization={normalization_type!r}, expected imagenet")
if [int(x) for x in data.get("report_timestamps", [])] != expected_timestamps:
    fail(f"report_timestamps={data.get('report_timestamps')!r}, expected {expected_timestamps}")
if int(data.get("primary_timestamp", -1)) != expected_primary_timestamp:
    fail(f"primary_timestamp={data.get('primary_timestamp')!r}, expected {expected_primary_timestamp}")

for field in ("samples_per_condition", "miou", "online_error", "miou_by_timestamp", "online_error_by_timestamp"):
    if field not in data:
        fail(f"missing field {field}")

validate_metric_dict("miou", data["miou"])
validate_metric_dict("online_error", data["online_error"])
validate_error_relation(data["miou"], data["online_error"], "online_error")

for timestamp in expected_timestamps:
    key = str(timestamp)
    if key not in data["miou_by_timestamp"]:
        fail(f"missing miou_by_timestamp[{key}]")
    if key not in data["online_error_by_timestamp"]:
        fail(f"missing online_error_by_timestamp[{key}]")
    validate_metric_dict(f"miou_by_timestamp[{key}]", data["miou_by_timestamp"][key])
    validate_metric_dict(f"online_error_by_timestamp[{key}]", data["online_error_by_timestamp"][key])
    validate_error_relation(data["miou_by_timestamp"][key], data["online_error_by_timestamp"][key], f"online_error_by_timestamp[{key}]")

if changed:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print(f"[OK] {method}: strict Table-5 schema validated for {path}")
PY
}

run_local_method() {
  local method="$1"
  local result_file="${RESULTS_DIR}/${method}_seed${SEED}.json"
  local -a cmd

  if [ -s "$result_file" ]; then
    if validate_result_schema "$method" "$result_file"; then
      echo "↷ Skip ${method}: found strict-protocol result ${result_file}"
      return 0
    fi
    echo "[INFO] Existing ${method} result is stale or incomplete; rerunning under strict protocol."
  fi

  cmd=(
    "$PYTHON_CMD" code/segmentation/run_acdc_segformer_table5.py
    --acdc-root "$ACDC_ROOT"
    --model-name-or-path "$SEGFORMER_MODEL"
    --method "$method"
    --split "$SPLIT"
    --rounds "$ROUNDS"
    --batch-size 1
    --input-width "$INPUT_WIDTH"
    --input-height "$INPUT_HEIGHT"
    --report-timestamps "$REPORT_TIMESTAMPS"
    --workers "$WORKERS"
    --seed "$SEED"
    --gpu "$GPU"
    --output-dir "$RESULTS_DIR"
  )
  if [ "$method" = "atlas" ]; then
    cmd+=(--atlas-method-variant "$ATLAS_METHOD_VARIANT")
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
    )
  fi

  "${cmd[@]}"
  validate_result_schema "$method" "$result_file"
}

run_external_adapter() {
  local method="$1"
  local result_file="${RESULTS_DIR}/${method}_seed${SEED}.json"
  local adapter_script="scripts/table5_external_adapters/${method}.sh"

  if [ -s "$result_file" ]; then
    if validate_result_schema "$method" "$result_file"; then
      echo "↷ Skip ${method}: found strict-protocol result ${result_file}"
      return 0
    fi
    echo "[INFO] Existing ${method} result is stale or incomplete; external adapter must regenerate it."
  fi

  if [ ! -x "$adapter_script" ]; then
    echo "[INFO] ${method} is marked as external_adaptation."
    echo "[INFO] Adapter script not found: ${adapter_script}"
    echo "[INFO] ${METHOD_REASON[$method]}"
    write_method_status "$method" "${METHOD_STATUS[$method]}" "${METHOD_REASON[$method]}" "${METHOD_REFERENCE[$method]}"
    return 0
  fi

  echo "[INFO] Running external adapter ${adapter_script}"
  TABLE5_ADAPTER_METHOD="$method" \
  TABLE5_RESULT_FILE="$result_file" \
  TABLE5_LOG_DIR="" \
  ACDC_ROOT="$ACDC_ROOT" \
  SEGFORMER_MODEL="$SEGFORMER_MODEL" \
  SPLIT="$SPLIT" \
  INPUT_WIDTH="$INPUT_WIDTH" \
  INPUT_HEIGHT="$INPUT_HEIGHT" \
  IMAGE_SIZE="$IMAGE_SIZE" \
  ROUNDS="$ROUNDS" \
  REPORT_TIMESTAMPS="$REPORT_TIMESTAMPS" \
  SEED="$SEED" \
  GPU="$GPU" \
  bash "$adapter_script"

  if [ ! -s "$result_file" ]; then
    echo "[WARN] ${method} adapter completed but result file missing: ${result_file}"
    write_method_status "$method" "${METHOD_STATUS[$method]}" "Adapter ran but did not create ${result_file}" "${METHOD_REFERENCE[$method]}"
    return 0
  fi

  if ! validate_result_schema "$method" "$result_file"; then
    write_method_status "$method" "invalid_result_schema" "Adapter result exists but does not satisfy table5_surgeon_cotta_v1 schema; rerun/fix adapter before using it." "${METHOD_REFERENCE[$method]}"
    return 1
  fi
}

build_registry_summary() {
  local summary_file="${RESULTS_DIR}/table5_registry_seed${SEED}.json"
  $PYTHON_CMD - <<PY
import json
import os

methods = "${METHODS_CSV}".split(",")
results_dir = "${RESULTS_DIR}"
summary = {
    "protocol": "table5_surgeon_cotta_v1",
    "legacy_protocol": "table5_unified_repo_v1",
    "seed": ${SEED},
    "acdc_root": "${ACDC_ROOT}",
    "split": "${SPLIT}",
    "model_name_or_path": "${SEGFORMER_MODEL}",
    "rounds": ${ROUNDS},
    "input_width": ${INPUT_WIDTH},
    "input_height": ${INPUT_HEIGHT},
    "image_size": "${IMAGE_SIZE}",
    "normalization": {
        "type": "imagenet",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "report_timestamps": "${REPORT_TIMESTAMPS}",
    "methods": {},
}
for method in methods:
    result_file = os.path.join(results_dir, f"{method}_seed${SEED}.json")
    status_file = os.path.join(results_dir, f"{method}_seed${SEED}.status.json")
    if os.path.isfile(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            summary["methods"][method] = {"kind": "result", "payload": json.load(f)}
    elif os.path.isfile(status_file):
        with open(status_file, "r", encoding="utf-8") as f:
            summary["methods"][method] = {"kind": "status", "payload": json.load(f)}
    else:
        summary["methods"][method] = {
            "kind": "missing",
            "payload": {"method": method, "status": "missing", "reason": "No result or status file was generated."},
        }
with open("${summary_file}", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"Saved registry summary to ${summary_file}")
PY
}

run_method() {
  local method="$1"
  local status="${METHOD_STATUS[$method]:-}"

  if [ -z "$status" ]; then
    echo "[ERROR] Unknown method: ${method}"
    exit 1
  fi

  case "$status" in
    local_supported)
      echo ""
      echo ">>> Running local method: ${method}"
      run_local_method "$method"
      ;;
    external_adaptation)
      echo ""
      echo ">>> Resolving external adaptation method: ${method}"
      run_external_adapter "$method"
      ;;
    unavailable)
      echo ""
      echo ">>> Recording unavailable method: ${method}"
      write_method_status "$method" "$status" "${METHOD_REASON[$method]}" "${METHOD_REFERENCE[$method]}"
      ;;
    *)
      echo "[ERROR] Unsupported status ${status} for method ${method}"
      exit 1
      ;;
  esac
}

validate_inputs() {
  if [ ! -d "$ACDC_ROOT" ]; then
    echo "[ERROR] ACDC root not found: ${ACDC_ROOT}"
    exit 1
  fi
  if [ ! -d "$SEGFORMER_MODEL" ]; then
    echo "[WARN] Model directory not found: ${SEGFORMER_MODEL}"
    echo "[WARN] Run scripts/download_table5_models.sh first if you want to use the local model cache."
  fi
}

print_header
validate_inputs

for method in "${REQUESTED_METHODS[@]}"; do
  run_method "$method"
done

build_registry_summary

echo ""
echo "Done. Table-5 run finished."
