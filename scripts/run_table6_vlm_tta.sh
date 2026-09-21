#!/bin/bash
# =============================================================================
# Table-6/Table-7 VLM TTA reproduction on CLIP ViT-B/16.
# Prompt settings:
#   zs   : zero-shot handcrafted / prompt-ensemble setting, no supervised prompt.
#   coop : CoOp-supervised learned prompt initialization, requires COOP_CKPT.
# Methods in each prompt setting: source, tpt, tda, adadem, atlas.
# Datasets by default: ImageNet-A / ImageNet-V2 / ImageNet-R / ImageNet-Sketch.
# =============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

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
BASELINE_DIR="code/baselines/tta-vlm-main"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
ARCH="${ARCH:-ViT-B/16}"
SEED="${SEED:-1}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-4}"
PRINT_FREQ="${PRINT_FREQ:-50}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
EPISODIC_BATCH_SIZE="${EPISODIC_BATCH_SIZE:-64}"
ONLINE_BATCH_SIZE="${ONLINE_BATCH_SIZE:-1}"
N_CTX="${N_CTX:-4}"
CTX_INIT="${CTX_INIT:-a_photo_of_a}"
CLIP_CACHE_ROOT="${CLIP_CACHE_ROOT:-$PROJECT_ROOT/models/clip/openai}"
DEFAULT_COOP_CKPT="$PROJECT_ROOT/models/coop/imagenet/vit_b16_ep50_nctx4_seed1/model.pth.tar-50"
PROMPT_SETTINGS_CSV="${PROMPT_SETTINGS:-zs,coop}"
METHODS_CSV="${METHODS:-source,atlas}"
TEST_SETS="${TEST_SETS:-A/V/R/K}"
COOP_CKPT="${COOP_CKPT:-$DEFAULT_COOP_CKPT}"
SAVE_PER_SAMPLE="${SAVE_PER_SAMPLE:-0}"
ADADEM_PI="${ADADEM_PI:-0.1}"
ATLAS_ADADEM_PI="${ATLAS_ADADEM_PI:-0.1}"
ATLAS_METHOD_VARIANT="${ATLAS_METHOD_VARIANT:-full}"
ATLAS_TTA_STEPS="${ATLAS_TTA_STEPS:-1}"
ATLAS_SELECTION_P="${ATLAS_SELECTION_P:-0.1}"
ATLAS_SELECTION_QUANTILE="${ATLAS_SELECTION_QUANTILE:-0.5}"
ATLAS_LR="${ATLAS_LR:-5e-3}"
ATLAS_ENTROPY_WEIGHT="${ATLAS_ENTROPY_WEIGHT:-0.35}"
ATLAS_SOURCE_ANCHOR_WEIGHT="${ATLAS_SOURCE_ANCHOR_WEIGHT:-0.02}"
ATLAS_VIEW_CONSISTENCY_WEIGHT="${ATLAS_VIEW_CONSISTENCY_WEIGHT:-0.05}"
ATLAS_VLM_RUNTIME="${ATLAS_VLM_RUNTIME:-episodic}"
ATLAS_SOURCE_OUTPUT="${ATLAS_SOURCE_OUTPUT:-auto}"
ATLAS_OUTPUT_FUSION="${ATLAS_OUTPUT_FUSION:-confidence}"
ATLAS_ADAPTED_OUTPUT_WEIGHT="${ATLAS_ADAPTED_OUTPUT_WEIGHT:-1.0}"
ATLAS_SOURCE_OUTPUT_WEIGHT="${ATLAS_SOURCE_OUTPUT_WEIGHT:-1.0}"
ATLAS_VIEW_OUTPUT_WEIGHT="${ATLAS_VIEW_OUTPUT_WEIGHT:-0.5}"
ATLAS_FUSION_TEMPERATURE="${ATLAS_FUSION_TEMPERATURE:-2.0}"
PROMPT_ENSEMBLE_CHUNK_SIZE="${PROMPT_ENSEMBLE_CHUNK_SIZE:-256}"
WRITE_LEGACY_TABLE7_ALIAS="${WRITE_LEGACY_TABLE7_ALIAS:-0}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_table6_vlm_tta.sh [--method METHOD[,METHOD...]] [--mode zs|coop]

Options:
  --method, --methods   Scope METHODS for this run, e.g. atlas.
  --mode                Scope prompt setting. Use zs for zero-shot or coop for CoOp.
                        sz is accepted as a backwards-compatible typo alias for zs.
  --help                Show this help.

Environment compatibility:
  METHODS=atlas PROMPT_SETTINGS=zs bash scripts/run_table6_vlm_tta.sh
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --method|--methods)
      if [ "$#" -lt 2 ]; then
        echo "[ERROR] $1 requires a value." >&2
        exit 1
      fi
      METHODS_CSV="$2"
      shift 2
      ;;
    --method=*|--methods=*)
      METHODS_CSV="${1#*=}"
      shift
      ;;
    --mode|--prompt-setting|--prompt-settings)
      if [ "$#" -lt 2 ]; then
        echo "[ERROR] $1 requires a value." >&2
        exit 1
      fi
      PROMPT_SETTINGS_CSV="$2"
      shift 2
      ;;
    --mode=*|--prompt-setting=*|--prompt-settings=*)
      PROMPT_SETTINGS_CSV="${1#*=}"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unsupported argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/table6_vlm_tta"
RUN_ID="${TIMESTAMP}_seed${SEED}"
RUN_DIR="${RESULTS_DIR}/runs/${RUN_ID}"
mkdir -p "$RESULTS_DIR" "$RUN_DIR/per_dataset"

weight_file_for_arch() {
  case "$1" in
    "ViT-B/16") echo "ViT-B-16.pt" ;;
    "ViT-L/14") echo "ViT-L-14.pt" ;;
    *)
      echo "[ERROR] Unsupported arch for official CLIP cache: $1" >&2
      exit 1
      ;;
  esac
}

normalize_prompt_setting() {
  case "$1" in
    zs|sz|zero|zero-shot|zeroshot|ZeroShot|zero_shot) echo "zs" ;;
    coop|CoOp|COOP) echo "coop" ;;
    *)
      echo "[ERROR] Unsupported prompt setting: $1. Use PROMPT_SETTINGS=zs,coop." >&2
      exit 1
      ;;
  esac
}

prompt_setting_label() {
  case "$1" in
    zs) echo "Zero-Shot" ;;
    coop) echo "CoOp" ;;
    *) echo "$1" ;;
  esac
}

coop_ctx_dim_for_arch() {
  case "$1" in
    "ViT-B/16") echo "512" ;;
    "ViT-L/14") echo "768" ;;
    *)
      echo "[ERROR] Unsupported arch for CoOp checkpoint validation: $1" >&2
      exit 1
      ;;
  esac
}

normalize_method() {
  case "$1" in
    source|baseline) echo "source" ;;
    tpt|TPT) echo "tpt" ;;
    tda|TDA) echo "tda" ;;
    adadem|AdaDEM|dem|DEM) echo "adadem" ;;
    atlas|ATLAS) echo "atlas" ;;
    coop|coop_adadem)
      echo "[ERROR] '${1}' is now a prompt setting, not a standalone method. Use PROMPT_SETTINGS=coop METHODS=source or METHODS=adadem." >&2
      exit 1
      ;;
    *)
      echo "[ERROR] Unsupported method: $1. Use METHODS=source,tpt,tda,adadem,atlas." >&2
      exit 1
      ;;
  esac
}

IFS=',' read -r -a RAW_PROMPT_SETTINGS <<< "$PROMPT_SETTINGS_CSV"
REQUESTED_PROMPT_SETTINGS=()
for raw_setting in "${RAW_PROMPT_SETTINGS[@]}"; do
  raw_setting="${raw_setting// /}"
  if [ -n "$raw_setting" ]; then
    REQUESTED_PROMPT_SETTINGS+=("$(normalize_prompt_setting "$raw_setting")")
  fi
done

IFS=',' read -r -a RAW_METHODS <<< "$METHODS_CSV"
REQUESTED_METHODS=()
for raw_method in "${RAW_METHODS[@]}"; do
  raw_method="${raw_method// /}"
  if [ -n "$raw_method" ]; then
    REQUESTED_METHODS+=("$(normalize_method "$raw_method")")
  fi
done

if [ "${#REQUESTED_PROMPT_SETTINGS[@]}" -eq 0 ]; then
  echo "[ERROR] No prompt settings requested."
  exit 1
fi
if [ "${#REQUESTED_METHODS[@]}" -eq 0 ]; then
  echo "[ERROR] No methods requested."
  exit 1
fi

if [ "$ATLAS_VLM_RUNTIME" != "episodic" ] && [ "$ATLAS_VLM_RUNTIME" != "instance" ]; then
  echo "[ERROR] Unsupported ATLAS_VLM_RUNTIME=${ATLAS_VLM_RUNTIME}. The live Table 6 ATLAS path is episodic only." >&2
  exit 1
fi

PROMPT_SETTINGS_JOINED=$(IFS=','; echo "${REQUESTED_PROMPT_SETTINGS[*]}")
METHODS_JOINED=$(IFS=','; echo "${REQUESTED_METHODS[*]}")
EXPECTED_RESULT_COUNT=$((${#REQUESTED_PROMPT_SETTINGS[@]} * ${#REQUESTED_METHODS[@]}))

CANONICAL_TABLE6_TEST_SETS="A/V/R/K"
CANONICAL_TABLE6_PROMPT_SETTINGS="zs,coop"
CANONICAL_TABLE6_METHODS="source,tpt,tda,adadem,atlas"

PARSE_ONLY_PUBLISH_TABLES=0
if [ "$PROMPT_SETTINGS_JOINED" = "$CANONICAL_TABLE6_PROMPT_SETTINGS" ] \
  && [ "$METHODS_JOINED" = "$CANONICAL_TABLE6_METHODS" ]; then
  PARSE_ONLY_PUBLISH_TABLES=1
fi

if [ "${TABLE6_PARSE_ONLY:-0}" = "1" ]; then
  echo "Prompt Settings: ${PROMPT_SETTINGS_JOINED}"
  echo "Methods:         ${METHODS_JOINED}"
  echo "Expected Rows:   ${EXPECTED_RESULT_COUNT}"
  echo "Publish Tables:  ${PARSE_ONLY_PUBLISH_TABLES}"
  exit 0
fi

NEEDS_COOP=0
for prompt_setting in "${REQUESTED_PROMPT_SETTINGS[@]}"; do
  if [ "$prompt_setting" = "coop" ]; then
    NEEDS_COOP=1
  fi
done
if [ "$NEEDS_COOP" = "1" ]; then
  if [ "$ARCH" != "ViT-B/16" ] && [ "$COOP_CKPT" = "$DEFAULT_COOP_CKPT" ]; then
    echo "[ERROR] Default CoOp checkpoint is only prepared for ViT-B/16."
    echo "[ERROR] Set COOP_CKPT to a checkpoint compatible with ARCH=${ARCH}."
    exit 1
  fi
  if [ -z "$COOP_CKPT" ] || [ ! -f "$COOP_CKPT" ]; then
    echo "[ERROR] CoOp prompt setting requires an ImageNet-trained CoOp ViT-B/16 M=4 checkpoint."
    echo "[ERROR] Expected default checkpoint: $DEFAULT_COOP_CKPT"
    echo "[ERROR] Run scripts/download_table6_coop_weights.sh first, or set COOP_CKPT=/absolute/path/to/coop_checkpoint.pth.tar."
    echo "[ERROR] Official CoOp model zoo: https://github.com/KaiyangZhou/CoOp"
    exit 1
  fi
  COOP_CTX_DIM="$(coop_ctx_dim_for_arch "$ARCH")"
  "$PYTHON_CMD" - "$COOP_CKPT" "$N_CTX" "$COOP_CTX_DIM" <<'PY'
import sys
import torch

path = sys.argv[1]
n_ctx = int(sys.argv[2])
ctx_dim = int(sys.argv[3])
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
keys = ["ctx", "prompt_learner.ctx", "module.prompt_learner.ctx", "model.prompt_learner.ctx"]
ctx = None
for key in keys:
    if isinstance(state, dict) and key in state:
        ctx = state[key]
        break
if ctx is None:
    raise SystemExit(f"[ERROR] Could not find CoOp prompt context in {path}")
if ctx.ndim == 3 and ctx.shape[0] == 1:
    ctx = ctx.squeeze(0)
expected = (n_ctx, ctx_dim)
if tuple(ctx.shape) != expected:
    raise SystemExit(f"[ERROR] CoOp prompt shape mismatch: checkpoint {tuple(ctx.shape)} vs runner {expected} in {path}")
print(f"Verified CoOp checkpoint: {path} ctx_shape={tuple(ctx.shape)}")
PY
fi

EXPECTED_WEIGHT_FILE="${CLIP_CACHE_ROOT}/$(weight_file_for_arch "$ARCH")"
if [ ! -f "$EXPECTED_WEIGHT_FILE" ]; then
  echo "[ERROR] Missing official CLIP weight file: $EXPECTED_WEIGHT_FILE"
  echo "[ERROR] Run scripts/download_table6_vlm_models.sh first."
  exit 1
fi

echo "=============================================="
echo "VLM TTA Prompt-Setting Matrix (CLIP ${ARCH})"
echo "=============================================="
echo "Conda Env:       sata"
echo "Baseline:        ${BASELINE_DIR}"
echo "Data Root:       ${DATA_ROOT}"
echo "Test Sets:       ${TEST_SETS}"
echo "Prompt Settings: ${PROMPT_SETTINGS_JOINED}"
echo "Methods:         ${METHODS_JOINED}"
echo "Seed:            ${SEED}"
echo "GPU:             ${GPU}"
echo "Workers:         ${WORKERS}"
echo "MaxSample:       ${MAX_SAMPLES}"
echo "N Ctx:           ${N_CTX}"
echo "Ctx Init:        ${CTX_INIT}"
echo "CLIP Root:       ${CLIP_CACHE_ROOT}"
echo "CoOp Ckpt:       ${COOP_CKPT:-<not set>}"
echo "Expected Rows:   ${EXPECTED_RESULT_COUNT}"
echo "Run ID:          ${RUN_ID}"
echo "Results:         ${RESULTS_DIR}"
echo "=============================================="

PRECHECK_FILE="${RUN_DIR}/preflight_seed${SEED}.json"
TTA_VLM_CLIP_DOWNLOAD_ROOT="$CLIP_CACHE_ROOT" "$PYTHON_CMD" - <<PY
import json
import os
import sys

project_root = "${PROJECT_ROOT}"
data_root = "${DATA_ROOT}"
requested_sets = [item for item in "${TEST_SETS}".split("/") if item]
sys.path.insert(0, os.path.join(project_root, "code", "baselines", "tta-vlm-main"))
from data.datautils import resolve_dataset_spec

payload = {
    "protocol": "vlm_tta_prompt_setting_matrix_v1",
    "requested_prompt_settings": "${PROMPT_SETTINGS_JOINED}".split(","),
    "requested_methods": "${METHODS_JOINED}".split(","),
    "requested_test_sets": requested_sets,
    "available_test_sets": [],
    "skipped_test_sets": {},
    "clip_cache_root": "${CLIP_CACHE_ROOT}",
    "required_weight_file": "${EXPECTED_WEIGHT_FILE}",
    "coop_ckpt": "${COOP_CKPT}",
    "n_ctx": int("${N_CTX}"),
    "ctx_init": "${CTX_INIT}",
    "skip_imagenet_1k": True,
}

for set_id in requested_sets:
    if set_id == "I":
        payload["skipped_test_sets"][set_id] = "ImageNet-1K is intentionally skipped for the requested OOD reproduction."
        continue
    try:
        spec = resolve_dataset_spec(set_id, data_root)
        payload["available_test_sets"].append(set_id)
        payload[set_id] = spec
    except Exception as exc:
        payload["skipped_test_sets"][set_id] = str(exc)

with open("${PRECHECK_FILE}", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

AVAILABLE_TEST_SETS="$($PYTHON_CMD - <<PY
import json
with open("${PRECHECK_FILE}", "r", encoding="utf-8") as f:
    payload = json.load(f)
print("/".join(payload["available_test_sets"]))
PY
)"

if [ -z "$AVAILABLE_TEST_SETS" ]; then
  echo "[ERROR] No runnable OOD test set is available under ${DATA_ROOT}"
  exit 1
fi

IFS='/' read -r -a EXECUTABLE_TEST_SETS <<< "$AVAILABLE_TEST_SETS"

RESULT_CACHE_PROTOCOL="vlm_tta_result_cache_v2"
PUBLISH_METHOD_ALIASES=0
PUBLISH_TABLE_ALIASES=0

if [ "$AVAILABLE_TEST_SETS" = "$CANONICAL_TABLE6_TEST_SETS" ]; then
  PUBLISH_METHOD_ALIASES=1
fi

if [ "$PUBLISH_METHOD_ALIASES" = "1" ] \
  && [ "$PROMPT_SETTINGS_JOINED" = "$CANONICAL_TABLE6_PROMPT_SETTINGS" ] \
  && [ "$METHODS_JOINED" = "$CANONICAL_TABLE6_METHODS" ]; then
  PUBLISH_TABLE_ALIASES=1
fi

if [ "$PUBLISH_METHOD_ALIASES" != "1" ]; then
  echo "[INFO] Scoped dataset run detected; top-level row aliases will not be updated."
fi
if [ "$PUBLISH_TABLE_ALIASES" != "1" ]; then
  echo "[INFO] Scoped prompt/method/dataset run detected; top-level table aliases will not be updated."
fi

method_summary_is_reusable() {
  local summary_json="$1"
  local prompt_setting="$2"
  local method="$3"

  if [ ! -s "$summary_json" ]; then
  return 1
  fi

  "$PYTHON_CMD" - \
  "$summary_json" \
  "$prompt_setting" \
  "$method" \
  "$ARCH" \
  "$SEED" \
  "$MAX_SAMPLES" \
  "$N_CTX" \
  "$CTX_INIT" \
  "$COOP_CKPT" \
  "$AVAILABLE_TEST_SETS" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

(
  summary_path,
  prompt_setting,
  method,
  arch,
  seed,
  max_samples,
  n_ctx,
  ctx_init,
  coop_ckpt,
  available_test_sets,
) = sys.argv[1:]

try:
  summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
except Exception:
  raise SystemExit(1)

expected_sets = [item for item in available_test_sets.split("/") if item]
row_id = f"{prompt_setting}_{method}"

if not expected_sets:
  raise SystemExit(1)
if summary.get("protocol") != "vlm_tta_prompt_setting_matrix_v1":
  raise SystemExit(1)
if summary.get("row_id") != row_id:
  raise SystemExit(1)
if summary.get("prompt_setting") != prompt_setting or summary.get("method") != method:
  raise SystemExit(1)
if summary.get("backbone") != arch:
  raise SystemExit(1)
if int(summary.get("seed", -1)) != int(seed):
  raise SystemExit(1)
if int(summary.get("n_ctx", -1)) != int(n_ctx):
  raise SystemExit(1)
if summary.get("ctx_init") != ctx_init:
  raise SystemExit(1)
if prompt_setting == "coop" and summary.get("coop_ckpt") != coop_ckpt:
  raise SystemExit(1)
if summary.get("effective_test_sets") != expected_sets:
  raise SystemExit(1)

per_dataset = summary.get("per_dataset")
if not isinstance(per_dataset, list) or len(per_dataset) != len(expected_sets):
  raise SystemExit(1)

for dataset_id, item in zip(expected_sets, per_dataset):
  if not isinstance(item, dict):
    raise SystemExit(1)
  if item.get("dataset") != dataset_id:
    raise SystemExit(1)
  if item.get("row_id") != row_id:
    raise SystemExit(1)
  if item.get("prompt_setting") != prompt_setting or item.get("method") != method:
    raise SystemExit(1)
  if item.get("backbone") != arch:
    raise SystemExit(1)
  if int(item.get("seed", -1)) != int(seed):
    raise SystemExit(1)
  if int(item.get("max_samples", -1)) != int(max_samples):
    raise SystemExit(1)
  if int(item.get("n_ctx", -1)) != int(n_ctx):
    raise SystemExit(1)
  if item.get("ctx_init") != ctx_init:
    raise SystemExit(1)
  metrics = item.get("metrics")
  if not isinstance(metrics, dict):
    raise SystemExit(1)
  if "tta_clean_acc" not in metrics and "clean_acc" not in metrics:
    raise SystemExit(1)
PY
}

prepare_reusable_method_summary() {
  local source_json="$1"
  local target_json="$2"

  "$PYTHON_CMD" - \
  "$source_json" \
  "$target_json" \
  "$PRECHECK_FILE" \
  "$RUN_ID" \
  "$CLIP_CACHE_ROOT" \
  "$COOP_CKPT" <<'PY'
import json
import sys
from pathlib import Path

source_path, target_path, precheck_path, run_id, clip_cache_root, coop_ckpt = sys.argv[1:]

summary = json.loads(Path(source_path).read_text(encoding="utf-8"))
preflight = json.loads(Path(precheck_path).read_text(encoding="utf-8"))
summary["run_id"] = run_id
summary["requested_test_sets"] = preflight["requested_test_sets"]
summary["effective_test_sets"] = preflight["available_test_sets"]
summary["skipped_test_sets"] = preflight["skipped_test_sets"]
summary["clip_cache_root"] = clip_cache_root
summary["coop_ckpt"] = coop_ckpt

output_path = Path(target_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Prepared reusable method summary at {output_path}")
PY
}

result_log_is_reusable() {
  local results_pt="$1"
  local prompt_setting="$2"
  local method="$3"
  local dataset_id="$4"
  local algorithm="$5"
  local runtime_setting="$6"
  local entrypoint="$7"
  local batch_size="$8"
  local metadata_pt
  local source_signature

  if [ ! -s "$results_pt" ]; then
    return 1
  fi

  metadata_pt="$(result_cache_metadata_path "$results_pt")"
  if [ ! -s "$metadata_pt" ]; then
  return 1
  fi

  source_signature="$(build_result_cache_signature "$prompt_setting" "$method" "$entrypoint")"

  TTA_VLM_CLIP_DOWNLOAD_ROOT="$CLIP_CACHE_ROOT" "$PYTHON_CMD" - \
  "$results_pt" \
  "$metadata_pt" \
  "$RESULT_CACHE_PROTOCOL" \
  "$prompt_setting" \
  "$method" \
  "$dataset_id" \
  "$algorithm" \
  "$runtime_setting" \
  "$entrypoint" \
  "$batch_size" \
  "$ARCH" \
  "$SEED" \
  "$MAX_SAMPLES" \
  "$N_CTX" \
  "$CTX_INIT" \
  "$COOP_CKPT" \
  "$source_signature" <<'PY' >/dev/null 2>&1
import sys
import json
import torch

(
  results_path,
  metadata_path,
  expected_protocol,
  prompt_setting,
  method,
  dataset_id,
  algorithm,
  runtime_setting,
  entrypoint,
  batch_size,
  arch,
  seed,
  max_samples,
  n_ctx,
  ctx_init,
  coop_ckpt,
  source_signature,
) = sys.argv[1:]

try:
  payload = torch.load(results_path, map_location="cpu")
except Exception:
    raise SystemExit(1)

if not isinstance(payload, dict):
    raise SystemExit(1)

if "tta_clean_acc" not in payload and "clean_acc" not in payload:
    raise SystemExit(1)

try:
  with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)
except Exception:
  raise SystemExit(1)

expected = {
  "cache_protocol": expected_protocol,
  "prompt_setting": prompt_setting,
  "method": method,
  "dataset": dataset_id,
  "backend_algorithm": algorithm,
  "runtime_setting": runtime_setting,
  "entrypoint": entrypoint,
  "batch_size": int(batch_size),
  "backbone": arch,
  "seed": int(seed),
  "max_samples": int(max_samples),
  "n_ctx": int(n_ctx),
  "ctx_init": ctx_init,
  "source_signature": source_signature,
}
for key, expected_value in expected.items():
  if metadata.get(key) != expected_value:
    raise SystemExit(1)

if prompt_setting == "coop":
  if metadata.get("coop_ckpt") != coop_ckpt:
    raise SystemExit(1)
elif metadata.get("coop_ckpt") not in (None, ""):
  raise SystemExit(1)
PY
}

result_cache_metadata_path() {
  local results_pt="$1"
  printf '%s.cache.json' "$results_pt"
}

build_result_cache_signature() {
  local prompt_setting="$1"
  local method="$2"
  local entrypoint="$3"

  TTA_VLM_CLIP_DOWNLOAD_ROOT="$CLIP_CACHE_ROOT" "$PYTHON_CMD" - \
  "$PROJECT_ROOT" \
  "$prompt_setting" \
  "$method" \
  "$entrypoint" \
  "$COOP_CKPT" <<'PY'
import json
import os
import sys

project_root, prompt_setting, method, entrypoint, coop_ckpt = sys.argv[1:]

dependencies = [
  os.path.join(project_root, "scripts", "run_table6_vlm_tta.sh"),
  os.path.join(project_root, "code", "baselines", "tta-vlm-main", entrypoint),
  os.path.join(project_root, "code", "baselines", "tta-vlm-main", "clip", "custom_clip.py"),
  os.path.join(project_root, "code", "baselines", "tta-vlm-main", "data", "datautils.py"),
]

method_dependencies = {
  "source": [
    os.path.join(project_root, "code", "baselines", "tta-vlm-main", "instance_method", "source.py"),
    os.path.join(project_root, "code", "atlas", "vlm_prompt_ensemble.py"),
  ],
  "tpt": [
    os.path.join(project_root, "code", "baselines", "tta-vlm-main", "instance_method", "tpt.py"),
  ],
  "adadem": [
    os.path.join(project_root, "code", "baselines", "tta-vlm-main", "instance_method", "tpt.py"),
  ],
  "tda": [
    os.path.join(project_root, "code", "baselines", "tta-vlm-main", "online_method", "tda.py"),
  ],
  "atlas": [
    os.path.join(project_root, "code", "atlas", "vlm_instance.py"),
    os.path.join(project_root, "code", "atlas", "vlm_prompt_ensemble.py"),
  ],
}
dependencies.extend(method_dependencies.get(method, []))
if prompt_setting == "coop":
  dependencies.append(coop_ckpt)

records = []
for path in dependencies:
  stat = os.stat(path)
  records.append({
    "path": os.path.relpath(path, project_root),
    "size": stat.st_size,
    "mtime_ns": stat.st_mtime_ns,
  })

print(json.dumps(records, sort_keys=True, separators=(",", ":")))
PY
}

write_result_cache_metadata() {
  local results_pt="$1"
  local prompt_setting="$2"
  local method="$3"
  local dataset_id="$4"
  local algorithm="$5"
  local runtime_setting="$6"
  local entrypoint="$7"
  local batch_size="$8"
  local metadata_pt
  local source_signature

  metadata_pt="$(result_cache_metadata_path "$results_pt")"
  source_signature="$(build_result_cache_signature "$prompt_setting" "$method" "$entrypoint")"

  SOURCE_SIGNATURE="$source_signature" \
  TTA_VLM_CLIP_DOWNLOAD_ROOT="$CLIP_CACHE_ROOT" "$PYTHON_CMD" - <<PY
import json
import os

source_signature = os.environ["SOURCE_SIGNATURE"]
metadata = {
  "cache_protocol": "${RESULT_CACHE_PROTOCOL}",
  "results_pt": "${results_pt}",
  "prompt_setting": "${prompt_setting}",
  "method": "${method}",
  "dataset": "${dataset_id}",
  "backend_algorithm": "${algorithm}",
  "runtime_setting": "${runtime_setting}",
  "entrypoint": "${entrypoint}",
  "batch_size": int("${batch_size}"),
  "backbone": "${ARCH}",
  "seed": int("${SEED}"),
  "max_samples": int("${MAX_SAMPLES}"),
  "n_ctx": int("${N_CTX}"),
  "ctx_init": "${CTX_INIT}",
  "coop_ckpt": "${COOP_CKPT}" if "${prompt_setting}" == "coop" else "",
  "source_signature": source_signature,
}
with open("${metadata_pt}", "w", encoding="utf-8") as f:
  json.dump(metadata, f, ensure_ascii=False, indent=2)
PY
}

run_single_dataset() {
  local prompt_setting="$1"
  local method="$2"
  local dataset_id="$3"
  local runtime_setting
  local entrypoint
  local algorithm
  local batch_size
  local row_id="${prompt_setting}_${method}"
  local output_root="$PROJECT_ROOT/$RESULTS_DIR/raw/${prompt_setting}/${method}/${dataset_id}"
  local per_dataset_json="$RUN_DIR/per_dataset/${row_id}_${dataset_id}_seed${SEED}.json"
  local results_pt
  local prompt_label
  local display_method
  local -a extra_args=()

  prompt_label="$(prompt_setting_label "$prompt_setting")"
  case "$method" in
    source) display_method="$prompt_label" ;;
    tpt) display_method="TPT" ;;
    tda) display_method="TDA" ;;
    adadem) display_method="AdaDEM" ;;
    atlas) display_method="ATLAS (ours)" ;;
    *) display_method="$method" ;;
  esac
  if [ "$prompt_setting" = "coop" ] && [ "$method" != "source" ]; then
    if [ "$method" = "atlas" ]; then
      display_method="ATLAS (CoOp)"
    else
      display_method="${display_method} (CoOp)"
    fi
  fi

  case "$method" in
    source)
      runtime_setting="episodic"
      entrypoint="instance_tta.py"
      batch_size="$EPISODIC_BATCH_SIZE"
      if [ "$prompt_setting" = "coop" ]; then
        algorithm="coop"
      else
        algorithm="source"
      fi
      ;;
    tpt)
      runtime_setting="episodic"
      entrypoint="instance_tta.py"
      algorithm="tpt"
      batch_size="$EPISODIC_BATCH_SIZE"
      ;;
    adadem)
      runtime_setting="episodic"
      entrypoint="instance_tta.py"
      algorithm="tpt"
      batch_size="$EPISODIC_BATCH_SIZE"
      extra_args+=(--adadem --adadem_pi "$ADADEM_PI")
      ;;
    tda)
      runtime_setting="online"
      entrypoint="online_tta.py"
      algorithm="tda"
      batch_size="$ONLINE_BATCH_SIZE"
      ;;
    atlas)
      algorithm="atlas"
      runtime_setting="episodic"
      entrypoint="instance_tta.py"
      batch_size="$EPISODIC_BATCH_SIZE"
      extra_args+=(
        --tta_steps "$ATLAS_TTA_STEPS"
        --selection_p "$ATLAS_SELECTION_P"
        --lr "$ATLAS_LR"
        --atlas_adadem_pi "$ATLAS_ADADEM_PI"
        --atlas_method_variant "$ATLAS_METHOD_VARIANT"
        --atlas_selection_quantile "$ATLAS_SELECTION_QUANTILE"
        --atlas_entropy_weight "$ATLAS_ENTROPY_WEIGHT"
        --atlas_source_anchor_weight "$ATLAS_SOURCE_ANCHOR_WEIGHT"
        --atlas_view_consistency_weight "$ATLAS_VIEW_CONSISTENCY_WEIGHT"
        --atlas_source_output "$ATLAS_SOURCE_OUTPUT"
        --atlas_output_fusion "$ATLAS_OUTPUT_FUSION"
        --atlas_adapted_output_weight "$ATLAS_ADAPTED_OUTPUT_WEIGHT"
        --atlas_source_output_weight "$ATLAS_SOURCE_OUTPUT_WEIGHT"
        --atlas_view_output_weight "$ATLAS_VIEW_OUTPUT_WEIGHT"
        --atlas_fusion_temperature "$ATLAS_FUSION_TEMPERATURE"
      )
      ;;
    *)
      echo "[ERROR] Unsupported method after normalization: ${method}"
      exit 1
      ;;
  esac

  if [ "$prompt_setting" = "coop" ]; then
    extra_args+=(--load "$COOP_CKPT")
  fi

  if [ "$runtime_setting" = "episodic" ]; then
    results_pt="${output_root}/${ARCH}/seed_${SEED}/${dataset_id}/results_log.pt"
  else
    results_pt="${output_root}/bs${batch_size}/${ARCH}/${dataset_id}/results_log.pt"
  fi

  if [ "$SAVE_PER_SAMPLE" = "1" ]; then
    extra_args+=(--save_per_sample)
  fi

  if [ "${RUN_FORCE:-0}" != "1" ] && result_log_is_reusable "$results_pt" "$prompt_setting" "$method" "$dataset_id" "$algorithm" "$runtime_setting" "$entrypoint" "$batch_size"; then
    echo "[SKIP] Reusing existing result log: ${results_pt}"
  else
    if [ "${RUN_FORCE:-0}" != "1" ] && [ -e "$results_pt" ]; then
      echo "[WARN] Existing result log is unreadable, incomplete, or stale/incompatible; rerunning: ${results_pt}"
    fi

    pushd "$BASELINE_DIR" >/dev/null
    TTA_VLM_CLIP_DOWNLOAD_ROOT="$CLIP_CACHE_ROOT" "$PYTHON_CMD" "$entrypoint" \
      --data "$DATA_ROOT" \
      --test_sets "$dataset_id" \
      -a "$ARCH" \
      -b "$batch_size" \
      -j "$WORKERS" \
      --gpu "$GPU" \
      --seed "$SEED" \
      --ctx_init "$CTX_INIT" \
      --n_ctx "$N_CTX" \
      --tta_steps 1 \
      --selection_p 0.1 \
      --lr 5e-3 \
      --max_samples "$MAX_SAMPLES" \
      -p "$PRINT_FREQ" \
      --algorithm "$algorithm" \
      --prompt_ensemble_chunk_size "$PROMPT_ENSEMBLE_CHUNK_SIZE" \
      --output_dir "$output_root" \
      "${extra_args[@]}"
    popd >/dev/null
  fi

  write_result_cache_metadata "$results_pt" "$prompt_setting" "$method" "$dataset_id" "$algorithm" "$runtime_setting" "$entrypoint" "$batch_size"

  TTA_VLM_CLIP_DOWNLOAD_ROOT="$CLIP_CACHE_ROOT" "$PYTHON_CMD" - <<PY
import json
import os
import torch

results_pt = "${results_pt}"
if not os.path.isfile(results_pt):
    raise RuntimeError(f"Missing result log: {results_pt}")
payload = torch.load(results_pt, map_location="cpu")

def to_jsonable(value):
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return str(value)

summary = {
    "row_id": "${row_id}",
    "prompt_setting": "${prompt_setting}",
    "prompt_setting_label": "${prompt_label}",
    "prompt_supervision": "CoOp-supervised learned prompt" if "${prompt_setting}" == "coop" else "zero-shot prompt",
    "method": "${method}",
    "display_method": "${display_method}",
    "backend_algorithm": "${algorithm}",
    "runtime_setting": "${runtime_setting}",
    "dataset": "${dataset_id}",
    "backbone": "${ARCH}",
    "batch_size": int("${batch_size}"),
    "seed": int("${SEED}"),
    "max_samples": int("${MAX_SAMPLES}"),
    "n_ctx": int("${N_CTX}"),
    "ctx_init": "${CTX_INIT}",
    "results_pt": results_pt,
    "metrics": {key: to_jsonable(value) for key, value in payload.items()},
}
with open("${PROJECT_ROOT}/${per_dataset_json}", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

build_method_summary() {
  local prompt_setting="$1"
  local method="$2"
  local row_id="${prompt_setting}_${method}"
  local summary_file="$RUN_DIR/${row_id}_seed${SEED}.json"

  "$PYTHON_CMD" - <<PY
import json
from pathlib import Path

preflight = json.loads(Path("${PRECHECK_FILE}").read_text(encoding="utf-8"))
dataset_order = preflight["available_test_sets"]
per_dataset = []
for dataset in dataset_order:
    path = Path("${PROJECT_ROOT}/${RUN_DIR}/per_dataset/${row_id}_" + dataset + "_seed${SEED}.json")
    if path.exists():
        per_dataset.append(json.loads(path.read_text(encoding="utf-8")))

def metric_value(item):
    metrics = item["metrics"]
    return metrics.get("tta_clean_acc", metrics.get("clean_acc"))

values = [metric_value(item) for item in per_dataset if metric_value(item) is not None]
summary = {
    "protocol": "vlm_tta_prompt_setting_matrix_v1",
    "row_id": "${row_id}",
    "prompt_setting": "${prompt_setting}",
    "method": "${method}",
    "display_method": per_dataset[0].get("display_method", "${method}") if per_dataset else "${method}",
    "backbone": "${ARCH}",
    "seed": int("${SEED}"),
    "run_id": "${RUN_ID}",
    "n_ctx": int("${N_CTX}"),
    "ctx_init": "${CTX_INIT}",
    "requested_test_sets": preflight["requested_test_sets"],
    "effective_test_sets": [item["dataset"] for item in per_dataset],
    "skipped_test_sets": preflight["skipped_test_sets"],
    "clip_cache_root": "${CLIP_CACHE_ROOT}",
    "coop_ckpt": "${COOP_CKPT}",
    "per_dataset": per_dataset,
    "ood_average": float(sum(values) / len(values)) if values else None,
}
summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
summary_path = Path("${PROJECT_ROOT}/${summary_file}")
summary_path.write_text(summary_text, encoding="utf-8")
print(f"Saved method summary to {summary_path}")

publish_method_alias = "${PUBLISH_METHOD_ALIASES}" == "1" and len(per_dataset) == len(dataset_order)
if publish_method_alias:
    alias_path = Path("${PROJECT_ROOT}/${RESULTS_DIR}/${row_id}_seed${SEED}.json")
    alias_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved top-level method alias to {alias_path}")
else:
    print("Skipped top-level method alias update for scoped or incomplete dataset coverage.")
PY
}

build_final_summary() {
  "$PYTHON_CMD" - <<PY
import json
from pathlib import Path

result_root = Path("${PROJECT_ROOT}/${RESULTS_DIR}")
run_dir = Path("${PROJECT_ROOT}/${RUN_DIR}")
preflight = json.loads(Path("${PRECHECK_FILE}").read_text(encoding="utf-8"))
dataset_order = preflight["available_test_sets"]
dataset_labels = {
    "A": "ImageNet-A",
    "V": "ImageNet-V2",
    "R": "ImageNet-R",
    "K": "ImageNet-Sketch",
}
prompt_labels = {
    "zs": "Zero-Shot",
    "coop": "CoOp",
}
prompt_settings = [item for item in "${PROMPT_SETTINGS_JOINED}".split(",") if item]
methods = [item for item in "${METHODS_JOINED}".split(",") if item]
method_summaries = []
for prompt_setting in prompt_settings:
    for method in methods:
        row_id = f"{prompt_setting}_{method}"
        path = run_dir / f"{row_id}_seed${SEED}.json"
        if path.exists():
            method_summaries.append(json.loads(path.read_text(encoding="utf-8")))

expected_result_count = len(prompt_settings) * len(methods)
if len(method_summaries) != expected_result_count:
    raise RuntimeError(
        f"Expected {expected_result_count} method summaries, found {len(method_summaries)} in {run_dir}"
    )

payload = {
    "protocol": "vlm_tta_prompt_setting_matrix_v1",
    "run_id": "${RUN_ID}",
    "backbone": "${ARCH}",
    "seed": int("${SEED}"),
    "max_samples": int("${MAX_SAMPLES}"),
    "n_ctx": int("${N_CTX}"),
    "ctx_init": "${CTX_INIT}",
    "requested_prompt_settings": prompt_settings,
    "requested_methods": methods,
    "expected_result_count": expected_result_count,
    "result_count": len(method_summaries),
    "requested_test_sets": preflight["requested_test_sets"],
    "effective_test_sets": dataset_order,
    "skipped_test_sets": preflight["skipped_test_sets"],
    "coop_ckpt": "${COOP_CKPT}",
    "method_summaries": method_summaries,
}

json_text = json.dumps(payload, ensure_ascii=False, indent=2)
run_json_path = run_dir / "table6_vlm_tta_vitb16_seed${SEED}.json"
run_json_path.write_text(json_text, encoding="utf-8")
print(f"Saved run JSON to {run_json_path}")

publish_table_aliases = "${PUBLISH_TABLE_ALIASES}" == "1"
write_legacy_table7_alias = "${WRITE_LEGACY_TABLE7_ALIAS}" == "1"
legacy_json_path = result_root / "table7_vitb16_seed${SEED}.json"
legacy_md_path = result_root / "table7_vitb16_seed${SEED}.md"

if legacy_json_path.exists() and not write_legacy_table7_alias:
  print(f"[WARN] Legacy alias exists and may be stale: {legacy_json_path}")
if legacy_md_path.exists() and not write_legacy_table7_alias:
  print(f"[WARN] Legacy alias exists and may be stale: {legacy_md_path}")

if publish_table_aliases:
  json_path = result_root / "table6_vlm_tta_vitb16_seed${SEED}.json"
  json_path.write_text(json_text, encoding="utf-8")
  print(f"Saved final JSON to {json_path}")
  if write_legacy_table7_alias:
    legacy_json_path.write_text(json_text, encoding="utf-8")
    print(f"Saved legacy JSON alias to {legacy_json_path}")
else:
  print("Skipped top-level final summary alias update for scoped prompt/method/dataset run.")

header = ["Method"] + [dataset_labels.get(dataset, dataset) for dataset in dataset_order] + ["OOD Average"]
rows = []
for summary in method_summaries:
    by_dataset = {item["dataset"]: item for item in summary["per_dataset"]}
    row = [summary.get("display_method", summary["method"])]
    for dataset in dataset_order:
        item = by_dataset.get(dataset)
        if item is None:
            row.append("-")
            continue
        metrics = item["metrics"]
        value = metrics.get("tta_clean_acc", metrics.get("clean_acc"))
        row.append("-" if value is None else f"{float(value):.2f}")
    avg = summary.get("ood_average")
    row.append("-" if avg is None else f"{float(avg):.2f}")
    rows.append(row)

lines = [
    "# VLM TTA Prompt-Setting Matrix Summary",
    "",
    f"- Run ID: {payload['run_id']}",
    f"- Backbone: {payload['backbone']}",
    f"- Seed: {payload['seed']}",
    f"- Max samples: {payload['max_samples']}",
    f"- Prompt context: n_ctx={payload['n_ctx']}, ctx_init={payload['ctx_init']}",
    f"- Result rows: {payload['result_count']} / {payload['expected_result_count']}",
    f"- CoOp checkpoint: {payload['coop_ckpt'] or '<not set>'}",
    "",
    "| " + " | ".join(header) + " |",
    "| " + " | ".join(["---"] * len(header)) + " |",
]
for row in rows:
    lines.append("| " + " | ".join(row) + " |")
lines.append("")
if preflight["skipped_test_sets"]:
    lines.append("## Skipped Test Sets")
    for key, value in preflight["skipped_test_sets"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")

md_text = "\n".join(lines)
run_md_path = run_dir / "table6_vlm_tta_vitb16_seed${SEED}.md"
run_md_path.write_text(md_text, encoding="utf-8")
print(f"Saved run Markdown to {run_md_path}")

if publish_table_aliases:
  md_path = result_root / "table6_vlm_tta_vitb16_seed${SEED}.md"
  md_path.write_text(md_text, encoding="utf-8")
  print(f"Saved final Markdown to {md_path}")
  if write_legacy_table7_alias:
    legacy_md_path.write_text(md_text, encoding="utf-8")
    print(f"Saved legacy Markdown alias to {legacy_md_path}")
PY
}

for prompt_setting in "${REQUESTED_PROMPT_SETTINGS[@]}"; do
  for method in "${REQUESTED_METHODS[@]}"; do
    row_id="${prompt_setting}_${method}"
    existing_summary_json="$PROJECT_ROOT/$RESULTS_DIR/${row_id}_seed${SEED}.json"
    run_summary_json="$PROJECT_ROOT/$RUN_DIR/${row_id}_seed${SEED}.json"

    if [ "${RUN_FORCE:-0}" != "1" ] && method_summary_is_reusable "$existing_summary_json" "$prompt_setting" "$method"; then
      echo ""
      echo ">>> Reusing existing JSON summary for $(prompt_setting_label "$prompt_setting") / ${method}"
      prepare_reusable_method_summary "$existing_summary_json" "$run_summary_json"
      continue
    fi

    for dataset_id in "${EXECUTABLE_TEST_SETS[@]}"; do
      echo ""
      echo ">>> Running $(prompt_setting_label "$prompt_setting") / ${method} on ${dataset_id}"
      run_single_dataset "$prompt_setting" "$method" "$dataset_id"
    done
    build_method_summary "$prompt_setting" "$method"
  done
done

build_final_summary

echo ""
echo "Done. VLM TTA prompt-setting matrix run finished."
