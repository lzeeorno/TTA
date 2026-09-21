#!/bin/bash

RUN_RETRY_COUNT="${RUN_RETRY_COUNT:-3}"
RUN_RETRY_DELAY="${RUN_RETRY_DELAY:-5}"
RUN_SLEEP_BETWEEN_EXPERIMENTS="${RUN_SLEEP_BETWEEN_EXPERIMENTS:-5}"

run_now() {
    date '+%Y-%m-%d %H:%M:%S'
}

run_timestamp() {
    date +%Y%m%d_%H%M%S
}

ensure_dir() {
    mkdir -p "$1"
}

print_rule() {
    printf '%s\n' "=========================================="
}

print_section() {
    local title="$1"
    print_rule
    printf '%s\n' "$title"
    print_rule
}

run_with_retry() {
    local log_file="$1"
    local label="$2"
    shift 2

    local save_logs="${RUN_SAVE_LOGS:-0}"
    local log_sink="/dev/null"
    if [ "$save_logs" = "1" ] && [ -n "$log_file" ]; then
        log_sink="$log_file"
        ensure_dir "$(dirname "$log_sink")"
    fi

    local attempt=1
    local max_attempts="${RUN_RETRY_COUNT}"
    local delay_seconds="${RUN_RETRY_DELAY}"
    local cmd_display
    cmd_display=$(printf '%q ' "$@")

    while (( attempt <= max_attempts )); do
        {
            echo "[$(run_now)] START ${label} (attempt ${attempt}/${max_attempts})"
            echo "[$(run_now)] CMD   ${cmd_display}"
        } | tee -a "$log_sink"

        if "$@" 2>&1 | tee -a "$log_sink"; then
            echo "[$(run_now)] SUCCESS ${label} (attempt ${attempt}/${max_attempts})" | tee -a "$log_sink"
            return 0
        fi

        local exit_code=${PIPESTATUS[0]}
        echo "[$(run_now)] FAIL ${label} (attempt ${attempt}/${max_attempts}, exit=${exit_code})" | tee -a "$log_sink"

        if (( attempt == max_attempts )); then
            return "$exit_code"
        fi

        echo "[$(run_now)] RETRY after ${delay_seconds}s" | tee -a "$log_sink"
        sleep "$delay_seconds"
        attempt=$((attempt + 1))
    done
}

infer_expected_result_file() {
    local config_path="$1"
    local method="$2"
    local seed="$3"
    local order_idx="$4"
    local has_corruption_order="$5"
    local scenario_arg="$6"
    local dataset_override="$7"
    local backbone_override="$8"
    local run_suffix="$9"

    python - "$config_path" "$method" "$seed" "$order_idx" "$has_corruption_order" "$scenario_arg" "$dataset_override" "$backbone_override" "$run_suffix" <<'PY'
import os
import sys
import yaml

config_path, method, seed, order_idx, has_order, scenario_arg, dataset_override, backbone_override, run_suffix = sys.argv[1:]


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    def deep_update(base, override):
        for k, v in override.items():
            if k == '_BASE_':
                continue
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_update(base[k], v)
            else:
                base[k] = v

    if isinstance(cfg, dict) and '_BASE_' in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg['_BASE_'])
        base_cfg = load_config(base_path)
        deep_update(base_cfg, cfg)
        cfg = base_cfg
    return cfg


cfg = load_config(config_path)

dataset_name = dataset_override if dataset_override else cfg.get('dataset', {}).get('name', 'unknown')
backbone = backbone_override if backbone_override else cfg.get('model', {}).get('backbone', 'unknown')
results_root = cfg.get('logging', {}).get('results_dir', './results')

shuffle = bool(cfg.get('dataset', {}).get('shuffle', False))
reset_each = bool(cfg.get('tta', {}).get('reset_each_corruption', False))

scenario = scenario_arg
if scenario in ('', 'None', None):
    scenario = cfg.get('tta', {}).get('scenario', 'normal')
if scenario in ('', None):
    scenario = 'normal'

if run_suffix not in ('', 'None', None):
    method = f'{method}_{run_suffix}'

if dataset_name in {'cifar10c', 'cifar100c', 'imagenetc'}:
    suffix = ''
    if shuffle:
        suffix += '_shuffle'
    if not reset_each:
        suffix += '_continual'
    if scenario != 'normal':
        suffix += f'_{scenario}'
    results_dir = os.path.join(results_root, f'{dataset_name}_{backbone}{suffix}')
else:
    results_dir = os.path.join(results_root, f'{dataset_name}_{backbone}')

if has_order == '1':
    filename = f'{method}_seed{seed}_order{order_idx}.json'
else:
    filename = f'{method}_seed{seed}.json'

print(os.path.join(results_dir, filename))
PY
}

run_python_experiment() {
    local log_file="$1"
    local label="$2"
    shift 2

    # Reduce CUDA memory fragmentation for long-running TTA jobs unless user already set it.
    if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    fi

    local method=""
    local config_path="configs/default.yaml"
    local seed="0"
    local order_idx="0"
    local has_corruption_order="0"
    local scenario_arg=""
    local dataset_override=""
    local backbone_override=""
    local run_suffix=""
    local has_batch_size="0"

    local args=("$@")
    local i=0
    while (( i < ${#args[@]} )); do
        case "${args[$i]}" in
            --method)
                if (( i + 1 < ${#args[@]} )); then
                    method="${args[$((i + 1))]}"
                fi
                i=$((i + 2))
                ;;
            --config)
                if (( i + 1 < ${#args[@]} )); then
                    config_path="${args[$((i + 1))]}"
                fi
                i=$((i + 2))
                ;;
            --seed)
                if (( i + 1 < ${#args[@]} )); then
                    seed="${args[$((i + 1))]}"
                fi
                i=$((i + 2))
                ;;
            --order-idx)
                if (( i + 1 < ${#args[@]} )); then
                    order_idx="${args[$((i + 1))]}"
                fi
                i=$((i + 2))
                ;;
            --corruption-order)
                has_corruption_order="1"
                i=$((i + 2))
                ;;
            --scenario)
                if (( i + 1 < ${#args[@]} )); then
                    scenario_arg="${args[$((i + 1))]}"
                fi
                i=$((i + 2))
                ;;
            --dataset)
                if (( i + 1 < ${#args[@]} )); then
                    dataset_override="${args[$((i + 1))]}"
                fi
                i=$((i + 2))
                ;;
            --backbone)
                if (( i + 1 < ${#args[@]} )); then
                    backbone_override="${args[$((i + 1))]}"
                fi
                i=$((i + 2))
                ;;
            --run-suffix)
                if (( i + 1 < ${#args[@]} )); then
                    run_suffix="${args[$((i + 1))]}"
                fi
                i=$((i + 2))
                ;;
            --batch-size)
                has_batch_size="1"
                i=$((i + 2))
                ;;
            *)
                i=$((i + 1))
                ;;
        esac
    done

    if [ -z "$method" ]; then
        method="triad"
    fi

    if [ "$has_batch_size" = "0" ]; then
        if [ "$method" = "surgeon" ]; then
            args+=(--batch-size 32)
        else
            args+=(--batch-size 64)
        fi
    fi

    local result_file=""
    result_file=$(infer_expected_result_file "$config_path" "$method" "$seed" "$order_idx" "$has_corruption_order" "$scenario_arg" "$dataset_override" "$backbone_override" "$run_suffix" 2>/dev/null || true)
    if [ "${RUN_FORCE:-0}" != "1" ] && [ -n "$result_file" ] && [ -s "$result_file" ]; then
        local save_logs="${RUN_SAVE_LOGS:-0}"
        local log_sink="/dev/null"
        if [ "$save_logs" = "1" ] && [ -n "$log_file" ]; then
            log_sink="$log_file"
            ensure_dir "$(dirname "$log_sink")"
        fi
        echo "[$(run_now)] SKIP ${label} (existing result: ${result_file})" | tee -a "$log_sink"
        return 0
    fi

    run_with_retry "$log_file" "$label" python code/main.py "${args[@]}"
}

sleep_between_experiments() {
    local seconds="${1:-$RUN_SLEEP_BETWEEN_EXPERIMENTS}"
    if (( seconds > 0 )); then
        sleep "$seconds"
    fi
}

generate_auto_summary() {
    local results_dir="$1"
    local output_dir="$2"
    local summary_name="$3"
    local methods_csv="${4:-}"
    local extra_args=("${@:5}")

    # Keep CSV/MD summary together with JSON results.
    output_dir="$results_dir"
    ensure_dir "$output_dir"
    python scripts/generate_run_summary.py \
        --results-dir "$results_dir" \
        --output-dir "$output_dir" \
        --summary-name "$summary_name" \
        ${methods_csv:+--methods "$methods_csv"} \
        "${extra_args[@]}"
}

generate_multi_dir_summary() {
    local summary_name="$1"
    local methods_csv="${2:-}"
    shift 2
    local result_dirs=("$@")
    local methods_args=()
    local summary_args=()
    local result_dir=""
    local result_dir_count=0

    if [ -n "$methods_csv" ]; then
        methods_args=(--methods "$methods_csv")
    fi

    for result_dir in "${result_dirs[@]}"; do
        if [ ! -d "$result_dir" ]; then
            continue
        fi
        echo "[Summary] ${result_dir}/${summary_name}.{csv,md}"
        python scripts/generate_run_summary.py \
            --results-dir "$result_dir" \
            --output-dir "$result_dir" \
            --summary-name "$summary_name" \
            "${methods_args[@]}"
        summary_args+=(--results-dir "$result_dir")
        result_dir_count=$((result_dir_count + 1))
    done

    if [ "$result_dir_count" -le 1 ]; then
        return 0
    fi

    ensure_dir "results"
    echo "[Summary] results/${summary_name}_overall.{csv,md}"
    python scripts/generate_run_summary.py \
        "${summary_args[@]}" \
        --output-dir "results" \
        --summary-name "${summary_name}_overall" \
        "${methods_args[@]}"
}
