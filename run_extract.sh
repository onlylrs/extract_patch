#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="auto"
foreground_action=0
output_root=""
run_id=""
input_list=""
args=()

while (($#)); do
    case "$1" in
        --foreground)
            mode="foreground"
            shift
            ;;
        --background)
            mode="background"
            shift
            ;;
        --preview|--inspect|--show-config)
            foreground_action=1
            args+=("$1")
            shift
            ;;
        --output)
            if (($# < 2)); then
                echo "--output requires a path" >&2
                exit 2
            fi
            output_root="$2"
            args+=("$1" "$2")
            shift 2
            ;;
        --output=*)
            output_root="${1#--output=}"
            args+=("$1")
            shift
            ;;
        --input-list)
            if (($# < 2)); then
                echo "--input-list requires a path" >&2
                exit 2
            fi
            input_list="$2"
            args+=("$1" "$2")
            shift 2
            ;;
        --input-list=*)
            input_list="${1#--input-list=}"
            args+=("$1")
            shift
            ;;
        --run-id)
            if (($# < 2)); then
                echo "--run-id requires a value" >&2
                exit 2
            fi
            run_id="$2"
            args+=("$1" "$2")
            shift 2
            ;;
        --run-id=*)
            run_id="${1#--run-id=}"
            args+=("$1")
            shift
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$output_root" ]]; then
    mkdir -p "$output_root"
fi

if [[ "$mode" == "foreground" || ("$mode" == "auto" && "$foreground_action" -eq 1) ]]; then
    exec python3 "${ROOT}/extract_patches.py" "${args[@]}"
fi

mkdir -p "${ROOT}/logs"
if [[ -n "$run_id" ]]; then
    safe_run_id="${run_id//\//_}"
    log_path="${ROOT}/logs/${safe_run_id}.log"
    nohup env \
        EXTRACT_PATCH_STDOUT_LOGGED=1 \
        EXTRACT_PATCH_LOG_PATH="$log_path" \
        python3 "${ROOT}/extract_patches.py" "${args[@]}" \
        > "$log_path" 2>&1 < /dev/null &
    pid=$!
else
    input_name="${input_list##*/}"
    input_name="${input_name%.*}"
    input_name="${input_name:-extract}"
    input_name="${input_name// /_}"
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    (
        child_run_id="${input_name}_${BASHPID}_${timestamp}"
        child_log_path="${ROOT}/logs/${child_run_id}.log"
        exec nohup env \
            EXTRACT_PATCH_STDOUT_LOGGED=1 \
            EXTRACT_PATCH_LOG_PATH="$child_log_path" \
            python3 "${ROOT}/extract_patches.py" "${args[@]}" \
            --run-id "$child_run_id" \
            > "$child_log_path" 2>&1
    ) < /dev/null &
    pid=$!
    run_id="${input_name}_${pid}_${timestamp}"
    log_path="${ROOT}/logs/${run_id}.log"
fi
disown "$pid" 2>/dev/null || true

echo "Started background extraction"
echo "PID: $pid"
echo "Log: $log_path"
