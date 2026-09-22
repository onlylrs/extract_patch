#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root=""
run_id=""
input_list=""
first_input=""
mode="extract"
args=()

while (($#)); do
    case "$1" in
        --foreground)
            echo "--foreground is deprecated; run_extract.sh always uses nohup" >&2
            shift
            ;;
        --background)
            shift
            ;;
        --preview)
            mode="preview"
            args+=("$1")
            shift
            ;;
        --center-preview)
            mode="center-preview"
            args+=("$1")
            shift
            ;;
        --inspect)
            mode="inspect"
            args+=("$1")
            shift
            ;;
        --show-config)
            mode="show-config"
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
        --config|--input-root|--n-patches|--set)
            if (($# < 2)); then
                echo "$1 requires a value" >&2
                exit 2
            fi
            args+=("$1" "$2")
            shift 2
            ;;
        --config=*|--input-root=*|--n-patches=*|--set=*|-*)
            args+=("$1")
            shift
            ;;
        *)
            if [[ -z "$first_input" ]]; then
                first_input="$1"
            fi
            args+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$output_root" ]]; then
    mkdir -p "$output_root"
fi

mkdir -p "${ROOT}/logs"
if [[ -n "$input_list" ]]; then
    input_name="${input_list##*/}"
    input_name="${input_name%.*}"
elif [[ -n "$first_input" ]]; then
    input_name="${first_input##*/}"
    input_name="${input_name%.*}"
else
    input_name="extract"
fi
input_name="${input_name:-extract}"
input_name="${input_name// /_}"
input_name="${input_name//\//_}"
date_stamp="$(date '+%Y%m%d')"
timestamp="$(date '+%Y%m%d_%H%M%S')"
log_prefix="${ROOT}/logs/${input_name}_${mode}_${date_stamp}"

# exec keeps the background subshell PID as the Python process PID, so the
# printed PID, log filename, and PID recorded inside the log all agree.
(
    task_pid="$BASHPID"
    task_log_path="${log_prefix}_${task_pid}.log"
    if [[ -z "$run_id" ]]; then
        args+=(--run-id "${input_name}_${task_pid}_${timestamp}")
    fi
    exec nohup env \
        EXTRACT_PATCH_STDOUT_LOGGED=1 \
        EXTRACT_PATCH_LOG_PATH="$task_log_path" \
        OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        python3 "${ROOT}/extract_patches.py" "${args[@]}" \
        > "$task_log_path" 2>&1 < /dev/null
) &
pid=$!
disown "$pid" 2>/dev/null || true
log_path="${log_prefix}_${pid}.log"

echo "Started background extraction"
echo "PID: $pid"
echo "Log: $log_path"
