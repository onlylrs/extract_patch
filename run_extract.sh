#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root=""
run_id=""
input_list=""
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
        --preview|--center-preview|--inspect|--show-config)
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

mkdir -p "${ROOT}/logs"
if [[ -z "$run_id" ]]; then
    input_name="${input_list##*/}"
    input_name="${input_name%.*}"
    input_name="${input_name:-extract}"
    input_name="${input_name// /_}"
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    run_id="${input_name}_$$_${timestamp}"
    args+=(--run-id "$run_id")
fi
safe_run_id="${run_id//\//_}"
log_path="${ROOT}/logs/${safe_run_id}.log"
nohup env \
    EXTRACT_PATCH_STDOUT_LOGGED=1 \
    EXTRACT_PATCH_LOG_PATH="$log_path" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    python3 "${ROOT}/extract_patches.py" "${args[@]}" \
    > "$log_path" 2>&1 < /dev/null &
pid=$!
disown "$pid" 2>/dev/null || true

echo "Started background extraction"
echo "PID: $pid"
echo "Log: $log_path"
