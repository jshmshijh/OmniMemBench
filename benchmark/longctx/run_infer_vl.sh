#!/bin/bash
# OmniMemBench VL evaluation - Stage 1: Inference only
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_IDS="$(cat "$SCRIPT_DIR/track_run_ids.txt")"
MAX_WORKERS=50
MAX_TOKENS=32768
LOG_DIR="${SCRIPT_DIR}/logs/logs_vl_infer"
PID_FILE="${LOG_DIR}/pids.txt"
mkdir -p "$LOG_DIR"
> "$PID_FILE"

COMMON_ARGS="--type vl --disable_judge --max_workers $MAX_WORKERS --max_tokens $MAX_TOKENS --resume"

# Configure your models here
MODELS=(
    "your-model-name"
)
TIERS="128k"

PIDS=()

cleanup() {
    echo "Caught signal, killing all jobs..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    exit 1
}
trap cleanup SIGINT SIGTERM

for model in "${MODELS[@]}"; do
    echo "[VL-Infer] Launching $model (tiers: $TIERS)"
    python -u "$SCRIPT_DIR/run_longctx.py" \
        --model "$model" --tiers "$TIERS" --run_ids "$RUN_IDS" \
        --output_base "${SCRIPT_DIR}/output" \
        $COMMON_ARGS \
        > "$LOG_DIR/${model}.log" 2>&1 &
    PIDS+=($!); echo $! >> "$PID_FILE"
done

echo "All VL inference jobs launched (${#PIDS[@]} models). PIDs: ${PIDS[*]}"
echo "Logs: $LOG_DIR/"

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[WARN] PID $pid exited with error"
        FAIL=$((FAIL+1))
    fi
done

echo "VL inference done. Failures: $FAIL/${#PIDS[@]}"
