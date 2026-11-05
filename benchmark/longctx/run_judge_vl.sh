#!/bin/bash
# OmniMemBench VL evaluation - Stage 2: Judge only
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_IDS="$(cat "$SCRIPT_DIR/track_run_ids.txt")"
JUDGE_MODEL="${JUDGE_MODEL:-glm-5.2}"
MAX_WORKERS=50
LOG_DIR="${SCRIPT_DIR}/logs/logs_vl_judge"
mkdir -p "$LOG_DIR"

# Configure models to judge (must match inference stage)
ALL_MODELS=(
    "your-model-name"
)
TIERS="128k"

FAIL=0
for model in "${ALL_MODELS[@]}"; do
    echo "[VL-Judge] $model"
    if python -u "$SCRIPT_DIR/run_longctx.py" \
        --model "$model" --tiers "$TIERS" --run_ids "$RUN_IDS" \
        --type vl --judge_only \
        --judge_model "$JUDGE_MODEL" \
        --max_workers "$MAX_WORKERS" \
        --output_base "${SCRIPT_DIR}/output" \
        2>&1 | tee "$LOG_DIR/${model}.log"; then
        echo "[VL-Judge] $model done"
    else
        echo "[WARN] $model judge failed"
        FAIL=$((FAIL+1))
    fi
done

echo "VL judge pass done. Failures: $FAIL/${#ALL_MODELS[@]}"
