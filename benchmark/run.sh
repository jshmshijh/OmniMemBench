#!/usr/bin/env bash
set -euo pipefail

# OmniMemBench unified benchmark runner
# Replaces test_smoke.sh, test_main_methods.sh, run_eval_methods.sh, etc.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults (override via environment or arguments)
PYTHON="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
MODE="smoke"
MEMORY="NaiveRAG"
TIER="128k"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
EMB_PORT="${EMB_PORT:-8010}"
LLM_PORT="${LLM_PORT:-8000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_RESULTS="${SAVE_RESULTS:-1}"

usage() {
    cat <<'USAGE'
Usage: ./run.sh [OPTIONS]

Options:
  --mode MODE         Evaluation mode: smoke, full, eval (default: smoke)
  --memory NAME       Memory method (or "all") (default: NaiveRAG)
  --tier TIER         Context tier: 128k, 256k, 512k, 1m (default: 128k)
  --gpu IDS           GPU device IDs (default: 0)
  --data-root PATH    Path to benchmark data (default: ./data)
  --num-workers N     Parallel workers (default: 4)
  --help              Show this help

Environment variables:
  LLM_API_BASE        LLM endpoint (default: http://localhost:8000/v1)
  LLM_API_KEY         LLM API key
  LLM_MODEL           Model name
  EMB_SERVER_URL      Embedding server (default: http://localhost:8010/v1)
  JUDGE_MODEL         Judge model for eval mode (default: glm-5.2)

Examples:
  ./run.sh --mode smoke --memory NaiveRAG --tier 128k
  ./run.sh --mode full --memory all --tier 128k,256k
  ./run.sh --mode eval --memory NaiveRAG
USAGE
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2;;
        --memory) MEMORY="$2"; shift 2;;
        --tier) TIER="$2"; shift 2;;
        --gpu) GPU="$2"; shift 2;;
        --data-root) DATA_ROOT="$2"; shift 2;;
        --num-workers) NUM_WORKERS="$2"; shift 2;;
        --help|-h) usage;;
        *) echo "Unknown option: $1"; usage;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

# Memory method list
if [[ "$MEMORY" == "all" ]]; then
    METHODS=(SlidingWindow NaiveRAG GenerativeAgents MemGPT Reflexion MultimodalRAG NeuralGraph AUGUSTUS UniversalRAG)
else
    IFS=',' read -ra METHODS <<< "$MEMORY"
fi

# Tier list
IFS=',' read -ra TIERS <<< "$TIER"

echo "=== OmniMemBench ==="
echo "Mode: $MODE | Methods: ${METHODS[*]} | Tiers: ${TIERS[*]} | GPU: $GPU"
echo ""

# Determine datasets
case "$MODE" in
    smoke)
        DATASETS="00001"
        ;;
    full|eval)
        DATASETS="--all_datasets"
        ;;
    *)
        echo "Unknown mode: $MODE"; usage;;
esac

# Run inference, or score existing results in eval mode, for each method and tier
for method in "${METHODS[@]}"; do
    for tier in "${TIERS[@]}"; do
        echo "--- Running: $method @ $tier ---"

        if [[ "$MODE" == "eval" ]]; then
            ARGS=(
                "$REPO_ROOT/omnimembench/engine/evaluate/evaluation.py"
                --enable_llm_judge
                --enable_retrieval_judge
                --judge_model "${JUDGE_MODEL:-glm-5.2}"
            )
        else
            ARGS=("$SCRIPT_DIR/run_bench.py")
            if [[ "$SAVE_RESULTS" == "1" ]]; then
                ARGS+=(--save_results --save_efficiency)
            fi
        fi

        ARGS+=(
            --llm_name "${LLM_MODEL:-}"
            --memory_name "$method"
            --tier "benchmark_${tier}"
            --data_root "$DATA_ROOT"
            --num_workers "$NUM_WORKERS"
        )

        if [[ "$DATASETS" == "--all_datasets" ]]; then
            ARGS+=(--all_datasets)
        else
            ARGS+=(--data_name "$DATASETS")
        fi

        $PYTHON "${ARGS[@]}"
        echo ""
    done
done

echo "=== Done ==="
