#!/bin/bash
# H200 ONE-COMMAND RUNNER for MoDST on POPE Dataset
# Usage: 
#   POPE_ROOT=/path/to/pope NUM_SAMPLES=500 bash scripts/run_pope_h200.sh

set -e # Fail fast

# 1. Setup Paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
echo "Repo Root: $REPO_ROOT"

# Default params
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$REPO_ROOT/runs/pope_$TIMESTAMP"
mkdir -p "$RUN_DIR"

: "${POPE_ROOT:?Error: POPE_ROOT env var must be set to the POPE dataset directory.}"
: "${IMAGES_ROOT:=$POPE_ROOT/coco}" # Default if not set
: "${NUM_SAMPLES:=500}"
: "${MAX_NEW_TOKENS:=64}"
: "${BATCH_SIZE:=8}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Try to increase file limit if possible/allowed
ulimit -n 4096 2>/dev/null || true

echo "=========================================================="
echo "Starting MoDST POPE Run on H200"
echo "Run Dir: $RUN_DIR"
echo "Num Samples: $NUM_SAMPLES"
echo "=========================================================="

# 2. Run Inference (VLM)
echo "[1/3] Running VLM Inference (3-pass)..."
python -u "$REPO_ROOT/run_modst_pope.py" \
    --pope_root "$POPE_ROOT" \
    --images_root "$IMAGES_ROOT" \
    --out_jsonl "$RUN_DIR/preds.jsonl" \
    --num_samples "$NUM_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size "$BATCH_SIZE" \
    --model_id "llava-hf/llava-1.5-7b-hf"

# 3. Run Scoring (NLI)
echo "[2/3] Running NLI Scoring..."
python -u "$REPO_ROOT/eval/nli_scorer.py" \
    --preds_jsonl "$RUN_DIR/preds.jsonl" \
    --batch_size 32

# 4. Run Evaluation
echo "[3/3] Running Evaluation & Plotting..."
python -u "$REPO_ROOT/eval/eval_pope.py" \
    --preds_jsonl "$RUN_DIR/preds_scored.jsonl" \
    --out_dir "$RUN_DIR"

echo "=========================================================="
echo "RUN COMPLETE!"
echo "Artifacts:"
echo "  Predictions: $RUN_DIR/preds_scored.jsonl"
echo "  Metrics:     $RUN_DIR/metrics.json"
echo "  Risk Curve:  $RUN_DIR/risk_coverage.png"
echo "  Examples:    $RUN_DIR/qual_examples.md"
echo "=========================================================="
