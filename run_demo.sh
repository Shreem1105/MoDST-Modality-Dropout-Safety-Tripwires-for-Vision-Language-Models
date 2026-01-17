#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

echo "============================================="
echo "   MoDST Experiment: Setup & Run Sequence    "
echo "============================================="

# 1. Update/Install Dependencies
echo "[1/3] Checking dependencies..."
pip install -r requirements.txt

# 2. Setup Data (Generate demo data if 'data/val_grounding.json' is missing)
echo "[2/3] Checking dataset..."
python scripts/setup_demo.py

# 3. Run the Experiment
echo "[3/3] Starting MoDST pipeline..."
python -m modst.experiments.run_modst --config configs/default.yaml

echo "============================================="
echo "   Experiment Completed Successfully!        "
echo "============================================="
