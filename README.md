# MoDST: Modality-Dropout Safety Tripwires for VLMs

Official experiment codebase for the ICLR 2026 Workshop on Principled Design for Trustworthy AI.

## Overview
MoDST is an inference-time safety mechanism that detects grounding and safety failures in Vision-Language Models (VLMs) by running controlled modality dropouts. This codebase provides a modular framework for:
- 3-pass VLM inference (Full, Text-only, Image-only).
- Tri-factor safety scoring (Grounding, Fusion, Safety transfer).
- Evaluation under distribution shift (Image perturbations).

## Directory Structure
```text
modst/
├── models/         # VLM Interfaces (LLaVA-1.5)
├── tripwire/       # Safety Tripwire implementations & scoring
├── datasets/       # Base loaders & Distribution shift (Perturbations)
├── baselines/      # Vanilla & Confidence-based policies
├── experiments/    # Main execution scripts
├── analysis/       # Metrics computation & Plotting
configs/            # Experiment YAML configurations
scripts/            # Helper utility scripts
tests/              # Logic verification tests
```

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
1. **Configure Experiment**: Edit `configs/default.yaml` to set your dataset paths and model settings.
2. **Run Experiment**:
   ```bash
   python -m modst.experiments.run_modst --config configs/default.yaml
   ```
3. **Analyze Results**:
   ```bash
   # Evaluation script (to be customized based on results.json)
   python -m modst.analysis.metrics
   ```

## Key Research Features
- **H200 Optimized**: Supports 4-bit quantization and efficient batching of MoDST passes.
- **Modular Scoring**: Easily weight Grounding vs. Fusion instability.
- **Robust Evaluation**: Built-in image perturbation suite (Noise, Occlusion, Blur) to test safety under shift.
