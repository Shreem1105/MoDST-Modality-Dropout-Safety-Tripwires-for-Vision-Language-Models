# MoDST: Modality-Dropout Safety Tripwires for VLMs
[![ICLR Workshop](https://img.shields.io/badge/ICLR_2026-Workshop-blue.svg)](https://iclr.cc)

Official experiment codebase for the **ICLR 2026 Workshop on Principled Design for Trustworthy AI**.

## Overview
MoDST is an inference-time safety mechanism that detects grounding and safety failures in Vision-Language Models (VLMs) by running controlled modality dropouts. This codebase provides a modular framework for:
- **3-pass VLM inference** (Full, Text-only, Image-only).
- **Tri-factor safety scoring** (Grounding, Fusion, Safety transfer).
- **Evaluation under distribution shift** (Image perturbations).

## Directory Structure
```text
modst/
├── models/         # VLM Interfaces (LLaVA-1.5, InstructBLIP)
├── tripwire/       # Safety Tripwire implementations & scoring
├── datasets/       # Base loaders & Distribution shift (Perturbations)
├── baselines/      # Vanilla & Confidence-based policies
├── experiments/    # Main execution scripts
├── analysis/       # Metrics computation & Plotting
configs/            # Experiment YAML configurations
scripts/            # Helper utility scripts (download_data.py, setup_demo.py)
tests/              # Logic verification tests
```

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start (Large Scale)
To reproduce the full workshop results:

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Acquire Data**:
   Download the standard evaluation dataset (COCO-Val 2017) formatted for MoDST:
   ```bash
   python scripts/download_data.py
   ```

3. **Run Experiment**:
   Execute the full pipeline on GPU (H200/A100 recommended):
   ```bash
   python -m modst.experiments.run_modst --config configs/default.yaml
   ```

## Key Research Features
- **H200 Optimized**: Supports 4-bit quantization and efficient batching of MoDST passes.
- **Modular Scoring**: Easily weight Grounding vs. Fusion instability.
- **Robust Evaluation**: Built-in image perturbation suite (Noise, Occlusion, Blur) to test safety under shift.

## Citation
If you find this code useful, please cite our workshop paper:
```bibtex
@inproceedings{modst2026trustworthy,
  title={MoDST: Modality-Dropout Safety Tripwires for Vision-Language Models},
  author={Dixit, Aradhya and Team},
  booktitle={ICLR 2026 Workshop on Principled Design for Trustworthy AI},
  year={2026}
}
```
