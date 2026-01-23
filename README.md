# ASNN-Goose: Adapted Spiking Neural Network

Distilling GPT-2 into a ternary-activation RWKV model for GPU-efficient inference.

## Current Status

| Metric | Value |
|--------|-------|
| Best PPL | 306.89 (v14.3) |
| Teacher PPL | 44.6 (GPT-2) |
| Gap | 6.9x |
| Parameters | 74M |
| Architecture | d=768, 5 layers |

## Core Idea

Ternary activations {-1, 0, +1} convert multiplications to additions/subtractions:
- `+1`: add weight
- `-1`: subtract weight
- `0`: skip entirely

This enables computational shortcuts unavailable to continuous networks.

## Key Innovations

1. **CTKD**: Curriculum Temperature KD with Gradient Reversal Layer
2. **FDD**: Feature Dynamics Distillation using CKA loss
3. **SpikingBrain**: Information encoding validation

## Project Structure

```
asnn-goose-prototype/
├── notebooks/          # Colab/Kaggle notebooks (v6-v15)
├── src/
│   ├── models/         # ASNN-Goose architecture
│   ├── training/       # Distillation infrastructure
│   ├── evaluation/     # SpikingBrain validation
│   └── utils/          # Visualization, checkpoints
├── knowledge/
│   ├── overview.md     # Project overview
│   └── roadmap.md      # Version history and roadmap
└── changelog.md        # Detailed version changelog
```

## Roadmap

| Version | Status | Focus |
|---------|--------|-------|
| v14.3 | DONE | PPL 306.89 (current best) |
| v15 | IN PROGRESS | SpikingBrain validation |
| v16 | PLANNED | Sparse operations |
| v17 | PLANNED | Efficiency benchmarks |
| v18 | PLANNED | Ablation studies |
| v19 | PLANNED | Publication |

## Quick Start

```bash
# Run on Colab/Kaggle
# Upload notebooks/asnn_goose_colab_v15.ipynb
# Enable GPU runtime
# Run all cells
```

## Hardware

- Target: Kaggle T4 (16GB VRAM)
- Training: ~4 hours for full run
- VRAM: ~3-6GB peak

## Citation

```bibtex
@article{asnn-goose-2026,
  title={ASNN-Goose: Adapted Spiking Neural Networks with RWKV-style Recurrence},
  author={Eptesicus Laboratories},
  year={2026}
}
```
