# ASNN-Goose: Adaptive Spiking Neural Network with Goose Backbone

A research prototype for Eptesicus Laboratories (Lumis-NEXT initiative).

## Overview

ASNN-Goose combines:
1. **RWKV-style recurrence** (Goose backbone) with delta-rule state updates
2. **Ternary spiking activations** ({-1, 0, +1}) replacing continuous activations
3. **Static weight quantization** (INT8)
4. **Memory-locked test-time training (TTT)** via LoRA adapters

## Project Structure

```
asnn-goose-prototype/
├── config.py               # All hyperparameters
├── src/
│   ├── models/             # Core model components
│   ├── training/           # Training infrastructure
│   ├── evaluation/         # Evaluation and analysis
│   ├── kernels/            # Kernel-level analysis
│   └── utils/              # Utilities
├── notebooks/              # Experiment notebooks
├── outputs/                # Figures, checkpoints, logs
└── tests/                  # Unit tests
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter lab notebooks/
```

## Hardware Target

- Kaggle T4 GPU (16GB VRAM, Turing architecture with Tensor Cores)
- ~30 GPU hours/week budget

## Key Experiments

1. **Baseline Teacher**: Train dense Goose model
2. **Ternary Ablation**: Compare fixed vs adaptive thresholds
3. **Distillation**: Train ASNN-Goose student
4. **TTT Evaluation**: Test trigger logic and drift controls
5. **Kernel Analysis**: Measure sparsity benefits
6. **Full Evaluation**: Publication-ready results

## Configuration

All hyperparameters are centralized in `config.py`. Key settings:

- `d_model`: 256 (hidden dimension)
- `n_layers`: 4 (recurrent layers)
- `ternary_threshold_init`: 0.5
- `lora_rank`: 8
- `weight_bits`: 8 (INT8)

## Citation

```bibtex
@article{asnn-goose-2024,
  title={ASNN-Goose: Adaptive Spiking Neural Networks with RWKV-style Recurrence},
  author={Eptesicus Laboratories},
  year={2024}
}
```
