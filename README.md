# ASNN-Goose

Shrunk GPT-2 (124M) into a 74M spiking model that runs fine on regular GPUs. Ternary activations (-1, 0, +1) plus RWKV recurrence.

## Why bother

LLMs work great but burn cash and watts. ASNN-Goose keeps most of the smarts while slashing compute. No fancy neuromorphic chips needed.

## How

- Neurons fire -1, 0, or +1. Multiplications turn into cheap adds, subs, or nothing.
- RWKV recurrence carries a tiny state instead of re-reading the whole context each token.
- Curriculum-temperature distillation plus CKA feature matching teaches the small net both outputs and internals from GPT-2.
- SpikingBrain checks that the spikes actually mean something.

## Numbers (Feb 2026)

- Best perplexity: 306.89 (v14.3)
- Teacher GPT-2: 44.6
- 74M params, 5 layers
- Train time: ~4h on a Kaggle T4
- Peak VRAM: 3–6 GB

## Repo

- notebooks/ — Colab/Kaggle runs (v6–v15)
- src/ — models, train, eval, utils
- knowledge/ — plain roadmaps
- changelog.md — what changed when

## Roadmap

- v15 — full SpikingBrain test
- v16 — sparse ops
- v17 — speed benchmarks
- v18 — ablations
- v19 — paper, done

## Quick start

Open the newest notebook in Colab/Kaggle, flip GPU on, run.

## Who wants it

Anyone playing with efficient spikes on normal hardware, devs who need small models, or folks curious about practical brain-style AI.

## Scope note

ASNN = Adapted Spiking Neural Network. GPU-first today. Partners with SpikySpace for future neuromorphic chips.




follow me on X: x.com/dttdrv
