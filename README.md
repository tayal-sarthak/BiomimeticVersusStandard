# CKAN Twin Experiment (CIFAR‑10)

This repository is a small project for running a **Twin Experiment** with **Convolutional Kolmogorov‑Arnold Networks (CKANs / C‑KANs)**.

The idea is simple:

- **Twin A (Control / Standard):** train from epoch 0 on sharp CIFAR‑10 images.
- **Twin B (Experiment / Biomimetic):** train with a **blur curriculum** that starts with “infant‑like” blurry vision and progressively sharpens.

After training, both models are evaluated on **CIFAR‑10‑C** corruptions (snow, fog, glass blur, defocus blur) to compare robustness.

## What this repo actually does

There are four scripts under `scripts/` that run the whole experiment end‑to‑end:

1. **Train Twin A:** `scripts/run_standard_cifar.py`
2. **Train Twin B:** `scripts/run_biomimetic_cifar.py`
3. **Evaluate both:** `scripts/evaluate_robustness.py`
4. **Train Twin C (Anti-biomimetic / reverse curriculum):** `scripts/run_antibiomimetic_cifar.py`

An optional orchestrator, `run_experiment.py`, runs the first 3 three back‑to‑back and stops immediately if any stage crashes.

### The biomimetic blur curriculum

Twin B uses a Gaussian blur curriculum:

- Start at $\sigma = 4.0$.
- Linearly decay blur strength to $\sigma = 0$ halfway through training.
- Second half of training is fully sharp.

This is the “Biomimetic Learning” hypothesis being tested.

## Directory layout (important)

This workspace is intentionally not a standard package. The code is loaded via `sys.path` and relative paths.

- `KANConv.py`, `KANLinear.py`, `convolution.py`: CKAN core implementation.
- `scripts/`: the three experiment scripts.
- `data1/`:
  - CIFAR downloads (standard CIFAR‑10 and the CIFAR‑10‑C `.npy` files)
  - saved checkpoints:
    - `data1/model_a_standard.pth`
    - `data1/model_b_biomimetic.pth`
    - `data1/model_c_antibiomimetic.pth`

## How to run (Windows / overnight)

Run from the workspace root (the folder that contains `scripts/` and `data1/`).

### Option A: run the full experiment

```bash
python run_experiment.py
```

### Option B: run steps manually

```bash
python scripts/run_standard_cifar.py
python scripts/run_biomimetic_cifar.py
python scripts/run_antibiomimetic_cifar.py
python scripts/evaluate_robustness.py
```

### GPU requirement

Training is configured to **require CUDA** (no CPU fallback) because CKANs are computationally heavy.

## Results

The evaluation script prints a small comparison table (Standard vs Biomimetic) and declares a winner based on mean accuracy over:

- `snow`
- `glass_blur`
- `defocus_blur`
- `fog`

## Provenance / attribution (read this)

This repository is **heavily derived** from upstream Convolutional‑KAN implementations.

- A large portion of the CKAN core files and the original README content were taken/adapted from:
  - **GitHub:** https://github.com/AntonioTepsich/Convolutional-KANs
  - **Authors:** Alexander Bodner, Antonio Tepsich, Jack Spolski, Santiago Pourteau
  - **Paper:** “Convolutional Kolmogorov‑Arnold Networks” (submitted June 19, 2024)

If you are looking for the foundational method, math details, and baselines, please consult the upstream repo/paper. This repo focuses on the **Twin Experiment** harness and scripts.

## License

MIT (see `LICENSE`).
