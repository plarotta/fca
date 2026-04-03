# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project investigating **Future-Conditioned Attention (FCA)** — an architectural modification to decoder-only transformers that adds an explicit, iterative internal inference loop for future belief states.

The core idea: add a secondary per-layer attention branch conditioned on a bottlenecked latent projection of the hidden state (`z`), fused back into the primary residual stream via a learned sigmoid gate (`g`). The bottleneck forces `z` to become a compressed "belief state" encoding future trajectory predictions, making the transformer's implicit future representations computationally actionable.

## Key Architectural Concepts

- **Bottleneck latent `z`**: Hidden state mapped through a small MLP to 1/4 or 1/8 of hidden dim. Acts as a continuous "thought token" — a differentiable alternative to Quiet-STaR's discrete rationale generation.
- **Gated fusion `g`**: Sigmoid gate that down-weights the FCA branch when `z` is uncertain, preventing hallucinated futures from corrupting the primary attention path.
- **Sparse FCA Schedule**: FCA branches restricted to middle-to-late layers (top third) only. Early layers handle detokenization and are highly sensitive to structural disruption. The "Prediction Ensembling" stage (middle-to-late) is where the network naturally transitions to forward-looking computation.
- **Future loss objective**: MSE against stop-gradient of future hidden states (not discrete vocabulary tokens). This forces `z` toward a true belief state. EMA on targets recommended to reduce gradient variance.
- **Lambda schedule**: Delayed ramp-up of the future prediction loss to avoid bootstrap collapse (training `z` on random noise from an unconverged network).

## Research Phases

- **Phase -1 (Probe Experiment)**: Linear probe baseline calibration. Not testing *if* future info exists (Future Lens already proves it does), but quantifying *when* (at which layer depth) and *how strongly* — the success metric for FCA is the delta in probe accuracy at earlier layers.
- **FCA-random-z ablation**: Critical confound control. Substituting `z` with Gaussian noise isolates whether gains come from the structured belief state vs. generic architectural regularization.

## Key Evaluation Metrics

- **Early Commitment / Confidence Saturation**: Measures how early the model's prediction entropy collapses to the correct answer. FCA should shift this curve leftward (earlier layers, earlier sequence positions).
- **Dynamic halting / early exit**: If `g` gates saturate and `z` entropy minimizes in middle layers, the model can skip remaining layers — measured via FLOP-to-accuracy ratio.
- **Self-speculative decoding**: `z` from the topmost FCA layer projected into draft tokens, eliminating need for separate draft models.

## Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Smoke test: full pipeline on mini synthetic data (~2 min on CPU)
bash scripts/smoke_test.sh cpu

# Train FCA model (example: primary config)
uv run python -m fca.train --out_dir results/fca-top-third --fca_layers 8 9 10 11

# Train baseline (nanoGPT)
bash scripts/run_baseline.sh

# Prepare OpenWebText
bash scripts/prepare_data.sh

# Extract hidden states + train probes for a checkpoint
bash scripts/run_probes.sh <checkpoint_path> <model_name> [device]

# Run all 5 FCA experiment variants
bash scripts/run_all_experiments.sh
```

## Architecture

Built on top of **nanoGPT** (vendored in `nanoGPT/`). FCA modules in `fca/` extend nanoGPT's `Block` and `GPT` classes:

- `fca/config.py` — `FCAConfig` dataclass (extends GPTConfig fields + FCA-specific params)
- `fca/model.py` — `BottleneckMLP`, `FCAAttention`, `FCAGate`, `FCABranch`, `FCABlock`, `FCAGPT`
- `fca/losses.py` — `EMATargetTracker`, `lambda_schedule`, `compute_future_loss`, `compute_total_loss`
- `fca/train.py` — Training script (argparse CLI, supports DDP, wandb, gradient accumulation)

`FCAGPT.forward()` returns `(logits, ce_loss, aux)` where `aux` contains `hidden_states`, `z_vectors`, `gate_values`, `future_preds` — these feed into `compute_total_loss()`.

Probing harness in `probes/`:
- `extract.py` — extracts per-layer hidden states from either `GPT` (via hooks) or `FCAGPT` (via aux dict)
- `train_probes.py` — trains linear probes for all (layer, lookahead) pairs, generates plots

## Current State

Phase 0 (architecture) and Phase -1 code (probing harness) are complete. Remaining: data prep, baseline training, and running experiments (requires GPU). See `PROGRESS.md` for detailed status.
