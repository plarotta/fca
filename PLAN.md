# FCA Implementation & Experimentation Plan

## Context

Implement and experimentally validate the Future-Conditioned Attention (FCA) architecture. FCA adds a secondary per-layer attention branch conditioned on a bottlenecked latent `z`, fused back via a learned sigmoid gate `g`. The hypothesis: making the transformer's implicit future representations explicit and computationally actionable improves representation quality, sample efficiency, and enables inference optimizations (early exit, self-speculative decoding).

**Stack**: nanoGPT fork, PyTorch, single A100/H100, OpenWebText, GPT-2 Small (124M: 12 layers, 768 hidden dim, 12 heads).

---

## Phase -1: Baseline + Probe Calibration

**Goal**: Train a standard GPT-2 Small baseline and establish quantitative Future Lens baselines at every layer.

### Steps

1. **Clone and configure nanoGPT** for GPT-2 Small on OpenWebText
   - Prepare OpenWebText via nanoGPT's `data/openwebtext/prepare.py`
   - Train baseline to convergence (~100K steps, batch size 64, context 1024)
   - Checkpoint the final model

2. **Build the probing harness** (`probes/`)
   - `probe.py`: Linear probe (single affine layer) — frozen hidden state at layer `l`, position `t` → predict token at `t+k`
   - `extract.py`: Forward-pass hook to dump hidden states at every layer for a fixed eval set (~10K sequences)
   - `train_probes.py`: Train independent probes for each `(layer, lookahead)` pair
   - Lookahead values: k = 1, 2, 4, 8
   - Layers: all 12 (0-indexed: 0..11)
   - Metric: top-1 accuracy

3. **Record baseline probe curves**
   - 48 probes total (12 layers x 4 lookaheads)
   - Plot: x = layer depth, y = probe accuracy, one curve per lookahead k

### Deliverable
- Trained baseline checkpoint
- Baseline probe accuracy matrix (12 x 4)
- Probe accuracy plots

---

## Phase 0: FCA Architecture Implementation

**Goal**: Implement FCA modules that compose with nanoGPT's `Block` and `GPT` classes.

### `fca/model.py`

**BottleneckMLP**: `h ∈ R^(B,T,768)` → `Linear(768,384) → GELU → Linear(384, bottleneck_dim)` → `z ∈ R^(B,T,192)`

**FCAAttention**: Secondary multi-head attention. Q from `z`, K/V from `h`. Causal mask. Output projected back to 768 dim.

**FCAGate**: `g = sigmoid(Linear(h))` → scalar per position ∈ [0,1]

**FCABlock** (wraps nanoGPT `Block`):
```python
h = h + attn(ln1(h))       # standard
h = h + mlp(ln2(h))        # standard
z = bottleneck_mlp(h)       # FCA branch
fca_out = fca_attention(z, h)
g = fca_gate(h)
h = h + g * fca_out         # gated fusion
```

**FCATransformer** (extends nanoGPT `GPT`):
- `fca_layers` list controls which layers get FCA branches
- Default: `[8, 9, 10, 11]` (top third)

### `fca/losses.py`

**Future loss (Option B + EMA)**:
```
target = stop_grad(EMA(h_{l+1}[t+1]))
prediction = linear_proj(z_l[t])
loss = MSE(prediction, target)
EMA decay = 0.999
```

**Lambda schedule**: `lambda(step) = min(1.0, step / 20000)`; `total_loss = CE + lambda * future_loss`

### `fca/config.py`
- Experiment configs as dataclasses
- Sparse schedule presets: `top_third`, `top_quarter`, `last_4`, `all_layers`

---

## Phase 1: Core Training Experiments

**Goal**: Train FCA variants and ablations.

| Run ID | FCA Layers | z Source | Future Loss | Lambda Schedule |
|--------|------------|----------|-------------|-----------------|
| `baseline` | None | N/A | N/A | N/A |
| `fca-top-third` | [8,9,10,11] | Learned bottleneck | MSE + EMA | Linear 20K warmup |
| `fca-random-z` | [8,9,10,11] | Gaussian noise | MSE + EMA | Linear 20K warmup |
| `fca-all-layers` | [0..11] | Learned bottleneck | MSE + EMA | Linear 20K warmup |
| `fca-no-lambda` | [8,9,10,11] | Learned bottleneck | MSE + EMA | Constant 1.0 from step 0 |
| `fca-no-ema` | [8,9,10,11] | Learned bottleneck | MSE (raw) | Linear 20K warmup |

All runs: same data, same total steps (~100K), same base hyperparameters.

**Training script** (`fca/train.py`): Fork nanoGPT's `train.py`, add future loss, lambda schedule, EMA management, gate/z logging. Checkpoint at 25K, 50K, 75K, 100K.

---

## Phase 2: Evaluation

**Goal**: Quantify FCA's impact.

- **2a. Perplexity**: Validation perplexity for all 6 runs
- **2b. Probe accuracy delta** (key metric): Rerun probing harness on all checkpoints. Success = future extractability peaks at earlier layers for FCA vs baseline
- **2c. Gate analysis**: Gate value distributions, per-layer trends over training
- **2d. Confidence saturation**: Early commitment metric — AULC of confidence curves on cloze-style eval sequences. FCA should shift curves leftward
- **2e. Belief state geometry** (optional): PCA on `z` vectors, check for structured clusters

---

## Phase 3: Advanced Experiments (contingent on Phase 2)

- **3a. Early exit simulation**: Entropy-based halting, accuracy vs FLOPs saved
- **3b. Self-speculative decoding**: Project `z` → draft tokens, measure acceptance rate
- **3c. Bottleneck dim sweep**: {48, 96, 192, 384}

---

## Project Structure

```
fca/
├── CLAUDE.md
├── PLAN.md                     # this file
├── PROGRESS.md                 # execution progress tracker
├── Research Plan Review and Prior Work.md
├── nanoGPT/                    # cloned nanoGPT (submodule or vendored)
│   ├── model.py
│   ├── train.py
│   └── data/openwebtext/
├── fca/
│   ├── __init__.py
│   ├── model.py                # FCA modules
│   ├── losses.py               # future loss, lambda, EMA
│   ├── config.py               # experiment configs
│   └── train.py                # training script
├── probes/
│   ├── probe.py
│   ├── extract.py
│   └── train_probes.py
├── eval/
│   ├── perplexity.py
│   ├── probe_delta.py
│   ├── confidence_saturation.py
│   ├── gate_analysis.py
│   └── belief_geometry.py
├── configs/
│   ├── baseline.yaml
│   ├── fca_top_third.yaml
│   ├── fca_random_z.yaml
│   ├── fca_all_layers.yaml
│   ├── fca_no_lambda.yaml
│   └── fca_no_ema.yaml
├── scripts/
│   ├── prepare_data.sh
│   ├── run_baseline.sh
│   ├── run_all_experiments.sh
│   └── run_probes.sh
└── results/
```

---

## Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Bottleneck dim | 192 (768/4) | Research plan recommendation |
| FCA layers | [8, 9, 10, 11] | Top third — Stage 3 "Prediction Ensembling" |
| Future loss target | stop_grad(EMA(h_{l+1}[t+1])) | Option B + EMA |
| EMA decay | 0.999 | Standard (BYOL) |
| Lambda warmup | 20,000 steps | ~20% of training |
| Future loss weight (max) | 1.0 | Start equal to CE loss |
| Batch size | 64 x 1024 tokens | Fits A100 80GB |
| Learning rate | 6e-4 (cosine decay) | nanoGPT default |
| Total steps | 100,000 | Standard convergence |

---

## Verification

- **Unit tests**: Correct shapes, gate ∈ [0,1], stop-grad on targets
- **Smoke test**: Overfit single batch — CE → 0, future loss decreases
- **Probe sanity**: Layer 11 k=1 probe ≈ model's own next-token accuracy
- **Gate sanity**: Starts ~0.5 (uninformed), develops layer-dependent structure after lambda ramp
- **Random-z sanity**: Perplexity within ±2% of baseline
 

note:

`pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`