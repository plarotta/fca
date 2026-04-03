# FCA Project Progress

> **Workflow rule**: Execute tasks in steps. After completing each step, update this file before moving on.

---

## Phase -1: Baseline + Probe Calibration

- [x] Clone and configure nanoGPT
- [ ] Prepare OpenWebText dataset (`bash scripts/prepare_data.sh`)
- [ ] Train GPT-2 Small baseline (`bash scripts/run_baseline.sh`)
- [x] Build probing harness (`probes/probe.py`, `extract.py`, `train_probes.py`) — 5/5 tests passing
- [ ] Run baseline probes (`bash scripts/run_probes.sh results/baseline/ckpt.pt baseline cuda`)
- [ ] Generate baseline probe accuracy plots

## Phase 0: FCA Architecture Implementation

- [x] `fca/config.py` — experiment config dataclasses
- [x] `fca/model.py` — BottleneckMLP
- [x] `fca/model.py` — FCAAttention
- [x] `fca/model.py` — FCAGate
- [x] `fca/model.py` — FCABlock (wraps nanoGPT Block)
- [x] `fca/model.py` — FCAGPT (extends nanoGPT GPT)
- [x] `fca/losses.py` — future hidden state loss (MSE + EMA + stop-grad)
- [x] `fca/losses.py` — lambda schedule
- [x] `fca/train.py` — training script (fork nanoGPT, add FCA losses + logging)
- [x] Unit tests — 17/17 passing (shape checks, gate bounds, causality, smoke overfit)

## Phase 1: Core Training Experiments

- [ ] Train `fca-top-third` (primary config)
- [ ] Train `fca-random-z` (critical ablation)
- [ ] Train `fca-all-layers` (placement ablation)
- [ ] Train `fca-no-lambda` (schedule ablation)
- [ ] Train `fca-no-ema` (target ablation)

## Phase 2: Evaluation

- [x] `eval/perplexity.py` — validation perplexity for all runs
- [x] `eval/probe_delta.py` — probe accuracy delta vs baseline
- [x] `eval/gate_analysis.py` — gate value distributions and trends
- [x] `eval/confidence_saturation.py` — early commitment / AULC metric
- [x] `eval/belief_geometry.py` — PCA on z vectors (optional)
- [ ] Generate comparison plots and summary tables (requires trained checkpoints)

## Phase 3: Advanced Experiments (contingent)

- [ ] Early exit simulation (accuracy vs FLOPs)
- [ ] Self-speculative decoding from z
- [ ] Bottleneck dimension sweep {48, 96, 192, 384}

---

## Log

| Date | Step | Notes |
|------|------|-------|
| 2026-04-02 | Project kickoff | Created PLAN.md, PROGRESS.md, CLAUDE.md. Research plan reviewed. |
| 2026-04-02 | Phase 0 complete | Implemented all FCA modules (config, model, losses, train). 17/17 unit tests passing. nanoGPT cloned. uv project initialized. |
| 2026-04-02 | Phase -1 code complete | Built probing harness (probe.py, extract.py, train_probes.py). 5/5 probe tests passing. Shell scripts for data prep, baseline training, experiment runs, and probe pipeline. |
| 2026-04-02 | Phase 2 eval scripts | Built all 5 eval scripts (perplexity, probe_delta, gate_analysis, confidence_saturation, belief_geometry). 22/22 total tests passing. |
| 2026-04-02 | Smoke test pipeline | Added `scripts/smoke_test.sh` — runs full pipeline (data gen → baseline → FCA → random-z → probes → eval) on CPU with tiny model in ~2 min. Verified end-to-end. |
