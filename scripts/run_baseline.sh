#!/bin/bash
# Train the baseline GPT-2 Small model using nanoGPT
# Run from project root: bash scripts/run_baseline.sh

set -e

TORCH_COMPILE=${TORCH_COMPILE:-False}
MAX_ITERS=${MAX_ITERS:-10000}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-${MAX_ITERS}}
WARMUP_ITERS=${WARMUP_ITERS:-200}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
EVAL_ITERS=${EVAL_ITERS:-100}

echo "=== Training baseline GPT-2 Small ==="
echo "=== Quick budget: max_iters=${MAX_ITERS}, eval_interval=${EVAL_INTERVAL}, eval_iters=${EVAL_ITERS} ==="
cd nanoGPT
python train.py \
    --out_dir=../results/baseline \
    --n_layer=12 \
    --n_head=12 \
    --n_embd=768 \
    --batch_size=12 \
    --block_size=1024 \
    --bias=False \
    --dropout=0.0 \
    --learning_rate=6e-4 \
    --max_iters="${MAX_ITERS}" \
    --lr_decay_iters="${LR_DECAY_ITERS}" \
    --warmup_iters="${WARMUP_ITERS}" \
    --min_lr=6e-5 \
    --eval_interval="${EVAL_INTERVAL}" \
    --eval_iters="${EVAL_ITERS}" \
    --gradient_accumulation_steps=40 \
    --device=cuda \
    --dtype=bfloat16 \
    --compile="${TORCH_COMPILE}" \
    --wandb_log=True \
    --wandb_project=fca \
    --wandb_run_name=baseline
cd ..
echo "=== Baseline training complete ==="
