#!/bin/bash
# Train the baseline GPT-2 Small model using nanoGPT
# Run from project root: bash scripts/run_baseline.sh

set -e

echo "=== Training baseline GPT-2 Small ==="
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
    --max_iters=100000 \
    --lr_decay_iters=100000 \
    --warmup_iters=2000 \
    --min_lr=6e-5 \
    --eval_interval=2000 \
    --eval_iters=200 \
    --gradient_accumulation_steps=40 \
    --device=cuda \
    --dtype=bfloat16 \
    --compile=True \
    --wandb_log=True \
    --wandb_project=fca \
    --wandb_run_name=baseline
cd ..
echo "=== Baseline training complete ==="
