#!/bin/bash
# Run all FCA training experiments sequentially.
# Run from project root: bash scripts/run_all_experiments.sh

set -e

echo "========================================="
echo "=== FCA Experiment Suite              ==="
echo "========================================="

echo ""
echo "=== 1/5: fca-top-third (primary) ==="
python -m fca.train \
    --out_dir results/fca-top-third \
    --fca_layers 8 9 10 11 \
    --bottleneck_dim 192 \
    --device cuda \
    --dtype bfloat16 \
    --compile True \
    --wandb_log \
    --wandb_run_name fca-top-third

echo ""
echo "=== 2/5: fca-random-z (critical ablation) ==="
python -m fca.train \
    --out_dir results/fca-random-z \
    --fca_layers 8 9 10 11 \
    --bottleneck_dim 192 \
    --random_z \
    --device cuda \
    --dtype bfloat16 \
    --compile True \
    --wandb_log \
    --wandb_run_name fca-random-z

echo ""
echo "=== 3/5: fca-all-layers (placement ablation) ==="
python -m fca.train \
    --out_dir results/fca-all-layers \
    --fca_layers 0 1 2 3 4 5 6 7 8 9 10 11 \
    --bottleneck_dim 192 \
    --device cuda \
    --dtype bfloat16 \
    --compile True \
    --wandb_log \
    --wandb_run_name fca-all-layers

echo ""
echo "=== 4/5: fca-no-lambda (schedule ablation) ==="
python -m fca.train \
    --out_dir results/fca-no-lambda \
    --fca_layers 8 9 10 11 \
    --bottleneck_dim 192 \
    --no_lambda_schedule \
    --device cuda \
    --dtype bfloat16 \
    --compile True \
    --wandb_log \
    --wandb_run_name fca-no-lambda

echo ""
echo "=== 5/5: fca-no-ema (target ablation) ==="
python -m fca.train \
    --out_dir results/fca-no-ema \
    --fca_layers 8 9 10 11 \
    --bottleneck_dim 192 \
    --no_ema_target \
    --device cuda \
    --dtype bfloat16 \
    --compile True \
    --wandb_log \
    --wandb_run_name fca-no-ema

echo ""
echo "========================================="
echo "=== All experiments complete          ==="
echo "========================================="
