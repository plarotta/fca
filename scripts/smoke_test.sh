#!/bin/bash
# =============================================================================
# Full pipeline smoke test on a tiny synthetic dataset.
# Runs every stage: data gen → baseline → FCA → probes → eval
# Takes ~2-5 minutes on CPU, ~1 min on GPU.
#
# Usage:
#   bash scripts/smoke_test.sh          # auto-detect device
#   bash scripts/smoke_test.sh cpu      # force CPU
#   bash scripts/smoke_test.sh cuda     # force GPU
# =============================================================================

set -e

DEVICE=${1:-cpu}
# Use MPS on Apple Silicon if available and no explicit choice
if [ "$DEVICE" = "cpu" ] && python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="cpu"  # MPS has limited op support, stay on CPU for safety
fi

SMOKE_DIR="results/smoke"
DATA_DIR="nanoGPT/data/mini"
BLOCK_SIZE=64
N_LAYER=4
N_HEAD=4
N_EMBD=128
VOCAB_SIZE=256
MAX_ITERS=100
BATCH_SIZE=4
GRAD_ACCUM=1
FCA_LAYERS="2 3"
BOTTLENECK_DIM=32
PROBE_SEQS=10
PROBE_EPOCHS=3
EVAL_ITERS=10

echo "========================================="
echo "  FCA Pipeline Smoke Test"
echo "  device=${DEVICE}"
echo "========================================="
echo ""

# -------------------------------------------------
# Step 1: Generate mini dataset
# -------------------------------------------------
echo ">>> Step 1/7: Generating mini dataset..."
uv run python scripts/generate_mini_data.py \
    --output_dir "${DATA_DIR}" \
    --n_train 50000 \
    --n_val 5000 \
    --vocab_size ${VOCAB_SIZE}
echo ""

# -------------------------------------------------
# Step 2: Train baseline (nanoGPT)
# -------------------------------------------------
echo ">>> Step 2/7: Training baseline..."
cd nanoGPT
uv run python train.py \
    --out_dir="../${SMOKE_DIR}/baseline" \
    --dataset=mini \
    --n_layer=${N_LAYER} \
    --n_head=${N_HEAD} \
    --n_embd=${N_EMBD} \
    --block_size=${BLOCK_SIZE} \
    --batch_size=${BATCH_SIZE} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --max_iters=${MAX_ITERS} \
    --lr_decay_iters=${MAX_ITERS} \
    --eval_interval=50 \
    --eval_iters=${EVAL_ITERS} \
    --log_interval=25 \
    --warmup_iters=10 \
    --always_save_checkpoint=True \
    --compile=False \
    --device=${DEVICE} \
    --wandb_log=False
cd ..
echo ""

# -------------------------------------------------
# Step 3: Train FCA (primary config)
# -------------------------------------------------
echo ">>> Step 3/7: Training FCA (top-third)..."
uv run python -m fca.train \
    --out_dir "${SMOKE_DIR}/fca-top-third" \
    --dataset mini \
    --n_layer ${N_LAYER} \
    --n_head ${N_HEAD} \
    --n_embd ${N_EMBD} \
    --block_size ${BLOCK_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_iters ${MAX_ITERS} \
    --lr_decay_iters ${MAX_ITERS} \
    --eval_interval 50 \
    --eval_iters ${EVAL_ITERS} \
    --log_interval 25 \
    --warmup_iters 10 \
    --fca_layers ${FCA_LAYERS} \
    --bottleneck_dim ${BOTTLENECK_DIM} \
    --fca_n_head ${N_HEAD} \
    --lambda_warmup_steps 50 \
    --checkpoint_interval ${MAX_ITERS} \
    --compile False \
    --device ${DEVICE}
echo ""

# -------------------------------------------------
# Step 4: Train FCA-random-z (ablation)
# -------------------------------------------------
echo ">>> Step 4/7: Training FCA (random-z ablation)..."
uv run python -m fca.train \
    --out_dir "${SMOKE_DIR}/fca-random-z" \
    --dataset mini \
    --n_layer ${N_LAYER} \
    --n_head ${N_HEAD} \
    --n_embd ${N_EMBD} \
    --block_size ${BLOCK_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_iters ${MAX_ITERS} \
    --lr_decay_iters ${MAX_ITERS} \
    --eval_interval 50 \
    --eval_iters ${EVAL_ITERS} \
    --log_interval 25 \
    --warmup_iters 10 \
    --fca_layers ${FCA_LAYERS} \
    --bottleneck_dim ${BOTTLENECK_DIM} \
    --fca_n_head ${N_HEAD} \
    --lambda_warmup_steps 50 \
    --random_z \
    --checkpoint_interval ${MAX_ITERS} \
    --compile False \
    --device ${DEVICE}
echo ""

# -------------------------------------------------
# Step 5: Extract hidden states + train probes
# -------------------------------------------------
echo ">>> Step 5/7: Extracting hidden states & training probes..."

for MODEL_NAME in baseline fca-top-third fca-random-z; do
    CKPT="${SMOKE_DIR}/${MODEL_NAME}/ckpt.pt"
    if [ ! -f "$CKPT" ]; then
        echo "  Skipping ${MODEL_NAME} (no checkpoint found)"
        continue
    fi

    HIDDEN_DIR="${SMOKE_DIR}/hidden_states/${MODEL_NAME}"
    echo "  Extracting: ${MODEL_NAME}..."
    uv run python -m probes.extract \
        --checkpoint "${CKPT}" \
        --data_path "${DATA_DIR}/val.bin" \
        --output_dir "${HIDDEN_DIR}" \
        --n_sequences ${PROBE_SEQS} \
        --batch_size 5 \
        --device ${DEVICE}

    echo "  Training probes: ${MODEL_NAME}..."
    uv run python -m probes.train_probes \
        --hidden_states "${HIDDEN_DIR}/hidden_states.pt" \
        --output_dir "${SMOKE_DIR}/probes" \
        --model_name "${MODEL_NAME}" \
        --vocab_size ${VOCAB_SIZE} \
        --hidden_dim ${N_EMBD} \
        --block_size ${BLOCK_SIZE} \
        --n_layers ${N_LAYER} \
        --device ${DEVICE} \
        --probe_epochs ${PROBE_EPOCHS} \
        --probe_batch_size 256
    echo ""
done

# -------------------------------------------------
# Step 6: Probe delta comparison
# -------------------------------------------------
echo ">>> Step 6/7: Computing probe deltas..."
BASELINE_JSON="${SMOKE_DIR}/probes/baseline_probes.json"
FCA_JSON="${SMOKE_DIR}/probes/fca-top-third_probes.json"
RANDOM_JSON="${SMOKE_DIR}/probes/fca-random-z_probes.json"

if [ -f "$BASELINE_JSON" ] && [ -f "$FCA_JSON" ]; then
    FCA_FILES=("$FCA_JSON")
    FCA_NAMES=("fca-top-third")
    if [ -f "$RANDOM_JSON" ]; then
        FCA_FILES+=("$RANDOM_JSON")
        FCA_NAMES+=("fca-random-z")
    fi
    uv run python -m eval.probe_delta \
        --baseline "$BASELINE_JSON" \
        --fca "${FCA_FILES[@]}" \
        --fca_names "${FCA_NAMES[@]}" \
        --output_dir "${SMOKE_DIR}/eval"
fi
echo ""

# -------------------------------------------------
# Step 7: Perplexity + confidence saturation
# -------------------------------------------------
echo ">>> Step 7/7: Running eval metrics..."

CKPTS=()
NAMES=()
for MODEL_NAME in baseline fca-top-third fca-random-z; do
    CKPT="${SMOKE_DIR}/${MODEL_NAME}/ckpt.pt"
    if [ -f "$CKPT" ]; then
        CKPTS+=("$CKPT")
        NAMES+=("$MODEL_NAME")
    fi
done

if [ ${#CKPTS[@]} -ge 1 ]; then
    echo "  Perplexity..."
    uv run python -m eval.perplexity \
        --checkpoints "${CKPTS[@]}" \
        --names "${NAMES[@]}" \
        --data_path "${DATA_DIR}/val.bin" \
        --batch_size ${BATCH_SIZE} \
        --eval_iters ${EVAL_ITERS} \
        --device ${DEVICE}
    echo ""

    echo "  Confidence saturation..."
    uv run python -m eval.confidence_saturation \
        --checkpoints "${CKPTS[@]}" \
        --names "${NAMES[@]}" \
        --data_path "${DATA_DIR}/val.bin" \
        --n_sequences 20 \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --output_dir "${SMOKE_DIR}/eval"
fi

echo ""
echo "========================================="
echo "  Smoke test complete!"
echo "  Results in: ${SMOKE_DIR}/"
echo "========================================="
