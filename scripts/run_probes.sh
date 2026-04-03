#!/bin/bash
# Extract hidden states and train probes for a given checkpoint.
# Usage: bash scripts/run_probes.sh <checkpoint_path> <model_name> [device]
# Example: bash scripts/run_probes.sh results/baseline/ckpt.pt baseline cuda

set -e

CHECKPOINT=${1:?Usage: run_probes.sh <checkpoint_path> <model_name> [device]}
MODEL_NAME=${2:?Usage: run_probes.sh <checkpoint_path> <model_name> [device]}
DEVICE=${3:-cuda}

HIDDEN_DIR="results/hidden_states/${MODEL_NAME}"
PROBE_DIR="results/probes"

echo "=== Step 1: Extracting hidden states from ${CHECKPOINT} ==="
uv run python -m probes.extract \
    --checkpoint "${CHECKPOINT}" \
    --data_path nanoGPT/data/openwebtext/val.bin \
    --output_dir "${HIDDEN_DIR}" \
    --n_sequences 1000 \
    --batch_size 32 \
    --device "${DEVICE}"

echo ""
echo "=== Step 2: Training probes ==="
uv run python -m probes.train_probes \
    --hidden_states "${HIDDEN_DIR}/hidden_states.pt" \
    --output_dir "${PROBE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --device "${DEVICE}" \
    --probe_epochs 10

echo ""
echo "=== Done. Results at ${PROBE_DIR}/${MODEL_NAME}_probes.json ==="
