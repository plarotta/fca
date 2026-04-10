#!/bin/bash
# Extract hidden states and train probes for a given checkpoint.
# Usage: bash scripts/run_probes.sh <checkpoint_path> <model_name> [device]
# Example: bash scripts/run_probes.sh results/baseline/ckpt.pt baseline cuda

set -euo pipefail

CHECKPOINT=${1:?Usage: run_probes.sh <checkpoint_path> <model_name> [device]}
MODEL_NAME=${2:?Usage: run_probes.sh <checkpoint_path> <model_name> [device]}
DEVICE=${3:-cuda}
PROBE_SEQUENCES=${PROBE_SEQUENCES:-500}
EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE:-32}
PROBE_EPOCHS=${PROBE_EPOCHS:-10}
KEEP_HIDDEN_STATES=${KEEP_HIDDEN_STATES:-0}

HIDDEN_DIR="results/hidden_states/${MODEL_NAME}"
PROBE_DIR="results/probes"
HIDDEN_STATES_PATH="${HIDDEN_DIR}/hidden_states.pt"

echo "=== Step 1: Extracting hidden states from ${CHECKPOINT} ==="
echo "=== Using PROBE_SEQUENCES=${PROBE_SEQUENCES} (set KEEP_HIDDEN_STATES=1 to retain dumps) ==="
python -m probes.extract \
    --checkpoint "${CHECKPOINT}" \
    --data_path nanoGPT/data/openwebtext/val.bin \
    --output_dir "${HIDDEN_DIR}" \
    --n_sequences "${PROBE_SEQUENCES}" \
    --batch_size "${EXTRACT_BATCH_SIZE}" \
    --device "${DEVICE}"

echo ""
echo "=== Step 2: Training probes ==="
python -m probes.train_probes \
    --hidden_states "${HIDDEN_STATES_PATH}" \
    --output_dir "${PROBE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --device "${DEVICE}" \
    --probe_epochs "${PROBE_EPOCHS}"

if [ "${KEEP_HIDDEN_STATES}" != "1" ] && [ -f "${HIDDEN_STATES_PATH}" ]; then
    echo ""
    echo "=== Step 3: Cleaning up hidden state dump to save disk ==="
    rm -f "${HIDDEN_STATES_PATH}"
fi

echo ""
echo "=== Done. Results at ${PROBE_DIR}/${MODEL_NAME}_probes.json ==="
