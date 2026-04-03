#!/bin/bash
# Run probes on all model checkpoints, then generate comparison plots.
# Run from project root: bash scripts/run_all_probes.sh [device]

set -e

DEVICE=${1:-cuda}
PROBE_DIR="results/probes"

MODELS=(
    "baseline:results/baseline/ckpt.pt"
    "fca-top-third:results/fca-top-third/ckpt.pt"
    "fca-random-z:results/fca-random-z/ckpt.pt"
    "fca-all-layers:results/fca-all-layers/ckpt.pt"
    "fca-no-lambda:results/fca-no-lambda/ckpt.pt"
    "fca-no-ema:results/fca-no-ema/ckpt.pt"
)

for entry in "${MODELS[@]}"; do
    IFS=':' read -r name ckpt <<< "$entry"
    if [ -f "$ckpt" ]; then
        echo "=== Running probes for ${name} ==="
        bash scripts/run_probes.sh "$ckpt" "$name" "$DEVICE"
        echo ""
    else
        echo "=== Skipping ${name} (checkpoint not found: ${ckpt}) ==="
    fi
done

echo "=== Generating comparison plot ==="
JSON_FILES=()
LABELS=()
for entry in "${MODELS[@]}"; do
    IFS=':' read -r name ckpt <<< "$entry"
    json="${PROBE_DIR}/${name}_probes.json"
    if [ -f "$json" ]; then
        JSON_FILES+=("$json")
        LABELS+=("$name")
    fi
done

if [ ${#JSON_FILES[@]} -ge 2 ]; then
    uv run python -m probes.train_probes \
        --compare "${JSON_FILES[@]}" \
        --compare_labels "${LABELS[@]}" \
        --output_dir "$PROBE_DIR" \
        --hidden_states dummy  # required arg but unused in compare mode
fi

echo "=== All probes complete ==="
