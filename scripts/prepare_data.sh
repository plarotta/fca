#!/bin/bash
# Download and tokenize OpenWebText for nanoGPT
# Run from project root: bash scripts/prepare_data.sh

set -e

echo "=== Preparing OpenWebText dataset ==="
cd nanoGPT
uv run python data/openwebtext/prepare.py
cd ..
echo "=== Done. Data at nanoGPT/data/openwebtext/{train,val}.bin ==="
