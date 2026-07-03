#!/bin/bash
# Atlas instance setup script
# Run on vast.ai instance after creation

set -e

echo "=== Atlas Instance Setup ==="

# Install dependencies
pip install -q torch transformers tokenizers datasets numpy

# Clone repo
cd /root
git clone https://github.com/blackwell-systems/attention-head-atlas.git atlas
cd atlas

# Copy tokenizers from merge-barriers (or download from R2)
mkdir -p /root/tokenizers
echo "Tokenizers needed in /root/tokenizers/:"
echo "  standard-64k.json"
echo "  structok-64k.json"

echo ""
echo "=== Ready ==="
echo "Next steps:"
echo "  1. Copy tokenizers to /root/tokenizers/"
echo "  2. Run: python3 eval/prep_data.py --all \\"
echo "       --tokenizer-a /root/tokenizers/standard-64k.json \\"
echo "       --tokenizer-b /root/tokenizers/structok-64k.json \\"
echo "       --output-dir /root/data/"
echo "  3. Run: python3 eval/train_atlas.py \\"
echo "       --tokenizer /root/tokenizers/standard-64k.json \\"
echo "       --data /root/data/atlas-standard-64k.bin \\"
echo "       --output-dir /root/runs/baseline/"
echo "  4. Run: python3 eval/probe_heads.py \\"
echo "       --checkpoint-dir /root/runs/baseline/checkpoints/ \\"
echo "       --tokenizer /root/tokenizers/standard-64k.json \\"
echo "       --output-dir /root/results/baseline/"
