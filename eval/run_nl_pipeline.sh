#!/bin/bash
# Full NL-barrier pipeline: download corpus, train tokenizer, pretokenize, train model, probe
# Runs on the same instance after seed2 finishes
set -e
cd /root

echo "=== NL BARRIER PIPELINE ==="
date

# Step 1: Download FineWeb corpus (if not already present)
if [ ! -d /root/corpus ] || [ "$(ls /root/corpus/*.txt 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "Step 1: Downloading FineWeb corpus..."
    python3 prep_data.py --download --corpus-dir /root/corpus/ --target-gb 5
else
    echo "Step 1: Corpus already present ($(ls /root/corpus/*.txt | wc -l) files)"
fi

# Step 2: Train NL-barrier tokenizer
echo "Step 2: Training NL-barrier tokenizer..."
python3 train_nl_tokenizer.py \
    --corpus-dir /root/corpus/ \
    --output /root/tokenizers/nl-barrier-64k.json \
    --vocab-size 65536

# Step 3: Pretokenize with NL-barrier tokenizer
echo "Step 3: Pretokenizing..."
python3 prep_data.py --tokenize \
    --tokenizer /root/tokenizers/nl-barrier-64k.json \
    --corpus-dir /root/corpus/ \
    --output /root/data/atlas-nl-barrier-64k.bin

echo "Bin:" && du -h /root/data/atlas-nl-barrier-64k.bin

# Step 4: Train
echo "Step 4: Training..."
python3 train_atlas.py \
    --tokenizer /root/tokenizers/nl-barrier-64k.json \
    --data /root/data/atlas-nl-barrier-64k.bin \
    --run-name nl-barrier \
    --r2-prefix atlas/runs/nl-barrier \
    --output-dir /root/runs/nl-barrier \
    --steps 20000

# Step 5: Probe locally
echo "Step 5: Probing..."
python3 probe_heads.py \
    --checkpoint-dir /root/runs/nl-barrier/checkpoints/ \
    --tokenizer /root/tokenizers/nl-barrier-64k.json \
    --probe-dir /root/probes/ \
    --size 410m \
    --output-dir /root/results/nl-barrier/

# Step 6: Upload results to R2
echo "Step 6: Uploading results to R2..."
python3 -c "
import boto3
from pathlib import Path
s3 = boto3.client('s3', endpoint_url=os.environ['R2_ENDPOINT'], aws_access_key_id=os.environ['R2_ACCESS_KEY'], aws_secret_access_key=os.environ['R2_SECRET_KEY'])

# Upload tokenizer
s3.upload_file('/root/tokenizers/nl-barrier-64k.json', 'structok-training', 'atlas/tokens/nl-barrier-64k.json')
print('Uploaded tokenizer')

# Upload pretokenized bin
s3.upload_file('/root/data/atlas-nl-barrier-64k.bin', 'structok-training', 'atlas/tokens/atlas-nl-barrier-64k.bin')
print('Uploaded bin')

# Upload probe results
for f in sorted(Path('/root/results/nl-barrier').glob('*.json')):
    s3.upload_file(str(f), 'structok-training', 'atlas/results/nl-barrier/%s' % f.name)
print('Uploaded %d results' % len(list(Path('/root/results/nl-barrier').glob('*.json'))))
"

echo "=== NL BARRIER PIPELINE COMPLETE ==="
date
