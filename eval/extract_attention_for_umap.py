#!/usr/bin/env python3
"""
Extract full attention matrices for token-level UMAP visualization.

For each probe text, runs a forward pass and saves the per-token attention
profile: for each token position, how much each of the 384 heads attends
TO that position (averaged across all query positions). This produces a
384-dimensional vector per token, suitable for UMAP projection.

Extends Wang et al. (2025b) "Embryology of a Language Model" from 16
dimensions (their 3M model) to 384 dimensions (our 410M model).

Provenance: written 2026-07-05 for the developmental atlas project.

Usage:
  python extract_attention_for_umap.py \
    --checkpoint step-20000.pt \
    --tokenizer standard-64k.json \
    --probe-dir probes/ \
    --output attention-baseline.npz \
    --run-name baseline
"""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path


SPACING_CHARS = set(' \t\n\r')
DELIMITER_CHARS = set('|@<>"\',:;\t\n{}[]()')
BRACKET_CHARS = set('([{}])')


def classify_token(decoded, token_id):
    """Classify a token following Wang et al. (2025b) Table 1 categories."""
    if not decoded:
        return 'other'
    if all(c in SPACING_CHARS for c in decoded):
        return 'spacing'
    if any(c in DELIMITER_CHARS for c in decoded):
        return 'delimiter'
    if any(c in BRACKET_CHARS for c in decoded):
        return 'bracket'
    if decoded[0] == ' ' and len(decoded) > 1 and decoded[1:].isalpha():
        return 'word_start'
    if decoded.isdigit() or (len(decoded) > 1 and decoded.replace('.','').replace('-','').isdigit()):
        return 'numeric'
    if decoded.isalpha():
        return 'word_part'
    return 'other'


def extract_attention(model, tokenizer, text, device):
    """Run forward pass, return per-token attention profile and token info."""
    token_ids = tokenizer.encode(text).ids[:512]
    seq_len = len(token_ids)

    input_ids = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)

    # Stack all attention: [layers, heads, seq, seq]
    # Each attention[l] is [1, heads, seq, seq], squeeze batch dim
    all_attn = torch.stack([a[0] for a in outputs.attentions])  # [24, 16, seq, seq]

    # For each token position k, compute how much each head attends TO k
    # Average across all query positions: mean over dim=-2 (queries)
    # Result: [24, 16, seq] = per-head, per-key-position attention received
    attn_received = all_attn.mean(dim=-2)  # [24, 16, seq]

    # Reshape to [seq, 384] (flatten layers x heads)
    token_vectors = attn_received.permute(2, 0, 1).reshape(seq_len, -1).cpu().numpy()

    # Classify each token
    token_types = []
    token_strs = []
    for tid in token_ids:
        decoded = tokenizer.decode([tid])
        token_types.append(classify_token(decoded, tid))
        token_strs.append(decoded)

    return token_vectors, token_types, token_strs, token_ids


def main():
    parser = argparse.ArgumentParser(description="Extract attention for UMAP")
    parser.add_argument("--checkpoint", type=str, help="Local checkpoint path")
    parser.add_argument("--r2-checkpoint", type=str, help="R2 key for checkpoint")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--probe-dir", default="probes/")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--run-name", default="unknown")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    probes = {}
    for name in ["prose", "code", "structured", "induction", "duplicates", "brackets"]:
        path = probe_dir / ("%s.txt" % name)
        if path.exists():
            probes[name] = path.read_text()

    # Load model
    from tokenizers import Tokenizer
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    config = GPTNeoXConfig(
        vocab_size=vocab_size, hidden_size=1024, num_hidden_layers=24,
        num_attention_heads=16, intermediate_size=4096,
        max_position_embeddings=2048, attn_implementation="eager")
    model = GPTNeoXForCausalLM(config).to(args.device)
    model.eval()

    # Load checkpoint
    cp_path = args.checkpoint
    if args.r2_checkpoint:
        cp_path = "/tmp/umap_checkpoint.pt"
        print("Downloading %s..." % args.r2_checkpoint)
        import boto3
        s3 = boto3.client("s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["R2_SECRET_KEY"])
        s3.download_file("structok-training", args.r2_checkpoint, cp_path)

    cp = torch.load(cp_path, map_location=args.device, weights_only=False)
    model.load_state_dict(cp.get("model_state_dict", cp), strict=False)
    del cp
    if args.r2_checkpoint and os.path.exists(cp_path):
        os.remove(cp_path)
    torch.cuda.empty_cache()

    print("Model loaded on %s" % args.device)

    # Extract attention for each probe
    all_vectors = []
    all_types = []
    all_strs = []
    all_probe_names = []

    for probe_name, text in probes.items():
        print("  %s..." % probe_name, end=" ", flush=True)
        vectors, types, strs, _ = extract_attention(model, tok, text, args.device)
        all_vectors.append(vectors)
        all_types.extend(types)
        all_strs.extend(strs)
        all_probe_names.extend([probe_name] * len(types))
        print("%d tokens" % len(types))

    all_vectors = np.concatenate(all_vectors, axis=0)

    print("\nTotal: %d tokens, %d dimensions" % all_vectors.shape)
    print("Token type distribution:")
    from collections import Counter
    for t, count in Counter(all_types).most_common():
        print("  %s: %d" % (t, count))

    # Save
    np.savez_compressed(args.output,
        vectors=all_vectors,
        types=np.array(all_types),
        strs=np.array(all_strs),
        probe_names=np.array(all_probe_names),
        run_name=args.run_name)

    print("\nSaved to %s (%.1f MB)" % (args.output, os.path.getsize(args.output) / 1e6))


if __name__ == "__main__":
    main()
