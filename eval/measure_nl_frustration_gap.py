#!/usr/bin/env python3
"""
Measure NL frustration gap: forced-clean tokenization at NL delimiter characters.

The original frustration gap measurement uses 16 structured-data barrier characters
(pipe, @, angle brackets, etc.) and shows 0pp on web text because those characters
barely appear in prose. This script measures the gap using NL structural characters
(period, hyphen, apostrophe, etc.) which appear in every sentence.

If the NL gap is nonzero on FineWeb, it means standard BPE is corrupting natural
language structural boundaries, not just structured data boundaries.

Measures both struct and NL gaps side by side on 3 probe texts for comparison.

Written for the atlas re-probe (v2 data with spacing behavior).
Provenance: Written 2026-07-04 for the developmental atlas project.

Usage:
  python measure_nl_frustration_gap.py \\
    --tokenizer /path/to/tokenizer.json \\
    --checkpoint /path/to/step-20000.pt \\
    --probe-dir ../probes/

  # R2 mode (downloads checkpoint, measures, deletes):
  python measure_nl_frustration_gap.py \\
    --tokenizer /path/to/tokenizer.json \\
    --r2-checkpoint atlas/runs/baseline/checkpoints/step-20000.pt \\
    --probe-dir ../probes/
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import numpy as np


NL_DELIMITER_CHARS = set(".'?!-\"();:")
STRUCT_DELIMITER_CHARS = set('|@<>"\',:;\t\n{}[]()')


def get_r2_client():
    """Return boto3 S3 client for R2."""
    import boto3
    return boto3.client("s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"])


R2_BUCKET = "structok-training"


def get_attention(model, token_ids, device):
    """Run forward pass and return attention weights per layer."""
    input_ids = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)
    return [a[0].cpu() for a in outputs.attentions]


def measure_char_attention(attentions, token_ids, tokenizer, seq_len, char_set):
    """Measure attention mass on positions containing any character in char_set."""
    positions = []
    for pos, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid])
        if any(c in char_set for c in decoded):
            positions.append(pos)

    if not positions:
        return [[0.0] * attentions[0].shape[0]] * len(attentions)

    mask = torch.zeros(seq_len, dtype=torch.bool)
    for p in positions:
        mask[p] = True

    scores = []
    for layer_attn in attentions:
        mass = layer_attn[:, :, mask].sum(dim=-1).mean(dim=-1)
        scores.append(mass.tolist())
    return scores


def segment_and_tokenize(text, tokenizer, char_set):
    """Tokenize with forced isolation at each character in char_set."""
    segments = []
    current = []
    for ch in text:
        if ch in char_set:
            if current:
                segments.append(''.join(current))
                current = []
            segments.append(ch)
        else:
            current.append(ch)
    if current:
        segments.append(''.join(current))

    all_ids = []
    for seg in segments:
        all_ids.extend(tokenizer.encode(seg).ids)
    return all_ids


def measure_gap(model, text, tokenizer, device, char_set, label):
    """Measure frustration gap for a given character set. Returns gap value."""
    normal_ids = tokenizer.encode(text).ids[:1024]
    normal_attn = get_attention(model, normal_ids, device)
    normal_scores = measure_char_attention(
        normal_attn, normal_ids, tokenizer, len(normal_ids), char_set)

    clean_ids = segment_and_tokenize(text, tokenizer, char_set)[:1024]
    clean_attn = get_attention(model, clean_ids, device)
    clean_scores = measure_char_attention(
        clean_attn, clean_ids, tokenizer, len(clean_ids), char_set)

    all_normal = [v for layer in normal_scores for v in layer]
    all_clean = [v for layer in clean_scores for v in layer]
    all_deltas = [c - n for n, c in zip(all_normal, all_clean)]

    gap = float(np.mean(all_deltas))
    woke = sum(1 for d in all_deltas if d > 0.05)

    # Count how many positions matched
    normal_count = sum(1 for pos, tid in enumerate(normal_ids)
                       if any(c in char_set for c in tokenizer.decode([tid])))

    print("  %s gap: %.4f (%.1f pp), %d/%d heads woke, %d/%d positions matched" % (
        label, gap, gap * 100, woke, len(all_deltas),
        normal_count, len(normal_ids)))

    return {
        "gap": round(gap, 6),
        "gap_pp": round(gap * 100, 2),
        "heads_woke": woke,
        "total_heads": len(all_deltas),
        "normal_mean": round(float(np.mean(all_normal)), 6),
        "clean_mean": round(float(np.mean(all_clean)), 6),
        "positions_matched": normal_count,
        "total_positions": len(normal_ids),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure NL frustration gap")
    parser.add_argument("--checkpoint", type=str, help="Local checkpoint path")
    parser.add_argument("--r2-checkpoint", type=str, help="R2 key for checkpoint")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--probe-dir", default="probes/")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--run-name", default="unknown", help="Run name for labeling")
    parser.add_argument("--size", default="410m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    probes = {
        "prose": (probe_dir / "prose.txt").read_text(),
        "code": (probe_dir / "code.txt").read_text(),
        "structured": (probe_dir / "structured.txt").read_text(),
    }

    # Load tokenizer
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
        cp_path = "/tmp/nl_gap_checkpoint.pt"
        print("Downloading %s..." % args.r2_checkpoint)
        s3 = get_r2_client()
        s3.download_file(R2_BUCKET, args.r2_checkpoint, cp_path)

    cp = torch.load(cp_path, map_location=args.device, weights_only=False)
    model.load_state_dict(cp.get("model_state_dict", cp), strict=False)
    del cp
    if args.r2_checkpoint and os.path.exists(cp_path):
        os.remove(cp_path)
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("NL FRUSTRATION GAP: %s" % args.run_name)
    print("=" * 60)

    results = {}
    for probe_name, text in probes.items():
        print("\nProbe: %s" % probe_name)
        struct_gap = measure_gap(model, text, tok, args.device,
                                  STRUCT_DELIMITER_CHARS, "struct")
        nl_gap = measure_gap(model, text, tok, args.device,
                              NL_DELIMITER_CHARS, "NL    ")
        results[probe_name] = {
            "struct_gap": struct_gap,
            "nl_gap": nl_gap,
        }

    # Summary
    struct_avg = np.mean([r["struct_gap"]["gap"] for r in results.values()])
    nl_avg = np.mean([r["nl_gap"]["gap"] for r in results.values()])
    print("\n" + "-" * 40)
    print("Average struct gap: %.4f (%.1f pp)" % (struct_avg, struct_avg * 100))
    print("Average NL gap:     %.4f (%.1f pp)" % (nl_avg, nl_avg * 100))

    if args.output:
        output = {
            "run": args.run_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "results": results,
            "summary": {
                "struct_gap_avg": round(float(struct_avg), 6),
                "nl_gap_avg": round(float(nl_avg), 6),
            },
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print("\nSaved to %s" % args.output)


if __name__ == "__main__":
    main()
