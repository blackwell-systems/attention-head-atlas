#!/usr/bin/env python3
"""
Multi-behavior head probing: classify every attention head across 8 behavior types.

Run at each training checkpoint to build the developmental atlas.

For each head, computes:
- positional_prev: attention mass on position n-1
- positional_p0: attention mass on position 0
- induction: copy score (attention to token after previous occurrence of current query token)
- delimiter: attention mass on delimiter token positions
- bracket: attention from close-bracket to matching open-bracket
- content: correlation between attention and embedding cosine similarity
- duplicate: attention to previous occurrences of same token
- dormant: HONOR metric (output norm relative to layer average)

Usage:
  python probe_heads.py \
    --checkpoint path/to/checkpoint.pt \
    --tokenizer path/to/tokenizer.json \
    --probe-dir ../probes/ \
    --output ../results/step-XXXXX.json

  # Batch mode (all checkpoints in a directory):
  python probe_heads.py \
    --checkpoint-dir path/to/checkpoints/ \
    --tokenizer path/to/tokenizer.json \
    --probe-dir ../probes/ \
    --output-dir ../results/
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np


MODEL_CONFIGS = {
    "410m": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_position_embeddings": 2048,
        "arch": "neox",
    },
}

DELIMITER_CHARS = set('|@<>"\',:;\t\n{}[]()')
BRACKET_OPEN = set('([{')
BRACKET_CLOSE = set(')]}')
BRACKET_PAIRS = {')': '(', ']': '[', '}': '{'}


def load_model(checkpoint_path, size, tokenizer_path):
    from tokenizers import Tokenizer
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    config = MODEL_CONFIGS[size]

    model_config = GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        max_position_embeddings=config["max_position_embeddings"],
        attn_implementation="eager",
    )
    model = GPTNeoXForCausalLM(model_config)

    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = cp.get("model_state_dict", cp)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tok


def get_attention(model, token_ids, device):
    """Run forward pass and return attention weights per layer."""
    input_ids = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)
    return [a[0].cpu() for a in outputs.attentions]  # list of [heads, seq, seq]


def measure_positional_prev(attentions, seq_len):
    """How much each head attends to position n-1."""
    scores = []
    for layer_attn in attentions:
        # layer_attn: [heads, seq, seq]
        # For each query position q, check attention to position q-1
        prev_mass = torch.zeros(layer_attn.shape[0])
        for q in range(1, seq_len):
            prev_mass += layer_attn[:, q, q-1]
        prev_mass /= (seq_len - 1)
        scores.append(prev_mass.tolist())
    return scores


def measure_positional_p0(attentions, seq_len):
    """How much each head attends to position 0."""
    scores = []
    for layer_attn in attentions:
        p0_mass = layer_attn[:, :, 0].mean(dim=-1)
        scores.append(p0_mass.tolist())
    return scores


def measure_induction(attentions, token_ids, seq_len):
    """Induction score: does head attend to token AFTER previous occurrence of query token?"""
    scores = []
    # Build occurrence map: for each position, where was this token seen before?
    prev_occurrence = {}  # token_id -> list of positions
    next_after_prev = [None] * seq_len  # for position q, the position after prev occurrence of token_ids[q]

    occurrence_map = {}
    for pos in range(seq_len):
        tid = token_ids[pos]
        if tid in occurrence_map:
            prev_pos = occurrence_map[tid][-1]
            if prev_pos + 1 < seq_len:
                next_after_prev[pos] = prev_pos + 1
        if tid not in occurrence_map:
            occurrence_map[tid] = []
        occurrence_map[tid].append(pos)

    for layer_attn in attentions:
        induction_mass = torch.zeros(layer_attn.shape[0])
        count = 0
        for q in range(seq_len):
            target = next_after_prev[q]
            if target is not None:
                induction_mass += layer_attn[:, q, target]
                count += 1
        if count > 0:
            induction_mass /= count
        scores.append(induction_mass.tolist())
    return scores


def measure_delimiter(attentions, token_ids, tokenizer, seq_len):
    """Attention mass on delimiter positions."""
    delim_positions = []
    for pos, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid])
        if any(c in DELIMITER_CHARS for c in decoded):
            delim_positions.append(pos)

    if not delim_positions:
        return [[0.0] * attentions[0].shape[0]] * len(attentions)

    scores = []
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for p in delim_positions:
        mask[p] = True

    for layer_attn in attentions:
        delim_mass = layer_attn[:, :, mask].sum(dim=-1).mean(dim=-1)
        scores.append(delim_mass.tolist())
    return scores


def measure_bracket(attentions, token_ids, tokenizer, seq_len):
    """Attention from close-bracket to matching open-bracket."""
    # Find bracket positions and match them
    decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
    stack = []
    matches = {}  # close_pos -> open_pos

    for pos, tok_str in enumerate(decoded_tokens):
        if any(c in BRACKET_OPEN for c in tok_str):
            stack.append(pos)
        elif any(c in BRACKET_CLOSE for c in tok_str):
            if stack:
                open_pos = stack.pop()
                matches[pos] = open_pos

    if not matches:
        return [[0.0] * attentions[0].shape[0]] * len(attentions)

    scores = []
    for layer_attn in attentions:
        bracket_mass = torch.zeros(layer_attn.shape[0])
        for close_pos, open_pos in matches.items():
            if close_pos < seq_len and open_pos < seq_len:
                bracket_mass += layer_attn[:, close_pos, open_pos]
        bracket_mass /= len(matches)
        scores.append(bracket_mass.tolist())
    return scores


def measure_duplicate(attentions, token_ids, seq_len):
    """Attention to previous occurrences of the same token."""
    # For each position, find all earlier positions with same token
    same_token_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for q in range(seq_len):
        for k in range(q):
            if token_ids[q] == token_ids[k]:
                same_token_mask[q, k] = True

    if not same_token_mask.any():
        return [[0.0] * attentions[0].shape[0]] * len(attentions)

    scores = []
    for layer_attn in attentions:
        # Mean attention to same-token positions
        dup_mass = (layer_attn * same_token_mask.unsqueeze(0)).sum(dim=-1).mean(dim=-1)
        scores.append(dup_mass.tolist())
    return scores


def measure_dormant(attentions, seq_len):
    """Approximate HONOR: heads with very concentrated attention on one position."""
    scores = []
    for layer_attn in attentions:
        # Max attention weight (how concentrated)
        max_attn = layer_attn.max(dim=-1).values.mean(dim=-1)  # [heads]
        # High max = concentrated (potentially dormant if on pos 0)
        scores.append(max_attn.tolist())
    return scores


def probe_checkpoint(model, tokenizer, probe_dir, device):
    """Run all probes on all probe texts, return full head classification."""
    probes = {
        "prose": (probe_dir / "prose.txt").read_text(),
        "code": (probe_dir / "code.txt").read_text(),
        "structured": (probe_dir / "structured.txt").read_text(),
        "induction": (probe_dir / "induction.txt").read_text(),
        "duplicates": (probe_dir / "duplicates.txt").read_text(),
        "brackets": (probe_dir / "brackets.txt").read_text(),
    }

    results = {}

    for probe_name, text in probes.items():
        token_ids = tokenizer.encode(text).ids[:1024]
        seq_len = len(token_ids)

        model = model.to(device)
        attentions = get_attention(model, token_ids, device)

        results[probe_name] = {
            "seq_len": seq_len,
            "positional_prev": measure_positional_prev(attentions, seq_len),
            "positional_p0": measure_positional_p0(attentions, seq_len),
            "induction": measure_induction(attentions, token_ids, seq_len),
            "delimiter": measure_delimiter(attentions, token_ids, tokenizer, seq_len),
            "bracket": measure_bracket(attentions, token_ids, tokenizer, seq_len),
            "duplicate": measure_duplicate(attentions, token_ids, seq_len),
            "dormant": measure_dormant(attentions, seq_len),
        }

    return results


def classify_heads(results, num_layers=24, num_heads=16):
    """Classify each head by its dominant behavior across all probes."""
    classifications = []

    for layer in range(num_layers):
        for head in range(num_heads):
            scores = {
                "positional_prev": np.mean([results[p]["positional_prev"][layer][head] for p in results]),
                "positional_p0": np.mean([results[p]["positional_p0"][layer][head] for p in results]),
                "induction": results["induction"]["induction"][layer][head],
                "delimiter": results["structured"]["delimiter"][layer][head],
                "bracket": results["brackets"]["bracket"][layer][head],
                "duplicate": results["duplicates"]["duplicate"][layer][head],
            }

            dominant = max(scores, key=scores.get)
            confidence = scores[dominant] / (sum(scores.values()) + 1e-10)

            classifications.append({
                "layer": layer,
                "head": head,
                "dominant": dominant,
                "confidence": round(confidence, 4),
                "scores": {k: round(v, 4) for k, v in scores.items()},
            })

    return classifications


def main():
    parser = argparse.ArgumentParser(description="Multi-behavior head probing")
    parser.add_argument("--checkpoint", type=str, help="Single checkpoint path")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory of checkpoints (batch mode)")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--probe-dir", default="probes/")
    parser.add_argument("--output", type=str, help="Single output path")
    parser.add_argument("--output-dir", type=str, help="Output directory (batch mode)")
    parser.add_argument("--size", default="410m", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)

    if args.checkpoint:
        # Single checkpoint mode
        print("Loading model from %s..." % args.checkpoint)
        model, tokenizer = load_model(args.checkpoint, args.size, args.tokenizer)

        print("Probing all behaviors...")
        t0 = time.time()
        results = probe_checkpoint(model, tokenizer, probe_dir, args.device)
        print("  Done (%.1fs)" % (time.time() - t0))

        print("Classifying heads...")
        classifications = classify_heads(results)

        # Summary
        type_counts = {}
        for c in classifications:
            t = c["dominant"]
            type_counts[t] = type_counts.get(t, 0) + 1

        print("\nHead type distribution:")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print("  %s: %d heads (%.1f%%)" % (t, count, count/384*100))

        output = {
            "checkpoint": args.checkpoint,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "classifications": classifications,
            "type_counts": type_counts,
            "raw_scores": results,
        }

        out_path = Path(args.output) if args.output else Path("results/probe-result.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print("\nSaved to %s" % out_path)

    elif args.checkpoint_dir:
        # Batch mode
        cp_dir = Path(args.checkpoint_dir)
        out_dir = Path(args.output_dir) if args.output_dir else Path("results/")
        out_dir.mkdir(parents=True, exist_ok=True)

        checkpoints = sorted(cp_dir.glob("step-*.pt")) + sorted(cp_dir.glob("*/checkpoint.pt"))
        print("Found %d checkpoints" % len(checkpoints))

        for cp_path in checkpoints:
            step_name = cp_path.stem if cp_path.stem.startswith("step") else cp_path.parent.stem
            out_path = out_dir / ("%s.json" % step_name)

            if out_path.exists():
                print("  Skipping %s (exists)" % step_name)
                continue

            print("  Probing %s..." % step_name)
            model, tokenizer = load_model(str(cp_path), args.size, args.tokenizer)
            results = probe_checkpoint(model, tokenizer, probe_dir, args.device)
            classifications = classify_heads(results)

            output = {
                "checkpoint": str(cp_path),
                "step": step_name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "classifications": classifications,
                "raw_scores": results,
            }

            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)

            del model
            torch.cuda.empty_cache()

        print("\nAll checkpoints probed. Results in %s" % out_dir)


if __name__ == "__main__":
    main()
