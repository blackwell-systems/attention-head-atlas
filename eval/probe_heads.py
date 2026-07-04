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
import os
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


def measure_entropy(attentions, seq_len):
    """Attention entropy per head. Low = concentrated, high = diffuse."""
    scores = []
    for layer_attn in attentions:
        # layer_attn: [heads, seq, seq] - causal mask means varying lengths
        entropy = torch.zeros(layer_attn.shape[0])
        for q in range(1, seq_len):  # skip position 0 (attends only to itself)
            probs = layer_attn[:, q, :q+1]  # [heads, q+1]
            # Clamp to avoid log(0)
            log_probs = torch.log(probs.clamp(min=1e-10))
            entropy -= (probs * log_probs).sum(dim=-1)
        entropy /= (seq_len - 1)
        scores.append(entropy.tolist())
    return scores


def segment_at_barriers(text):
    """Split text at barrier characters, keeping delimiters as separate segments."""
    segments = []
    current = []
    for ch in text:
        if ch in DELIMITER_CHARS:
            if current:
                segments.append(''.join(current))
                current = []
            segments.append(ch)
        else:
            current.append(ch)
    if current:
        segments.append(''.join(current))
    return segments


def tokenize_forced_clean(text, tokenizer):
    """Tokenize with forced delimiter isolation using the tokenizer's own vocabulary."""
    segments = segment_at_barriers(text)
    all_ids = []
    for seg in segments:
        encoded = tokenizer.encode(seg)
        all_ids.extend(encoded.ids)
    return all_ids


def measure_frustration_gap(model, text, tokenizer, device):
    """Measure delimiter attention under normal vs forced-clean tokenization.
    Returns per-head delta (clean - normal) and aggregate stats."""
    # Normal tokenization
    normal_ids = tokenizer.encode(text).ids[:1024]
    normal_attn = get_attention(model, normal_ids, device)
    normal_delim = measure_delimiter(normal_attn, normal_ids, tokenizer, len(normal_ids))

    # Forced-clean tokenization
    clean_ids = tokenize_forced_clean(text, tokenizer)[:1024]
    clean_attn = get_attention(model, clean_ids, device)
    clean_delim = measure_delimiter(clean_attn, clean_ids, tokenizer, len(clean_ids))

    # Compute per-head delta
    deltas = []
    for layer_idx in range(len(normal_delim)):
        layer_deltas = []
        for head_idx in range(len(normal_delim[layer_idx])):
            d = clean_delim[layer_idx][head_idx] - normal_delim[layer_idx][head_idx]
            layer_deltas.append(round(d, 4))
        deltas.append(layer_deltas)

    # Flatten for stats
    all_normal = [v for layer in normal_delim for v in layer]
    all_clean = [v for layer in clean_delim for v in layer]
    all_deltas = [v for layer in deltas for v in layer]

    return {
        "normal_mean": round(np.mean(all_normal), 4),
        "clean_mean": round(np.mean(all_clean), 4),
        "gap": round(np.mean(all_deltas), 4),
        "min_delta": round(min(all_deltas), 4),
        "max_delta": round(max(all_deltas), 4),
        "heads_woke_up": sum(1 for d in all_deltas if d > 0.05),
        "total_heads": len(all_deltas),
        "per_layer_deltas": deltas,
    }


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

    model = model.to(device)
    for probe_name, text in probes.items():
        token_ids = tokenizer.encode(text).ids[:1024]
        seq_len = len(token_ids)

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
            "entropy": measure_entropy(attentions, seq_len),
        }

    # Frustration gap: forced-clean vs normal on structured text
    results["frustration_gap"] = measure_frustration_gap(
        model, probes["structured"], tokenizer, device)

    return results


def classify_heads(results, num_layers=24, num_heads=16):
    """Classify each head by its dominant behavior across all probes."""
    probe_names = [p for p in results if p != "frustration_gap"]
    classifications = []

    for layer in range(num_layers):
        for head in range(num_heads):
            scores = {
                "positional_prev": np.mean([results[p]["positional_prev"][layer][head] for p in probe_names]),
                "positional_p0": np.mean([results[p]["positional_p0"][layer][head] for p in probe_names]),
                "induction": results["induction"]["induction"][layer][head],
                "delimiter": results["structured"]["delimiter"][layer][head],
                "bracket": results["brackets"]["bracket"][layer][head],
                "duplicate": results["duplicates"]["duplicate"][layer][head],
            }

            # Mean entropy across probes
            entropy = np.mean([results[p]["entropy"][layer][head] for p in probe_names])

            # Specialization index: max / sum (1.0 = pure specialist, ~0.17 = uniform)
            score_vals = list(scores.values())
            total = sum(score_vals) + 1e-10
            spec_index = max(score_vals) / total

            dominant = max(scores, key=scores.get)
            confidence = scores[dominant] / total

            # "None of the above": all scores below threshold
            is_unclassified = bool(max(score_vals) < 0.05)

            classifications.append({
                "layer": layer,
                "head": head,
                "dominant": "unclassified" if is_unclassified else dominant,
                "confidence": round(float(confidence), 4),
                "specialization_index": round(float(spec_index), 4),
                "entropy": round(float(entropy), 4),
                "unclassified": is_unclassified,
                "scores": {k: round(float(v), 4) for k, v in scores.items()},
            })

    return classifications


def get_r2_client():
    """Create boto3 S3 client for R2."""
    import boto3
    import os
    return boto3.client("s3",
        endpoint_url=os.environ.get("R2_ENDPOINT",
            "https://b5e39abd50c5b82163c5fe72db9b880e.r2.cloudflarestorage.com"),
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY",
            "d77b3d0a3829377b3b71ffc11f610435"),
        aws_secret_access_key=os.environ.get("R2_SECRET_KEY",
            "9206e3609275a5b8655d5c5b0f3faf536415e324f4493cfe3ce2b4ffb53e0244"))


R2_BUCKET = "structok-training"


def list_r2_checkpoints(r2_prefix):
    """List checkpoint files under an R2 prefix."""
    s3 = get_r2_client()
    prefix = "%s/checkpoints/" % r2_prefix
    resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
    keys = []
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".pt"):
            keys.append(key)
    return sorted(keys)


def download_r2_checkpoint(r2_key, local_path):
    """Download a checkpoint from R2."""
    s3 = get_r2_client()
    s3.download_file(R2_BUCKET, r2_key, str(local_path))


def upload_r2(local_path, r2_key):
    """Upload file to R2."""
    s3 = get_r2_client()
    s3.upload_file(str(local_path), R2_BUCKET, r2_key)


def main():
    parser = argparse.ArgumentParser(description="Multi-behavior head probing")
    parser.add_argument("--checkpoint", type=str, help="Single checkpoint path")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory of checkpoints (batch mode)")
    parser.add_argument("--r2-prefix", type=str, help="R2 prefix to probe (e.g. atlas/runs/baseline)")
    parser.add_argument("--r2-output-prefix", type=str, help="R2 prefix for results upload")
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

    elif args.r2_prefix:
        # R2 streaming mode: download checkpoint, probe on GPU, upload result, delete local
        out_dir = Path(args.output_dir) if args.output_dir else Path("/tmp/atlas-results/")
        out_dir.mkdir(parents=True, exist_ok=True)
        r2_out = args.r2_output_prefix or args.r2_prefix.replace("/runs/", "/results/")

        cp_keys = list_r2_checkpoints(args.r2_prefix)
        print("Found %d checkpoints on R2 under %s" % (len(cp_keys), args.r2_prefix), flush=True)

        # Create model ONCE, load to GPU, swap state_dict per checkpoint
        from tokenizers import Tokenizer as Tok
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        tokenizer = Tok.from_file(args.tokenizer)
        vocab_size = tokenizer.get_vocab_size()
        cfg = MODEL_CONFIGS[args.size]
        model_config = GPTNeoXConfig(
            vocab_size=vocab_size,
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            intermediate_size=cfg["intermediate_size"],
            max_position_embeddings=cfg["max_position_embeddings"],
            attn_implementation="eager",
        )
        model = GPTNeoXForCausalLM(model_config).to(args.device)
        model.eval()
        print("Model on %s" % args.device, flush=True)

        tmp_cp = Path("/tmp/atlas-probe-checkpoint.pt")

        for cp_key in cp_keys:
            step_name = Path(cp_key).stem
            r2_result_key = "%s/%s.json" % (r2_out, step_name)

            print("  Probing %s..." % step_name, end=" ", flush=True)
            t0 = time.time()

            # Download checkpoint
            download_r2_checkpoint(cp_key, tmp_cp)

            # Swap state_dict (stays on GPU)
            cp = torch.load(tmp_cp, map_location=args.device, weights_only=False)
            model.load_state_dict(cp.get("model_state_dict", cp), strict=False)
            os.remove(tmp_cp)
            del cp

            # Probe
            results = probe_checkpoint(model, tokenizer, probe_dir, args.device)
            classifications = classify_heads(results)

            output = {
                "checkpoint": cp_key,
                "step": step_name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "classifications": classifications,
                "raw_scores": results,
            }

            out_path = out_dir / ("%s.json" % step_name)
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)

            # Upload result to R2
            upload_r2(out_path, r2_result_key)
            os.remove(out_path)

            print("%.1fs" % (time.time() - t0), flush=True)

        print("\nAll %d checkpoints probed. Results on R2 under %s" % (len(cp_keys), r2_out))

    elif args.checkpoint_dir:
        # Local batch mode
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
