#!/usr/bin/env python3
"""
Multi-behavior head probing: classify every attention head across 7 behavior types.

Run at each training checkpoint to build the developmental atlas.

For each head, computes 7 behavior scores used for classification:
- positional_prev: attention mass on position n-1
- positional_p0: attention mass on position 0
- induction: copy score (attention to token after previous occurrence of current query token)
- delimiter: attention mass on delimiter token positions
- bracket: attention from close-bracket to matching open-bracket
- duplicate: attention to previous occurrences of same token
- spacing: attention mass on whitespace positions (space, newline, tab, carriage return)

Plus 2 auxiliary metrics (not used for classification):
- dormant: HONOR metric approximation (max attention concentration)
- entropy: per-head attention entropy

Usage:
  # Single checkpoint:
  python probe_heads.py \\
    --checkpoint path/to/checkpoint.pt \\
    --tokenizer path/to/tokenizer.json \\
    --probe-dir ../probes/ \\
    --output ../results/step-XXXXX.json

  # Local batch (all checkpoints in a directory):
  python probe_heads.py \\
    --checkpoint-dir path/to/checkpoints/ \\
    --tokenizer path/to/tokenizer.json \\
    --probe-dir ../probes/ \\
    --output-dir ../results/

  # R2 streaming (download, probe, upload, delete):
  python probe_heads.py \\
    --r2-prefix atlas/runs/baseline \\
    --tokenizer path/to/tokenizer.json \\
    --probe-dir ../probes/

  # Re-probe (overwrite existing results):
  python probe_heads.py \\
    --r2-prefix atlas/runs/baseline \\
    --tokenizer path/to/tokenizer.json \\
    --probe-dir ../probes/ \\
    --force

  # Also save results locally:
  python probe_heads.py \\
    --r2-prefix atlas/runs/baseline \\
    --tokenizer path/to/tokenizer.json \\
    --probe-dir ../probes/ \\
    --save-local ../results/baseline/
"""

import argparse
import gc
import json
import os
import shutil
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
SPACING_CHARS = set(' \t\n\r')
BRACKET_OPEN = set('([{')
BRACKET_CLOSE = set(')]}')
BRACKET_PAIRS = {')': '(', ']': '[', '}': '{'}

# Minimum free disk space (bytes) before downloading a checkpoint.
# 410M checkpoint is ~1.7 GB; result JSON is ~800 KB. 3 GB gives margin.
MIN_DISK_BYTES = 3 * 1024 * 1024 * 1024


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
    next_after_prev = [None] * seq_len

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
    same_token_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for q in range(seq_len):
        for k in range(q):
            if token_ids[q] == token_ids[k]:
                same_token_mask[q, k] = True

    if not same_token_mask.any():
        return [[0.0] * attentions[0].shape[0]] * len(attentions)

    scores = []
    for layer_attn in attentions:
        dup_mass = (layer_attn * same_token_mask.unsqueeze(0)).sum(dim=-1).mean(dim=-1)
        scores.append(dup_mass.tolist())
    return scores


def measure_spacing(attentions, token_ids, tokenizer, seq_len):
    """Attention mass on whitespace positions (space, newline, tab, carriage return)."""
    space_positions = []
    for pos, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid])
        if any(c in SPACING_CHARS for c in decoded):
            space_positions.append(pos)

    if not space_positions:
        return [[0.0] * attentions[0].shape[0]] * len(attentions)

    scores = []
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for p in space_positions:
        mask[p] = True

    for layer_attn in attentions:
        space_mass = layer_attn[:, :, mask].sum(dim=-1).mean(dim=-1)
        scores.append(space_mass.tolist())
    return scores


def measure_dormant(attentions, seq_len):
    """Approximate HONOR: heads with very concentrated attention on one position."""
    scores = []
    for layer_attn in attentions:
        max_attn = layer_attn.max(dim=-1).values.mean(dim=-1)  # [heads]
        scores.append(max_attn.tolist())
    return scores


def measure_entropy(attentions, seq_len):
    """Attention entropy per head. Low = concentrated, high = diffuse."""
    scores = []
    for layer_attn in attentions:
        entropy = torch.zeros(layer_attn.shape[0])
        for q in range(1, seq_len):  # skip position 0 (attends only to itself)
            probs = layer_attn[:, q, :q+1]  # [heads, q+1]
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
    normal_ids = tokenizer.encode(text).ids[:1024]
    normal_attn = get_attention(model, normal_ids, device)
    normal_delim = measure_delimiter(normal_attn, normal_ids, tokenizer, len(normal_ids))

    clean_ids = tokenize_forced_clean(text, tokenizer)[:1024]
    clean_attn = get_attention(model, clean_ids, device)
    clean_delim = measure_delimiter(clean_attn, clean_ids, tokenizer, len(clean_ids))

    deltas = []
    for layer_idx in range(len(normal_delim)):
        layer_deltas = []
        for head_idx in range(len(normal_delim[layer_idx])):
            d = clean_delim[layer_idx][head_idx] - normal_delim[layer_idx][head_idx]
            layer_deltas.append(round(d, 4))
        deltas.append(layer_deltas)

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
            "spacing": measure_spacing(attentions, token_ids, tokenizer, seq_len),
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
                "spacing": np.mean([results[p]["spacing"][layer][head] for p in probe_names]),
            }

            entropy = np.mean([results[p]["entropy"][layer][head] for p in probe_names])

            score_vals = list(scores.values())
            total = sum(score_vals) + 1e-10
            spec_index = max(score_vals) / total

            dominant = max(scores, key=scores.get)
            confidence = scores[dominant] / total

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


# ── R2 helpers ──

_r2_client = None


def get_r2_client():
    """Return cached boto3 S3 client for R2. Created once, reused for all operations."""
    global _r2_client
    if _r2_client is None:
        import boto3
        _r2_client = boto3.client("s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["R2_SECRET_KEY"])
    return _r2_client


R2_BUCKET = "structok-training"


def list_r2_checkpoints(r2_prefix):
    """List checkpoint files under an R2 prefix, with pagination."""
    s3 = get_r2_client()
    prefix = "%s/checkpoints/" % r2_prefix
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".pt"):
                keys.append(key)
    return sorted(keys)


def list_r2_results(r2_out):
    """List existing result step names under an R2 output prefix."""
    s3 = get_r2_client()
    existing = set()
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=r2_out + "/"):
            for obj in page.get("Contents", []):
                existing.add(Path(obj["Key"]).stem)
    except Exception:
        pass
    return existing


def download_r2_checkpoint(r2_key, local_path, max_retries=5):
    """Download a checkpoint from R2 with retries and exponential backoff."""
    s3 = get_r2_client()
    for attempt in range(1, max_retries + 1):
        try:
            s3.download_file(R2_BUCKET, r2_key, str(local_path))
            if local_path.exists() and local_path.stat().st_size > 0:
                return True
            raise Exception("Download produced empty file")
        except Exception as e:
            if attempt < max_retries:
                wait = 5 * attempt
                print("  (retry %d/%d in %ds: %s)" % (attempt, max_retries, wait, e), flush=True)
                time.sleep(wait)
            else:
                raise Exception("Failed to download %s after %d attempts: %s" % (r2_key, max_retries, e))


def upload_r2_verified(local_path, r2_key, max_retries=3):
    """Upload file to R2 and verify the upload by checking remote size."""
    s3 = get_r2_client()
    local_size = local_path.stat().st_size

    for attempt in range(1, max_retries + 1):
        try:
            s3.upload_file(str(local_path), R2_BUCKET, r2_key)
            resp = s3.head_object(Bucket=R2_BUCKET, Key=r2_key)
            remote_size = resp["ContentLength"]
            if remote_size == local_size:
                return True
            raise Exception("Size mismatch: local %d vs remote %d" % (local_size, remote_size))
        except Exception as e:
            if attempt < max_retries:
                wait = 3 * attempt
                print("  (upload retry %d/%d in %ds: %s)" % (attempt, max_retries, wait, e), flush=True)
                time.sleep(wait)
            else:
                raise Exception("Failed to upload %s after %d attempts: %s" % (r2_key, max_retries, e))


def check_disk_space(path, min_bytes=MIN_DISK_BYTES):
    """Check that the filesystem containing path has enough free space."""
    usage = shutil.disk_usage(str(path))
    if usage.free < min_bytes:
        free_gb = usage.free / (1024 ** 3)
        need_gb = min_bytes / (1024 ** 3)
        raise RuntimeError(
            "Insufficient disk space: %.1f GB free, need %.1f GB. "
            "Free space before continuing." % (free_gb, need_gb))


def format_eta(seconds):
    """Format seconds as HH:MM:SS or MM:SS."""
    if seconds < 0:
        return "??:??"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return "%d:%02d:%02d" % (h, m, s)
    return "%d:%02d" % (m, s)


def cleanup_tmp(tmp_cp):
    """Remove temporary checkpoint file if it exists."""
    try:
        if tmp_cp.exists():
            os.remove(tmp_cp)
    except OSError:
        pass


def gpu_cleanup():
    """Free GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def is_valid_result(path):
    """Check if a result JSON file is valid (parseable and has classifications)."""
    try:
        with open(path) as f:
            d = json.load(f)
        return "classifications" in d and len(d["classifications"]) == 384
    except (json.JSONDecodeError, OSError, KeyError):
        return False


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
    parser.add_argument("--save-local", type=str, help="Also save results to this local directory")
    parser.add_argument("--force", action="store_true", help="Re-probe even if results already exist")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be probed without doing it")
    parser.add_argument("--size", default="410m", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)

    # Validate probe directory
    required_probes = ["prose.txt", "code.txt", "structured.txt", "induction.txt", "duplicates.txt", "brackets.txt"]
    missing = [p for p in required_probes if not (probe_dir / p).exists()]
    if missing:
        print("ERROR: Missing probe files in %s: %s" % (probe_dir, ', '.join(missing)))
        return

    if args.checkpoint:
        # ── Single checkpoint mode ──
        print("Loading model from %s..." % args.checkpoint)
        model, tokenizer = load_model(args.checkpoint, args.size, args.tokenizer)

        print("Probing all behaviors...")
        t0 = time.time()
        results = probe_checkpoint(model, tokenizer, probe_dir, args.device)
        print("  Done (%.1fs)" % (time.time() - t0))

        print("Classifying heads...")
        classifications = classify_heads(results)

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
        # ── R2 streaming mode ──
        out_dir = Path(args.output_dir) if args.output_dir else Path("/tmp/atlas-results/")
        out_dir.mkdir(parents=True, exist_ok=True)
        default_r2_out = args.r2_prefix.replace("/runs/", "/results/")

        # If --force and no explicit output prefix, auto-version to avoid
        # destroying original results. Original data backs published findings.
        if args.r2_output_prefix:
            r2_out = args.r2_output_prefix
        elif args.force:
            existing_at_default = list_r2_results(default_r2_out)
            if existing_at_default:
                # Find next available version: results-v2, results-v3, ...
                base = default_r2_out.rstrip("/")
                version = 2
                while True:
                    candidate = "%s-v%d" % (base, version)
                    if not list_r2_results(candidate):
                        break
                    version += 1
                r2_out = candidate
                print("Original results exist at %s/ (%d files)" % (default_r2_out, len(existing_at_default)), flush=True)
                print("Auto-versioned output to: %s/" % r2_out, flush=True)
            else:
                r2_out = default_r2_out
        else:
            r2_out = default_r2_out

        local_save_dir = None
        if args.save_local:
            local_save_dir = Path(args.save_local)
            local_save_dir.mkdir(parents=True, exist_ok=True)

        # List checkpoints
        cp_keys = list_r2_checkpoints(args.r2_prefix)
        total_checkpoints = len(cp_keys)
        print("Found %d checkpoints on R2 under %s" % (total_checkpoints, args.r2_prefix), flush=True)
        if total_checkpoints == 0:
            print("ERROR: No checkpoints found. Check R2 prefix.")
            return

        # Determine which to probe
        existing = set() if args.force else list_r2_results(r2_out)
        to_probe = [k for k in cp_keys if args.force or Path(k).stem not in existing]
        skipped = total_checkpoints - len(to_probe)
        print("To probe: %d, skipping: %d (existing)%s" % (
            len(to_probe), skipped,
            " [--force: re-probing all to new prefix]" if args.force else ""), flush=True)

        if not to_probe:
            print("Nothing to do. All checkpoints already probed.")
            return

        # Dry run: show plan and exit
        if args.dry_run:
            print("\n[DRY RUN] Would probe %d checkpoints:" % len(to_probe))
            for k in to_probe[:10]:
                print("  %s" % Path(k).stem)
            if len(to_probe) > 10:
                print("  ... and %d more" % (len(to_probe) - 10))
            print("\nEstimate: ~15s/step on A100, ~%s total" % format_eta(len(to_probe) * 15))
            print("Results would go to: R2 %s/" % r2_out)
            if r2_out != default_r2_out:
                print("Original results preserved at: R2 %s/" % default_r2_out)
            if local_save_dir:
                print("Local copy to: %s/" % local_save_dir)
            return

        check_disk_space(out_dir)

        # Create model ONCE on GPU, swap state_dict per checkpoint
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

        def probe_one_r2(cp_key, step_name, r2_result_key):
            """Download checkpoint from R2, probe, upload result. Returns duration on success."""
            check_disk_space(out_dir)
            download_r2_checkpoint(cp_key, tmp_cp)

            cp = torch.load(tmp_cp, map_location=args.device, weights_only=False)
            model.load_state_dict(cp.get("model_state_dict", cp), strict=False)
            cleanup_tmp(tmp_cp)
            del cp
            gpu_cleanup()

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

            upload_r2_verified(out_path, r2_result_key)

            if local_save_dir:
                shutil.copy2(str(out_path), str(local_save_dir / ("%s.json" % step_name)))

            os.remove(out_path)

        completed = 0
        failed = 0
        failed_steps = []
        durations = []
        run_start = time.time()

        for idx, cp_key in enumerate(to_probe):
            step_name = Path(cp_key).stem
            r2_result_key = "%s/%s.json" % (r2_out, step_name)

            if durations:
                eta_str = format_eta((len(to_probe) - idx) * sum(durations) / len(durations))
            else:
                eta_str = "??:??"
            print("  [%d/%d, ETA %s] %s..." % (idx + 1, len(to_probe), eta_str, step_name),
                  end=" ", flush=True)

            t0 = time.time()

            try:
                probe_one_r2(cp_key, step_name, r2_result_key)
                dur = time.time() - t0
                durations.append(dur)
                completed += 1
                print("%.1fs" % dur, flush=True)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("OOM, clearing cache...", end=" ", flush=True)
                    cleanup_tmp(tmp_cp)
                    gpu_cleanup()
                    try:
                        probe_one_r2(cp_key, step_name, r2_result_key)
                        dur = time.time() - t0
                        durations.append(dur)
                        completed += 1
                        print("recovered, %.1fs" % dur, flush=True)
                        continue
                    except Exception:
                        pass

                failed += 1
                failed_steps.append(step_name)
                print("FAILED: %s" % e, flush=True)
                cleanup_tmp(tmp_cp)
                gpu_cleanup()

            except Exception as e:
                failed += 1
                failed_steps.append(step_name)
                print("FAILED: %s" % e, flush=True)
                cleanup_tmp(tmp_cp)
                gpu_cleanup()

        # ── Summary ──
        total_time = time.time() - run_start
        print("\n" + "=" * 60)
        print("PROBING COMPLETE: %s" % args.r2_prefix)
        print("=" * 60)
        print("  Completed: %d" % completed)
        print("  Failed:    %d" % failed)
        print("  Skipped:   %d (already existed)" % skipped)
        print("  Total:     %d checkpoints" % total_checkpoints)
        print("  Time:      %s" % format_eta(total_time))
        if durations:
            print("  Avg/step:  %.1fs" % (sum(durations) / len(durations)))
        print("  Results:   R2 %s/" % r2_out)
        if r2_out != default_r2_out:
            print("  Original:  R2 %s/ (preserved)" % default_r2_out)
        if local_save_dir:
            print("  Local:     %s/" % local_save_dir)
        if failed_steps:
            print("\n  FAILED STEPS (re-run with same command to retry):")
            for s in failed_steps:
                print("    %s" % s)
        print()

    elif args.checkpoint_dir:
        # ── Local batch mode ──
        cp_dir = Path(args.checkpoint_dir)
        out_dir = Path(args.output_dir) if args.output_dir else Path("results/")
        out_dir.mkdir(parents=True, exist_ok=True)

        checkpoints = sorted(cp_dir.glob("step-*.pt")) + sorted(cp_dir.glob("*/checkpoint.pt"))
        total_checkpoints = len(checkpoints)
        print("Found %d checkpoints" % total_checkpoints)

        if total_checkpoints == 0:
            print("ERROR: No checkpoints found in %s" % cp_dir)
            return

        # Determine which to probe (skip valid existing unless --force)
        to_probe = []
        skipped = 0
        for cp_path in checkpoints:
            step_name = cp_path.stem if cp_path.stem.startswith("step") else cp_path.parent.stem
            out_path = out_dir / ("%s.json" % step_name)
            if not args.force and out_path.exists() and is_valid_result(out_path):
                skipped += 1
            else:
                to_probe.append(cp_path)

        if not args.force:
            corrupt = sum(1 for cp in to_probe
                          if (out_dir / ("%s.json" % (cp.stem if cp.stem.startswith("step") else cp.parent.stem))).exists())
            if corrupt > 0:
                print("  %d existing results are corrupt/incomplete, will re-probe" % corrupt)

        print("To probe: %d, skipping: %d (valid existing)" % (len(to_probe), skipped), flush=True)

        if not to_probe:
            print("Nothing to do. All checkpoints already probed.")
            return

        # Dry run: show plan and exit
        if args.dry_run:
            print("\n[DRY RUN] Would probe %d checkpoints:" % len(to_probe))
            for cp in to_probe[:10]:
                print("  %s" % cp.stem)
            if len(to_probe) > 10:
                print("  ... and %d more" % (len(to_probe) - 10))
            print("\nEstimate: ~15s/step on A100, ~%s total" % format_eta(len(to_probe) * 15))
            print("Results would go to: %s/" % out_dir)
            return

        check_disk_space(out_dir)

        # Create model once
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

        completed = 0
        failed = 0
        failed_steps = []
        durations = []
        run_start = time.time()

        for idx, cp_path in enumerate(to_probe):
            step_name = cp_path.stem if cp_path.stem.startswith("step") else cp_path.parent.stem
            out_path = out_dir / ("%s.json" % step_name)

            if durations:
                eta_str = format_eta((len(to_probe) - idx) * sum(durations) / len(durations))
            else:
                eta_str = "??:??"
            print("  [%d/%d, ETA %s] %s..." % (idx + 1, len(to_probe), eta_str, step_name),
                  end=" ", flush=True)

            t0 = time.time()
            try:
                check_disk_space(out_dir)

                cp = torch.load(str(cp_path), map_location=args.device, weights_only=False)
                model.load_state_dict(cp.get("model_state_dict", cp), strict=False)
                del cp
                gpu_cleanup()

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

                completed += 1
                dur = time.time() - t0
                durations.append(dur)
                print("%.1fs" % dur, flush=True)

                gpu_cleanup()

            except Exception as e:
                failed += 1
                failed_steps.append(step_name)
                print("FAILED: %s" % e, flush=True)
                gpu_cleanup()

        total_time = time.time() - run_start
        print("\n" + "=" * 60)
        print("PROBING COMPLETE: %s" % cp_dir)
        print("=" * 60)
        print("  Completed: %d" % completed)
        print("  Failed:    %d" % failed)
        print("  Skipped:   %d (valid existing)" % skipped)
        print("  Total:     %d checkpoints" % total_checkpoints)
        print("  Time:      %s" % format_eta(total_time))
        if durations:
            print("  Avg/step:  %.1fs" % (sum(durations) / len(durations)))
        print("  Results:   %s/" % out_dir)
        if failed_steps:
            print("\n  FAILED STEPS (re-run with --force to retry):")
            for s in failed_steps:
                print("    %s" % s)
        print()


if __name__ == "__main__":
    main()
