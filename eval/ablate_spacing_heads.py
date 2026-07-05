#!/usr/bin/env python3
"""
Ablation study: zero spacing heads and measure perplexity change.

Tests whether spacing heads are counterproductive (removal improves PPL),
neutral (no change), or productive (removal degrades PPL).

Methodology follows the coupling paper (Blackwell, 2026a): zero-ablation
of output projection weights, paired with random head controls.

Usage:
  python ablate_spacing_heads.py \
    --checkpoint step-20000.pt \
    --tokenizer standard-64k.json \
    --classifications results/baseline-v2-excess/step-20000.json \
    --probe-dir probes/

  # R2 mode:
  python ablate_spacing_heads.py \
    --r2-checkpoint atlas/runs/baseline/checkpoints/step-20000.pt \
    --tokenizer tokenizers/standard-64k.json \
    --classifications results/baseline-v2-excess/step-20000.json \
    --probe-dir probes/

Provenance: written 2026-07-04 for the developmental atlas project.
Adapted from the 18-phase ablation protocol in the coupling paper.
"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import torch
import numpy as np


MODEL_CONFIGS = {
    "410m": {
        "hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16,
        "intermediate_size": 4096, "max_position_embeddings": 2048, "arch": "neox",
    },
    "410m-llama": {
        "hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16,
        "num_key_value_heads": 4, "intermediate_size": 2816,
        "max_position_embeddings": 2048, "rope_theta": 500000.0, "arch": "llama",
    },
    "1.3b-llama": {
        "hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 32,
        "num_key_value_heads": 8, "intermediate_size": 5632,
        "max_position_embeddings": 2048, "rope_theta": 500000.0, "arch": "llama",
    },
}


def load_model(checkpoint_path, tokenizer_path, device, size="410m"):
    """Load model (NeoX or Llama)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    cfg = MODEL_CONFIGS[size]
    arch = cfg.get("arch", "neox")

    if arch == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            intermediate_size=cfg["intermediate_size"],
            max_position_embeddings=cfg["max_position_embeddings"],
            rope_theta=cfg.get("rope_theta", 10000.0),
            attn_implementation="eager",
        )
        model = LlamaForCausalLM(config).to(device)
    else:
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        config = GPTNeoXConfig(
            vocab_size=vocab_size,
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            intermediate_size=cfg["intermediate_size"],
            max_position_embeddings=cfg["max_position_embeddings"],
            attn_implementation="eager",
        )
        model = GPTNeoXForCausalLM(config).to(device)

    cp = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(cp.get("model_state_dict", cp), strict=False)
    model.eval()
    return model, tok


def get_head_list(classifications_path, head_type):
    """Get list of (layer, head) tuples for a given dominant type."""
    with open(classifications_path) as f:
        data = json.load(f)
    heads = []
    for c in data["classifications"]:
        if c["dominant"] == head_type:
            heads.append((c["layer"], c["head"]))
    return heads


def zero_heads(model, heads, arch="neox", hidden_size=1024, num_heads=16):
    """Zero output projection weights for specified heads. Returns a deep copy."""
    ablated = copy.deepcopy(model)
    head_dim = hidden_size // num_heads

    for layer, head in heads:
        if arch == "llama":
            dense = ablated.model.layers[layer].self_attn.o_proj
        else:
            dense = ablated.gpt_neox.layers[layer].attention.dense
        start = head * head_dim
        end = start + head_dim
        dense.weight.data[:, start:end] = 0.0

    return ablated


def measure_ppl(model, tokenizer, texts, device):
    """Measure perplexity on a list of texts. Returns per-text and mean PPL."""
    ppls = []
    for text in texts:
        ids = tokenizer.encode(text).ids[:1024]
        input_ids = torch.tensor([ids], device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()

        ppl = float(np.exp(loss))
        ppls.append(ppl)

    return ppls, float(np.mean(ppls))


def run_ablation(model, tokenizer, texts, heads_to_ablate, label, device,
                 arch="neox", hidden_size=1024, num_heads=16):
    """Run one ablation: zero heads, measure PPL, report."""
    ablated = zero_heads(model, heads_to_ablate, arch=arch,
                        hidden_size=hidden_size, num_heads=num_heads)
    ppls, mean_ppl = measure_ppl(ablated, tokenizer, texts, device)
    del ablated
    torch.cuda.empty_cache()
    return mean_ppl, ppls


def main():
    parser = argparse.ArgumentParser(description="Ablation study on spacing heads")
    parser.add_argument("--checkpoint", type=str, help="Local checkpoint path")
    parser.add_argument("--r2-checkpoint", type=str, help="R2 key for checkpoint")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--classifications", required=True, help="Excess-corrected classifications JSON")
    parser.add_argument("--probe-dir", default="probes/")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for control ablations")
    parser.add_argument("--n-random-trials", type=int, default=5, help="Number of random control trials")
    parser.add_argument("--size", default="410m", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    probe_dir = Path(args.probe_dir)

    # Load texts
    texts = {}
    for name in ["prose", "code", "structured", "induction", "duplicates", "brackets"]:
        path = probe_dir / ("%s.txt" % name)
        if path.exists():
            texts[name] = path.read_text()
    if (probe_dir / "prose_punctuated.txt").exists():
        texts["prose_punctuated"] = (probe_dir / "prose_punctuated.txt").read_text()

    print("Loaded %d probe texts" % len(texts))

    # Load checkpoint
    cp_path = args.checkpoint
    if args.r2_checkpoint:
        cp_path = "/tmp/ablation_checkpoint.pt"
        print("Downloading %s..." % args.r2_checkpoint)
        import boto3
        s3 = boto3.client("s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["R2_SECRET_KEY"])
        s3.download_file("structok-training", args.r2_checkpoint, cp_path)

    print("Loading model (%s)..." % args.size)
    model, tokenizer = load_model(cp_path, args.tokenizer, args.device, size=args.size)

    if args.r2_checkpoint and os.path.exists(cp_path):
        os.remove(cp_path)

    cfg = MODEL_CONFIGS[args.size]
    arch = cfg.get("arch", "neox")
    num_layers = cfg["num_hidden_layers"]
    num_heads = cfg["num_attention_heads"]
    total_heads = num_layers * num_heads

    # Get head lists
    spacing_heads = get_head_list(args.classifications, "spacing")
    p0_heads = get_head_list(args.classifications, "positional_p0")
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    non_spacing_heads = [h for h in all_heads if h not in spacing_heads]

    print("\nSpacing heads: %d" % len(spacing_heads))
    print("P0 heads: %d" % len(p0_heads))
    print("Total heads: %d" % len(all_heads))

    text_list = list(texts.values())
    text_names = list(texts.keys())

    # Baseline (no ablation)
    print("\n=== BASELINE (no ablation) ===")
    baseline_ppls, baseline_mean = measure_ppl(model, tokenizer, text_list, args.device)
    print("Mean PPL: %.1f" % baseline_mean)
    for name, ppl in zip(text_names, baseline_ppls):
        print("  %s: %.1f" % (name, ppl))

    results = {
        "baseline": {"mean": baseline_mean, "per_text": dict(zip(text_names, baseline_ppls))},
    }

    # Ablate spacing heads
    print("\n=== ABLATE SPACING HEADS (%d heads) ===" % len(spacing_heads))
    spacing_mean, spacing_ppls = run_ablation(model, tokenizer, text_list, spacing_heads, "spacing", args.device,
                                       arch=arch, hidden_size=cfg["hidden_size"], num_heads=num_heads)
    spacing_delta = (spacing_mean - baseline_mean) / baseline_mean * 100
    print("Mean PPL: %.1f (%+.1f%%)" % (spacing_mean, spacing_delta))
    for name, ppl, base in zip(text_names, spacing_ppls, baseline_ppls):
        delta = (ppl - base) / base * 100
        print("  %s: %.1f (%+.1f%%)" % (name, ppl, delta))

    results["spacing_ablation"] = {
        "heads": len(spacing_heads),
        "mean": spacing_mean,
        "delta_pct": round(spacing_delta, 2),
        "per_text": dict(zip(text_names, spacing_ppls)),
    }

    # Ablate P0 heads
    if p0_heads:
        print("\n=== ABLATE P0 HEADS (%d heads) ===" % len(p0_heads))
        p0_mean, p0_ppls = run_ablation(model, tokenizer, text_list, p0_heads, "p0", args.device,
                                   arch=arch, hidden_size=cfg["hidden_size"], num_heads=num_heads)
        p0_delta = (p0_mean - baseline_mean) / baseline_mean * 100
        print("Mean PPL: %.1f (%+.1f%%)" % (p0_mean, p0_delta))
        results["p0_ablation"] = {
            "heads": len(p0_heads),
            "mean": p0_mean,
            "delta_pct": round(p0_delta, 2),
            "per_text": dict(zip(text_names, p0_ppls)),
        }

    # Random controls (same number of heads as spacing)
    print("\n=== RANDOM CONTROLS (%d trials, %d heads each) ===" % (args.n_random_trials, len(spacing_heads)))
    random_deltas = []
    for trial in range(args.n_random_trials):
        random_heads = random.sample(non_spacing_heads, min(len(spacing_heads), len(non_spacing_heads)))
        rand_mean, _ = run_ablation(model, tokenizer, text_list, random_heads, "random_%d" % trial, args.device,
                                       arch=arch, hidden_size=cfg["hidden_size"], num_heads=num_heads)
        rand_delta = (rand_mean - baseline_mean) / baseline_mean * 100
        random_deltas.append(rand_delta)
        print("  Trial %d: %.1f (%+.1f%%)" % (trial + 1, rand_mean, rand_delta))

    results["random_controls"] = {
        "n_trials": args.n_random_trials,
        "heads_per_trial": len(spacing_heads),
        "deltas_pct": [round(d, 2) for d in random_deltas],
        "mean_delta_pct": round(float(np.mean(random_deltas)), 2),
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Baseline PPL:          %.1f" % baseline_mean)
    print("Spacing ablation:      %.1f (%+.1f%%)" % (spacing_mean, spacing_delta))
    if p0_heads:
        print("P0 ablation:           %.1f (%+.1f%%)" % (p0_mean, p0_delta))
    print("Random control (mean): %+.1f%%" % np.mean(random_deltas))
    print()

    if spacing_delta < 0:
        print("FINDING: Removing spacing heads IMPROVES perplexity.")
        print("Spacing heads are COUNTERPRODUCTIVE (same pattern as stranded heads at 1.3B).")
    elif spacing_delta < np.mean(random_deltas):
        print("FINDING: Removing spacing heads hurts LESS than removing random heads.")
        print("Spacing heads contribute less than average to model performance.")
    else:
        print("FINDING: Removing spacing heads hurts MORE than removing random heads.")
        print("Spacing heads are productive despite their apparent waste.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print("\nSaved to %s" % args.output)


if __name__ == "__main__":
    main()
