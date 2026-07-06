#!/usr/bin/env python3
"""
Activation patching experiment for unclassified attention heads.

Identifies what the 95 unclassified heads in the merge-barrier comparison
model are doing by patching each head's activations from a corrupted input
into a clean forward pass and measuring what breaks.

Six behavior categories:
  1. Subject-verb agreement (syntax)
  2. Coreference / entity tracking
  3. Semantic similarity
  4. Local context / n-gram
  5. Clause boundary
  6. Positional / distance

Protocol:
  For each (head, behavior, input_pair):
    1. Run clean input -> capture head H's pre-dense activations
    2. Run corrupted input -> capture head H's pre-dense activations
    3. Run clean input again, but inject H's corrupted activations
    4. Measure logit shift at the critical token position

The patching mechanism hooks into the dense layer's input (the merged
attention output before projection), replaces head H's slice with the
corrupted version, and measures the downstream effect.

Usage:
  python patch_unclassified_heads.py \
    --checkpoint path/to/comparison-step-20000.pt \
    --tokenizer path/to/structok-64k.json \
    --classifications results/comparison-v2-excess/step-20000.json \
    --output results/patching/patching-comparison.json
"""

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer


NUM_LAYERS = 24
NUM_HEADS = 16
HEAD_DIM = 64  # 1024 / 16


def load_model(checkpoint_path, tokenizer_path, device):
    """Load GPT-NeoX 410M model."""
    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()

    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    config = GPTNeoXConfig(
        vocab_size=vocab_size, hidden_size=1024, num_hidden_layers=24,
        num_attention_heads=16, intermediate_size=4096,
        max_position_embeddings=2048, attn_implementation="eager")
    model = GPTNeoXForCausalLM(config).to(device)
    model.eval()

    cp = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(cp.get("model_state_dict", cp))
    print("Loaded step %s" % cp.get("step", "?"))
    del cp
    torch.cuda.empty_cache()
    return model, tok


def get_unclassified_heads(classifications_path):
    """Load classifications and return list of (layer, head) for unclassified heads."""
    with open(classifications_path) as f:
        data = json.load(f)
    heads = []
    for c in data["classifications"]:
        if c["dominant"] == "unclassified":
            heads.append((c["layer"], c["head"]))
    return heads


def tokenize(tok, text):
    """Tokenize text and return token IDs."""
    return tok.encode(text).ids


# ── Activation patching machinery ──

class ActivationCapture:
    """Captures the input to the dense layer (merged attention output) at a specific layer."""

    def __init__(self, model, layer_idx):
        self.layer_idx = layer_idx
        self.captured = None
        self.hook = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(
            self._hook_fn
        )

    def _hook_fn(self, module, input):
        # input is a tuple, first element is the merged attention output
        # shape: (batch, seq_len, hidden_size)
        self.captured = input[0].detach().clone()

    def remove(self):
        self.hook.remove()


class ActivationPatcher:
    """Patches one head's slice in the dense layer input with corrupted activations."""

    def __init__(self, model, layer_idx, head_idx, corrupted_activations):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.corrupted = corrupted_activations
        start = head_idx * HEAD_DIM
        end = start + HEAD_DIM
        self.start = start
        self.end = end
        self.hook = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(
            self._hook_fn
        )

    def _hook_fn(self, module, input):
        # Replace head's slice with corrupted version
        # input[0] shape: (batch, seq_len, hidden_size)
        patched = input[0].clone()
        seq_len = min(patched.shape[1], self.corrupted.shape[1])
        patched[:, :seq_len, self.start:self.end] = self.corrupted[:, :seq_len, self.start:self.end]
        return (patched,) + input[1:]

    def remove(self):
        self.hook.remove()


@torch.no_grad()
def run_patching_experiment(model, tok, clean_text, corrupted_text,
                            critical_pos, correct_token_id, incorrect_token_id,
                            target_layer, target_head, device):
    """
    Run one activation patching experiment.

    Returns the patching effect: how much the logit difference
    (correct - incorrect) changes when head H's activations are
    patched from the corrupted input.

    Negative effect means head H contributes to getting this right.
    """
    # Step 1: Run clean input, get baseline logits
    clean_ids = tokenize(tok, clean_text)
    clean_input = torch.tensor([clean_ids], device=device)

    clean_logits = model(clean_input).logits[0]
    if critical_pos >= clean_logits.shape[0]:
        return None
    clean_diff = (clean_logits[critical_pos, correct_token_id] -
                  clean_logits[critical_pos, incorrect_token_id]).item()

    # Step 2: Run corrupted input, capture head H's activations
    corrupted_ids = tokenize(tok, corrupted_text)
    corrupted_input = torch.tensor([corrupted_ids], device=device)

    capture = ActivationCapture(model, target_layer)
    model(corrupted_input)
    corrupted_activations = capture.captured
    capture.remove()

    # Step 3: Run clean input with head H patched from corrupted
    patcher = ActivationPatcher(model, target_layer, target_head, corrupted_activations)
    patched_logits = model(clean_input).logits[0]
    patcher.remove()

    if critical_pos >= patched_logits.shape[0]:
        return None
    patched_diff = (patched_logits[critical_pos, correct_token_id] -
                    patched_logits[critical_pos, incorrect_token_id]).item()

    # Patching effect: how much the logit difference changed
    # Negative means head H helps get this right (patching hurts)
    effect = patched_diff - clean_diff
    return effect


# ── Input pair generators ──

def make_agreement_pairs(rng):
    """Subject-verb agreement pairs. Singular subject, plural attractor."""
    templates = [
        ("The %s that chased the %s %s", "was", "were"),
        ("The %s near the %s %s", "is", "are"),
        ("The %s behind the %s %s", "has", "have"),
        ("The %s beside the %s %s", "runs", "run"),
        ("The %s above the %s %s", "falls", "fall"),
    ]
    singular = ["cat", "dog", "bird", "tree", "stone", "river", "tower", "lamp", "chair", "cloud"]
    plural = ["dogs", "birds", "trees", "stones", "rivers", "towers", "lamps", "chairs", "clouds", "hills"]
    pairs = []
    for _ in range(20):
        template, correct_verb, incorrect_verb = rng.choice(templates)
        subj = rng.choice(singular)
        attr = rng.choice(plural)
        clean = template % (subj, attr, correct_verb)
        corrupted = template % (subj, attr, incorrect_verb)
        pairs.append({
            "clean": clean,
            "corrupted": corrupted,
            "correct_token": correct_verb,
            "incorrect_token": incorrect_verb,
            "behavior": "agreement",
        })
    return pairs


def make_coreference_pairs(rng):
    """Coreference / entity tracking pairs."""
    templates = [
        ("%s told %s that %s was right",),
        ("%s gave %s a gift and %s smiled",),
        ("%s saw %s and then %s left",),
        ("%s helped %s because %s was kind",),
    ]
    female = ["Alice", "Carol", "Elena", "Grace", "Maria"]
    male = ["Bob", "David", "Frank", "Henry", "James"]
    pairs = []
    for _ in range(20):
        template = rng.choice(templates)[0]
        f = rng.choice(female)
        m = rng.choice(male)
        # Clean: pronoun matches first entity (female)
        clean = template % (f, m, "she")
        corrupted = template % (f, m, "he")
        pairs.append({
            "clean": clean,
            "corrupted": corrupted,
            "correct_token": "she",
            "incorrect_token": "he",
            "behavior": "coreference",
        })
    return pairs


def make_semantic_pairs(rng):
    """Semantic similarity / plausibility pairs."""
    templates_and_swaps = [
        ("The dog chased the %s across the yard", "ball", "lamp"),
        ("She drank a cup of %s after dinner", "coffee", "gravel"),
        ("He read the %s before sleeping", "book", "fence"),
        ("The bird landed on the %s in the garden", "branch", "hammer"),
        ("They cooked %s for the family meal", "pasta", "cardboard"),
        ("The child played with the %s at the park", "swing", "invoice"),
        ("He drove the %s to the store", "car", "pillow"),
        ("She planted %s in the garden bed", "flowers", "batteries"),
        ("The fish swam through the %s water", "clear", "wooden"),
        ("He wore a %s to the meeting", "suit", "ladder"),
    ]
    pairs = []
    for _ in range(20):
        template, coherent, incoherent = rng.choice(templates_and_swaps)
        clean = template % coherent
        corrupted = template % incoherent
        pairs.append({
            "clean": clean,
            "corrupted": corrupted,
            "correct_token": coherent,
            "incorrect_token": incoherent,
            "behavior": "semantic",
        })
    return pairs


def make_local_context_pairs(rng):
    """Local context / n-gram prediction pairs."""
    templates = [
        ("Once upon a %s", "time", "brick"),
        ("The United %s", "States", "Pencil"),
        ("Thank you very %s", "much", "purple"),
        ("As a matter of %s", "fact", "cheese"),
        ("In the middle of the %s", "night", "fork"),
        ("On the other %s", "hand", "cloud"),
        ("At the end of the %s", "day", "spoon"),
        ("For the first %s", "time", "lamp"),
        ("In front of the %s", "house", "verb"),
        ("Out of the %s", "blue", "math"),
    ]
    pairs = []
    for _ in range(20):
        template, correct, incorrect = rng.choice(templates)
        clean = template % correct
        corrupted = template % incorrect
        pairs.append({
            "clean": clean,
            "corrupted": corrupted,
            "correct_token": correct,
            "incorrect_token": incorrect,
            "behavior": "local_context",
        })
    return pairs


def make_clause_boundary_pairs(rng):
    """Clause boundary detection pairs."""
    clauses_a = [
        "When the rain stopped",
        "After the sun set",
        "Before the bell rang",
        "While the wind blew",
        "Since the door closed",
    ]
    clauses_b = [
        "the children played outside",
        "the crowd went home",
        "the workers took a break",
        "the animals hid in the barn",
        "the lights turned on",
    ]
    pairs = []
    for _ in range(20):
        a = rng.choice(clauses_a)
        b = rng.choice(clauses_b)
        clean = "%s, %s" % (a, b)
        corrupted = "%s %s" % (a, b)  # remove comma (clause boundary)
        # The "correct" token here is the comma; we measure if the head
        # is sensitive to clause boundary presence
        pairs.append({
            "clean": clean,
            "corrupted": corrupted,
            "correct_token": ",",
            "incorrect_token": None,  # handled specially
            "behavior": "clause_boundary",
        })
    return pairs


def make_positional_pairs(rng):
    """Positional / distance-based attention pairs."""
    words = ["alpha", "beta", "gamma", "delta", "epsilon",
             "zeta", "eta", "theta", "iota", "kappa"]
    pairs = []
    for _ in range(20):
        seq = rng.sample(words, 8)
        clean = " ".join(seq)
        corrupted = " ".join(reversed(seq))
        pairs.append({
            "clean": clean,
            "corrupted": corrupted,
            "correct_token": seq[-1],  # last word in clean order
            "incorrect_token": seq[0],  # last word in reversed order
            "behavior": "positional",
        })
    return pairs


BEHAVIOR_GENERATORS = {
    "agreement": make_agreement_pairs,
    "coreference": make_coreference_pairs,
    "semantic": make_semantic_pairs,
    "local_context": make_local_context_pairs,
    "clause_boundary": make_clause_boundary_pairs,
    "positional": make_positional_pairs,
}


def find_token_id(tok, word):
    """Find the token ID for a word (with and without leading space)."""
    # Try with leading space (most common in BPE)
    ids_space = tokenize(tok, " " + word)
    if len(ids_space) == 1:
        return ids_space[0]
    # Try without
    ids_bare = tokenize(tok, word)
    if len(ids_bare) == 1:
        return ids_bare[0]
    # Return the first token of the space version
    if ids_space:
        return ids_space[0]
    if ids_bare:
        return ids_bare[0]
    return None


def find_critical_position(tok, text, target_word):
    """Find the token position just BEFORE the target word appears."""
    ids = tokenize(tok, text)
    target_ids = tokenize(tok, " " + target_word)
    if not target_ids:
        target_ids = tokenize(tok, target_word)

    target_first = target_ids[0] if target_ids else None

    for i, tid in enumerate(ids):
        if tid == target_first and i > 0:
            return i - 1  # position before the target

    # Fallback: return second-to-last position
    return max(0, len(ids) - 2)


def main():
    parser = argparse.ArgumentParser(description="Activation patching for unclassified heads")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--classifications", required=True,
                       help="Excess-corrected classifications JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--include-controls", action="store_true",
                       help="Also patch 20 random non-unclassified heads as controls")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    model, tok = load_model(args.checkpoint, args.tokenizer, device)

    # Get unclassified heads
    unclassified = get_unclassified_heads(args.classifications)
    print("Unclassified heads: %d" % len(unclassified))

    # Optionally add control heads
    control_heads = []
    if args.include_controls:
        with open(args.classifications) as f:
            all_data = json.load(f)
        classified = [(c["layer"], c["head"]) for c in all_data["classifications"]
                      if c["dominant"] != "unclassified"]
        rng_ctrl = random.Random(99)
        control_heads = rng_ctrl.sample(classified, min(20, len(classified)))
        print("Control heads: %d" % len(control_heads))

    all_heads = [(lh, "unclassified") for lh in unclassified] + \
                [(lh, "control") for lh in control_heads]

    # Generate input pairs
    rng = random.Random(42)
    all_pairs = {}
    for behavior, generator in BEHAVIOR_GENERATORS.items():
        all_pairs[behavior] = generator(rng)
        print("Generated %d pairs for %s" % (len(all_pairs[behavior]), behavior))

    total_experiments = len(all_heads) * sum(len(p) for p in all_pairs.values())
    print("\nTotal experiments: %d" % total_experiments)
    print("=" * 60)

    # Run patching
    results = {}
    completed = 0

    for (layer, head), head_type in all_heads:
        head_key = "L%02dH%02d" % (layer, head)
        head_results = {"layer": layer, "head": head, "type": head_type, "behaviors": {}}

        for behavior, pairs in all_pairs.items():
            effects = []

            for pair in pairs:
                correct_word = pair["correct_token"]
                incorrect_word = pair["incorrect_token"]

                if correct_word is None or incorrect_word is None:
                    # For clause_boundary, measure overall logit entropy change
                    # instead of specific token comparison
                    completed += 1
                    continue

                correct_id = find_token_id(tok, correct_word)
                incorrect_id = find_token_id(tok, incorrect_word)

                if correct_id is None or incorrect_id is None:
                    completed += 1
                    continue

                critical_pos = find_critical_position(tok, pair["clean"], correct_word)

                effect = run_patching_experiment(
                    model, tok,
                    pair["clean"], pair["corrupted"],
                    critical_pos, correct_id, incorrect_id,
                    layer, head, device
                )

                if effect is not None:
                    effects.append(effect)
                completed += 1

            if effects:
                mean_effect = sum(effects) / len(effects)
                abs_mean = sum(abs(e) for e in effects) / len(effects)
                head_results["behaviors"][behavior] = {
                    "mean_effect": round(mean_effect, 4),
                    "abs_mean_effect": round(abs_mean, 4),
                    "n_valid": len(effects),
                    "effects": [round(e, 4) for e in effects],
                }

            if completed % 100 == 0:
                effect_val = mean_effect if effects else 0
                print("  [%d/%d] %s layer %d head %d, %s: %.4f" % (
                    completed, total_experiments, head_type, layer, head,
                    behavior, effect_val))

        # Classify: which behavior has the largest absolute patching effect?
        if head_results["behaviors"]:
            ranked = sorted(head_results["behaviors"].items(),
                          key=lambda x: x[1]["abs_mean_effect"], reverse=True)
            head_results["dominant_behavior"] = ranked[0][0]
            head_results["dominant_effect"] = ranked[0][1]["abs_mean_effect"]
            head_results["ranking"] = [(b, d["abs_mean_effect"]) for b, d in ranked]
        else:
            head_results["dominant_behavior"] = "unknown"
            head_results["dominant_effect"] = 0
            head_results["ranking"] = []

        results[head_key] = head_results

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    behavior_counts = {}
    for head_key, hr in results.items():
        if hr["type"] != "unclassified":
            continue
        dom = hr["dominant_behavior"]
        behavior_counts[dom] = behavior_counts.get(dom, 0) + 1

    print("\nUnclassified head behavior distribution:")
    for behavior, count in sorted(behavior_counts.items(), key=lambda x: -x[1]):
        print("  %-20s %d heads" % (behavior, count))

    print("\nPer-head classifications:")
    for head_key, hr in sorted(results.items()):
        if hr["type"] != "unclassified":
            continue
        print("  %s: %s (effect: %.4f)" % (
            head_key, hr["dominant_behavior"], hr["dominant_effect"]))

    if args.include_controls:
        print("\nControl head classifications:")
        for head_key, hr in sorted(results.items()):
            if hr["type"] != "control":
                continue
            print("  %s (%s): dominant patching = %s (%.4f)" % (
                head_key, hr["type"], hr["dominant_behavior"], hr["dominant_effect"]))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "model": args.checkpoint,
        "classifications": args.classifications,
        "unclassified_count": len(unclassified),
        "control_count": len(control_heads),
        "behaviors_tested": list(BEHAVIOR_GENERATORS.keys()),
        "pairs_per_behavior": 20,
        "total_experiments": total_experiments,
        "behavior_distribution": behavior_counts,
        "heads": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print("\nSaved to %s" % output_path)


if __name__ == "__main__":
    main()
