#!/usr/bin/env python3
"""
Completion-based downstream benchmark for raw language models.

Unlike benchmark_downstream.py (which requires instruction following),
this benchmark measures next-token prediction accuracy on tasks where
the correct continuation is determined by structure, not instruction.

Five tasks:
  1. Bracket closing: predict the correct closing bracket
  2. JSON structure: predict structural tokens in JSON (: , " { })
  3. Pattern continuation: predict the next token in a repeating sequence
  4. Structural token accuracy: per-token-type accuracy on structured text
  5. Whitespace prediction: predict space/newline at word boundaries

All tasks measure top-1 greedy prediction accuracy. The comparison model
(merge barriers) should outperform the baseline on structural tokens
because it has more bracket/delimiter heads and fewer spacing heads.

Usage:
  python benchmark_completion.py \
    --checkpoint path/to/step-20000.pt \
    --tokenizer path/to/standard-64k.json \
    --output results/completion-baseline.json \
    --model-name baseline
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer


SPACING_CHARS = set(' \t\n\r')
DELIMITER_CHARS = set('|@<>"\',:;\t{}[]()')
BRACKET_OPEN = {'(': ')', '[': ']', '{': '}'}
BRACKET_CLOSE = set(')]}\u003e')


def load_model(checkpoint_path, tokenizer_path, device, size="410m"):
    """Load model and tokenizer."""
    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()

    if size == "410m-llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(
            vocab_size=vocab_size, hidden_size=1024, num_hidden_layers=24,
            num_attention_heads=16, num_key_value_heads=4,
            intermediate_size=2816, max_position_embeddings=2048,
            rope_theta=500000.0, use_cache=False,
            attn_implementation="eager")
        model = LlamaForCausalLM(config).to(device)
    else:
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


@torch.no_grad()
def get_next_token(model, tok, text, device):
    """Get the model's top-1 predicted next token."""
    ids = tok.encode(text).ids
    if len(ids) > 2040:
        ids = ids[-2040:]
    input_ids = torch.tensor([ids], device=device)
    logits = model(input_ids).logits[0, -1]
    next_id = torch.argmax(logits).item()
    return tok.decode([next_id]), next_id


@torch.no_grad()
def get_per_token_accuracy(model, tok, text, device):
    """Run text through model, return per-position top-1 accuracy and token info."""
    ids = tok.encode(text).ids
    if len(ids) > 2048:
        ids = ids[:2048]
    input_ids = torch.tensor([ids], device=device)
    logits = model(input_ids).logits[0]  # (seq_len, vocab)
    preds = torch.argmax(logits[:-1], dim=-1)  # predict position i+1 from position i
    targets = torch.tensor(ids[1:], device=device)
    correct = (preds == targets).cpu().tolist()

    results = []
    for i, (target_id, is_correct) in enumerate(zip(ids[1:], correct)):
        decoded = tok.decode([target_id])
        token_type = classify_token(decoded)
        results.append({
            "position": i + 1,
            "token": decoded,
            "type": token_type,
            "correct": is_correct,
        })
    return results


def classify_token(decoded):
    """Classify a token for structural analysis."""
    if not decoded:
        return "other"
    stripped = decoded.strip()
    if not stripped or all(c in SPACING_CHARS for c in decoded):
        return "spacing"
    if stripped in BRACKET_OPEN or stripped in BRACKET_CLOSE:
        return "bracket"
    if all(c in DELIMITER_CHARS for c in stripped):
        return "delimiter"
    if stripped.isdigit():
        return "numeric"
    return "content"


# ── Task 1: Bracket closing ──

def task_bracket_closing(model, tok, device, rng):
    """Test if model predicts correct closing bracket."""
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    results = []

    for _ in range(100):
        depth = rng.randint(1, 4)
        # Build nested brackets
        opens = []
        seq = ""
        for _ in range(depth):
            o, c = rng.choice(pairs)
            seq += o
            opens.append(c)

        # Add some content inside
        content = rng.choice(["x", "data", "1", "value", "item"])
        seq += content

        # Close all but the last one
        for c in reversed(opens[1:]):
            seq += c

        # Model should predict the final closing bracket
        expected_close = opens[0]
        predicted, _ = get_next_token(model, tok, seq, device)
        predicted_stripped = predicted.strip()

        correct = predicted_stripped == expected_close
        results.append({
            "sequence": seq,
            "expected": expected_close,
            "predicted": predicted_stripped,
            "correct": correct,
            "depth": depth,
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    return {"accuracy": accuracy, "total": len(results), "examples": results}


# ── Task 2: JSON structure prediction ──

def task_json_structure(model, tok, device, rng):
    """Test if model predicts JSON structural tokens correctly."""
    names = ["Alice", "Bob", "Carol", "David", "Elena", "Frank"]
    cities = ["Portland", "Austin", "Denver", "Chicago", "Seattle"]
    jobs = ["teacher", "engineer", "doctor", "artist", "chef"]

    results = []
    for _ in range(100):
        name = rng.choice(names)
        age = rng.randint(20, 60)
        city = rng.choice(cities)
        job = rng.choice(jobs)

        # Test different structural predictions
        test_type = rng.choice(["colon", "comma", "close_brace", "close_quote"])

        if test_type == "colon":
            # After a key, predict :
            prompt = '{"name"'
            expected = ":"
        elif test_type == "comma":
            # After a value, predict ,
            prompt = '{"name": "%s"' % name
            expected = ","
        elif test_type == "close_brace":
            # After last value, predict }
            prompt = '{"name": "%s", "age": %d' % (name, age)
            # Could be , or } depending on whether more fields follow
            # Use } as expected since we're testing structural understanding
            expected = "}"
        else:
            # After string content, predict closing "
            prompt = '{"name": "%s' % name
            expected = '"'

        predicted, _ = get_next_token(model, tok, prompt, device)
        predicted_stripped = predicted.strip()
        correct = expected in predicted_stripped or predicted_stripped == expected

        results.append({
            "prompt_suffix": prompt[-30:],
            "test_type": test_type,
            "expected": expected,
            "predicted": predicted_stripped,
            "correct": correct,
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    by_type = {}
    for r in results:
        t = r["test_type"]
        if t not in by_type:
            by_type[t] = {"correct": 0, "total": 0}
        by_type[t]["total"] += 1
        if r["correct"]:
            by_type[t]["correct"] += 1
    for t in by_type:
        by_type[t]["accuracy"] = by_type[t]["correct"] / by_type[t]["total"]

    return {"accuracy": accuracy, "total": len(results), "by_type": by_type, "examples": results}


# ── Task 3: Pattern continuation ──

def task_pattern_continuation(model, tok, device, rng):
    """Test if model continues repeating token patterns."""
    words = ["apple", "red", "north", "alpha", "one", "dog", "sun", "key"]
    results = []

    for _ in range(100):
        # Pick 2-4 words for the pattern
        pattern_len = rng.randint(2, 4)
        pattern = rng.sample(words, pattern_len)

        # Repeat 3-5 times, then give partial
        repeats = rng.randint(3, 5)
        seq = " ".join(pattern * repeats)
        # Add partial pattern
        partial_len = rng.randint(0, pattern_len - 1)
        if partial_len > 0:
            seq += " " + " ".join(pattern[:partial_len])

        expected = pattern[partial_len]
        predicted, _ = get_next_token(model, tok, seq, device)
        predicted_stripped = predicted.strip().lower()

        correct = expected.lower() in predicted_stripped or predicted_stripped.startswith(expected.lower())
        results.append({
            "pattern": pattern,
            "partial_len": partial_len,
            "expected": expected,
            "predicted": predicted_stripped,
            "correct": correct,
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    return {"accuracy": accuracy, "total": len(results), "examples": results}


# ── Task 4: Structural token accuracy ──

def task_structural_accuracy(model, tok, device, rng):
    """Measure per-token-type prediction accuracy on structured text."""
    # Generate varied structured text
    texts = []

    # JSON objects
    for _ in range(10):
        obj = {"name": rng.choice(["Alice", "Bob", "Carol"]),
               "age": rng.randint(20, 60),
               "city": rng.choice(["Portland", "Austin", "Denver"])}
        texts.append(json.dumps(obj))

    # Nested JSON
    for _ in range(5):
        obj = {"user": {"name": rng.choice(["Alice", "Bob"]),
                        "scores": [rng.randint(1, 100) for _ in range(3)]},
               "active": rng.choice(["true", "false"])}
        texts.append(json.dumps(obj))

    # Code-like text
    code_snippets = [
        "func main() {\n    x := getValue()\n    if x > 0 {\n        process(x)\n    }\n}",
        "for i := 0; i < len(items); i++ {\n    result[i] = transform(items[i])\n}",
        "type Config struct {\n    Name string\n    Port int\n    Debug bool\n}",
    ]
    texts.extend(code_snippets)

    # Bracket sequences
    for _ in range(5):
        depth = rng.randint(2, 5)
        seq = ""
        stack = []
        for _ in range(depth * 2):
            if rng.random() < 0.5 and len(stack) < depth:
                b = rng.choice(["(", "[", "{"])
                seq += b
                stack.append(BRACKET_OPEN[b])
            elif stack:
                seq += stack.pop()
        while stack:
            seq += stack.pop()
        texts.append(seq)

    # Aggregate per-token-type accuracy
    type_stats = {}
    all_results = []
    for text in texts:
        token_results = get_per_token_accuracy(model, tok, text, device)
        for tr in token_results:
            t = tr["type"]
            if t not in type_stats:
                type_stats[t] = {"correct": 0, "total": 0}
            type_stats[t]["total"] += 1
            if tr["correct"]:
                type_stats[t]["correct"] += 1
        all_results.extend(token_results)

    for t in type_stats:
        type_stats[t]["accuracy"] = type_stats[t]["correct"] / max(type_stats[t]["total"], 1)

    overall = sum(s["correct"] for s in type_stats.values()) / max(sum(s["total"] for s in type_stats.values()), 1)
    return {"accuracy": overall, "by_type": type_stats, "total_tokens": len(all_results)}


# ── Task 5: Whitespace prediction ──

def task_whitespace_prediction(model, tok, device, rng):
    """Test if model predicts spaces between words correctly."""
    sentences = [
        "The quick brown fox jumps over the lazy dog",
        "She walked down the long winding road toward the city",
        "Every morning he drinks coffee and reads the newspaper",
        "The old stone bridge crossed the river near the village",
        "Small birds gathered on the wire above the garden",
        "The library was quiet except for the sound of pages turning",
        "Rain fell steadily through the night and into the morning",
        "They built the house on the hill overlooking the valley",
        "The market was crowded with people buying fresh produce",
        "A single light burned in the window of the top floor",
    ]

    results = []
    for _ in range(100):
        sent = rng.choice(sentences)
        words = sent.split()
        # Pick a random word boundary
        cut_idx = rng.randint(2, len(words) - 2)
        prefix = " ".join(words[:cut_idx])

        predicted, _ = get_next_token(model, tok, prefix, device)

        # The next token should start with a space (word boundary)
        expected_next_word = words[cut_idx]
        has_space = predicted.startswith(" ")
        word_correct = expected_next_word.lower() in predicted.strip().lower()

        results.append({
            "prefix_last_word": words[cut_idx - 1],
            "expected_next": expected_next_word,
            "predicted": predicted,
            "has_space": has_space,
            "word_correct": word_correct,
        })

    space_accuracy = sum(r["has_space"] for r in results) / len(results)
    word_accuracy = sum(r["word_correct"] for r in results) / len(results)
    return {
        "space_accuracy": space_accuracy,
        "word_accuracy": word_accuracy,
        "total": len(results),
        "examples": results,
    }


TASKS = {
    "bracket_closing": ("Bracket closing prediction", task_bracket_closing),
    "json_structure": ("JSON structural token prediction", task_json_structure),
    "pattern_continuation": ("Repeating pattern continuation", task_pattern_continuation),
    "structural_accuracy": ("Per-token-type accuracy on structured text", task_structural_accuracy),
    "whitespace_prediction": ("Whitespace boundary prediction", task_whitespace_prediction),
}


def main():
    parser = argparse.ArgumentParser(description="Completion-based benchmark")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--size", default="410m", choices=["410m", "410m-llama"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    model, tok = load_model(args.checkpoint, args.tokenizer, device, args.size)
    rng = random.Random(42)

    print("\nRunning %d completion tasks" % len(TASKS))
    print("=" * 60)

    all_results = {}
    for name, (desc, fn) in TASKS.items():
        print("\n[%s] %s..." % (name, desc))
        result = fn(model, tok, device, rng)
        all_results[name] = result

        if "accuracy" in result:
            print("  Accuracy: %.1f%%" % (result["accuracy"] * 100))
        if "by_type" in result:
            for t, s in sorted(result["by_type"].items()):
                print("    %-12s %.1f%% (%d/%d)" % (t, s["accuracy"] * 100, s["correct"], s["total"]))
        if "space_accuracy" in result:
            print("  Space: %.1f%%  Word: %.1f%%" % (result["space_accuracy"] * 100, result["word_accuracy"] * 100))

    print("\n" + "=" * 60)
    print("SUMMARY: %s" % args.model_name)
    print("=" * 60)
    for name, (desc, _) in TASKS.items():
        r = all_results[name]
        if "accuracy" in r:
            print("  %-30s %.1f%%" % (name, r["accuracy"] * 100))
        if "space_accuracy" in r:
            print("  %-30s space=%.1f%% word=%.1f%%" % (name, r["space_accuracy"] * 100, r["word_accuracy"] * 100))
    print("=" * 60)

    output = {
        "model": args.model_name,
        "checkpoint": args.checkpoint,
        "results": all_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nSaved to %s" % args.output)


if __name__ == "__main__":
    main()
