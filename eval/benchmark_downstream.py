#!/usr/bin/env python3
"""
Downstream task benchmark for the developmental attention head atlas.

Evaluates whether structural differences in attention head populations (bracket
specialists, delimiter heads, spacing heads) translate into measurable accuracy
differences on downstream tasks that require structural understanding.

Five tasks, each with 100 programmatically generated examples:
  1. Bracket matching (balanced vs unbalanced sequences)
  2. Duplicate word detection in lists
  3. Field extraction from JSON objects
  4. Sentence boundary detection
  5. Prose QA (factual questions about generated passages)

All test data is generated deterministically (seed=42) with no external datasets.
Adapted from the experiment design in EXPERIMENT-DESIGN.md.

Usage:
  # Local checkpoint:
  python benchmark_downstream.py \\
    --checkpoint path/to/step-20000.pt \\
    --tokenizer path/to/standard-64k.json \\
    --output results/benchmark-baseline.json \\
    --model-name baseline

  # R2 checkpoint:
  python benchmark_downstream.py \\
    --r2-checkpoint atlas/runs/baseline/step-20000.pt \\
    --tokenizer path/to/standard-64k.json \\
    --output results/benchmark-baseline.json \\
    --model-name baseline
"""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM


# ---------------------------------------------------------------------------
# Word lists and value pools for data generation
# ---------------------------------------------------------------------------

COMMON_WORDS = [
    "apple", "bridge", "castle", "dragon", "eagle", "forest", "garden",
    "harbor", "island", "jungle", "kettle", "lantern", "mountain", "needle",
    "ocean", "palace", "quiver", "river", "sunset", "tower", "umbrella",
    "valley", "window", "crystal", "zenith", "anchor", "barrel", "candle",
    "dolphin", "ember", "falcon", "glacier", "helmet", "ivory", "jasmine",
    "kite", "lemon", "marble", "notebook", "olive", "pebble", "rabbit",
    "silver", "temple", "violin", "walnut", "basket", "compass", "feather",
    "lantern", "mirror", "pepper", "saddle", "timber", "voyage", "whistle",
]

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Elena", "Frank", "Grace", "Henry",
    "Irene", "James", "Karen", "Leo", "Maria", "Nathan", "Olivia", "Paul",
    "Quinn", "Rachel", "Samuel", "Tina", "Uma", "Victor", "Wendy", "Xavier",
]

CITIES = [
    "Portland", "Austin", "Denver", "Chicago", "Seattle", "Boston", "Miami",
    "Phoenix", "Atlanta", "Nashville", "Raleigh", "Orlando", "Dallas",
    "Minneapolis", "Detroit", "Charlotte", "Memphis", "Tampa", "Baltimore",
    "Milwaukee", "Sacramento", "Cleveland", "Pittsburgh", "Cincinnati",
]

JOB_TITLES = [
    "teacher", "engineer", "doctor", "artist", "chef", "nurse", "pilot",
    "lawyer", "writer", "designer", "mechanic", "architect", "librarian",
    "accountant", "therapist", "firefighter", "pharmacist", "dentist",
]

COMPANIES = [
    "Apex Corp", "Bright Labs", "Core Systems", "Delta Works", "Echo Group",
    "Forge Inc", "Grid Solutions", "Horizon Tech", "Ion Research", "Jade Dynamics",
]

COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange", "black", "white",
    "silver", "teal", "coral", "indigo", "maroon", "olive", "navy",
]

HOBBIES = [
    "painting", "hiking", "cooking", "reading", "cycling", "swimming",
    "gardening", "photography", "chess", "running", "fishing", "yoga",
]

STATUS_VALUES = ["active", "inactive", "pending", "verified", "suspended"]


# ---------------------------------------------------------------------------
# R2 download
# ---------------------------------------------------------------------------

def download_from_r2(r2_key, local_path):
    """Download a file from R2 storage."""
    import boto3
    s3 = boto3.client("s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"])
    print(f"Downloading from R2: {r2_key}")
    s3.download_file("structok-training", r2_key, str(local_path))
    print(f"Downloaded to {local_path} ({os.path.getsize(local_path) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, vocab_size, device):
    """Load a GPT-NeoX 410M model from a .pt checkpoint."""
    config = GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=2048,
        attn_implementation="eager",
    )
    model = GPTNeoXForCausalLM(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    step = checkpoint.get("step", "unknown")
    loss = checkpoint.get("loss", "unknown")
    print(f"Loaded checkpoint: step={step}, loss={loss}")

    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Greedy generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=10, device="cuda"):
    """Greedy-decode up to max_new_tokens from the prompt."""
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    # Truncate to fit within max_position_embeddings
    if input_ids.shape[1] > 2048 - max_new_tokens:
        input_ids = input_ids[:, -(2048 - max_new_tokens):]

    generated_ids = []
    for _ in range(max_new_tokens):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        generated_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Stop if we hit a newline or end of text
        decoded = tokenizer.decode([next_token.item()])
        if "\n" in decoded:
            break

    response = tokenizer.decode(generated_ids).strip()
    return response


# ---------------------------------------------------------------------------
# Task 1: Bracket matching
# ---------------------------------------------------------------------------

def generate_balanced_brackets(rng, depth, length_target):
    """Generate a balanced bracket sequence of approximately the target length."""
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    stack = []
    result = []

    for _ in range(length_target):
        # Decide whether to open or close
        can_open = len(result) + len(stack) + 1 <= length_target
        can_close = len(stack) > 0

        if can_open and (not can_close or rng.random() < 0.6):
            pair = rng.choice(pairs)
            result.append(pair[0])
            stack.append(pair[1])
            if len(stack) >= depth:
                # Start closing to respect depth
                pass
        elif can_close:
            result.append(stack.pop())
        else:
            break

    # Close remaining
    while stack:
        result.append(stack.pop())

    return "".join(result)


def make_bracket_examples(rng, n=100):
    """Generate bracket matching examples. Half balanced, half unbalanced."""
    examples = []
    for i in range(n):
        depth = rng.randint(2, 6)
        length = rng.randint(10, 30)
        seq = generate_balanced_brackets(rng, depth, length)

        if i < n // 2:
            # Balanced
            is_balanced = True
        else:
            # Unbalanced: remove one bracket at a random position
            if len(seq) > 1:
                remove_idx = rng.randint(0, len(seq) - 1)
                seq = seq[:remove_idx] + seq[remove_idx + 1:]
            is_balanced = False

        prompt = (
            "Is this bracket sequence balanced? Answer yes or no.\n\n"
            f"Sequence: {seq}\n\n"
            "Answer:"
        )
        expected = "yes" if is_balanced else "no"
        examples.append({
            "prompt": prompt,
            "expected": expected,
            "sequence": seq,
            "is_balanced": is_balanced,
        })

    rng.shuffle(examples)
    return examples


def score_bracket(response, expected):
    """Check if response starts with the correct yes/no."""
    response_lower = response.strip().lower()
    return response_lower.startswith(expected)


# ---------------------------------------------------------------------------
# Task 2: Duplicate detection
# ---------------------------------------------------------------------------

def make_duplicate_examples(rng, n=100):
    """Generate duplicate word detection examples."""
    examples = []
    for _ in range(n):
        list_size = rng.randint(8, 12)
        words = rng.sample(COMMON_WORDS, list_size)
        # Pick one word to duplicate
        dup_idx = rng.randint(0, len(words) - 1)
        dup_word = words[dup_idx]
        # Insert duplicate at a different position
        insert_pos = rng.randint(0, len(words))
        while insert_pos == dup_idx:
            insert_pos = rng.randint(0, len(words))
        words.insert(insert_pos, dup_word)

        word_list_str = ", ".join(words)
        prompt = (
            "Which word appears twice in this list?\n\n"
            f"List: {word_list_str}\n\n"
            "The duplicate word is:"
        )
        examples.append({
            "prompt": prompt,
            "expected": dup_word,
            "words": words,
        })

    return examples


def score_duplicate(response, expected):
    """Check if response contains the duplicate word."""
    return expected.lower() in response.strip().lower()


# ---------------------------------------------------------------------------
# Task 3: Field extraction
# ---------------------------------------------------------------------------

def make_field_extraction_examples(rng, n=100):
    """Generate JSON field extraction examples."""
    examples = []
    for _ in range(n):
        # Build a JSON object with 5-7 fields
        obj = {}
        fields_available = [
            ("name", lambda: rng.choice(FIRST_NAMES)),
            ("age", lambda: rng.randint(20, 65)),
            ("city", lambda: rng.choice(CITIES)),
            ("status", lambda: rng.choice(STATUS_VALUES)),
            ("score", lambda: rng.randint(50, 100)),
            ("job", lambda: rng.choice(JOB_TITLES)),
            ("color", lambda: rng.choice(COLORS)),
        ]
        num_fields = rng.randint(5, 7)
        selected_fields = rng.sample(fields_available, num_fields)

        for field_name, value_fn in selected_fields:
            obj[field_name] = value_fn()

        # Pick a field to ask about
        ask_field = rng.choice(list(obj.keys()))
        expected_value = str(obj[ask_field])
        json_string = json.dumps(obj, indent=2)

        prompt = (
            f'What is the value of the "{ask_field}" field in this data?\n\n'
            f'{json_string}\n\n'
            'Answer:'
        )
        examples.append({
            "prompt": prompt,
            "expected": expected_value,
            "field": ask_field,
            "json_obj": obj,
        })

    return examples


def score_field_extraction(response, expected):
    """Check if response contains the correct field value."""
    return expected.lower() in response.strip().lower()


# ---------------------------------------------------------------------------
# Task 4: Sentence boundary detection
# ---------------------------------------------------------------------------

SENTENCE_STARTERS = [
    "The", "A", "One", "Several", "Many", "Most", "Some", "Each",
    "Every", "This", "That", "These", "Those", "Our", "Their",
]

SENTENCE_TEMPLATES = [
    "{starter} {adj} {noun} {verb} near the {place}.",
    "{starter} old {noun} {verb} beside the {place}.",
    "{starter} {noun} quickly {verb} across the {place}.",
    "{starter} small {noun} {verb} through the {place}.",
    "{starter} {adj} {noun} {verb} behind the {place}.",
]

ADJECTIVES = [
    "large", "bright", "quiet", "tall", "round", "swift", "calm",
    "bold", "warm", "sharp", "soft", "dark", "clear", "fresh",
]

NOUNS = [
    "river", "tower", "bridge", "garden", "forest", "mountain",
    "village", "harbor", "meadow", "canyon", "valley", "island",
]

VERBS_PAST = [
    "flowed", "stood", "stretched", "bloomed", "grew", "rose",
    "appeared", "rested", "turned", "moved", "shifted", "settled",
]

PLACES = [
    "city gate", "stone wall", "iron fence", "wooden bridge",
    "north hill", "south road", "east shore", "west field",
    "old mill", "tall cliff", "deep pond", "wide plaza",
]


def make_sentence(rng, template_list, starter=None):
    """Generate a random sentence from templates."""
    template = rng.choice(template_list)
    if starter is None:
        starter = rng.choice(SENTENCE_STARTERS)
    return template.format(
        starter=starter,
        adj=rng.choice(ADJECTIVES),
        noun=rng.choice(NOUNS),
        verb=rng.choice(VERBS_PAST),
        place=rng.choice(PLACES),
    )


def make_sentence_boundary_examples(rng, n=100):
    """Generate sentence boundary detection examples."""
    examples = []
    for _ in range(n):
        sent1 = make_sentence(rng, SENTENCE_TEMPLATES)
        # Ensure second sentence starts with a distinct word
        second_starter = rng.choice(SENTENCE_STARTERS)
        sent2 = make_sentence(rng, SENTENCE_TEMPLATES, starter=second_starter)
        passage = f"{sent1} {sent2}"
        first_word_sent2 = second_starter

        prompt = (
            "What is the first word of the second sentence?\n\n"
            f"{passage}\n\n"
            "Answer:"
        )
        examples.append({
            "prompt": prompt,
            "expected": first_word_sent2,
            "passage": passage,
            "sentence1": sent1,
            "sentence2": sent2,
        })

    return examples


def score_sentence_boundary(response, expected):
    """Check if response starts with (or contains) the correct first word."""
    response_clean = response.strip().lower().strip('"\'')
    expected_lower = expected.lower()
    return response_clean.startswith(expected_lower) or expected_lower in response_clean


# ---------------------------------------------------------------------------
# Task 5: Prose QA
# ---------------------------------------------------------------------------

QUESTIONS_AND_FIELDS = [
    ("What is {name}'s age?", "age"),
    ("Where does {name} live?", "city"),
    ("What is {name}'s job?", "job"),
    ("What color does {name} like best?", "color"),
    ("What is {name}'s hobby?", "hobby"),
    ("What company does {name} work for?", "company"),
]


def make_prose_qa_examples(rng, n=100):
    """Generate prose QA examples from templates."""
    examples = []
    for _ in range(n):
        name = rng.choice(FIRST_NAMES)
        age = rng.randint(22, 60)
        city = rng.choice(CITIES)
        job = rng.choice(JOB_TITLES)
        company = rng.choice(COMPANIES)
        color = rng.choice(COLORS)
        hobby = rng.choice(HOBBIES)

        passage = (
            f"{name} is {age} years old and lives in {city}. "
            f"They work as a {job} at {company}. "
            f"Their favorite color is {color}. "
            f"In their free time, they enjoy {hobby}."
        )

        fields = {
            "name": name,
            "age": str(age),
            "city": city,
            "job": job,
            "company": company,
            "color": color,
            "hobby": hobby,
        }

        q_template, q_field = rng.choice(QUESTIONS_AND_FIELDS)
        question = q_template.format(name=name)
        expected = fields[q_field]

        prompt = (
            "Answer the question based on the passage.\n\n"
            f"Passage: {passage}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        examples.append({
            "prompt": prompt,
            "expected": expected,
            "passage": passage,
            "question": question,
            "field": q_field,
        })

    return examples


def score_prose_qa(response, expected):
    """Check if response contains the correct answer."""
    return expected.lower() in response.strip().lower()


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

TASKS = {
    "bracket_matching": {
        "generate": make_bracket_examples,
        "score": score_bracket,
        "description": "Bracket sequence balance detection",
    },
    "duplicate_detection": {
        "generate": make_duplicate_examples,
        "score": score_duplicate,
        "description": "Duplicate word identification in lists",
    },
    "field_extraction": {
        "generate": make_field_extraction_examples,
        "score": score_field_extraction,
        "description": "JSON field value extraction",
    },
    "sentence_boundary": {
        "generate": make_sentence_boundary_examples,
        "score": score_sentence_boundary,
        "description": "Sentence boundary detection",
    },
    "prose_qa": {
        "generate": make_prose_qa_examples,
        "score": score_prose_qa,
        "description": "Factual question answering over passages",
    },
}


def run_task(task_name, task_info, model, tokenizer, device, verbose=False):
    """Run a single evaluation task and return results."""
    rng = random.Random(42)
    examples = task_info["generate"](rng)
    score_fn = task_info["score"]

    correct = 0
    total = len(examples)
    example_results = []

    for i, ex in enumerate(examples):
        response = generate(model, tokenizer, ex["prompt"], max_new_tokens=10, device=device)
        is_correct = score_fn(response, ex["expected"])
        if is_correct:
            correct += 1

        result = {
            "index": i,
            "expected": ex["expected"],
            "response": response,
            "correct": is_correct,
        }
        example_results.append(result)

        if verbose and not is_correct:
            print(f"  [{task_name}] #{i}: expected='{ex['expected']}', got='{response}'")

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "examples": example_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Downstream task benchmark for attention head atlas models."
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to local .pt checkpoint file")
    parser.add_argument("--r2-checkpoint", type=str, default=None,
                        help="R2 key for checkpoint (downloads to temp dir)")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer .json file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write results JSON")
    parser.add_argument("--model-name", type=str, default="model",
                        help="Name for this model in results (e.g. 'baseline', 'structok')")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (default: cuda if available, else cpu)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print incorrect examples during evaluation")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=list(TASKS.keys()),
                        help="Run only specific tasks (default: all)")
    args = parser.parse_args()

    # Resolve device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Resolve checkpoint
    checkpoint_path = args.checkpoint
    tmp_dir = None
    if args.r2_checkpoint:
        tmp_dir = tempfile.mkdtemp(prefix="atlas-benchmark-")
        checkpoint_path = os.path.join(tmp_dir, os.path.basename(args.r2_checkpoint))
        download_from_r2(args.r2_checkpoint, checkpoint_path)
    elif checkpoint_path is None:
        parser.error("Either --checkpoint or --r2-checkpoint is required")

    # Load tokenizer and model
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, vocab_size, device)

    # Clean up downloaded checkpoint to free disk
    if tmp_dir is not None:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Cleaned up temporary checkpoint download")

    # Select tasks
    task_names = args.tasks if args.tasks else list(TASKS.keys())

    # Run all tasks
    print(f"\nRunning {len(task_names)} tasks, 100 examples each")
    print("=" * 60)

    all_results = {}
    total_correct = 0
    total_examples = 0

    for task_name in task_names:
        task_info = TASKS[task_name]
        print(f"\n[{task_name}] {task_info['description']}...")
        result = run_task(task_name, task_info, model, tokenizer, device,
                          verbose=args.verbose)
        all_results[task_name] = result
        total_correct += result["correct"]
        total_examples += result["total"]
        print(f"  Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")

    overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS SUMMARY: {args.model_name}")
    print("=" * 60)
    for task_name in task_names:
        r = all_results[task_name]
        bar = "#" * int(r["accuracy"] * 40)
        print(f"  {task_name:25s}  {r['accuracy']:6.1%}  ({r['correct']:3d}/{r['total']:3d})  {bar}")
    print("-" * 60)
    print(f"  {'OVERALL':25s}  {overall_accuracy:6.1%}  ({total_correct:3d}/{total_examples:3d})")
    print("=" * 60)

    # Save results
    output_data = {
        "model": args.model_name,
        "checkpoint": args.checkpoint or args.r2_checkpoint,
        "tokenizer": args.tokenizer,
        "device": device,
        "overall_accuracy": round(overall_accuracy, 4),
        "results": all_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
