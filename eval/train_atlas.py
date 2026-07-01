#!/usr/bin/env python3
"""
Training script for the attention head atlas.
Trains a 410M GPT-NeoX model with frequent checkpointing for developmental analysis.

Checkpoint schedule:
- Every 50 steps for first 2000 (40 checkpoints, captures emergence)
- Every 200 steps from 2000-20000 (90 checkpoints, captures stabilization)
- Total: 130 checkpoints

After training, run probe_heads.py in batch mode on the checkpoint directory.

Usage:
  python train_atlas.py \
    --tokenizer ../tokenizers/standard-64k.json \
    --data ../data/tokens.bin \
    --steps 20000 \
    --output-dir ../runs/baseline/
"""

import argparse
import json
import mmap
import struct
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Train model for head atlas")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True, help="Path to pretokenized .bin file")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--output-dir", default="runs/baseline/")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    # Create model
    config = GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=2048,
        use_cache=False,
    )
    model = GPTNeoXForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    params = sum(p.numel() for p in model.parameters())
    print("=" * 60)
    print("ATTENTION HEAD ATLAS: Training")
    print("=" * 60)
    print("  Model: GPT-NeoX 410M (%d params)" % params)
    print("  Tokenizer: %s (%d vocab)" % (args.tokenizer, vocab_size))
    print("  Data: %s" % args.data)
    print("  Steps: %d" % args.steps)
    print("  Checkpoints: every 50 steps (0-2000), every 200 steps (2000+)")
    print("  Output: %s" % output_dir)
    print()

    # Load data (memory-mapped)
    data_file = open(args.data, 'rb')
    mm = mmap.mmap(data_file.fileno(), 0, access=mmap.ACCESS_READ)
    total_tokens = len(mm) // 2
    num_sequences = total_tokens // args.seq_length
    print("  Data: %d tokens, %d sequences" % (total_tokens, num_sequences))
    print()

    # Training loop
    model.train()
    t0 = time.time()
    log_entries = []

    for step in range(1, args.steps + 1):
        # Random sequence
        seq_idx = torch.randint(0, num_sequences - 1, (1,)).item()
        offset = seq_idx * args.seq_length * 2
        raw = mm[offset:offset + args.seq_length * 2]
        tokens = [t[0] for t in struct.iter_unpack('<H', raw)]

        input_ids = torch.tensor([tokens[:-1]], device=device)
        labels = torch.tensor([tokens[1:]], device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if step % 10 == 0:
            elapsed = time.time() - t0
            print("  step %5d/%d | loss %.4f | %.1f steps/s" % (
                step, args.steps, loss.item(), step / elapsed))

        # Checkpoint schedule
        should_checkpoint = False
        if step <= 2000 and step % 50 == 0:
            should_checkpoint = True
        elif step > 2000 and step % 200 == 0:
            should_checkpoint = True

        if should_checkpoint:
            cp_path = checkpoint_dir / ("step-%05d.pt" % step)
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "loss": loss.item(),
            }, cp_path)

            log_entries.append({"step": step, "loss": loss.item(), "time": time.time() - t0})

            # Save log incrementally
            with open(output_dir / "training_log.json", "w") as f:
                json.dump(log_entries, f)

            print("    [checkpoint saved: step %d]" % step)

    mm.close()
    data_file.close()

    print("\nTraining complete. %d checkpoints saved." % len(log_entries))
    print("Next: run probe_heads.py --checkpoint-dir %s" % checkpoint_dir)


if __name__ == "__main__":
    main()
