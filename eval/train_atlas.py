#!/usr/bin/env python3
"""
Training script for the attention head atlas.
Trains a 410M GPT-NeoX model with frequent checkpointing for developmental analysis.

Checkpoint schedule:
- Every 50 steps for first 2000 (40 checkpoints, captures emergence)
- Every 200 steps from 2000-20000 (90 checkpoints, captures stabilization)
- Total: 130 checkpoints

Checkpoints are uploaded to R2 immediately after saving and deleted locally
to keep disk usage flat (~2 GB working space).

After training, run probe_heads.py in batch mode on R2 checkpoints.

Usage:
  python train_atlas.py \
    --tokenizer /root/tokenizers/standard-64k.json \
    --data /root/data/tokens.bin \
    --steps 20000 \
    --run-name baseline \
    --r2-prefix atlas/runs/baseline
"""

import argparse
import json
import mmap
import os
import struct
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer


def get_r2_client():
    """Create boto3 S3 client for R2."""
    import boto3
    return boto3.client("s3",
        endpoint_url=os.environ.get("R2_ENDPOINT",
            "https://b5e39abd50c5b82163c5fe72db9b880e.r2.cloudflarestorage.com"),
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY",
            "d77b3d0a3829377b3b71ffc11f610435"),
        aws_secret_access_key=os.environ.get("R2_SECRET_KEY",
            "9206e3609275a5b8655d5c5b0f3faf536415e324f4493cfe3ce2b4ffb53e0244"))


R2_BUCKET = "structok-training"


def upload_to_r2(local_path, r2_key):
    """Upload file to R2 and return success."""
    try:
        s3 = get_r2_client()
        s3.upload_file(str(local_path), R2_BUCKET, r2_key)
        return True
    except Exception as e:
        print("    WARNING: R2 upload failed for %s: %s" % (r2_key, e))
        return False


def main():
    parser = argparse.ArgumentParser(description="Train model for head atlas")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True, help="Path to pretokenized .bin file")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--run-name", default="baseline", help="Run name (baseline or comparison)")
    parser.add_argument("--r2-prefix", default=None, help="R2 prefix for uploads (e.g. atlas/runs/baseline)")
    parser.add_argument("--output-dir", default=None, help="Local output dir (optional, for keeping checkpoints locally)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--keep-local", action="store_true", help="Keep checkpoints locally after R2 upload")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a single temp checkpoint path to keep disk flat
    tmp_checkpoint = Path("/tmp/atlas-checkpoint.pt")

    # Optional local output
    if args.output_dir:
        output_dir = Path(args.output_dir)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("/tmp/atlas-output")
        output_dir.mkdir(parents=True, exist_ok=True)

    r2_prefix = args.r2_prefix or ("atlas/runs/%s" % args.run_name)

    # Load tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    # Create model
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
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
    print("ATTENTION HEAD ATLAS: Training (%s)" % args.run_name)
    print("=" * 60)
    print("  Model: GPT-NeoX 410M (%d params)" % params)
    print("  Tokenizer: %s (%d vocab)" % (args.tokenizer, vocab_size))
    print("  Data: %s" % args.data)
    print("  Steps: %d" % args.steps)
    print("  R2 prefix: %s" % r2_prefix)
    print("  Keep local: %s" % args.keep_local)
    print()

    # Load data (memory-mapped)
    data_file = open(args.data, 'rb')
    mm = mmap.mmap(data_file.fileno(), 0, access=mmap.ACCESS_READ)
    total_tokens = len(mm) // 2
    num_sequences = total_tokens // args.seq_length
    print("  Data: %d tokens, %d sequences" % (total_tokens, num_sequences))
    print()

    # Save step-0 checkpoint (random initialization, before any training)
    torch.save({
        "step": 0,
        "model_state_dict": model.state_dict(),
        "loss": float("nan"),
    }, tmp_checkpoint)
    r2_key = "%s/checkpoints/step-00000.pt" % r2_prefix
    if upload_to_r2(tmp_checkpoint, r2_key):
        print("  [step 0 (init) -> R2]")
        if not args.keep_local:
            os.remove(tmp_checkpoint)

    # Training loop
    model.train()
    t0 = time.time()
    log_entries = [{"step": 0, "loss": None, "time": 0, "r2_uploaded": True}]
    checkpoints_saved = 0

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
            print("  step %5d/%d | loss %.4f | ppl %.1f | %.1f steps/s" % (
                step, args.steps, loss.item(), torch.exp(loss).item(), step / elapsed))

        # Checkpoint schedule
        should_checkpoint = False
        if step <= 2000 and step % 50 == 0:
            should_checkpoint = True
        elif step > 2000 and step % 200 == 0:
            should_checkpoint = True

        if should_checkpoint:
            step_name = "step-%05d" % step

            # Save checkpoint (model weights only, no optimizer - saves ~50% disk)
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "loss": loss.item(),
            }, tmp_checkpoint)

            cp_size = os.path.getsize(tmp_checkpoint) / 1e6

            # Upload to R2
            r2_key = "%s/checkpoints/%s.pt" % (r2_prefix, step_name)
            uploaded = upload_to_r2(tmp_checkpoint, r2_key)

            if uploaded and not args.keep_local:
                os.remove(tmp_checkpoint)
                print("    [step %d -> R2 (%.0f MB), deleted local]" % (step, cp_size))
            elif args.keep_local and args.output_dir:
                local_path = Path(args.output_dir) / "checkpoints" / ("%s.pt" % step_name)
                os.rename(tmp_checkpoint, local_path)
                print("    [step %d -> R2 + local (%.0f MB)]" % (step, cp_size))
            else:
                print("    [step %d saved (%.0f MB), R2 %s]" % (
                    step, cp_size, "ok" if uploaded else "FAILED"))

            log_entries.append({
                "step": step,
                "loss": loss.item(),
                "time": round(time.time() - t0, 1),
                "r2_uploaded": uploaded,
            })
            checkpoints_saved += 1

            # Save log incrementally
            log_path = output_dir / "training_log.json"
            with open(log_path, "w") as f:
                json.dump(log_entries, f)

    mm.close()
    data_file.close()

    # Upload training log
    log_path = output_dir / "training_log.json"
    upload_to_r2(log_path, "%s/training_log.json" % r2_prefix)

    print("\nTraining complete. %d checkpoints saved to R2." % checkpoints_saved)
    print("R2 prefix: %s" % r2_prefix)


if __name__ == "__main__":
    main()
