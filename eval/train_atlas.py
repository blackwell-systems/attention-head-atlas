#!/usr/bin/env python3
"""
Training script for the attention head atlas.
Trains a 410M GPT-NeoX model with frequent checkpointing for developmental analysis.

Checkpoint schedule:
- Step 0 (random init, before any training)
- Every 50 steps for first 2000 (40 checkpoints, captures emergence)
- Every 200 steps from 2000-20000 (90 checkpoints, captures stabilization)
- Total: 131 checkpoints

All checkpoints are saved locally during training. After training completes,
all checkpoints are uploaded to R2 with retry logic and verification.
Requires ~250 GB disk.

Usage:
  python train_atlas.py \
    --tokenizer /root/tokenizers/standard-64k.json \
    --data /root/data/tokens.bin \
    --steps 20000 \
    --run-name baseline \
    --r2-prefix atlas/runs/baseline \
    --output-dir /root/runs/baseline
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


def upload_to_r2_with_retry(local_path, r2_key, max_retries=3):
    """Upload file to R2 with retries and verification."""
    s3 = get_r2_client()
    local_size = os.path.getsize(local_path)

    for attempt in range(1, max_retries + 1):
        try:
            s3.upload_file(str(local_path), R2_BUCKET, r2_key)

            # Verify upload by checking size on R2
            resp = s3.head_object(Bucket=R2_BUCKET, Key=r2_key)
            r2_size = resp["ContentLength"]
            if r2_size != local_size:
                print("    VERIFY FAIL: %s local=%d r2=%d (attempt %d)" % (
                    r2_key, local_size, r2_size, attempt))
                continue
            return True
        except Exception as e:
            print("    UPLOAD FAIL: %s attempt %d: %s" % (r2_key, attempt, e))
            if attempt < max_retries:
                time.sleep(5 * attempt)  # backoff

    return False


def upload_all_to_r2(output_dir, r2_prefix):
    """Upload all checkpoints, training log, and probes to R2 after training."""
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"

    checkpoints = sorted(checkpoint_dir.glob("step-*.pt"))
    print("\n=== Uploading %d checkpoints to R2 ===" % len(checkpoints))

    uploaded = 0
    failed = 0
    for cp in checkpoints:
        r2_key = "%s/checkpoints/%s" % (r2_prefix, cp.name)
        size_mb = os.path.getsize(cp) / 1e6
        print("  %s (%.0f MB)..." % (cp.name, size_mb), end=" ", flush=True)
        if upload_to_r2_with_retry(cp, r2_key):
            print("OK")
            uploaded += 1
        else:
            print("FAILED")
            failed += 1

    # Upload training log
    log_path = output_dir / "training_log.json"
    if log_path.exists():
        r2_key = "%s/training_log.json" % r2_prefix
        upload_to_r2_with_retry(log_path, r2_key)
        print("  training_log.json uploaded")

    print("\n=== Upload complete: %d uploaded, %d failed ===" % (uploaded, failed))
    if failed > 0:
        print("WARNING: %d checkpoints failed to upload. Local copies preserved." % failed)
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Train model for head atlas")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True, help="Path to pretokenized .bin file")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--run-name", default="baseline", help="Run name (baseline or comparison)")
    parser.add_argument("--r2-prefix", default=None, help="R2 prefix for uploads")
    parser.add_argument("--output-dir", required=True, help="Local output directory for checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--skip-upload", action="store_true", help="Skip R2 upload after training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    print("  Output: %s" % output_dir)
    print("  R2 prefix: %s" % r2_prefix)
    print()

    # Save step-0 checkpoint (random initialization, before any training)
    step0_path = checkpoint_dir / "step-00000.pt"
    torch.save({
        "step": 0,
        "model_state_dict": model.state_dict(),
        "loss": float("nan"),
    }, step0_path)
    print("  [step 0 (random init) saved: %.0f MB]" % (os.path.getsize(step0_path) / 1e6))

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
    log_entries = [{"step": 0, "loss": None, "time": 0}]
    checkpoints_saved = 1  # step 0

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
            cp_path = checkpoint_dir / ("step-%05d.pt" % step)
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "loss": loss.item(),
            }, cp_path)

            cp_size = os.path.getsize(cp_path) / 1e6
            print("    [checkpoint saved: step %d, %.0f MB]" % (step, cp_size))

            log_entries.append({
                "step": step,
                "loss": loss.item(),
                "time": round(time.time() - t0, 1),
            })
            checkpoints_saved += 1

            # Save log incrementally
            with open(output_dir / "training_log.json", "w") as f:
                json.dump(log_entries, f)

    mm.close()
    data_file.close()

    elapsed = time.time() - t0
    print("\nTraining complete. %d checkpoints saved locally in %.0f minutes." % (
        checkpoints_saved, elapsed / 60))

    # Disk usage
    total_size = sum(f.stat().st_size for f in checkpoint_dir.glob("*.pt"))
    print("Total checkpoint size: %.1f GB" % (total_size / 1e9))

    # Upload to R2
    if not args.skip_upload:
        success = upload_all_to_r2(output_dir, r2_prefix)
        if success:
            print("\nAll checkpoints verified on R2. Safe to destroy instance.")
        else:
            print("\nSome uploads failed. DO NOT destroy instance until resolved.")
    else:
        print("\nSkipping R2 upload (--skip-upload). Upload manually before destroying instance.")


if __name__ == "__main__":
    main()
