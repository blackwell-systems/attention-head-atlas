#!/usr/bin/env python3
"""
Circuit causality experiment: can coupling an isolated head to a circuit
member prevent P0 collapse?

Modifies the training loop to add a regularization term that encourages
doomed heads (those that collapse into P0 in the baseline) to correlate
their attention patterns with nearby circuit members.

Two runs:
  1. Coupled: regularization on 8 doomed heads paired with circuit members
  2. Control: same regularization on 8 random non-doomed heads

If coupled doomed heads avoid P0 collapse while control heads don't change,
circuits are causally protective.

Based on train_atlas.py with the addition of attention coupling loss.

Usage:
  python train_circuit_causality.py \
    --tokenizer /root/standard-64k.json \
    --data /root/atlas-standard-64k.bin \
    --run-name circuit-coupled \
    --r2-prefix atlas/runs/circuit-coupled \
    --output-dir /root/runs/circuit-coupled \
    --coupling-config coupled \
    --coupling-lambda 0.01 \
    --steps 20000
"""

import argparse
import json
import mmap
import os
import struct
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


# ── Coupling configurations ──
# Identified from baseline P0 analysis and circuit correlation data.

# Best candidates: doomed heads with highest correlation to non-spacing
# circuit members, plus late sinkers that resist collapse the longest.
COUPLED_PAIRS = [
    # (doomed_layer, doomed_head, partner_layer, partner_head, note)
    (4,  3,  6,  9, "highest corr to positional_prev circuit (0.851), sinks step 700"),
    (10, 13, 12, 15, "corr 0.563 to delimiter circuit, sinks step 300"),
    (1,  4,  5,  1, "corr 0.460 to delimiter circuit, sinks step 200"),
    (0,  5,  5,  1, "corr 0.291 to delimiter circuit, sinks step 1300"),
    (0, 12,  3,  0, "late sinker (step 19800), corr 0.703 to spacing circuit"),
    (0, 15, 15,  4, "late sinker (step 17800), corr 0.640 to spacing circuit"),
    (21, 15, 14, 2, "late sinker (step 5200), corr 0.653 to spacing circuit"),
    (22,  9, 15,  4, "corr 0.745 to spacing circuit, sinks step 850"),
]

# Control: 8 random non-P0 heads paired with circuit members.
# These heads are productive (delimiter, spacing, positional_prev).
# Regularization should not change their behavior.
CONTROL_PAIRS = [
    # (head_layer, head_head, partner_layer, partner_head, note)
    (3,  5,  6,  9, "delimiter head, paired with positional_prev circuit"),
    (7,  2,  5,  1, "spacing head, paired with delimiter circuit"),
    (9,  8, 12, 15, "spacing head, paired with delimiter circuit"),
    (11, 6, 15,  4, "spacing head, paired with spacing circuit"),
    (15, 9,  3,  0, "spacing head, paired with spacing circuit"),
    (18, 2, 14,  2, "spacing head, paired with spacing circuit"),
    (6,  3,  5,  1, "delimiter head, paired with delimiter circuit"),
    (8, 11,  6,  9, "positional_prev head, paired with positional_prev circuit"),
]


def get_coupling_pairs(config):
    """Return list of (doomed_layer, doomed_head, partner_layer, partner_head)."""
    if config == "coupled":
        return [(l, h, pl, ph) for l, h, pl, ph, _ in COUPLED_PAIRS]
    elif config == "control":
        return [(l, h, pl, ph) for l, h, pl, ph, _ in CONTROL_PAIRS]
    elif config == "none":
        return []
    else:
        raise ValueError("Unknown coupling config: %s" % config)


def compute_coupling_loss(model, coupling_pairs, input_ids, device):
    """
    Compute attention coupling loss between doomed heads and their partners.

    For each pair, extract the attention pattern (the attention weights after
    softmax) for both heads, and compute negative cosine similarity.
    Minimizing this loss encourages the doomed head's attention pattern to
    correlate with the partner's.

    Returns scalar coupling loss.
    """
    if not coupling_pairs:
        return torch.tensor(0.0, device=device)

    # Forward pass with attention outputs
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)
    attentions = outputs.attentions  # tuple of (batch, num_heads, seq, seq)

    total_loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    for doomed_layer, doomed_head, partner_layer, partner_head in coupling_pairs:
        if doomed_layer >= len(attentions) or partner_layer >= len(attentions):
            continue

        # Extract attention patterns: (seq, seq) for each head
        doomed_attn = attentions[doomed_layer][0, doomed_head].detach()
        partner_attn = attentions[partner_layer][0, partner_head].detach()

        # Flatten to vectors and compute cosine similarity
        d_flat = doomed_attn.reshape(-1)
        p_flat = partner_attn.reshape(-1)

        # Negative cosine similarity (minimize = maximize similarity)
        cos_sim = F.cosine_similarity(d_flat.unsqueeze(0), p_flat.unsqueeze(0))
        total_loss = total_loss - cos_sim

        n_pairs += 1

    if n_pairs > 0:
        total_loss = total_loss / n_pairs

    return total_loss


def get_r2_client():
    """Create boto3 S3 client for R2."""
    import boto3
    return boto3.client("s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"])


R2_BUCKET = "structok-training"


def upload_to_r2_with_retry(local_path, r2_key, max_retries=3):
    """Upload file to R2 with retries and verification."""
    s3 = get_r2_client()
    local_size = os.path.getsize(local_path)

    for attempt in range(1, max_retries + 1):
        try:
            s3.upload_file(str(local_path), R2_BUCKET, r2_key)
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
                time.sleep(5 * attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description="Train with circuit coupling")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True, help="Path to pretokenized .bin file")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--run-name", default="circuit-coupled")
    parser.add_argument("--r2-prefix", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--coupling-config", required=True,
                       choices=["coupled", "control", "none"],
                       help="Which heads to couple: coupled (doomed+partners), "
                            "control (non-doomed+partners), none (baseline)")
    parser.add_argument("--coupling-lambda", type=float, default=0.01,
                       help="Coupling loss weight (default 0.01)")
    parser.add_argument("--coupling-every", type=int, default=10,
                       help="Compute coupling loss every N steps (saves compute)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    r2_prefix = args.r2_prefix or ("atlas/runs/%s" % args.run_name)

    # Load tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    # Create model (NeoX 410M, same as baseline)
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

    # Get coupling pairs
    coupling_pairs = get_coupling_pairs(args.coupling_config)

    params = sum(p.numel() for p in model.parameters())
    print("=" * 60)
    print("CIRCUIT CAUSALITY EXPERIMENT")
    print("=" * 60)
    print("  Model: GPT-NeoX 410M (%d params)" % params)
    print("  Tokenizer: %s (%d vocab)" % (args.tokenizer, vocab_size))
    print("  Data: %s" % args.data)
    print("  Steps: %d" % args.steps)
    print("  Coupling config: %s" % args.coupling_config)
    print("  Coupling lambda: %g" % args.coupling_lambda)
    print("  Coupling every: %d steps" % args.coupling_every)
    print("  Coupling pairs: %d" % len(coupling_pairs))
    for dl, dh, pl, ph in coupling_pairs:
        print("    L%02dH%02d -> L%02dH%02d" % (dl, dh, pl, ph))
    print("  Output: %s" % output_dir)
    print("  R2 prefix: %s" % r2_prefix)
    print()

    # Background uploader
    upload_queue = []
    upload_lock = threading.Lock()
    upload_results = {"uploaded": 0, "failed": 0}
    training_active = [True]

    def background_uploader():
        while True:
            item = None
            with upload_lock:
                if upload_queue:
                    item = upload_queue.pop(0)
            if item is None:
                time.sleep(1)
                if not training_active[0]:
                    break
                continue
            local_path, r2_key = item
            if upload_to_r2_with_retry(local_path, r2_key):
                upload_results["uploaded"] += 1
            else:
                upload_results["failed"] += 1

    uploader_thread = threading.Thread(target=background_uploader, daemon=True)
    uploader_thread.start()

    def queue_upload(local_path, r2_key):
        with upload_lock:
            upload_queue.append((str(local_path), r2_key))

    # Save step-0
    step0_path = checkpoint_dir / "step-00000.pt"
    torch.save({
        "step": 0,
        "model_state_dict": model.state_dict(),
        "loss": float("nan"),
        "coupling_config": args.coupling_config,
        "coupling_lambda": args.coupling_lambda,
        "coupling_pairs": [(dl, dh, pl, ph) for dl, dh, pl, ph in coupling_pairs],
    }, step0_path)
    print("  [step 0 saved: %.0f MB]" % (os.path.getsize(step0_path) / 1e6))
    queue_upload(step0_path, "%s/checkpoints/step-00000.pt" % r2_prefix)

    # Load data
    data_file = open(args.data, 'rb')
    mm = mmap.mmap(data_file.fileno(), 0, access=mmap.ACCESS_READ)
    total_tokens = len(mm) // 2
    num_sequences = total_tokens // args.seq_length
    print("  Data: %d tokens, %d sequences" % (total_tokens, num_sequences))
    print()

    # Training loop
    model.train()
    t0 = time.time()
    log_entries = [{"step": 0, "loss": None, "coupling_loss": None, "time": 0}]
    checkpoints_saved = 0

    for step in range(1, args.steps + 1):
        # Random sequence
        seq_idx = torch.randint(0, num_sequences - 1, (1,)).item()
        offset = seq_idx * args.seq_length * 2
        raw = mm[offset:offset + args.seq_length * 2]
        tokens = [t[0] for t in struct.iter_unpack('<H', raw)]

        input_ids = torch.tensor([tokens[:-1]], device=device)
        labels = torch.tensor([tokens[1:]], device=device)

        # Standard LM loss
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            lm_loss = outputs.loss

        # Coupling loss (every N steps to save compute)
        coupling_loss_val = 0.0
        if coupling_pairs and step % args.coupling_every == 0:
            model.eval()
            c_loss = compute_coupling_loss(model, coupling_pairs, input_ids, device)
            coupling_loss_val = c_loss.item()
            total_loss = lm_loss + args.coupling_lambda * c_loss
            model.train()
        else:
            total_loss = lm_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if step % 10 == 0:
            elapsed = time.time() - t0
            if coupling_loss_val != 0:
                print("  step %5d/%d | lm_loss %.4f | coupling %.4f | %.1f steps/s" % (
                    step, args.steps, lm_loss.item(), coupling_loss_val, step / elapsed))
            else:
                print("  step %5d/%d | lm_loss %.4f | %.1f steps/s" % (
                    step, args.steps, lm_loss.item(), step / elapsed))

        # Checkpoint schedule (same as atlas)
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
                "loss": lm_loss.item(),
                "coupling_loss": coupling_loss_val,
                "coupling_config": args.coupling_config,
                "coupling_lambda": args.coupling_lambda,
            }, cp_path)

            cp_size = os.path.getsize(cp_path) / 1e6
            print("    [checkpoint saved: step %d, %.0f MB]" % (step, cp_size))

            r2_key = "%s/checkpoints/step-%05d.pt" % (r2_prefix, step)
            queue_upload(cp_path, r2_key)

            log_entries.append({
                "step": step,
                "loss": lm_loss.item(),
                "coupling_loss": coupling_loss_val,
                "time": round(time.time() - t0, 1),
            })
            checkpoints_saved += 1

            with open(output_dir / "training_log.json", "w") as f:
                json.dump(log_entries, f)

    mm.close()
    data_file.close()

    elapsed = time.time() - t0
    print("\nTraining complete. %d checkpoints in %.0f minutes." % (
        checkpoints_saved, elapsed / 60))

    # Wait for uploads
    training_active[0] = False
    print("\nWaiting for R2 uploads...")
    with upload_lock:
        remaining = len(upload_queue)
    while remaining > 0:
        print("  %d remaining..." % remaining)
        time.sleep(10)
        with upload_lock:
            remaining = len(upload_queue)
    uploader_thread.join(timeout=60)

    print("Uploads: %d OK, %d failed" % (
        upload_results["uploaded"], upload_results["failed"]))

    # Verify on R2
    if not args.skip_upload:
        print("\nVerifying R2...")
        s3 = get_r2_client()
        verified = 0
        for cp in sorted(checkpoint_dir.glob("step-*.pt")):
            r2_key = "%s/checkpoints/%s" % (r2_prefix, cp.name)
            try:
                resp = s3.head_object(Bucket=R2_BUCKET, Key=r2_key)
                if resp["ContentLength"] == os.path.getsize(cp):
                    verified += 1
            except Exception:
                pass
        print("Verified: %d / %d" % (verified, checkpoints_saved + 1))


if __name__ == "__main__":
    main()
