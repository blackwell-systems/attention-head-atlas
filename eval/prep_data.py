#!/usr/bin/env python3
"""
Download SlimPajama sample and pretokenize for atlas training.

Downloads ~5GB from HuggingFace, tokenizes with the specified tokenizer,
writes a binary file of uint16 token IDs.

Usage:
  # Download corpus (once):
  python prep_data.py --download --output-dir /root/corpus/

  # Pretokenize (run two in parallel for both tokenizers):
  python prep_data.py --tokenize \
    --tokenizer /root/tokenizers/standard-64k.json \
    --corpus-dir /root/corpus/ \
    --output /root/data/atlas-standard-64k.bin

  python prep_data.py --tokenize \
    --tokenizer /root/tokenizers/structok-64k.json \
    --corpus-dir /root/corpus/ \
    --output /root/data/atlas-structok-64k.bin

  # All-in-one (download + tokenize both):
  python prep_data.py --all \
    --tokenizer-a /root/tokenizers/standard-64k.json \
    --tokenizer-b /root/tokenizers/structok-64k.json \
    --output-dir /root/data/
"""

import argparse
import os
import struct
import time
from pathlib import Path


def download_corpus(output_dir, target_gb=5):
    """Download SlimPajama sample from HuggingFace."""
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading SlimPajama (~%dGB sample)..." % target_gb)
    t0 = time.time()

    # Stream SlimPajama and write chunks
    ds = load_dataset(
        "cerebras/SlimPajama-627B",
        split="train",
        streaming=True,
    )

    chunk_idx = 0
    total_bytes = 0
    target_bytes = target_gb * 1024 * 1024 * 1024
    chunk_texts = []
    chunk_size = 0
    chunk_limit = 100 * 1024 * 1024  # 100MB per chunk file

    for example in ds:
        text = example["text"]
        if not text or len(text) < 100:
            continue

        chunk_texts.append(text)
        chunk_size += len(text.encode("utf-8"))

        if chunk_size >= chunk_limit:
            chunk_path = output_dir / ("chunk-%04d.txt" % chunk_idx)
            with open(chunk_path, "w") as f:
                f.write("\n\n".join(chunk_texts))
            total_bytes += chunk_size
            chunk_idx += 1
            chunk_texts = []
            chunk_size = 0

            elapsed = time.time() - t0
            print("  chunk %d | %.1f GB | %.0f MB/s" % (
                chunk_idx, total_bytes / 1e9, total_bytes / 1e6 / elapsed))

            if total_bytes >= target_bytes:
                break

    # Write remaining
    if chunk_texts:
        chunk_path = output_dir / ("chunk-%04d.txt" % chunk_idx)
        with open(chunk_path, "w") as f:
            f.write("\n\n".join(chunk_texts))
        total_bytes += chunk_size
        chunk_idx += 1

    elapsed = time.time() - t0
    print("Downloaded %.1f GB in %d chunks (%.0fs)" % (
        total_bytes / 1e9, chunk_idx, elapsed))
    return chunk_idx


def tokenize_corpus(tokenizer_path, corpus_dir, output_path):
    """Tokenize all chunk files and write binary."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    corpus_dir = Path(corpus_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = sorted(corpus_dir.glob("chunk-*.txt"))
    print("Tokenizing %d chunks with %s..." % (len(chunks), Path(tokenizer_path).name))
    t0 = time.time()

    total_tokens = 0
    with open(output_path, "wb") as out:
        for i, chunk_path in enumerate(chunks):
            text = chunk_path.read_text()
            encoded = tok.encode(text)
            token_ids = encoded.ids

            for tid in token_ids:
                if tid > 65535:
                    tid = 0  # clamp to uint16
                out.write(struct.pack("<H", tid))

            total_tokens += len(token_ids)

            if (i + 1) % 5 == 0:
                elapsed = time.time() - t0
                print("  chunk %d/%d | %d tokens | %.0f tok/s" % (
                    i + 1, len(chunks), total_tokens, total_tokens / elapsed))

    elapsed = time.time() - t0
    size_gb = os.path.getsize(output_path) / 1e9
    print("Done: %d tokens, %.1f GB, %.0fs" % (total_tokens, size_gb, elapsed))
    print("Saved to %s" % output_path)


def main():
    parser = argparse.ArgumentParser(description="Atlas data preparation")
    parser.add_argument("--download", action="store_true", help="Download SlimPajama")
    parser.add_argument("--tokenize", action="store_true", help="Tokenize corpus")
    parser.add_argument("--all", action="store_true", help="Download + tokenize both")
    parser.add_argument("--output-dir", default="/root/data/")
    parser.add_argument("--corpus-dir", default="/root/corpus/")
    parser.add_argument("--tokenizer", help="Tokenizer JSON (for --tokenize)")
    parser.add_argument("--tokenizer-a", help="First tokenizer (for --all)")
    parser.add_argument("--tokenizer-b", help="Second tokenizer (for --all)")
    parser.add_argument("--output", help="Output bin path (for --tokenize)")
    parser.add_argument("--target-gb", type=int, default=5)
    args = parser.parse_args()

    if args.download:
        download_corpus(args.corpus_dir, args.target_gb)

    elif args.tokenize:
        if not args.tokenizer or not args.output:
            parser.error("--tokenize requires --tokenizer and --output")
        tokenize_corpus(args.tokenizer, args.corpus_dir, args.output)

    elif args.all:
        import subprocess

        if not args.tokenizer_a or not args.tokenizer_b:
            parser.error("--all requires --tokenizer-a and --tokenizer-b")

        # Step 1: Download
        download_corpus(args.corpus_dir, args.target_gb)

        # Step 2: Tokenize both in parallel
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name_a = Path(args.tokenizer_a).stem
        name_b = Path(args.tokenizer_b).stem

        print("\nTokenizing both in parallel...")
        proc_a = subprocess.Popen([
            "python3", __file__, "--tokenize",
            "--tokenizer", args.tokenizer_a,
            "--corpus-dir", args.corpus_dir,
            "--output", str(output_dir / ("atlas-%s.bin" % name_a)),
        ])
        proc_b = subprocess.Popen([
            "python3", __file__, "--tokenize",
            "--tokenizer", args.tokenizer_b,
            "--corpus-dir", args.corpus_dir,
            "--output", str(output_dir / ("atlas-%s.bin" % name_b)),
        ])

        proc_a.wait()
        proc_b.wait()

        if proc_a.returncode == 0 and proc_b.returncode == 0:
            print("\nBoth tokenizations complete.")
        else:
            print("\nERROR: tokenization failed (a=%d, b=%d)" % (
                proc_a.returncode, proc_b.returncode))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
