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
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
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


def _tokenize_chunk(args):
    """Tokenize a single chunk file. Worker function for multiprocessing."""
    chunk_path, tokenizer_path, out_path = args
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    text = Path(chunk_path).read_text()

    # Split into ~1 MB segments to avoid tokenizer OOM on large inputs
    segment_size = 1_000_000
    all_ids = []
    for i in range(0, len(text), segment_size):
        segment = text[i:i + segment_size]
        encoded = tok.encode(segment)
        all_ids.extend(encoded.ids)

    data = struct.pack("<%dH" % len(all_ids),
                       *[min(tid, 65535) for tid in all_ids])
    with open(out_path, "wb") as f:
        f.write(data)
    return len(all_ids)


def tokenize_corpus(tokenizer_path, corpus_dir, output_path):
    """Tokenize all chunk files in parallel and concatenate."""
    from multiprocessing import Pool, cpu_count

    corpus_dir = Path(corpus_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = sorted(corpus_dir.glob("chunk-*.txt"))
    workers = min(12, cpu_count(), len(chunks))
    print("Tokenizing %d chunks with %s using %d workers..." % (
        len(chunks), Path(tokenizer_path).name, workers))
    t0 = time.time()

    # Tokenize each chunk to a temp file in parallel
    tmp_dir = Path("/tmp/atlas-tok-tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    work = [(str(c), tokenizer_path, str(tmp_dir / ("%s.bin" % c.stem)))
            for c in chunks]

    with Pool(workers) as pool:
        token_counts = pool.map(_tokenize_chunk, work)

    total_tokens = sum(token_counts)
    elapsed = time.time() - t0
    print("  Tokenized %d tokens in %.0fs (%.0f tok/s)" % (
        total_tokens, elapsed, total_tokens / elapsed))

    # Concatenate in order
    print("  Concatenating %d chunk bins..." % len(chunks))
    with open(output_path, "wb") as out:
        for chunk_path in chunks:
            tmp_path = tmp_dir / ("%s.bin" % chunk_path.stem)
            out.write(tmp_path.read_bytes())
            tmp_path.unlink()

    tmp_dir.rmdir()
    size_gb = os.path.getsize(output_path) / 1e9
    print("Done: %d tokens, %.1f GB, %.0fs" % (total_tokens, size_gb, time.time() - t0))
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
