#!/usr/bin/env python3
"""
Train a BPE tokenizer with natural-language-optimized merge barriers.

Barrier characters: . ' ? ! - " ( ) ; :
These are the structural delimiters of natural language (sentence boundaries,
contractions, parentheticals, clause separators). Selected based on adversarial
surface analysis across 43 production tokenizers.

Usage:
  python train_nl_tokenizer.py \
    --corpus-dir /root/corpus/ \
    --output /root/tokenizers/nl-barrier-64k.json \
    --vocab-size 65536
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers


NL_BARRIERS = ['.', "'", '?', '!', '-', '"', '(', ')', ';', ':']


def main():
    parser = argparse.ArgumentParser(description="Train NL-barrier tokenizer")
    parser.add_argument("--corpus-dir", required=True, help="Directory of text files")
    parser.add_argument("--output", required=True, help="Output tokenizer JSON path")
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--max-files", type=int, default=20, help="Max corpus files to use")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    corpus_files = sorted(corpus_dir.glob("chunk-*.txt"))[:args.max_files]
    if not corpus_files:
        corpus_files = sorted(corpus_dir.glob("*.txt"))[:args.max_files]

    if not corpus_files:
        print("ERROR: No text files found in %s" % corpus_dir)
        return

    total_mb = sum(f.stat().st_size for f in corpus_files) / 1e6
    print("Training NL-barrier tokenizer")
    print("  Barriers: %s" % ' '.join(repr(c) for c in NL_BARRIERS))
    print("  Corpus: %d files (%.0f MB)" % (len(corpus_files), total_mb))
    print("  Vocab size: %d" % args.vocab_size)

    # Build pre-tokenizer: isolate each NL barrier character
    splits = [pre_tokenizers.Split(c, behavior="isolated") for c in NL_BARRIERS]
    splits.append(pre_tokenizers.ByteLevel(add_prefix_space=False))

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(splits)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<eos>"],
        show_progress=True,
    )

    print("Training...")
    tokenizer.train(files=[str(f) for f in corpus_files], trainer=trainer)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output)
    print("Saved to %s (%d vocab)" % (args.output, tokenizer.get_vocab_size()))

    # Verify barriers
    test = '"Hello," she said. "It\'s a self-contained (parenthetical) test!'
    encoded = tokenizer.encode(test)
    tokens = [tokenizer.decode([t]) for t in encoded.ids]
    print("\nVerification: %s" % test)
    print("Tokens: %s" % tokens)

    for c in NL_BARRIERS:
        if c in test:
            found = any(t.strip() == c for t in tokens)
            print("  %s isolated: %s" % (repr(c), found))


if __name__ == "__main__":
    main()
