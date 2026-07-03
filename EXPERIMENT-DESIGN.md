# Attention Head Atlas: Experiment Design

## Hypothesis

Attention heads develop specialized behaviors in a predictable developmental sequence during transformer training. The tokenizer determines not just which specializations emerge, but when and in what order they appear.

## Experimental Design

Two training runs that differ ONLY in the tokenizer:

| | Baseline (standard BPE) | Comparison (merge barriers) |
|---|---|---|
| Architecture | GPT-NeoX 410M (24 layers, 16 heads, 384 total) | Same |
| Tokenizer | standard-64k (no barriers) | structok-64k (16 merge barriers) |
| Vocab size | ~65,536 | 65,539 |
| Training data | SlimPajama 5GB sample | Same source corpus |
| tokens.bin | Different (different tokenizer) | Different (different tokenizer) |
| Batch size | 8 | 8 |
| Learning rate | 3e-4 flat | 3e-4 flat |
| Steps | 20,000 | 20,000 |
| Context length | 2,048 | 2,048 |
| Precision | bf16 | bf16 |

## Corpus

FineWeb (HuggingFaceFW/fineweb, sample-10BT split), ~5GB sample. High-quality web corpus used by modern production models. NOT the structured-data-heavy corpus from merge-barriers experiments. This ensures findings generalize to how production models develop.

## Checkpoint Schedule

- Every 50 steps for steps 0-2,000 (40 checkpoints, captures emergence)
- Every 200 steps for steps 2,000-20,000 (90 checkpoints, captures stabilization)
- Total: 130 checkpoints per run

## Probing

At each checkpoint, every head (384) is probed across 8 behavior types on 6 fixed probe texts:

| Behavior | Metric | Probe text |
|----------|--------|-----------|
| Positional (previous token) | Attention mass on position n-1 | All probes |
| Positional (P0 sink) | Attention mass on position 0 | All probes |
| Induction | Copy score: attention to token after previous occurrence | induction.txt |
| Delimiter/structural | Attention mass on delimiter token positions | structured.txt |
| Bracket matching | Attention from close-bracket to matching open-bracket | brackets.txt |
| Content (semantic) | Correlation between attention and embedding similarity | prose.txt |
| Duplicate token | Attention to previous occurrences of same token | duplicates.txt |
| Dormant | Max attention concentration (HONOR approximation) | All probes |

Each head receives a continuous score vector (not a label), plus:
- Specialization index: max(scores) / sum(scores)
- Top-2 behaviors with confidence
- Context-conditional scores (per probe text)

## R2 Storage Schema

```
atlas/
  tokens/
    standard-64k.json                    # tokenizer definition
    structok-64k.json                    # tokenizer definition
    atlas-standard-64k.bin               # pretokenized corpus (standard)
    atlas-structok-64k.bin               # pretokenized corpus (barriers)
  runs/
    baseline/
      checkpoints/
        step-00050.pt                    # 130 checkpoints
        step-00100.pt
        ...
        step-20000.pt
      training_log.json                  # step, loss, time for each checkpoint
    comparison/
      checkpoints/
        step-00050.pt
        ...
        step-20000.pt
      training_log.json
  results/
    baseline/
      step-00050.json                    # probe results per checkpoint
      step-00100.json
      ...
      step-20000.json
    comparison/
      step-00050.json
      ...
      step-20000.json
  probes/
    prose.txt                            # fixed probe texts (versioned)
    code.txt
    structured.txt
    induction.txt
    duplicates.txt
    brackets.txt
```

## Provenance

- Corpus: FineWeb (HuggingFace HuggingFaceFW/fineweb, sample-10BT, 5GB sample)
- Tokenizers: standard-64k.json from merge-barriers run-002, structok-64k.json from structok repo
- Training script: `eval/train_atlas.py`
- Probing script: `eval/probe_heads.py`
- All probe texts committed to repo and archived to R2

## Key Questions

1. What develops first? Positional heads before induction? Delimiter before content?
2. Is there a fixed developmental order or does it depend on the tokenizer?
3. Do heads transition between types during training?
4. When do heads "commit" to a specialization (irreversibly)?
5. Does the merge-barrier tokenizer change the developmental sequence, not just the outcome?

## Estimated Cost

- Training: ~2 hours per run on 4090 ($0.40/hr = ~$0.80)
- Probing: ~1 hour per run (~$0.40)
- Total for both runs: ~$2.50

## Relationship to Prior Work

This project extends the merge-barriers research (DOI: 10.5281/zenodo.20925910). During that work, we measured delimiter head emergence at step ~1,000 and gradient-attention coupling over 20K steps. The atlas asks: what ELSE is developing simultaneously, and does the tokenizer change the sequence?
