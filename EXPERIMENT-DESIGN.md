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
| Training data | FineWeb 5GB sample | Same source corpus |
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
- Step 0 (random init, before any training)
- Total: 131 checkpoints per run

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
    baseline/checkpoints/step-00000.pt through step-20000.pt   # 131 checkpoints COMPLETE
    comparison/checkpoints/step-00000.pt through step-20000.pt # 131 checkpoints COMPLETE
    seed2/checkpoints/                                          # 131 checkpoints PLANNED
  results/
    baseline/step-00000.json through step-20000.json           # 131 probe results COMPLETE
    comparison/step-00000.json through step-20000.json         # 131 probe results COMPLETE
    seed2/                                                      # PLANNED
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
6. Is the developmental order deterministic or stochastic across random seeds?

## Run 3: Seed Variation

Tests whether the developmental sequence observed in baseline is deterministic (same order every time) or stochastic (seed-dependent, per Baherwani et al. 2026).

| | Seed variation (seed2) |
|---|---|
| Architecture | GPT-NeoX 410M (same as baseline) |
| Tokenizer | standard-64k (same as baseline) |
| Training data | FineWeb 5GB sample (same pretokenized bin as baseline) |
| Random init | Different (new instance, new PyTorch default init) |
| Steps | 20,000 |
| R2 prefix | `atlas/runs/seed2` |
| Results prefix | `atlas/results/seed2` |

Everything identical to baseline except the random number generator seed. If the developmental timeline matches baseline, the sequence is architecture-determined. If it differs, emergence is stochastic and single-run observations cannot be generalized.

## Estimated Cost

- Baseline + Comparison: ~$5 (completed)
- Seed variation: ~$2.50
- Total: ~$7.50

## Status (2026-07-04)

| Run | Training | Probing | Analysis |
|-----|----------|---------|----------|
| Baseline (standard BPE) | COMPLETE (131 checkpoints on R2) | COMPLETE (131 results on R2) | 8 findings documented |
| Comparison (merge barriers) | COMPLETE (131 checkpoints on R2) | COMPLETE (131 results on R2) | 8 findings documented |
| Seed variation (seed2) | IN PROGRESS | PLANNED (local, after training) | - |

All probe results also committed to `results/` in this repo. 6 visualization charts in `charts/`. Analysis scripts run locally (no GPU needed).

## Roadmap

### Next: Structok Corpus Atlas (Run 4)

Run the atlas on the structok corpus (14% JSON, 8% GCF, 13% code) instead of FineWeb. Same methodology, same checkpoint schedule, different corpus. The frustration gap should appear on this corpus, and we'd see exactly WHEN it emerges at 50-step granularity. This directly extends the stranded paper from 8 data points to 130.

The FineWeb atlas and the structok-corpus atlas together tell the full story: "the developmental sequence is similar but the frustration gap only appears when structured data is present in training."

Estimated cost: ~$2.50. Requires the structok corpus pretokenized bins (already on R2 from merge-barriers experiments).

### Fix: Excess Score Correction (post-hoc, free)

The current classification uses raw scores, not excess scores. The merge-barriers paper solved this exact problem with the excess score methodology: subtract the base rate of delimiter positions from the raw delimiter attention. A head that scores 0.30 on delimiter when 0.30 of positions are delimiters has excess 0.00 (no specialization). A head that scores 0.30 when 0.10 of positions are delimiters has excess 0.20 (genuine specialist).

Apply the same correction to all behavior types. For each probe text, compute the base rate for each behavior (what fraction of positions are delimiter positions, what fraction are duplicate tokens, etc.) and subtract.

This is a post-hoc recomputation on the existing data. No new experiments needed. The raw scores in the JSONs contain everything required. This fix improves both the head type classification accuracy and sharpens the specialization index (removes base-rate inflation that makes everything look moderate).

### Fix: Extreme Probe Texts (re-probe step-20000 only)

Add more probe texts with extreme characteristics to separate specialists from generalists:
- A probe with zero delimiters (pure prose, no punctuation)
- A probe with very high delimiter density (dense JSON or GCF)
- A probe designed to maximally trigger induction (long repeated sequences)

The current 6 probes all contain moderate amounts of everything. Extreme probes would produce clearer specialization signals. Only requires re-probing the step-20000 checkpoint to validate the methodology, not all 131.

## Relationship to Prior Work

This project extends the merge-barriers research (DOI: 10.5281/zenodo.20925910). During that work, we measured delimiter head emergence at step ~1,000 and gradient-attention coupling over 20K steps. The atlas asks: what ELSE is developing simultaneously, and does the tokenizer change the sequence?
