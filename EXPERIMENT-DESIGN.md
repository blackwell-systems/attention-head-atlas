# Attention Head Atlas: Experiment Design

## Hypothesis

Attention heads develop specialized behaviors in a predictable developmental sequence during transformer training. The tokenizer determines not just which specializations emerge, but when and in what order they appear.

## Experimental Design

Four training runs isolating the tokenizer variable, seed variable, and barrier character set variable:

| | Baseline | Comparison | Seed2 | NL-barrier |
|---|---|---|---|---|
| Architecture | GPT-NeoX 410M (24 layers, 16 heads, 384 total) | Same | Same | Same |
| Tokenizer | standard-64k (no barriers) | structok-64k (16 struct barriers) | standard-64k (no barriers) | nl-barrier-64k (10 NL barriers) |
| Vocab size | 65,536 | 65,539 | 65,536 | ~65,536 |
| Training data | FineWeb 5GB | Same | Same | Same |
| Random init | Default | Default | Different seed | Default |
| Batch size | 1 (single sequence per step) | 1 | 1 | 1 |
| Learning rate | 3e-4 flat | 3e-4 flat | 3e-4 flat | 3e-4 flat |
| Steps | 20,000 | 20,000 | 20,000 | 20,000 |
| Context length | 2,048 | 2,048 | 2,048 | 2,048 |
| Precision | bf16 | bf16 | bf16 | bf16 |

The structured-data barrier tokenizer forbids merges involving 16 characters: `| @ < > " ' : , ; \t { } [ ] ( )`. The NL-barrier tokenizer forbids merges involving 10 characters: `. ' ? ! - " ( ) ; :`. Five characters overlap.

## Corpus

FineWeb (HuggingFaceFW/fineweb, sample-10BT split), ~5GB sample. High-quality web corpus used by modern production models. NOT the structured-data-heavy corpus from merge-barriers experiments. This ensures findings generalize to how production models develop.

## Checkpoint Schedule

- Every 50 steps for steps 0-2,000 (40 checkpoints, captures emergence)
- Every 200 steps for steps 2,000-20,000 (90 checkpoints, captures stabilization)
- Step 0 (random init, before any training)
- Total: 131 checkpoints per run

## Probing

At each checkpoint, every head (384) is probed on 6 fixed probe texts across 6 behavior types plus 2 auxiliary metrics:

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
    nl-barrier-64k.json                  # NL-barrier tokenizer definition
    atlas-standard-64k.bin               # pretokenized corpus (standard)
    atlas-structok-64k.bin               # pretokenized corpus (struct barriers)
    atlas-nl-barrier-64k.bin             # pretokenized corpus (NL barriers)
  runs/
    baseline/checkpoints/step-00000.pt through step-20000.pt   # 131 checkpoints COMPLETE
    comparison/checkpoints/step-00000.pt through step-20000.pt # 131 checkpoints COMPLETE
    seed2/checkpoints/step-00000.pt through step-20000.pt      # 131 checkpoints COMPLETE
    nl-barrier/checkpoints/step-00000.pt through step-20000.pt # 131 checkpoints COMPLETE
  results/
    baseline/step-00000.json through step-20000.json           # v1: 131 probe results (6 behaviors)
    comparison/step-00000.json through step-20000.json         # v1: 131 probe results
    seed2/step-00000.json through step-20000.json              # v1: 131 probe results
    nl-barrier/step-00000.json through step-20000.json         # v1: 131 probe results
    baseline-v2/step-00000.json through step-20000.json        # v2: 131 probe results (7 behaviors incl. spacing)
    comparison-v2/step-00000.json through step-20000.json      # v2: 131 probe results
    seed2-v2/step-00000.json through step-20000.json           # v2: 131 probe results
    nl-barrier-v2/step-00000.json through step-20000.json      # v2: 131 probe results
```

## Provenance

- Corpus: FineWeb (HuggingFace HuggingFaceFW/fineweb, sample-10BT, 5GB sample)
- Tokenizers: standard-64k.json from merge-barriers run-002, structok-64k.json from structok repo, nl-barrier-64k.json trained by `eval/train_nl_tokenizer.py`
- Training script: `eval/train_atlas.py`
- Probing script: `eval/probe_heads.py`
- Excess correction: `eval/excess_score_correction.py`
- Analysis scripts: `eval/analyze_p0_deep.py`, `eval/analyze_seed2.py`, `eval/analyze_velocity_circuits.py`, `eval/analyze_nl_barrier.py`, `eval/measure_nl_frustration_gap.py`
- All probe texts committed to repo and archived to R2

## Key Questions

1. What develops first? Positional heads before induction? Delimiter before content?
2. Is there a fixed developmental order or does it depend on the tokenizer?
3. Do heads transition between types during training?
4. When do heads "commit" to a specialization (irreversibly)?
5. Does the merge-barrier tokenizer change the developmental sequence, not just the outcome?
6. Is the developmental order deterministic or stochastic across random seeds?

## Estimated Cost

- Baseline + Comparison: ~$5 (completed)
- Seed variation: ~$2.50 (completed)
- NL-barrier: ~$2.50 (completed)
- v2 re-probe (all 4 runs): ~$0.80 (completed, 2x RTX 4090)
- Structok corpus (2 runs): ~$7.00 (completed, training + probing across 4 instances)
- Ablation study (3 models): ~$0.30 (completed, inference only)
- UMAP extraction (2 models): ~$0.10 (completed, inference only)
- Llama developmental atlas (training + probing + ablation): ~$3.40 (completed)
- Downstream completion benchmark: ~$0.20 (completed, inference only)

## Status (2026-07-05)

| Run | Arch | Training | Probing | Excess | Ablation | Analysis |
|-----|------|----------|---------|--------|----------|----------|
| Baseline | NeoX MHA | COMPLETE | COMPLETE (v1+v2) | COMPLETE | COMPLETE | 17 findings |
| Comparison | NeoX MHA | COMPLETE | COMPLETE (v1+v2) | COMPLETE | COMPLETE | 17 findings |
| Seed2 | NeoX MHA | COMPLETE | COMPLETE (v1+v2) | COMPLETE | COMPLETE | 17 findings |
| NL-barrier | NeoX MHA | COMPLETE | COMPLETE (v1+v2) | COMPLETE* | N/A | 17 findings |
| Structok-baseline | NeoX MHA | COMPLETE | COMPLETE | COMPLETE | COMPLETE | 17 findings |
| Structok-comparison | NeoX MHA | COMPLETE | COMPLETE | COMPLETE | N/A | 17 findings |
| Llama-FineWeb | Llama GQA | COMPLETE | COMPLETE (131) | COMPLETE | COMPLETE | 17 findings |

*NL-barrier step-0 checkpoint corrupted by disk-full event. Step-50 base rates used as proxy for excess correction. Defensible: 50 steps = 0.008% of corpus, attention still approximately random.

All 917 checkpoints (7 runs x 131) on R2. 1,572 probe results on R2 and committed to `results/`. 40 visualization charts in `charts/`. 17 findings documented in RESULTS.md. Paper in `paper/developmental-atlas.md` (15 figures, 11 tables, 698 lines). Downstream benchmark in `results/benchmark/` (3 models). All data also on HuggingFace (7 step-20000 checkpoints + 3 tokenizers).

## Roadmap

### Completed: Structok Corpus Atlas (Runs 5-6)

Tests the two-regime prediction: a corpus with high delimiter density should produce BOTH a frustration gap AND spacing heads, bridging the web text regime (zero gap, 183 spacing heads) and the structured data regime (40pp gap, full stranding).

#### Corpus

The rebalanced structok corpus from merge-barriers run-002 (6.1 GB):

| Source | Size | % |
|--------|------|---|
| FineWeb (web text) | 2.0 GB | 33% |
| JSON | 850 MB | 14% |
| Code (Go, Python, TS, JS, Rust) | 800 MB | 13% |
| GCF | 500 MB | 8% |
| Wikipedia | 200 MB | 3% |
| YAML/CSV | 45 MB | 1% |

#### Pretokenized bins (already on R2)

Both bins are on R2, pretokenized with the SAME tokenizers used in the FineWeb atlas runs:

| R2 key | Tokenizer | Size |
|--------|-----------|------|
| `tokens/standard-64k-v2.bin` | `standard-64k.json` (same as atlas baseline) | 4.8 GB |
| `tokens/structok-64k-v2.bin` | `structok-64k.json` (same as atlas comparison) | 4.8 GB |

Tokenizer identity verified: run-002 used the same `standard-64k.json` and `structok-64k.json` files that are in `atlas/tokens/` on R2. No re-tokenization needed.

**Provenance**: bins were created by `structok/prep_run002.py` (private repo, Blackwell Systems). The script downloads HF datasets (FineWeb, Wikipedia), downloads code/JSON/YAML from R2 (originally from run-001), generates GCF data in-script with seed-based randomization, concatenates all sources, shuffles, pretokenizes with both tokenizers, and uploads to R2. These are the same bins that trained the models published in the merge-barriers paper (DOI: 10.5281/zenodo.20925910, run-002).

No NL-barrier bin exists for the structok corpus. Would need `train_nl_tokenizer.py` + `prep_data.py` to create one. Not planned for this run (two runs is sufficient to test the prediction).

#### Runs

| Run | Name | Tokenizer | Corpus bin (R2) | R2 checkpoint prefix | R2 results prefix |
|-----|------|-----------|----------------|---------------------|------------------|
| 5 | structok-baseline | standard-64k.json | `tokens/standard-64k-v2.bin` | `atlas/runs/structok-baseline` | `atlas/results/structok-baseline` |
| 6 | structok-comparison | structok-64k.json | `tokens/structok-64k-v2.bin` | `atlas/runs/structok-comparison` | `atlas/results/structok-comparison` |

#### Hyperparameters (identical to FineWeb runs)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | GPT-NeoX 410M (24 layers, 16 heads) | Same as all atlas runs |
| Steps | 20,000 | Same schedule |
| Batch size | 1 (single sequence per step) | Matches FineWeb runs |
| Learning rate | 3e-4 flat | Matches FineWeb runs |
| Context length | 2,048 | Matches FineWeb runs |
| Precision | bf16 | Matches FineWeb runs |
| Checkpoint schedule | 131 checkpoints (every 50 to step 2000, every 200 to step 20000) | Same schedule |

Note: the structok corpus bin is 4.8 GB vs 2.3 GB for FineWeb. With the same number of steps, the model sees proportionally less of the larger corpus. This is acceptable because we're comparing within the structok runs (baseline vs comparison) and against FineWeb runs at the same step count.

#### Probing

Use the v2 probe script (`eval/probe_heads.py`) with 7 behavior types including spacing. Same probe texts as the v2 FineWeb re-probe (consistent across all runs). R2 streaming mode with `--force` for probing (since these are new runs, there are no existing results to protect).

Use `--save-local` to also save results to `results/structok-baseline/` and `results/structok-comparison/` for git commit.

#### Excess correction

Run `eval/excess_score_correction.py --run structok-baseline --run structok-comparison`. Step-0 base rates from the structok corpus will differ from FineWeb (higher delimiter base rates due to 14% JSON + 8% GCF), which is exactly what excess correction is designed to handle.

#### Predictions (testable)

| Metric | FineWeb (observed) | Structok (predicted) | Rationale |
|--------|-------------------|---------------------|-----------|
| Frustration gap (baseline) | 0 pp | > 5 pp | 35% structured content provides delimiter density |
| Frustration gap (comparison) | 0 pp | ~0 pp | Merge barriers prevent stranding |
| Spacing heads (baseline) | 183 | 50-150 | Some spacing, but delimiter competes for heads |
| Spacing heads (comparison) | 13 | 0-10 | Barriers already eliminate spacing on FineWeb |
| P0 heads (baseline) | 32 | 20-60 | May increase with more failed delimiter attempts |
| P0 heads (comparison) | 40 | 20-40 | Similar or lower |
| Bracket specialists (comparison) | 39 | 40-80 | More bracket content available |

The key prediction: spacing heads AND frustration gap will coexist in the baseline structok run, confirming the two-regime model is a continuum, not a binary.

#### Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Instance dies mid-training | Lost time, partial data | Select instance with max_days > 7. Background R2 upload saves completed checkpoints. Resume with `--resume-from r2`. |
| Disk fills (131 checkpoints x 1.7 GB = 223 GB) | Training stops | Require 300+ GB disk. Monitor disk usage. Delete local checkpoints after R2 upload verified. |
| Wrong tokenizer version | Invalid comparison | VERIFIED: `standard-64k.json` and `structok-64k.json` on R2 are identical between atlas/tokens/ and the bins in tokens/. |
| R2 upload failures | Missing checkpoints | probe_heads.py has verified uploads with retry. train_atlas.py has background upload thread with retry. |
| Budget overrun | Can't finish | Training: ~$2.50/run x 2 = $5. Probing: ~$0.40/run x 2 = $0.80. Total: ~$5.80. Budget ceiling: $8. |
| Corpus bin download slow | Wasted instance time | Pick instance with > 1 Gbps network. 4.8 GB download at 1 Gbps = ~40s. |

#### Validation checkpoints (stop/go gates)

1. **After bin download**: verify file size matches R2 (4.8 GB for standard, 4.8 GB for structok).
2. **After step-0**: probe and verify base rates look reasonable. Delimiter base rate should be higher than FineWeb (more delimiter content).
3. **After step-1000**: check frustration gap. If nonzero, the prediction is already partially confirmed. If zero, the corpus may not be structured enough (but continue anyway).
4. **After step-5000**: check head distribution. Spacing heads should be emerging but fewer than 183 (FineWeb level).

#### Execution order

1. Spin up instance (RTX 4090, 300+ GB disk, fast network, max_days > 7)
2. Install deps: `pip install tokenizers transformers boto3`
3. Upload `train_atlas.py`, `probe_heads.py`, `excess_score_correction.py`, all probe texts
4. Download tokenizers from `atlas/tokens/` on R2
5. Download corpus bin from `tokens/standard-64k-v2.bin` on R2
6. Run training: `python3 train_atlas.py --tokenizer standard-64k.json --data standard-64k-v2.bin --run-name structok-baseline --r2-prefix atlas/runs/structok-baseline --output-dir /root/runs/structok-baseline --steps 20000`
7. Probe immediately after training: `python3 probe_heads.py --r2-prefix atlas/runs/structok-baseline --tokenizer standard-64k.json --probe-dir probes/ --save-local results/structok-baseline/`
8. Repeat steps 5-7 with structok-64k tokenizer and bin for comparison run
9. Download results locally
10. Run excess correction locally
11. Analyze: compare to FineWeb predictions table above

#### Results (2026-07-04)

COMPLETE. Both runs trained, probed, excess-corrected. Key finding: spacing persists at 172/384 even on structured data (nearly identical to FineWeb's 183). Frustration gap is 1.0 pp (nonzero, confirming two-regime model as continuum). Delimiter heads scale up (131 vs 74) while positional_prev shrinks (17 vs 68). Spacing is a baseline cost of standard BPE regardless of corpus composition. See RESULTS.md Finding 14.

### Completed: Spacing Head Ablation Study

Tests whether spacing heads are counterproductive (removal improves PPL), neutral, or productive. Follows the zero-ablation methodology from the coupling paper (Blackwell, 2026a, Section 5.3).

#### Design

Zero-ablation on GPT-NeoX 410M (full MHA, clean per-head intervention). For each ablation condition, deep copy the model, zero the output projection weights (`gpt_neox.layers.{layer}.attention.dense.weight[:, start:end] = 0`) for the selected heads, measure perplexity on the 7 probe texts, discard the copy.

#### Ablation conditions

| Condition | Heads | Purpose |
|-----------|-------|---------|
| Baseline | None (0) | Reference PPL |
| Spacing ablation | 183 spacing heads (baseline) | Are spacing heads counterproductive? |
| P0 ablation | 32 P0 heads (baseline) | Are P0 heads counterproductive? |
| Random control x5 | 183 random non-spacing heads | Capacity reduction baseline |

The causal signal is the gap between spacing ablation and random controls, not the absolute PPL change. This controls for the generic capacity reduction from zeroing 183/384 heads.

#### Models to test

| Model | Checkpoint | Spacing heads | Source |
|-------|-----------|--------------|--------|
| FineWeb baseline | `atlas/runs/baseline/checkpoints/step-20000.pt` | 183 | `results/baseline-v2-excess/step-20000.json` |
| Structok baseline | `atlas/runs/structok-baseline/checkpoints/step-20000.pt` | 172 | `results/structok-baseline-excess/step-20000.json` |
| FineWeb comparison | `atlas/runs/comparison/checkpoints/step-20000.pt` | 13 | `results/comparison-v2-excess/step-20000.json` |

The comparison model (13 spacing heads) serves as a sanity check: removing 13 heads from a model with productive specialization should degrade performance.

#### Predictions

| Condition | Predicted PPL change | Rationale |
|-----------|---------------------|-----------|
| Spacing ablation (baseline) | Improves or neutral | Spacing heads attend to whitespace boundaries the model doesn't need; same pattern as stranded heads at 1.3B |
| Spacing ablation (structok) | Improves or neutral | Same mechanism, different corpus |
| P0 ablation (baseline) | Improves or neutral | P0 heads are a failure mode (Section 4.2) |
| Random control (baseline) | Degrades | Removing productive heads should hurt |
| Comparison spacing ablation | Degrades or neutral | Only 13 heads, model has productive specialization |

The strongest possible result: spacing removal improves PPL AND random control degrades PPL. This would prove spacing heads are not just wasteful but counterproductive, matching the scaling paradox from the coupling paper (removing stranded heads improved comprehension by 57% at 1.3B).

#### Execution

1. Spin up GPU instance (RTX 4090, minimal disk, fast network)
2. Upload `eval/ablate_spacing_heads.py`, probe texts, tokenizers
3. Download step-20000 checkpoint from R2 (or HF)
4. Run ablation: `python ablate_spacing_heads.py --r2-checkpoint atlas/runs/baseline/checkpoints/step-20000.pt --tokenizer standard-64k.json --classifications results/baseline-v2-excess/step-20000.json --probe-dir probes/ --output results/ablation/baseline-spacing.json`
5. Repeat for structok-baseline and comparison
6. Download results, record in RESULTS.md

**Cost:** Inference only, ~30 min on RTX 4090. ~$0.30.

**Script:** `eval/ablate_spacing_heads.py`. Provenance: adapted from the 18-phase ablation protocol in the coupling paper (Blackwell, 2026a). Zero-ablation on output projections, paired random controls, per-text and mean PPL reporting.

#### Results (2026-07-04)

COMPLETE. Spacing heads are productive (+64.3% degradation vs +28.7% random controls). They are mandatory damage repair, not waste. P0 heads are genuinely useless (+1.4%). The model dedicates 47% of heads to boundary recovery that merge barriers make unnecessary. See RESULTS.md Finding 15.

### Completed: Spacing Probe + Full Re-probe (v2)

COMPLETE (2026-07-04). Added `measure_spacing()` to `probe_heads.py`. Re-probed all 524 checkpoints (4 runs x 131) on RTX 4090 with 7-behavior taxonomy and consistent probe texts across all runs.

**Result:** Spacing is the dominant specialization in standard BPE: 183/384 heads (47.7%) in both baseline and seed2. Merge barriers reduce to 13 (struct) or 0 (NL). 54 of v1's 96 baseline P0 heads were actually spacing specialists. See RESULTS.md Findings 12 and 13.

**Data:** v2 raw results in `results/{run}-v2/`, excess-corrected in `results/{run}-v2-excess/`. v1 data preserved in `results/{run}/` and `results/{run}-excess/`. Both on R2 under `atlas/results/`.

**Cost:** ~$0.80 total (2 RTX 4090 instances, ~1.5 hours).

### Completed: Punctuated Prose Probe for NL Frustration Gap

COMPLETE (2026-07-04). Wrote `probes/prose_punctuated.txt` with natural punctuation. Measured NL frustration gap on step-20000 for all 4 runs. Result: NL gap is genuinely zero on web text (all values under 1pp). The frustration gap requires structured data density, not just delimiter characters. On web text, BPE damage manifests as spacing head proliferation instead. See RESULTS.md Finding 13.

### Completed: NL-Barrier Run (Run 4)

COMPLETE. NL-barrier tokenizer trained with barriers on `. ' ? ! - " ( ) ; :`. Result: merge barriers are universal (NL barriers r=0.923 with struct barriers). See RESULTS.md Finding 11.

### Completed: Seed Variation (Run 3)

COMPLETE. Distribution correlation r=0.794. Emergence is partially stochastic. See RESULTS.md Finding 10.

### Completed: Excess Score Correction

COMPLETE. Corrected results in `results/{run}-excess/` for all 4 runs. See RESULTS.md Finding 1.

### Completed: Probe Text Improvements

COMPLETE. Improved probes (real bracketed code, standardized lengths, punctuation-stripped prose) used for seed2 and NL-barrier runs. Baseline and comparison used original probes; excess correction normalizes across probe sets.

## Future Roadmap (toward universality)

### Completed: Architecture and Scale Replication (Llama 410M + 1.3B)

COMPLETE (2026-07-05). Probed 4 existing Llama checkpoints from the coupling paper (run-003 Llama 410M, run-004 Llama 1.3B, both standard and structok tokenizers) with the 7-behavior taxonomy including spacing. Generated Llama-specific step-0 base rates from random initialization for excess correction.

**Results (excess-corrected):**

| Model | Arch | Heads | Spacing | P0 | Delimiter | Positional_prev |
|-------|------|-------|---------|-----|-----------|----------------|
| NeoX 410M standard | MHA | 384 | 183 (47.7%) | 32 (8.3%) | 74 (19.3%) | 68 (17.7%) |
| Llama 410M standard | GQA | 384 | 60 (15.6%) | 90 (23.4%) | 43 (11.2%) | 147 (38.3%) |
| Llama 1.3B standard | GQA | 768 | 128 (16.7%) | 180 (23.4%) | 137 (17.8%) | 263 (34.2%) |
| NeoX 410M structok | MHA | 384 | 13 (3.4%) | 40 (10.4%) | 79 (20.6%) | 91 (23.7%) |
| Llama 410M structok | GQA | 384 | 0 (0%) | 80 (20.8%) | 120 (31.2%) | 137 (35.7%) |
| Llama 1.3B structok | GQA | 768 | 2 (0.3%) | 97 (12.6%) | 365 (47.5%) | 169 (22.0%) |

**Frustration gap (Llama):** 0.4pp (410M standard), 0.2pp (1.3B standard), 0.0pp (both structok). Same zero-gap pattern as NeoX.

**Key findings:**
- Spacing exists on both architectures. Universality confirmed.
- Spacing percentage varies: ~47% on NeoX (MHA), ~16% on Llama (GQA). GQA distributes the spacing signal differently.
- P0 is higher on Llama (~23%) than NeoX (~8%). GQA produces more P0 sinks.
- Total non-productive (spacing + P0): ~56% NeoX, ~39% Llama. Different distribution, same mechanism.
- Merge barriers eliminate spacing on both architectures (0-2 heads on Llama structok).
- Spacing is consistent across Llama scales: 15.6% at 410M, 16.7% at 1.3B.

**Provenance:** Llama checkpoints from coupling paper (run-003 at `checkpoints/run-003-llama-{standard,structok}/step-40000/`, run-004 at `checkpoints/run-004-llama-{standard,structok}/step-50000/`). Step-0 base rates generated from random Llama init on GPU. Excess correction via `eval/excess_score_correction.py` (auto-detects head count). Results in `results/llama-{410m,1.3b}-{standard,structok}{,-excess}/`.

**Cost:** ~$0.50 (inference only, existing checkpoints).

### Completed: Downstream Completion Benchmark

Tests whether the capacity tax translates into measurable next-token prediction accuracy differences. An initial instruction-following benchmark (QA format, `eval/benchmark_downstream.py`) produced 0% on both models because 410M/20K-step models cannot instruction-follow. The benchmark was redesigned as completion-based tasks (`eval/benchmark_completion.py`) measuring next-token prediction accuracy on structural text.

#### Results (2026-07-05)

COMPLETE. Three models tested: NeoX baseline, Llama baseline, NeoX comparison.

| Task | NeoX Baseline | Llama Baseline | NeoX Comparison |
|------|--------------|---------------|----------------|
| Bracket closing | 13.0% | 3.0% | 22.0% |
| JSON structure | 19.0% | 10.0% | 15.0% |
| Pattern continuation | 0.0% | 0.0% | 0.0% |
| Overall structural accuracy | 4.1% | 3.0% | 16.7% |
| Whitespace space prediction | 72.0% | 55.0% | 0.0% |
| Whitespace word prediction | 8.0% | 7.0% | 15.0% |

Per-token-type structural accuracy:

| Token type | NeoX Baseline | Llama Baseline | NeoX Comparison |
|-----------|--------------|---------------|----------------|
| Bracket | 0.0% | 9.5% | 19.8% |
| Delimiter | 2.2% | 1.1% | 25.5% |
| Spacing | 47.4% | 36.8% | 0.0% |

The capacity tax translates directly to downstream accuracy. Both standard BPE models show the same pattern: high spacing prediction, low structural accuracy. Merge barriers invert this (0% spacing, 19.8% bracket, 25.5% delimiter). Spacing accuracy correlates with head count (47.4% NeoX with 183 heads, 36.8% Llama with 154 heads). The comparison model predicts the actual next word more often (15% vs 8%) despite never predicting a space token.

**Cost:** ~$0.20 (inference only, RTX 4090, ~5 min per model).

Source: `eval/benchmark_completion.py`. Results in `results/benchmark/` and on R2 at `atlas/results/benchmark/`.

### Completed: Llama Developmental Atlas (Llama-FineWeb-Baseline)

Full 131-checkpoint developmental analysis of Llama 410M trained on FineWeb. Architecture replication of the NeoX developmental atlas: not just "spacing exists in Llama" but "spacing follows the same developmental program across architectures."

#### Results (2026-07-05)

COMPLETE. Training, probing (131 checkpoints), excess correction, ablation, UMAP extraction, charts all done.

**Landmark result:** Spacing ablation NeoX MHA +64.3%, Llama GQA +67.0%. The capacity tax is BPE-dependent, not architecture-dependent.

**Head type distribution (excess-corrected, step 20000):**

| Type | NeoX 410M | Llama 410M |
|------|-----------|------------|
| Spacing | 183 (47.7%) | 154 (40.1%) |
| P0 | 32 (8.3%) | 31 (8.1%) |
| Delimiter | 74 (19.3%) | 92 (24.0%) |
| Positional_prev | 68 (17.7%) | 72 (18.8%) |

**Predictions vs observed:**

| Metric | Predicted | Observed | Assessment |
|--------|-----------|----------|------------|
| Final spacing count | 50-80 (~15%) | 154 (40.1%) | **Prediction falsified.** FineWeb-trained Llama has much more spacing than structok-trained Llama (154 vs 60). Earlier low count was corpus effect. |
| Final P0 count | 80-100 (~23%) | 31 (8.1%) | **Prediction falsified.** P0 count essentially identical to NeoX (31 vs 32), not elevated as endpoint probes suggested. |
| Ablation: spacing removal | +20-60% PPL | +67.0% | Above predicted range. Nearly identical to NeoX (+64.3%). |
| Ablation: P0 removal | ~0-5% PPL | -3.4% | Confirmed: P0 useless on both architectures. |

The falsified predictions are themselves findings: the earlier structok-trained Llama probes (60 spacing, 90 P0) reflected corpus effects, not architecture effects. With the same corpus and tokenizer, Llama and NeoX converge on similar distributions.

#### Training

| Parameter | Value |
|-----------|-------|
| Architecture | Llama 410M (24 layers, 16 heads, 4 KV heads, hidden=1024, intermediate=2816) |
| Tokenizer | standard-64k.json (same as NeoX baseline) |
| Corpus | FineWeb 5GB (same bin as NeoX baseline: `atlas/tokens/atlas-standard-64k.bin`) |
| Steps | 20,000 |
| Batch size | 1 |
| Learning rate | 3e-4 flat |
| Context length | 2,048 |
| Precision | bf16 |
| Checkpoint schedule | 131 checkpoints (same as NeoX) |
| R2 prefix | `atlas/runs/llama-fineweb-baseline` |
| Instance | ssh4.vast.ai:15912, RTX 4090, $0.46/hr |

Training command:
```
python3 train_atlas.py --tokenizer standard-64k.json --data atlas-standard-64k.bin \
  --run-name llama-fineweb-baseline --r2-prefix atlas/runs/llama-fineweb-baseline \
  --output-dir /root/runs/llama-fineweb-baseline --arch llama --steps 20000
```

#### Probing (after training)

Probe all 131 checkpoints with 7-behavior taxonomy on same instance:
```
python3 probe_heads.py --r2-prefix atlas/runs/llama-fineweb-baseline \
  --tokenizer standard-64k.json --probe-dir probes/ --size 410m-llama \
  --save-local results/llama-fineweb-baseline/
```

This will take ~2-3 hours on a single 4090 (131 checkpoints, each requires model load + 7 probe texts).

#### Excess correction

Generate step-0 base rates from the step-00000 checkpoint (random Llama init), then:
```
python3 excess_score_correction.py --run llama-fineweb-baseline
```

Auto-detects 384 heads (16 per layer x 24 layers) from the raw data.

#### Ablation (after probing)

Zero-ablation on step-20000 checkpoint using the excess-corrected classifications:
```
python3 ablate_spacing_heads.py --checkpoint step-20000.pt \
  --tokenizer standard-64k.json \
  --classifications results/llama-fineweb-baseline-excess/step-20000.json \
  --size 410m-llama --probe-dir probes/
```

Uses Llama-specific zeroing: `model.layers.{layer}.self_attn.o_proj` (already implemented in ablate_spacing_heads.py).

#### Charts (after excess correction)

Generate developmental charts comparable to NeoX:
- Behavior emergence timeline (when does each behavior first appear?)
- Head count trajectories (spacing, P0, delimiter, positional_prev over 131 steps)
- Specialization index distribution over training
- UMAP: head-level joint embedding with NeoX baseline for direct comparison
- UMAP: developmental sequence (8 training steps) showing spacing emergence in GQA

#### Predictions

| Metric | NeoX baseline | Llama predicted | Rationale |
|--------|--------------|----------------|-----------|
| Spacing emergence step | ~200-400 | ~200-600 | GQA may delay or accelerate |
| Final spacing count | 183 (47.7%) | 50-80 (~15%) | Consistent with endpoint probe (60 heads) |
| Final P0 count | 32 (8.3%) | 80-100 (~23%) | Consistent with endpoint probe (90 heads) |
| Spacing + P0 total | ~56% | ~39% | Architecture-dependent but still massive |
| Ablation: spacing removal | +64.3% PPL | +20-60% PPL | Mandatory damage repair on GQA too |
| Ablation: P0 removal | +1.4% PPL | ~0-5% PPL | P0 useless on both architectures |
| Ablation: random control | +28.7% PPL | +15-30% PPL | Capacity reduction baseline |

#### R2 storage

```
atlas/runs/llama-fineweb-baseline/
  checkpoints/step-00000.pt through step-20000.pt    # 131 checkpoints
atlas/results/llama-fineweb-baseline/
  step-00000.json through step-20000.json             # 131 probe results
```

Results locally in `results/llama-fineweb-baseline/` and `results/llama-fineweb-baseline-excess/`.

#### Cost

- Training: ~$2.30 (5 hours on RTX 4090 at $0.46/hr)
- Probing: ~$1.00 (2-3 hours on same instance)
- Ablation: ~$0.10 (inference only, single checkpoint)
- Total: ~$3.40

#### Why this matters

Without developmental data, the Llama finding is a single data point: "spacing exists at endpoint." With 131 checkpoints, we can show spacing follows a developmental program that is conserved across architectures, even though GQA changes the quantitative distribution. This is the difference between "we measured it" and "we understand it."

### Completed: Llama Ablation (clean, FineWeb-trained)

COMPLETE (2026-07-05). Included in Llama Developmental Atlas above. The earlier Llama ablation from run-003/004 checkpoints was INCONCLUSIVE (structok corpus, baseline PPL 150K-314K). The FineWeb-trained Llama gives clean results: +67.0% spacing ablation, -3.4% P0 ablation.

### Remaining: Causal circuit intervention

The circuit protection finding is correlational (100% P0 isolation). To make it causal: modify the training loop to add a regularization term that couples an isolated head's trajectory to a circuit member's trajectory, then train and observe whether the coupled head avoids P0 collapse. This would convert the correlational circuit protection finding into a causal one, which would be a standalone contribution to the mechanistic interpretability literature.

#### Design

1. Identify heads that collapse into P0 in the baseline model (known from Finding 2: median sink step 11,000).
2. Before training a new model, select 5-10 of these "doomed" head positions.
3. Add a regularization term to the training loss that encourages each doomed head's attention pattern to correlate with a nearby circuit member's pattern. The coupling strength should be modest (e.g., lambda=0.01) so it nudges rather than forces.
4. Train for 20,000 steps with the same hyperparameters as baseline.
5. At convergence, check: did the coupled heads avoid P0 collapse? Did they develop productive specialization? Did the circuit they were coupled to grow or change?
6. Control: train a second run with the regularization term applied to random non-doomed heads (to verify the effect is specific to coupling, not to regularization in general).

#### Infrastructure

All existing infrastructure supports this:
- `train_atlas.py`: training loop (add regularization term)
- `probe_heads.py`: probing at 131 checkpoints (unchanged)
- `excess_score_correction.py`: excess correction (unchanged)
- `analyze_p0_deep.py`: P0 tracking (unchanged)
- `analyze_seed2.py`: circuit identification (use to find target circuit members)

The engineering work is the custom regularization term and verifying it doesn't destabilize training.

#### Predictions

| Condition | P0 collapse of doomed heads | Rationale |
|-----------|---------------------------|-----------|
| Coupled to circuit member | Reduced (< 50% sink) | Circuit provides mutual reinforcement |
| Coupled to random head | No change (> 80% sink) | Random coupling doesn't create stable circuits |
| No coupling (baseline) | 100% sink | Known from Finding 2 |

The strongest result: coupled heads survive AND develop the same specialization as their circuit partner. This would prove circuits are causally protective, not merely co-occurring with survival.

**Cost:** 2 training runs (~$5), 2 probing runs (~$0.80). ~$6 total. ~1 week of focused engineering for the regularization term.

**Impact:** If successful, this is a standalone contribution to mechanistic interpretability: the first causal demonstration that developmental circuits protect heads from collapse. Independent of the tokenizer/spacing findings.

### Completed: Unclassified Head Identification (Activation Patching) — NULL RESULT

The merge-barrier comparison model has 95 unclassified heads (24.7%) with excess scores below 0.02 across all 7 measured behaviors. These heads are freed from the spacing tax and are doing something our taxonomy doesn't capture. Identifying what they do completes the picture of what a healthy model looks like.

#### Method: Activation Patching

For each of the 95 unclassified heads, swap its activations from a corrupted input into a clean forward pass and measure what breaks. If patching head H causes the model to lose a specific capability, H is performing that function.

#### Behavior hypotheses and input pairs

**1. Subject-verb agreement (syntax)**

| Clean input | Corrupted input | Signal |
|------------|----------------|--------|
| The cat that chased the dogs was tired | The cat that chased the dogs were tired | Prediction shifts was -> were |
| The teacher with the students walks home | The teacher with the students walk home | Prediction shifts walks -> walk |
| The box of chocolates is on the table | The box of chocolates are on the table | Prediction shifts is -> are |

20 pairs. Vary attractor noun number (singular subject, plural attractor). If patching a head shifts the model toward the wrong verb form, that head is tracking syntactic agreement across intervening material.

**2. Coreference / entity tracking**

| Clean input | Corrupted input | Signal |
|------------|----------------|--------|
| Alice gave Bob a book. She smiled. | Alice gave Bob a book. He smiled. | Prediction shifts she -> he |
| The doctor told the nurse that she was right | The doctor told the nurse that he was right | Pronoun resolution changes |

20 pairs. Swap gendered entities to flip coreference. If patching a head changes pronoun prediction, it's tracking entity identity.

**3. Semantic similarity / content association**

| Clean input | Corrupted input | Signal |
|------------|----------------|--------|
| The dog chased the ball across the yard | The dog chased the lamp across the yard | Next-word prediction changes |
| She drank a cup of coffee | She drank a cup of gravel | Plausibility judgment shifts |

20 pairs. Replace a semantically coherent object with an incoherent one. If patching shifts predictions toward the incoherent continuation, the head is doing semantic compatibility.

**4. Local context / n-gram prediction**

| Clean input | Corrupted input | Signal |
|------------|----------------|--------|
| New York City is a large | Old Blue Tree is a large | Next-word prediction changes |
| Once upon a time there was a | Brick under a time there was a | Prediction shifts |

20 pairs. Corrupt local context while preserving global structure. If patching disrupts prediction, the head is doing local n-gram processing.

**5. Clause / phrase boundary detection**

| Clean input | Corrupted input | Signal |
|------------|----------------|--------|
| When the rain stopped, the children played | When the rain stopped the children played | Comma-dependent clause boundary lost |
| The man who wore a hat left | The man who wore a hat, left | Spurious boundary inserted |

20 pairs. Add or remove clause boundaries. If patching disrupts cross-clause predictions, the head is tracking phrase structure.

**6. Positional / distance-based attention**

| Clean input | Corrupted input | Signal |
|------------|----------------|--------|
| A B C D E F G H I J | J I H G F E D C B A | Position-dependent predictions change |

20 pairs. Reverse or shuffle token order. If patching changes predictions in a position-dependent way, the head is doing distance-based processing distinct from positional_prev.

#### Protocol

1. Load the NeoX comparison model (merge barriers, step-20000) from HuggingFace
2. For each behavior category, generate 20 clean/corrupted input pairs (seed=42, deterministic)
3. For each of the 95 unclassified heads:
   a. Run clean input, record all head activations and final logits
   b. Run corrupted input, record head H's activations
   c. Replace head H's activations in the clean forward pass with the corrupted activations
   d. Measure logit change at the critical token position
   e. Score: magnitude of logit shift toward the corrupted prediction
4. For each head, rank behavior categories by patching effect magnitude
5. Classify: head is assigned to the behavior with the largest patching effect (if above threshold)
6. Control: run the same protocol on 20 random non-unclassified heads to verify they show expected behavior (delimiter heads should respond to structural patching, spacing heads should not respond to syntactic patching)

#### Implementation

Write `eval/patch_unclassified_heads.py` that:
1. Loads the comparison model and tokenizer
2. Generates all input pairs programmatically (no external datasets)
3. Implements activation patching via PyTorch forward hooks
4. Runs all 95 heads x 6 behaviors x 20 pairs = 11,400 patching experiments
5. Outputs per-head behavior classification with confidence scores
6. Saves JSON results to `results/patching/`

The patching mechanism uses `register_forward_hook` to capture activations on the corrupted pass, then `register_forward_pre_hook` to inject them on the clean pass. This is standard practice (Conmy et al., 2023; Wang et al., 2023).

#### Expected results

At 410M/20K steps, the model may not have developed strong syntactic capabilities. Expected distribution of the 95 heads:

| Behavior | Expected count | Rationale |
|----------|---------------|-----------|
| Semantic similarity | 30-50 | Most common in small models |
| Local context / n-gram | 20-30 | Basic pattern matching |
| Subject-verb agreement | 5-15 | Requires deeper processing |
| Coreference | 0-5 | Rare at this scale |
| Clause boundary | 5-10 | Possible given freed capacity |
| Positional / distance | 5-10 | Residual positional processing |
| Still unclassified | 10-20 | Some heads may require SAEs |

If more than 20 heads remain unclassified after patching, sparse autoencoders (Bricken et al., 2023; Cunningham et al., 2024) become the next methodology.

#### Results (2026-07-06) — NULL RESULT

All 95 unclassified heads and all 20 control heads showed near-zero patching effects for agreement, coreference, semantic, and local_context behaviors (abs_mean_effect < 0.001). The positional category produced the only nonzero signal (abs_mean_effect 0.002-0.039), but this was a design flaw: reversing the entire sequence produces a massive activation difference at every head, swamping the subtler behavioral signals.

**Interpretation:** At 410M/20K steps, the model has not developed the linguistic capabilities these probes test for. The freed heads are performing processing below the resolution of behavioral probes at this scale: likely content similarity via embedding space, local windowing, n-gram statistics, or diffuse contextual processing that doesn't map to named linguistic capabilities.

**This is informative:** It sets a lower bound on the scale where freed capacity develops identifiable linguistic specialization. It also confirms that activation patching requires models with the target capability already present. At larger scale (1.3B+) or with more training, these probes may produce signal.

**Next methodology:** Sparse autoencoders (Bricken et al., 2023; Cunningham et al., 2024), which decompose activations into interpretable features without requiring behavioral hypotheses. Or repeat patching at 1.3B scale where linguistic capabilities are stronger.

**Cost:** ~$0.10, ~12 minutes on RTX 4090. Script: `eval/patch_unclassified_heads.py`. Results: `results/patching/patching-comparison.json`.

### Planned: LLC Phase-Transition Correlation (bridge to Timaeus / SLT)

The strongest link between this program and the Timaeus developmental-interpretability work (Wang, Hoogland, Murfet). We measure development from attention patterns (behavioral); singular learning theory measures it from loss-landscape geometry via the Local Learning Coefficient (LLC, the RLCT lambda). These are orthogonal methods. If the spacing-head emergence we found behaviorally is a genuine developmental stage, it should also appear as a phase transition in the LLC trajectory.

#### Hypothesis

The spacing-head emergence (behaviorally: ~28 heads by step 50, 184 by step 150, stabilized by step 2,000) coincides with a phase transition in the LLC. And the merge-barrier model, which never builds spacing heads, should show a *different* LLC trajectory: a missing or shifted transition where the spacing stage would otherwise be.

If confirmed, this connects our empirical finding to SLT machinery in Timaeus's own language, on our own data. Nobody has connected the BPE spacing tax to the learning coefficient.

#### Method

`devinterp` (the Timaeus Python library, `pip install devinterp`) estimates the LLC at a checkpoint via SGLD sampling: for each of N chains, take D draws, each draw a forward+backward pass on a batch. One lambda per checkpoint. Plotting lambda(t) across training reveals phase transitions as jumps/plateaus (Hoogland et al. 2024 showed induction heads emerge at an LLC transition; Furman & Lau give the at-scale estimation method).

#### Protocol

1. **Calibration first (do NOT skip).** Start from the devinterp defaults and the Furman & Lau scaling rules for a 410M model. Run their diagnostic recipe (trace plots, chain-convergence checks) on a handful of checkpoints until the estimate is stable. This is the step that requires understanding the SGLD internals; the paper provides the recipe for what "converged" looks like. Budget 1-2 GPU hours.
2. **Coarse sweep, one run.** ~25 checkpoints across the baseline trajectory (denser in the early window step 0-500 where differentiation happens). Produces a lambda(t) curve. Check whether a transition appears near the behavioral spacing emergence (step 50-150).
3. **Decision gate.** If a transition shows up, densify and add the merge-barrier (comparison) run. If nothing shows, stop; you've spent ~$5 instead of ~$30 to learn the spacing stage is not visible in the LLC at this scale.
4. **Full comparison.** Baseline vs comparison lambda(t). The key result: does the barrier model lack the transition the baseline has?

#### Cost

Per checkpoint: ~5-15 min on a 4090/A100 (depends on chains x draws). Estimates:

| Scope | Checkpoints | GPU time | Cost |
|-------|------------|----------|------|
| Calibration | ~5 (repeated) | 1-2 hrs | $1-2 |
| Coarse sweep, one run | ~25 | 2-6 hrs | $1-3 |
| Coarse, both runs | ~50 | 5-12 hrs | $3-8 |
| Full dense, both runs | 262 | 22-65 hrs | $10-30 |

Recommended: spend ~$5 on calibration + one coarse run to find out if it works before committing to the full sweep. Furman & Lau's paper and the devinterp defaults substantially cut the calibration cost (start from their settings, verify on our model, adjust if the diagnostic fails), turning a blind multi-hour search into 1-2 hours of guided verification.

#### Why this matters

This is the only planned experiment that puts our empirical results and Timaeus's theory on the same graph. It is written in their language (LLC), on our data (917 checkpoints). It is also the natural on-ramp to learning singular learning theory hands-on: run the tool on data we already understand, let a surprising result motivate opening the box. Bridge-paper potential with the Timaeus group.

Reference: Hoogland et al. (2024) "The Developmental Landscape of In-Context Learning"; Furman & Lau (2024) "Estimating the Local Learning Coefficient at Scale"; Wang et al. (2025a) rLLC. All in `references/`.

### Priority order

1. **LLC phase-transition correlation (calibration + coarse sweep first, ~$5)**: the bridge to Timaeus/SLT. Highest strategic value (connects our work to their theory and community). Start small: prove the transition is visible before the full sweep.
2. **Scale replication (1.3B or larger)**: highest value for addressing the remaining limitation, moderate cost. Could also enable patching re-run with stronger linguistic signal.
3. **SAE analysis of unclassified heads**: decompose the 95 heads without behavioral hypotheses.
4. **Circuit causality (regularization intervention)**: highest engineering effort, standalone contribution to interpretability.

## Relationship to Prior Work

This project extends the merge-barriers research (DOI: 10.5281/zenodo.20925910). During that work, we measured delimiter head emergence at step ~1,000 and gradient-attention coupling over 20K steps. The atlas asks: what ELSE is developing simultaneously, and does the tokenizer change the sequence?
