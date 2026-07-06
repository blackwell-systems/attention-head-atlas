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
- Llama developmental atlas (training + probing + ablation): ~$3.40 (in progress)

## Status (2026-07-04)

| Run | Training | v1 Probing | v2 Probing | Excess (v1) | Excess (v2) | Analysis |
|-----|----------|-----------|-----------|-------------|-------------|----------|
| Baseline | COMPLETE | COMPLETE (old probes) | COMPLETE (spacing) | COMPLETE | COMPLETE | 13 findings |
| Comparison | COMPLETE | COMPLETE (old probes) | COMPLETE (spacing) | COMPLETE | COMPLETE | 13 findings |
| Seed2 | COMPLETE | COMPLETE (new probes) | COMPLETE (spacing) | COMPLETE | COMPLETE | 13 findings |
| NL-barrier | COMPLETE | COMPLETE (new probes) | COMPLETE (spacing) | COMPLETE (step-50 proxy*) | COMPLETE | 13 findings |

*NL-barrier step-0 checkpoint corrupted by disk-full event. Step-50 base rates used as proxy for excess correction in both v1 and v2. Defensible: 50 steps = 0.008% of corpus, attention still approximately random.

All 524 checkpoints on R2. 1048 probe results (524 v1 + 524 v2) on R2 and committed to `results/`. Excess-corrected results in `results/{run}-excess/` (v1) and `results/{run}-v2-excess/` (v2). 9 visualization charts in `charts/` (from v1 data; v2 charts pending). 13 findings documented in RESULTS.md. Paper draft in `paper/developmental-atlas.md` (needs revision for v2 findings).

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

### Remaining: Downstream Task Benchmark

Tests whether the capacity tax translates into measurable task performance differences (accuracy, not just PPL). Uses existing NeoX 410M baseline vs comparison checkpoints on FineWeb. Inference only, no training.

#### Design principles

- Same architecture, same training data, only tokenizer differs (baseline vs comparison)
- Tasks simple enough for 410M (not MMLU or HumanEval, too hard at this scale)
- Mix of structural tasks (where boundary recovery matters) and prose tasks (where it shouldn't)
- Accuracy on each task, not PPL. Binary correct/incorrect. 100 examples per task.
- The comparison model should win on structural tasks and tie on prose

#### Proposed tasks

| Task | What it tests | Expected effect | How to measure |
|------|--------------|-----------------|----------------|
| Bracket matching | Given nested `([{}])`, is it balanced? | Large: baseline 4 bracket heads, comparison 39 | Generate 100 balanced/unbalanced bracket sequences, prompt model to predict "balanced" or "unbalanced", measure accuracy |
| Duplicate detection | Which word appears twice in this list? | Large: ablation showed +405% degradation when spacing removed | Generate 100 word lists with one duplicate, prompt model to identify it, measure accuracy |
| Field extraction | What is the value of field X in this JSON/GCF payload? | Large: delimiter heads + structural processing | Generate 100 small payloads (5-10 fields), ask for a specific field value, measure exact match accuracy |
| Code completion | Complete this Python function | Medium: code depends on bracket/delimiter boundaries | 100 simple function completions (reverse a list, sum elements), measure whether output is syntactically valid |
| Sentence boundary | Where does the second sentence start? | Medium: spacing-dependent | 100 two-sentence passages, ask which word starts the second sentence, measure accuracy |
| Prose QA | Answer a question about a paragraph | Small: NL is redundant, spacing shouldn't matter | 100 simple factual questions about short paragraphs, measure accuracy |

#### Implementation

Write `eval/benchmark_downstream.py` that:
1. Loads a checkpoint and tokenizer
2. For each task, generates 100 test examples programmatically (no external dataset dependency)
3. Runs each example through the model using greedy next-token prediction
4. Scores accuracy per task
5. Reports per-task accuracy for baseline vs comparison
6. Outputs JSON with all results

#### Checkpoints to test

| Model | Checkpoint | Source |
|-------|-----------|--------|
| NeoX 410M baseline (standard BPE) | HF: blackwell-systems/attention-head-atlas/baseline-step-20000.pt | FineWeb trained |
| NeoX 410M comparison (merge barriers) | HF: blackwell-systems/attention-head-atlas/comparison-step-20000.pt | FineWeb trained |

#### Expected results

The comparison model should show:
- Higher accuracy on bracket matching (39 bracket heads vs 4)
- Higher accuracy on duplicate detection (spacing heads freed for other work)
- Higher accuracy on field extraction (more delimiter heads)
- Similar or higher accuracy on code completion
- Similar accuracy on prose QA (no NL cost from merge barriers)

If the comparison model wins on structural tasks with no regression on prose, the capacity tax claim has downstream task validation.

#### Cost

Inference only. Can run on any GPU instance or locally on CPU (slower). No training needed. The checkpoints are on HuggingFace.

### Remaining: Llama Developmental Atlas (Llama-FineWeb-Baseline)

Full 131-checkpoint developmental analysis of Llama 410M trained on FineWeb. This is the architecture replication of the NeoX developmental atlas: not just "spacing exists in Llama" but "spacing follows the same developmental program across architectures."

#### Purpose

The NeoX atlas has developmental trajectories for 4 runs (baseline, comparison, seed2, NL-barrier) plus 2 structok corpus runs. All are MHA. Adding the Llama developmental trajectory answers:

1. **When do spacing heads emerge in GQA?** Same training step as MHA, or earlier/later?
2. **Does GQA change the emergence dynamics?** With 4 KV heads shared across 16 query heads, do spacing heads cluster within KV groups or distribute evenly?
3. **Does the spacing fin appear in Llama UMAP?** Wang et al.'s spacing fin was observed on MHA. Does it manifest differently with GQA?
4. **Is the P0 surge simultaneous across architectures?** NeoX P0 emerges around step 200-400. Same in Llama?

#### Training

Currently running (2026-07-05). Llama 410M on FineWeb with standard-64k tokenizer.

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

### Remaining: Llama Ablation (clean, FineWeb-trained)

The Llama ablation from run-003/004 checkpoints was INCONCLUSIVE because those models were trained on structok corpus but probed on FineWeb texts (distribution mismatch, baseline PPL 150K-314K). The llama-fineweb-baseline run fixes this: same corpus as NeoX baseline, clean ablation comparison.

**Cost:** Included in Llama Developmental Atlas above (ablation runs on the same step-20000 checkpoint).

### 3. Downstream task impact

The ablation measures PPL change, not task accuracy. Showing that merge barriers improve actual benchmark scores (MMLU, HumanEval, structured data comprehension tasks) would connect the capacity tax to metrics model providers already track. This bridges the gap between "heads are reallocated" and "the reallocation matters for performance."

**Cost:** Depends on benchmark suite. Evaluation only, no training.

### 4. Causal circuit intervention

The circuit protection finding is correlational (100% P0 isolation). To make it causal: artificially couple an isolated head into a circuit during training (e.g., by adding a regularization term that correlates its trajectory with a circuit member) and show it survives instead of collapsing into P0. This would establish that circuits are causally protective, not merely co-occurring with survival.

**Cost:** Requires custom training loop modification. Moderate effort.

### Priority order

Experiment 2 (1.3B spacing probe) is the highest-value, lowest-cost next step. The checkpoint exists. The probe script exists. One inference run answers whether spacing scales. If it does, that single data point eliminates the most common objection to the paper.

## Relationship to Prior Work

This project extends the merge-barriers research (DOI: 10.5281/zenodo.20925910). During that work, we measured delimiter head emergence at step ~1,000 and gradient-attention coupling over 20K steps. The atlas asks: what ELSE is developing simultaneously, and does the tokenizer change the sequence?
