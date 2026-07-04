# Developmental Atlas of Attention Head Specialization

**Paper:** "Developmental Atlas of Attention Head Specialization: How Transformers Organize During Training and Why Circuits Protect Against Collapse"

**A comprehensive map of when and how attention heads specialize during transformer training, how the tokenizer shapes that process, and why heads that wire into circuits survive while isolated heads collapse.**

## The Gap

Every existing study of attention head specialization probes ONE behavior type in isolation:

| Paper | What they tracked | What they missed |
|-------|------------------|-----------------|
| Olsson et al. 2022 | Induction heads emerge at step X | Everything else |
| Gu et al. 2025 | Dormancy emerges at ~55B tokens | What the non-dormant heads are doing |
| Wang et al. 2025 | Head differentiation via learning coefficient | Only a 2-layer toy model |
| Blackwell 2026 (ours) | Delimiter heads by step ~1000 | Other specialization types |

Nobody has produced a **full developmental atlas**: tracking ALL head behaviors simultaneously from step 0 to convergence in a realistic model. We don't know:

- What develops first? Positional heads before induction? Delimiter before content?
- Is there a fixed developmental order or does it depend on data/tokenizer?
- Do heads transition between types during training?
- When do heads "commit" to a specialization (irreversibly)?
- Does the tokenizer change the developmental sequence (not just the outcome)?

## The Project

Train a single 410M transformer, checkpoint frequently, and at each checkpoint classify every head across multiple behavior types simultaneously.

### Behaviors to probe at each checkpoint

| Behavior | How to measure | Expected emergence |
|----------|---------------|-------------------|
| **Positional (previous token)** | Attention mass on position n-1 | Very early (step 0-100) |
| **Positional (first token / P0 sink)** | Attention mass on position 0 | Early (Gu: by 55B tokens) |
| **Induction** | Copy score: does head attend to token after previous occurrence? | Early-mid (Olsson: phase change) |
| **Delimiter/structural** | Attention mass on delimiter token positions | Early (our data: step ~1000) |
| **Syntactic (bracket matching)** | Attention from close-bracket to open-bracket | Mid? |
| **Content (semantic similarity)** | Correlation between attention and embedding similarity | Mid-late? |
| **Duplicate token** | Attention to previous occurrences of same token | Unknown |
| **Dormant (attention sink)** | HONOR metric < threshold (Sandoval-Segura 2025) | Early, then stabilizes |

### Training setup

- Architecture: GPT-NeoX 410M (same scale as merge-barriers run-002)
- Corpus: **SlimPajama** or **RedPajama-v2** (standard pretraining mix approximating production models: ~70% web, ~10% code, ~5% academic, ~5% books, ~5% Wikipedia, ~5% misc). NOT the structured-data-heavy corpus from merge-barriers experiments. Ensures findings generalize to how production models develop.
- Tokenizer: standard BPE (GPT-NeoX default or similar)
- Checkpoints: every 50 steps for first 2000, every 200 steps to 20000
- Probe data: fixed set of texts covering all behavior types (structured data, code, prose, brackets, repeated tokens)
- **Baseline** (primary): standard tokenizer, standard corpus. The "normal" developmental atlas.
- **Comparison** (second run): merge-barrier tokenizer, same corpus. Tests whether the tokenizer changes the developmental SEQUENCE, not just the outcome.

### Polysemanticity (heads do multiple things)

Heads are not single-function units. A head might be 60% positional on prose but 80% delimiter on structured data. Forcing a single label is a simplification that hides this.

**Approach: score vectors, not labels.**

For each checkpoint, for each head (24 layers x 16 heads = 384):
- Full score vector across all 8 behavior types (continuous, not classified)
- **Specialization index**: how concentrated the score vector is (high = specialist, low = generalist). Computed as max(scores) / sum(scores). A head at [0.8, 0.1, 0.05, 0.05, 0, 0, 0, 0] has high specialization. A head at [0.15, 0.14, 0.13, 0.12, 0.12, 0.12, 0.11, 0.11] is a generalist.
- **Top-2 behaviors** with confidence: "primarily delimiter (0.6), secondarily positional (0.25)" is more honest than "delimiter head"
- **Context-conditional scores**: measured separately on each probe text. A head might be positional on prose but delimiter on JSON. The stranding experiment already showed this (same head, different behavior on different input).

**Key developmental questions this enables:**
- Do heads START as generalists and BECOME specialists over training?
- Does specialization index increase monotonically, or do heads oscillate?
- When a head "commits" to a specialization, is it irreversible?
- Do polysemantic heads occupy specific layers (early = generalist, late = specialist)?

**Future extensions (not v1):**
- Sparse Autoencoders (SAEs) on head outputs to decompose features within a single head
- QK/OV circuit decomposition into independent subspace directions (Elhage et al., 2022)
- Causal scrubbing: vary input features and measure which output dimensions respond

### Visualizations

- **Developmental timeline**: x=training step, y=head index, color=dominant behavior type. Shows when each head commits.
- **Phase transitions**: when do groups of heads shift behavior simultaneously?
- **Tokenizer comparison**: same timeline for Model A vs Model B. Do heads specialize in the same order?
- **Layer depth analysis**: do early layers specialize first, or is it distributed?

## Why This Matters

1. **Developmental interpretability** is a young subfield. This would be the first comprehensive atlas at realistic scale.
2. **Training efficiency**: if we know WHEN heads commit, we could intervene (change learning rate, inject data) at critical periods.
3. **Extends our merge-barriers work**: the tokenizer changes the outcome (which heads develop). Does it also change the SEQUENCE?
4. **Practical for model providers**: understanding head development could inform curriculum learning, data mixing schedules, and early stopping decisions.

## Relationship to Published Work

This is the third paper in a research program on tokenizer-attention coupling:

1. **Tokenizer-Attention Coupling** (DOI: 10.5281/zenodo.20925910): Proves BPE merge decisions permanently shape which attention heads develop. 3 domains, 2 architectures, 2 scales.
2. **Stranded Attention** (DOI: 10.5281/zenodo.21158886): Characterizes the frustration gap: 40pp of structural attention capacity permanently locked away. 384/384 heads affected.
3. **Developmental Atlas** (this paper): Tracks ALL head types simultaneously from step 0 to convergence. Discovers developmental circuits, P0 failure cascade, and that NL characters have 265x larger adversarial surfaces than structured data delimiters.

The atlas extends both prior papers: the frustration gap is domain-dependent (0pp on web text, 40pp on structured data), but the damage mechanism still operates via P0 head collapse. It also extends Gu et al. (ICLR 2025) by answering two of their stated open questions about attention sinks.

Target: cs.LG on arXiv. Standalone paper with cross-references to the prior two.

## Status (2026-07-04)

- [x] Baseline training + probing (131 checkpoints, standard BPE)
- [x] Comparison training + probing (131 checkpoints, merge barriers)
- [x] Seed variation training + probing (131 checkpoints, different init)
- [x] Excess score correction applied to all 3 runs
- [x] Developmental circuit discovery
- [x] P0 deep analysis (failure cascade, circuit isolation, Gu et al. connection)
- [x] Velocity-based circuit discovery
- [x] NL adversarial surface analysis
- [x] 10 findings documented in RESULTS.md
- [x] 8 visualization charts
- [ ] NL barriers experiment (in progress on vast.ai)
- [ ] Structok corpus run (planned)

## Repository Structure

```
attention-head-atlas/
  eval/                           # Scripts
    train_atlas.py                # Training with step-0, background R2 upload, resume
    probe_heads.py                # 8-behavior probing + entropy + frustration gap
    excess_score_correction.py    # Post-hoc base-rate correction
    prep_data.py                  # FineWeb download + parallel pretokenization
    train_nl_tokenizer.py         # Train NL-barrier tokenizer
    run_nl_pipeline.sh            # Full NL-barrier experiment pipeline
    ascii-adversarial-surface.py  # 43-tokenizer adversarial surface scan
    analyze_seed2.py              # Seed variation analysis
    analyze_p0_deep.py            # P0 failure cascade analysis
    analyze_velocity_circuits.py  # Derivative-based circuit discovery
  probes/                         # Probe texts (6 standard + 2 extreme)
  results/                        # Probe results
    baseline/                     # 131 raw results
    baseline-excess/              # 131 excess-corrected
    comparison/                   # 131 raw results
    comparison-excess/            # 131 excess-corrected
    seed2/                        # 131 raw results (new probes)
    seed2-excess/                 # 131 excess-corrected
  charts/                         # Visualization
    generate_atlas.py             # Chart generation (4-run support)
    *.png                         # 8 charts
  references/                     # 9 prior art PDFs
  tokenizers/                     # NL-barrier tokenizer
  EXPERIMENT-DESIGN.md            # Design, R2 schema, roadmap
  RESULTS.md                      # 10 findings
```

## Repository Structure

```
attention-head-atlas/
  eval/                         # Scripts
    train_atlas.py              # Training with step-0 checkpoint, background R2 upload, resume support
    probe_heads.py              # Multi-behavior probing: 8 types + entropy + frustration gap
    excess_score_correction.py  # Post-hoc base-rate correction on existing results
    prep_data.py                # Download FineWeb corpus + parallel pretokenization
    ascii-adversarial-surface.py # 43-tokenizer adversarial surface scan (from merge-barriers)
  probes/                       # Probe texts (input for probing)
    prose.txt                   # Natural language, no punctuation (~256 tokens)
    code.txt                    # Go function with structs (~256 tokens)
    structured.txt              # GCF tabular data with pipe delimiters (~256 tokens)
    induction.txt               # Repeated sentences for induction testing (~256 tokens)
    duplicates.txt              # Repeated words for duplicate-token testing (~256 tokens)
    brackets.txt                # Real Go code with balanced brackets (~256 tokens)
    dense_json.txt              # Extreme: high delimiter density nested JSON
    heavy_induction.txt         # Extreme: maximally repeated sequences
  results/                      # Probe results (JSON per checkpoint)
    baseline/                   # 131 raw probe results (standard BPE)
    comparison/                 # 131 raw probe results (merge barriers)
    baseline-excess/            # 131 excess-corrected results
    comparison-excess/          # 131 excess-corrected results
    ascii-adversarial-surface-43-tokenizers-20260625.json  # 43-tokenizer scan
  charts/                       # Visualization
    generate_atlas.py           # Chart generation script (raw + excess, light + dark)
    README.md                   # Chart interpretation guide
    *.png                       # Generated charts
  references/                   # Prior art PDFs (8 papers)
  EXPERIMENT-DESIGN.md          # Experiment design, R2 schema, roadmap
  RESULTS.md                    # Findings narrative (9 findings)
```

## Scripts

| Script | Purpose | Provenance |
|--------|---------|------------|
| `eval/train_atlas.py` | Train GPT-NeoX 410M with 131 checkpoints (step-0 through 20000). Saves locally, uploads to R2 in background thread with retry. Supports `--resume-from r2`. | Written for this project |
| `eval/probe_heads.py` | Probe every head across 8 behavior types on 6 probe texts. Supports local checkpoints (`--checkpoint-dir`) or R2 streaming (`--r2-prefix`) with skip logic for existing results. GPU inference, model reuse across checkpoints. | Written for this project |
| `eval/excess_score_correction.py` | Subtract step-0 base rates from raw scores to reveal genuine specialization. Reads existing results, writes corrected results to `*-excess/` directories. | Adapted from merge-barriers excess score methodology (Blackwell, 2026) |
| `eval/prep_data.py` | Download FineWeb corpus from HuggingFace, pretokenize with both tokenizers in parallel using multiprocessing (12 workers, 1 MB segments). | Written for this project |
| `eval/ascii-adversarial-surface.py` | Scan all 94 printable ASCII characters across 43 tokenizer vocabularies for merge entries. | Copied from gcf repo (`eval/ascii-adversarial-surface.py`) |
| `charts/generate_atlas.py` | Generate all visualization charts from probe results. Supports raw and excess-corrected data, light and dark themes. | Written for this project |

## Data on R2

All training checkpoints and probe results are archived on Cloudflare R2 under the `structok-training` bucket with prefix `atlas/`.

```
atlas/tokens/         # Pretokenized bins + tokenizer JSONs
atlas/runs/baseline/  # 131 training checkpoints (standard BPE)
atlas/runs/comparison/ # 131 training checkpoints (merge barriers)
atlas/runs/seed2/     # 131 training checkpoints (seed variation, in progress)
atlas/results/baseline/ # 131 probe results
atlas/results/comparison/ # 131 probe results
atlas/results/seed2/  # Probe results (pending)
```

## Infrastructure

- Training: GPT-NeoX 410M on A100 PCIE (vast.ai)
- Corpus: FineWeb (HuggingFaceFW/fineweb, sample-10BT, ~5 GB)
- Tokenizers: standard-64k.json (no barriers), structok-64k.json (16 merge barriers)
- Analysis: local (no GPU needed for post-hoc analysis and chart generation)
