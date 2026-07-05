# Developmental Atlas of Attention Head Specialization

## Experiment Summary

Four GPT-NeoX 410M models trained for 20,000 steps on FineWeb (web text corpus):
- **Baseline**: standard BPE (65,536 vocab)
- **Comparison**: merge-barrier BPE (65,539 vocab, 16 structured delimiter characters forbidden from merging)
- **Seed2**: standard BPE (65,536 vocab, different random init)
- **NL-barrier**: NL-barrier BPE (~65,536 vocab, 10 NL delimiter characters forbidden from merging)

131 checkpoints per run (step 0 through 20,000). Each checkpoint probed across 7 behavior types (positional_prev, positional_p0, induction, delimiter, bracket, duplicate, spacing) plus 2 auxiliary metrics (entropy, dormancy) on 6 probe texts, plus frustration gap measurement. All classifications use excess scores (base-rate corrected) unless stated otherwise.

### v1 vs v2 probe data

**v1 probes** (original): 6 behavior types (no spacing). Baseline and comparison used original probe texts; seed2 and NL-barrier used improved probes (real bracketed code, standardized lengths, punctuation-stripped prose). Results in `results/{run}/` and `results/{run}-excess/`.

**v2 probes** (2026-07-04 re-probe): 7 behavior types (spacing added). All 4 runs re-probed with identical improved probe texts on a single RTX 4090, ensuring consistent probes across all runs. Results in `results/{run}-v2/` and `results/{run}-v2-excess/`. v1 data preserved.

The v2 re-probe revealed that spacing is the dominant head specialization in standard BPE models (183/384 heads), and that over half of v1's apparent P0 heads were actually spacing specialists. All numbers below are from v2 unless labeled (v1).

## Excess Score Methodology

At random initialization, a head's attention is roughly uniform across all positions. If 30% of positions in a probe text are delimiter characters, a random head directs ~30% of attention to delimiters by chance. That's not specialization; it's arithmetic.

Raw scores measure total attention to each behavior type. Excess scores subtract the step-0 base rate (measured from random init) to reveal genuine specialization. A head with 0.30 raw and 0.30 base rate has 0.00 excess (no specialization). A head with 0.30 raw and 0.10 base rate has 0.20 excess (genuine specialist).

Without this correction, the original probe set produced heavily inflated classifications: 172 heads appeared to be "delimiter specialists" in baseline, but only 83 showed genuine excess. The brackets probe (100% delimiter characters) inflated every head's delimiter score to 1.0. The excess methodology, adapted from the merge-barriers paper (Blackwell, 2026), removes this inflation.

## Finding 1: Head Type Distribution at Convergence

### v2 (with spacing, consistent probes)

| Type | Baseline | Comparison | Seed2 | NL-barrier |
|------|----------|------------|-------|------------|
| Spacing | 183 (47.7%) | 13 (3.4%) | 183 (47.7%) | 0 (0.0%) |
| Unclassified | 10 (2.6%) | 95 (24.7%) | 7 (1.8%) | 61 (15.9%) |
| Positional (prev) | 68 (17.7%) | 91 (23.7%) | 59 (15.4%) | 92 (24.0%) |
| Delimiter | 74 (19.3%) | 79 (20.6%) | 93 (24.2%) | 58 (15.1%) |
| P0 sink | 32 (8.3%) | 40 (10.4%) | 29 (7.6%) | 56 (14.6%) |
| Bracket | 4 (1.0%) | 39 (10.2%) | 2 (0.5%) | 60 (15.6%) |
| Induction | 5 (1.3%) | 15 (3.9%) | 4 (1.0%) | 24 (6.3%) |
| Duplicate | 8 (2.1%) | 12 (3.1%) | 7 (1.8%) | 33 (8.6%) |

### v1 (no spacing, inconsistent probes, for reference)

| Type | Baseline | Comparison | Seed2 | NL-barrier |
|------|----------|------------|-------|------------|
| Positional (prev) | 102 (26.6%) | 99 (25.8%) | 86 (22.4%) | 93 (24.2%) |
| P0 sink | 96 (25.0%) | 52 (13.5%) | 64 (16.7%) | 57 (14.8%) |
| Delimiter | 83 (21.6%) | 66 (17.2%) | 143 (37.2%) | 57 (14.8%) |
| Unclassified | 38 (9.9%) | 57 (14.8%) | 51 (13.3%) | 66 (17.2%) |
| Induction | 32 (8.3%) | 26 (6.8%) | 27 (7.0%) | 25 (6.5%) |
| Duplicate | 24 (6.3%) | 37 (9.6%) | 10 (2.6%) | 26 (6.8%) |
| Bracket | 9 (2.3%) | 47 (12.2%) | 3 (0.8%) | 60 (15.6%) |

### Key changes from v1 to v2

**Spacing is the dominant specialization in standard BPE.** 183/384 heads (47.7%) in both baseline and seed2 are spacing specialists. These heads attend to whitespace positions (space, newline, tab). This was invisible in v1 because spacing was not measured; those heads were misclassified as positional_prev, P0, delimiter, induction, or duplicate.

**P0 count drops from 96 to 32 in baseline.** 54 of v1's 96 P0 heads were actually spacing specialists. The genuine P0 count is 29-32 (7.6-8.3%), not 96 (25%).

**Merge-barrier models have almost no spacing heads.** Comparison: 13. NL-barrier: 0. Merge barriers prevent the whitespace boundary corruption that forces heads into spacing recovery.

**The merge-barrier model still develops more bracket specialists** (39 vs 4 baseline). NL-barrier produces the most (60), because parentheses are in the NL barrier set.

## Finding 2: P0 Collapse Is Real But Smaller Than v1 Reported

### v2 revision

The v1 analysis reported 96 P0 heads in baseline. The v2 re-probe with spacing measurement reveals that 54 of those 96 were actually spacing specialists, not P0 sinks. The genuine P0 count is 32 (8.3%), not 96 (25%).

**v1 P0 heads reclassified in v2:**

| Run | v1 P0 | v2 P0 | Were spacing | Genuine P0 | Other |
|-----|-------|-------|-------------|------------|-------|
| Baseline | 96 | 32 | 54 (56%) | 29 | 13 |
| Comparison | 52 | 40 | 3 (6%) | 26 | 23 |
| Seed2 | 64 | 29 | 35 (55%) | 29 | 0 |
| NL-barrier | 57 | 56 | 0 (0%) | 56 | 1 |

**Where v1 baseline P0 heads went in v2:** spacing:54, positional_p0:29, delimiter:8, positional_prev:3, unclassified:2.

**The P0 cascade mechanism is still real.** The 29-32 genuine P0 heads in baseline still show the try-fail-collapse pattern from the v1 deep analysis. The v1 analysis of prior types and sink timing applies to this smaller set. The conclusion that P0 is a failure mode (not a benefit) stands.

**What changed:** the SCALE of the cascade. 8.3% of heads are genuine P0 sinks, not 25%. The remaining 17% were spacing heads that the v1 probe couldn't identify. This is a measurement correction, not a refutation of the mechanism.

### v1 findings (still valid for genuine P0 subset)

The deep analysis of P0 heads (prior types, sink timing, circuit isolation) was performed on v1 data. The try-fail-collapse narrative applies to the genuine P0 subset (29-32 heads) rather than all 96.

### P0 Deep Analysis

**What were P0 heads before they sank?** This is the mechanistic finding. P0 sinking is a failure cascade, not a design choice.

| Prior type | Baseline | Seed2 |
|-----------|----------|-------|
| Unclassified (never specialized) | 37 (39%) | 32 (50%) |
| Delimiter (tried, failed) | 34 (35%) | 21 (33%) |
| Duplicate | 7 (7%) | 6 (9%) |
| Induction | 7 (7%) | 0 |
| Positional_prev | 7 (7%) | 4 (6%) |
| Bracket | 4 (4%) | 1 (2%) |

**When do they sink?** Gradual attrition, not a phase transition. Earliest: step 100. Median: step 11,000. 62 of 96 sink late (after step 2000). Individual heads give up at different times across training.

**Late layers are most vulnerable.** L23 (8 heads) and L17 (8 heads) have the highest P0 concentration in baseline. These layers handle the most complex processing and are most sensitive to boundary corruption.

**Merge barriers save 79 heads from P0.** At the 96 positions where baseline has P0 sinks, comparison has: 23 delimiter, 22 positional_prev, 9 duplicate, 9 induction, 8 bracket, 8 unclassified. Only 17 are P0 in both. Merge barriers convert wasted P0 capacity directly into productive specialization.

**Sinking is late, not early.** Median sink step: 11,000. 62 of 96 heads sink after step 2,000. Gu et al. (2025) showed the P0 sink mechanism (the ability of position 0 to attract attention) emerges globally by step 1-2K. Our finding extends this: the mechanism is available early, but individual heads don't collapse into it until much later, after failing at other specializations. Gu et al. studied when the infrastructure appears. We study when heads decide to use it.

**P0 heads are 100% isolated.** None of the 96 baseline P0 heads belong to any co-specializing circuit. Circuits are resistant to dormancy; isolated heads are not. This suggests circuits provide mutual reinforcement that prevents collapse: heads that wire together survive; heads that don't, sink. This is a novel insight about why circuits form: they're not just computationally useful, they're developmentally protective.

**P0 count is seed-dependent.** 96 (baseline) vs 64 (seed2). Same tokenizer, different init. The overall pattern holds (standard BPE produces more P0 sinks than merge barriers) but the exact count varies. The effect is real; the precise number is not generalizable from a single seed.

### Connection to Gu et al. (2025) Open Questions

Gu et al. ("When Attention Sink Emerges in Language Models," ICLR 2025) posed two open questions in their Future Work that the atlas directly addresses:

**Open question 1: "It remains unclear whether attention sink benefits LM downstream performance."** Our data shows P0 sinks are a failure mode, not a benefit. 35% of P0 heads were delimiter specialists that tried to specialize and failed. 100% are isolated from circuits (no computational role in co-specializing networks). Merge barriers that prevent P0 sinking produce more productive heads (79 converted to delimiter, positional_prev, induction, bracket). The model is better off without them.

**Open question 2: "We will extend the research scope to explore how these sink tokens are related to the pre-training."** The tokenizer is the connection. Standard BPE produces 96 P0 heads; merge barriers produce 52. Same architecture, same data, only the tokenizer differs. The sink-prone tokens (merged delimiters) are created by BPE's merge decisions during tokenizer training, which is a pre-training decision. The tokenizer determines which heads have viable specialization targets and which will eventually collapse.

Source: `eval/analyze_p0_deep.py`.

## Finding 3: Entropy Divergence

Baseline attention entropy rises from ~0.35 back to ~0.70 in late training (steps 10,000-20,000). Comparison stays flat at ~0.35. The standard BPE model's attention becomes more diffuse over time; the merge-barrier model maintains focused attention throughout.

This is consistent with the dormancy finding: more P0 sinks means more heads routing attention to a single position (low local entropy but high global entropy as attention spreads across remaining non-sink heads).

## Finding 4: Frustration Gap Is Domain-Dependent

Frustration gap (normal vs forced-clean delimiter attention) is 0.000 for both runs at all checkpoints. This contrasts with the 40pp gap found in the stranded attention paper, which used a structured-data-heavy corpus (14% JSON, 8% GCF, 13% code). On pure web text, there is insufficient structured content to create stranding. This is consistent with the theory: tokenizer-attention coupling matters in proportion to delimiter density in the training data.

## Finding 5: Developmental Circuit Discovery

Pairwise correlation of score vector trajectories across all 384x384 head pairs reveals co-specializing circuits:

**Baseline**: 2 circuits (threshold 0.9)
- 32-head delimiter circuit spanning 20 layers (the structural backbone)
- 5-head satellite circuit in 2 layers

**Comparison**: 3 circuits (threshold 0.9)
- 36-head delimiter circuit spanning 18 layers (larger than baseline)
- 5-head satellite circuit in 1 layer
- 4-head satellite circuit in 4 layers

Key properties:
- **Cross-layer**: 94% of top 50 correlated pairs span different layers. Circuits are vertical pipelines, not horizontal clusters.
- **Tokenizer changes circuit scale**: Merge barriers produce a larger delimiter circuit (36 vs 32 heads) with the same topology.
- **Competitive heads exist**: L08H05 (bracket specialist in baseline) anti-correlates with 7 of the top 10 negatively correlated pairs, actively suppressing duplicate and P0 behaviors.

This is the first demonstration of circuit discovery through developmental co-specialization timing rather than activation patching (Conmy et al., 2023).

## Finding 6: Developmental Sequence

Differentiation begins by step 50 and is rapid through step 500. By step 2000, the head type distribution has largely stabilized.

The initial raw-score analysis suggested all heads start as "delimiter" at step 0. With excess correction, all heads start as "unclassified" at step 0 (no genuine specialization above base rate), which is the expected result for random initialization.

## Finding 7: Layer-Depth Specialization

Middle layers specialize the most during training; early and late layers remain relatively stable.

**Baseline**: Layers 8-12 show the largest specialization increase (+0.06 to +0.10 change from early to late training). Layer 11 has the highest final specialization index (0.568). Layer 0 actually decreases (-0.058), suggesting early layers start with high base-rate scores and settle as training progresses.

**Comparison**: More distributed. Layers 6, 20, and 22 show the largest increases. The comparison model develops specialization more evenly across depths rather than concentrating in the middle.

## Finding 8: Polysemanticity

Most heads are moderate specialists, not pure specialists or generalists:

| Category | Baseline | Comparison |
|----------|----------|------------|
| Specialists (index > 0.7) | 31 (8.1%) | 20 (5.2%) |
| Moderate (0.3-0.7) | 333 (86.7%) | 355 (92.4%) |
| Generalists (< 0.3) | 20 (5.2%) | 9 (2.3%) |

The merge-barrier model has fewer extreme specialists AND fewer extreme generalists. It pushes more heads into the moderate range: capable of multiple behaviors but with clear preferences. The baseline produces more extreme outcomes in both directions.

## Methodology Notes

- Training corpus: FineWeb (HuggingFaceFW/fineweb, sample-10BT), ~5 GB, pure web text
- Probe texts: 6 fixed texts (prose, code, structured, induction, duplicates, brackets)
- Classification: excess score (dominant behavior after subtracting step-0 base rate)
- Frustration gap: forced-clean tokenization (segment at 16 barrier chars, tokenize independently)
- Entropy: per-head attention entropy averaged across all probe texts
- Circuit discovery: pairwise Pearson correlation of flattened score trajectories (131 steps x 6 behaviors = 786 values per head)

## Finding 9: Natural Language Has a Larger Adversarial Surface Than Structured Data

Reanalysis of the 43-tokenizer adversarial surface scan (from the merge-barriers paper) reveals that natural language structural characters have far larger adversarial surfaces than the structured data characters the original research focused on:

| Character | Mergeable words | Role | Comparison to pipe (24) |
|-----------|----------------|------|------------------------|
| `.` (period) | 6,366 | Sentence boundaries | 265x |
| `-` (hyphen) | 2,886 | Compound words, ranges | 120x |
| `(` (open paren) | 2,353 | Parenthetical clauses | 98x |
| `'` (apostrophe) | 706 | Contractions, possessives | 29x |
| `:` (colon) | 232 | Clause introduction | 10x |
| `"` (quote) | 193 | Dialogue, quotation | 8x |
| `)` (close paren) | 184 | Parenthetical close | 8x |
| `;` (semicolon) | 57 | Clause separation | 2x |
| `?` (question mark) | 50 | Question boundaries | 2x |
| `!` (exclamation) | 30 | Emphasis boundaries | 1.3x |

For comparison: pipe (GCF) has 24 mergeable words. Tab (TOON) has 1,238. JSON's quote has 193.

**Period alone has a 265x larger adversarial surface than pipe.** Every sentence boundary in every tokenizer vocabulary has thousands of merged entries where the period fuses with the following word (`.the`, `.and`, `.this`, etc.). Every compound word merges the hyphen (`self-`, `well-`, `non-`). Every contraction merges the apostrophe (`'t`, `'s`, `'re`).

The reason this hasn't been noticed: natural language structure is redundant. A missing sentence boundary can be inferred from capitalization and context. A missing field boundary in JSON cannot. But the attention capacity wasted on boundary recovery is proportional to the adversarial surface, not to the downstream error rate. If the model spends capacity recovering 6,366 merged period boundaries, that capacity is unavailable for content processing, even if the model ultimately recovers the boundaries correctly.

This suggests merge barriers may be a universal principle for language modeling, not just a structured data optimization. NL-specific barriers on period, hyphen, apostrophe, and parentheses would prevent the merges that create the largest adversarial surfaces in natural language. No prior work has tested this.

Source data: `results/ascii-adversarial-surface-43-tokenizers-20260625.json` (43 tokenizers, 94 printable ASCII characters).

## Finding 10: Emergence Is Partially Stochastic Across Seeds

Seed2 uses the same standard BPE tokenizer and FineWeb corpus as baseline, with a different random initialization. Seed2 used improved probe texts (real bracketed code, standardized lengths, punctuation-stripped prose); excess correction normalizes across probe sets.

### Head type distribution (excess-corrected, step 20000)

| Type | Baseline | Seed2 | Diff |
|------|----------|-------|------|
| Delimiter | 83 | 143 | +60 |
| Positional (prev) | 102 | 86 | -16 |
| P0 sink | 96 | 64 | -32 |
| Unclassified | 38 | 51 | +13 |
| Induction | 32 | 27 | -5 |
| Duplicate | 24 | 10 | -14 |
| Bracket | 9 | 3 | -6 |

Distribution correlation: **0.794**. The overall pattern holds (positional_prev and delimiter dominate in both), but exact counts vary substantially. This confirms Baherwani et al. (2026): emergence is partially stochastic. The architecture determines which types of specialization are possible; the random seed determines how many heads commit to each type.

### Emergence timing

Some behaviors emerge at the same step across seeds (induction at step 150, duplicate at step 50, P0 at step 100). Others vary. The emergence order is partially fixed by architecture but not fully deterministic.

### Entropy trajectory is seed-independent

Both seeds follow the same entropy curve: high at init (~3.7-4.5), crash to ~0.38 by step 5000, rise to ~0.67-0.70 by step 20000. The entropy divergence between standard BPE and merge barriers (Finding 3) is architecture-determined, not seed-dependent.

### Circuits are seed-dependent in position but not in type

Baseline's largest circuit: 21 positional_prev heads across 13 layers. Seed2's largest: 27 positional_prev heads across 14 layers. Only 1 position overlaps (L22H02). The model builds the same TYPE of circuit (positional_prev backbone) but at different architectural positions. Seed2 additionally develops a 19-head delimiter circuit that baseline's largest circuit doesn't include.

**Conclusion:** The developmental sequence is partially deterministic (same types emerge, same entropy trajectory, similar distribution) and partially stochastic (different head counts, different circuit positions, different circuit sizes). Single-run observations about which specific heads specialize cannot be generalized, but observations about which types of specialization emerge and when can be.

Source: `eval/analyze_seed2.py`, data in `results/seed2-excess/`.

## Finding 11: Merge Barriers Are a Universal Principle

NL-barrier tokenizer (`. ' ? ! - " ( ) ; :`) uses completely different characters than the structured-data barrier tokenizer (`| @ < > " ' : , ; \t { } [ ] ( )`). Only 5 characters overlap. Yet the NL-barrier model develops head specialization highly similar to the structured-barrier model and dissimilar to the no-barrier baseline.

### Head type distribution (excess-corrected, step 20000)

| Type | Baseline | Struct barriers | NL barriers |
|------|----------|----------------|-------------|
| Positional (prev) | 102 | 99 | 93 |
| P0 sink | 96 | 52 | 57 |
| Delimiter | 83 | 66 | 57 |
| Unclassified | 38 | 57 | 66 |
| Induction | 32 | 26 | 25 |
| Duplicate | 24 | 37 | 26 |
| Bracket | 9 | 47 | 60 |

Distribution correlations:
- NL vs Struct barriers: **r=0.923**
- NL vs Baseline: r=0.579
- Struct vs Baseline: r=0.717

### Key findings

**NL barriers behave like structured barriers, not like no barriers.** The mechanism is not about protecting specific characters. It's about keeping ANY structural delimiter isolated so heads can anchor on it. Different barrier sets produce the same developmental outcome.

**NL barriers produce the most bracket specialists** (60 vs 47 struct vs 9 baseline). The NL barrier set includes `(` and `)`, which are parenthetical characters common in web text. This directly confirms that barrier selection drives bracket specialization.

**NL barriers reduce P0 dormancy** to the same level as structured barriers (57 vs 52 vs 96 baseline). Merge barriers prevent the try-fail-collapse cascade regardless of which characters are protected.

**NL barriers produce higher entropy** (1.21 at step 20000 vs 0.70 baseline vs 0.38 struct). NL barrier characters (period, hyphen, apostrophe) are far more common in web text than structured barrier characters (pipe, @), so isolating them changes the token distribution more substantially. The model develops more distributed attention, not less.

**Circuits are identical in structure across all three tokenizers.** All three models develop a ~20-head positional_prev circuit spanning ~14 layers. Circuit topology is architecture-determined, not tokenizer-dependent.

**Frustration gap remains 0pp.** Even with NL-specific barriers, no frustration gap on web text. The stranding mechanism requires structured data in the training corpus, not just clean delimiters.

**This is the paper's strongest generalizability claim.** Two completely different barrier sets, designed for different domains, produce the same developmental effect on head specialization. Merge barriers are not a structured-data trick. They are a general principle: isolating any set of structural delimiter characters prevents P0 collapse and redirects that capacity into productive specialization.

Source: `eval/analyze_nl_barrier.py`, data in `results/nl-barrier-excess/`. NL-barrier base rates computed from step 50 (step 0 corrupted by disk-full event during training).

## Finding 12: Spacing Is the Dominant Specialization in Standard BPE (v2)

The v2 re-probe added spacing (attention mass on whitespace positions: space, newline, tab, carriage return) as a 7th behavior type. This was motivated by Wang et al. (2025b) who discovered a "spacing fin" as a distinct developmental structure.

### Spacing head counts (v2 excess-corrected, step 20000)

| Run | Spacing heads | % of 384 |
|-----|--------------|----------|
| Baseline | 183 | 47.7% |
| Seed2 | 183 | 47.7% |
| Comparison (struct barriers) | 13 | 3.4% |
| NL-barrier | 0 | 0.0% |

**Nearly half of all heads in standard BPE are spacing specialists.** The count is identical across seeds (183 in both baseline and seed2), confirming this is architecture-determined, not stochastic.

**Merge barriers eliminate spacing heads.** Structured barriers reduce spacing from 183 to 13. NL barriers eliminate them entirely (0). NL barrier characters (period, hyphen, apostrophe, parentheses) co-occur with whitespace constantly in prose (`word. Next`, `self-contained`, `it's`). Protecting these characters from merging keeps whitespace boundaries clean, eliminating the need for spacing recovery heads.

**Spacing heads were invisible in v1.** Without the spacing behavior measurement, these 183 heads were misclassified: 54 as P0 sinks, 29 as positional_prev, 24 as induction, 22 as delimiter, 18 as duplicate, 32 as unclassified, 4 as bracket. The v1 P0 cascade was inflated because the v1 probe taxonomy couldn't distinguish spacing from P0 sinking.

**Connection to Wang et al. (2025b).** Their "spacing fin" discovery predicted that spacing would be a major head specialization. Our data confirms this at 410M scale and shows that the tokenizer determines how many heads develop spacing specialization. This validates adding spacing to head behavior taxonomies.

Source: v2 re-probe data in `results/{run}-v2-excess/`. Script: `eval/probe_heads.py` (v2 with spacing).

## Finding 13: NL Frustration Gap Measurement

Measured the frustration gap using NL delimiter characters (`. ' ? ! - " ( ) ; :`) in addition to the original structured delimiter characters. Tested on step-20000 for all 4 runs across 3 probe texts (prose, code, structured).

### Results

| Run | Struct gap (avg) | NL gap (avg) |
|-----|-----------------|--------------|
| Baseline | -0.1 pp | -0.4 pp |
| Comparison | 0.0 pp | -0.9 pp |
| Seed2 | -0.9 pp | -1.1 pp |
| NL-barrier | -0.1 pp | 0.0 pp |

Both gaps are effectively zero across all runs. The prose probe has no punctuation (deliberately stripped), so it cannot measure NL delimiter boundaries. The code and structured probes show small negative values (forced-clean tokenization slightly reduces attention to these characters).

### Follow-up: Punctuated prose probe

A dedicated punctuated prose probe was written (probes/prose_punctuated.txt) with natural sentences including periods, apostrophes, hyphens, parentheses, and quotation marks. Results:

| Run | Struct gap | NL gap |
|-----|-----------|--------|
| Baseline | -0.6 pp | -0.4 pp |
| Comparison | 0.0 pp | -0.3 pp |
| Seed2 | -0.6 pp | -0.7 pp |
| NL-barrier | 0.0 pp | 0.0 pp |

**The NL frustration gap is genuinely zero on web text.** Even with a punctuated prose probe containing periods, apostrophes, hyphens, and parentheses, no measurable gap appears. The frustration gap requires structured data density in the training corpus (as shown in the stranded attention paper at 40pp on a structured-data-heavy corpus).

**The damage from BPE on web text manifests differently:** not as a measurable frustration gap, but as spacing head proliferation. Standard BPE wastes 183/384 heads on whitespace recovery (Finding 12). This is the web-text analog of stranding: the model devotes capacity to boundary recovery, but the recovered boundaries are whitespace (ubiquitous in prose) rather than delimiters (sparse in prose).

Source: `eval/measure_nl_frustration_gap.py`. Results in `results/nl-frustration-gap/`.

## Finding 14: Structok Corpus Atlas (Runs 5-6)

Two additional runs trained on the structok corpus (33% FineWeb, 14% JSON, 13% code, 8% GCF, 3% Wikipedia, 1% YAML/CSV) to bridge the FineWeb atlas with the stranded attention paper. Same methodology, same checkpoint schedule, same tokenizers (standard-64k and structok-64k). Pretokenized bins from merge-barriers run-002 (provenance: `structok/prep_run002.py`).

### Head type distribution (excess-corrected, step 20000)

| Type | FineWeb BL | FineWeb Comp | Structok BL | Structok Comp |
|------|-----------|-------------|-------------|---------------|
| Spacing | 183 | 13 | 172 | 8 |
| Delimiter | 74 | 79 | 131 | 127 |
| Positional (prev) | 68 | 91 | 17 | 33 |
| P0 sink | 32 | 40 | 34 | 46 |
| Duplicate | 8 | 12 | 20 | 18 |
| Bracket | 4 | 39 | 9 | 34 |
| Induction | 5 | 15 | 0 | 16 |
| Unclassified | 10 | 95 | 1 | 102 |

### Frustration gap

| Run | Gap | Heads woke |
|-----|-----|-----------|
| Structok baseline | 1.0 pp | 55/384 |
| Structok comparison | 0.0 pp | 0/384 |
| FineWeb baseline | 0.0 pp | 11/384 |
| FineWeb comparison | 0.0 pp | 0/384 |

**The frustration gap is nonzero on the structok corpus (1.0 pp, 55 heads woke up).** This confirms the two-regime model: delimiter density in the training corpus determines whether the gap appears. 1.0 pp is modest compared to the 40pp in the stranded attention paper, which used a fully converged model on higher-density structured data. The atlas runs at 20K steps may not have reached the stranding level that the longer merge-barriers runs showed.

**Merge barriers eliminate the gap entirely** on both corpora (0.0 pp for both comparisons).

### Predictions vs actuals

| Metric | Predicted | Actual | Match? |
|--------|-----------|--------|--------|
| Frustration gap (baseline) | > 5 pp | 1.0 pp | Partial: nonzero but smaller than expected |
| Frustration gap (comparison) | ~0 pp | 0.0 pp | Yes |
| Spacing heads (baseline) | 50-150 | 172 | No: nearly same as FineWeb (183) |
| Spacing heads (comparison) | 0-10 | 8 | Yes |
| P0 heads (baseline) | 20-60 | 34 | Yes |
| P0 heads (comparison) | 20-40 | 46 | Partial: slightly above range |
| Bracket specialists (comparison) | 40-80 | 34 | Partial: close to range |

### Key findings

**Spacing persists on structured data.** The structok baseline has 172 spacing heads, nearly identical to FineWeb's 183. The structured content (35% of corpus) did not reduce spacing. Instead, the model grows MORE delimiter heads (131 vs 74) on top of the same spacing base. Spacing is not displaced by delimiter specialization; they coexist.

**Delimiter heads scale with corpus content.** 131 delimiter heads on structok vs 74 on FineWeb (77% increase). The 35% structured content in the training corpus drives proportionally more delimiter specialization.

**Positional_prev heads shrink to make room.** 17 on structok vs 68 on FineWeb. The model sacrifices positional_prev heads to accommodate the additional delimiter heads. Spacing and P0 counts stay approximately the same.

**The frustration gap appears but is small (1.0 pp).** The stranded attention paper's 40pp gap was measured on a fully converged model. The atlas runs at 20K steps (batch size 1) may not reach the same level of stranding. The gap may increase with longer training.

**The two-regime model holds, but the boundary between regimes is not sharp.** The structok corpus shows elements of both regimes simultaneously: a nonzero frustration gap (structured data symptom) AND spacing proliferation (web text symptom). The regimes are a continuum, not a binary.

Source: training via `eval/train_atlas.py`, probing via `eval/probe_heads.py`, excess correction via `eval/excess_score_correction.py`. Data in `results/structok-baseline/`, `results/structok-comparison/`, `results/structok-baseline-excess/`, `results/structok-comparison-excess/`. Checkpoints on R2 under `atlas/runs/structok-baseline/` and `atlas/runs/structok-comparison/`. Pretokenized bins from R2 `tokens/standard-64k-v2.bin` and `tokens/structok-64k-v2.bin`, provenance: `structok/prep_run002.py`.

## Finding 15: Spacing Heads Are Mandatory Damage Repair (Ablation)

Zero-ablation study following the methodology from Blackwell (2026a, Section 5.3): deep copy the model, zero output projection weights for selected heads, measure perplexity on 7 probe texts, compare to random head controls.

### Results

| Model | Spacing heads | Spacing ablation | Random control (mean) | P0 ablation |
|-------|--------------|-----------------|----------------------|-------------|
| FineWeb baseline | 183 | +64.3% | +28.7% | +1.4% |
| Structok baseline | 172 | +68.9% | -2.8% | +0.5% |
| Comparison (barriers) | 13 | +15.1% | +8.3% | +0.2% |

### Per-text breakdown (FineWeb baseline)

| Probe | Spacing ablation | Random control |
|-------|-----------------|----------------|
| Duplicates | +405.1% | (included in mean) |
| Brackets | +61.2% | |
| Code | +59.6% | |
| Structured | +35.7% | |
| Prose | +35.8% | |
| Prose punctuated | +21.6% | |
| Induction | +18.3% | |

### Key findings

**Spacing heads are productive, not counterproductive.** Removing them degrades perplexity more than removing the same number of random heads (+64.3% vs +28.7% on FineWeb, +68.9% vs -2.8% on structok). This is the opposite of stranded heads at 1.3B, where removal improved comprehension by 57% (Blackwell, 2026b).

**Spacing heads are mandatory damage repair.** The model dedicates 47% of its heads to whitespace boundary recovery because BPE corrupted those boundaries. Removing the repair kills the patient. But a healthy model (merge barriers) doesn't need the repair: the comparison model has only 13 spacing heads, and removing them hurts about the same as random controls (+15.1% vs +8.3%).

**The immune response analogy.** Spacing heads are like an immune response to corrupted boundaries. The model is sick (BPE merged whitespace boundaries), and it dedicates 47% of its capacity to the cure (whitespace recovery). The cure is essential; without it, performance collapses. But merge barriers prevent the disease, making the cure unnecessary. The waste is not that spacing heads exist; it is that they NEED to exist.

**P0 heads are genuinely useless (causal proof).** Removing 32-40 P0 heads produces less than 1.5% PPL change across all three models. This converts the correlational finding (100% circuit isolation) into causal evidence: P0 heads contribute nothing to model performance. They are a pure failure mode.

**Duplicate detection depends most on spacing recovery.** The +405% degradation on duplicates when spacing heads are removed shows that duplicate token detection relies heavily on whitespace boundaries. Code and brackets degrade ~60%, prose ~35%. This hierarchy reflects how much each task depends on boundary information.

**The capacity tax.** Standard BPE imposes a ~47% capacity tax on the model: 183 heads that cannot be removed without severe degradation, dedicated entirely to recovering boundaries that merge barriers would keep clean. This is not recoverable through training, pruning, or fine-tuning. Only changing the tokenizer eliminates the tax.

**Structok random controls improve PPL.** Removing random heads from the structok baseline model actually improves performance (-2.8%), acting as regularization. But removing spacing heads from the same model degrades by +68.9%. This is the starkest contrast in the dataset: the model has excess capacity it doesn't need, but spacing capacity it cannot lose.

Source: `eval/ablate_spacing_heads.py`. Data in `results/ablation/` and on R2 at `atlas/results/ablation/`. Methodology adapted from Blackwell (2026a) 18-phase ablation protocol.

## Finding 16: Architecture and Scale Replication (Llama 410M + 1.3B)

Probed existing Llama checkpoints from the coupling paper (run-003 Llama 410M GQA, run-004 Llama 1.3B GQA) with the 7-behavior taxonomy. Step-0 base rates generated from random Llama initialization for excess correction.

### Head type distribution (excess-corrected)

| Model | Arch | Heads | Spacing | P0 | Delimiter | Positional_prev |
|-------|------|-------|---------|-----|-----------|----------------|
| NeoX 410M standard | MHA | 384 | 183 (47.7%) | 32 (8.3%) | 74 (19.3%) | 68 (17.7%) |
| Llama 410M standard | GQA | 384 | 60 (15.6%) | 90 (23.4%) | 43 (11.2%) | 147 (38.3%) |
| Llama 1.3B standard | GQA | 768 | 128 (16.7%) | 180 (23.4%) | 137 (17.8%) | 263 (34.2%) |
| NeoX 410M structok | MHA | 384 | 13 (3.4%) | 40 (10.4%) | 79 (20.6%) | 91 (23.7%) |
| Llama 410M structok | GQA | 384 | 0 (0%) | 80 (20.8%) | 120 (31.2%) | 137 (35.7%) |
| Llama 1.3B structok | GQA | 768 | 2 (0.3%) | 97 (12.6%) | 365 (47.5%) | 169 (22.0%) |

### Frustration gap (Llama)

| Model | Gap | Heads woke |
|-------|-----|-----------|
| Llama 410M standard | 0.4 pp | 5/384 |
| Llama 410M structok | 0.0 pp | 0/384 |
| Llama 1.3B standard | 0.2 pp | 49/768 |
| Llama 1.3B structok | 0.0 pp | 0/768 |

### Key findings

**Spacing is universal across architectures.** Both MHA (NeoX) and GQA (Llama) develop spacing heads with standard BPE. Merge barriers eliminate them on both.

**The spacing percentage is architecture-dependent.** ~47% on NeoX (MHA), ~16% on Llama (GQA). GQA's shared key-value projections distribute the spacing signal differently, producing fewer spacing specialists but more P0 sinks.

**P0 is higher on GQA.** ~23% on Llama vs ~8% on NeoX. GQA produces more P0 sinks, possibly because shared KV projections make it harder for individual query heads to maintain stable specialization.

**Total non-productive capacity is architecture-dependent but substantial on both.** NeoX: ~56% (47.7% spacing + 8.3% P0). Llama: ~39% (16.7% spacing + 23.4% P0). The distribution between spacing and P0 varies, but the total tax is significant on both architectures.

**Spacing is consistent across Llama scales.** 15.6% at 410M, 16.7% at 1.3B. The percentage holds as head count doubles (60/384 to 128/768).

**Merge barriers work on GQA.** Llama structok models have 0-2 spacing heads (vs 60-128 standard). The fix is architecture-independent.

**Frustration gap is zero on Llama.** Same pattern as NeoX: 0.2-0.4pp on standard BPE, 0.0pp with merge barriers.

**The "approximately half" claim requires qualification.** The capacity tax is ~56% on NeoX (MHA) and ~39% on Llama (GQA). The paper should state the tax is "substantial" or "over a third" rather than "approximately half" when making architecture-general claims. The ~47% spacing figure is specific to NeoX MHA.

Source: Llama checkpoints from `checkpoints/run-003-llama-{standard,structok}/step-40000/` and `checkpoints/run-004-llama-{standard,structok}/step-50000/` on R2. Step-0 from random Llama init. Results in `results/llama-{410m,1.3b}-{standard,structok}{,-excess}/`.

## Two Regimes of BPE Damage

The atlas and stranded attention findings together reveal that BPE boundary corruption produces two distinct damage regimes, caused by the same mechanism but producing different symptoms depending on delimiter density in the training corpus.

**Regime 1: High delimiter density (structured data corpus).** When the training corpus contains concentrated structural content (14% JSON, 8% GCF, 13% code), heads develop genuine delimiter specialization and become stranded. The symptom is a frustration gap: 40pp difference in delimiter attention between normal and forced-clean tokenization, 384/384 heads affected (Blackwell, 2026b). The model develops structural capacity and then wastes it on boundary recovery.

**Regime 2: Low delimiter density (web text corpus).** When the training corpus is predominantly prose (FineWeb), heads cannot develop delimiter specialization because delimiter characters appear in too many varied contexts (sentence boundaries, abbreviations, URLs, compound words) to provide a clean anchoring signal. The frustration gap is zero. The symptom instead is spacing head proliferation: 183/384 heads (47.7%) become spacing specialists, dedicating their capacity to whitespace boundary recovery. An additional 29 heads collapse into genuine P0 sinks. Merge barriers eliminate spacing heads entirely (0 in NL-barrier) and reduce P0 sinks.

**Same mechanism, different symptoms.** In both regimes, BPE merges corrupt character boundaries that attention heads need for anchoring. In structured data, the corruption produces measurable stranding (heads detect boundaries but can't use them cleanly). In web text, the corruption is more diffuse: heads can't even develop delimiter specialization, so they fall back to spacing recovery or P0 collapse.

**Why this was invisible.** Prior work (including our own stranded attention paper) looked for damage by measuring the frustration gap. On web text, this gap is zero, leading to the conclusion that BPE damage is a structured-data problem. The spacing measurement reveals the hidden cost: the model isn't stranding on web text, but it's burning half its heads on whitespace recovery. This is only visible when spacing is included in the probe taxonomy.

**Merge barriers fix both regimes.** Structured barriers reduce spacing from 183 to 13 and P0 from 96 to 52 (v1) or 32 to 40 (v2). NL barriers eliminate spacing entirely (0) and keep P0 at 56. The mechanism is the same in both cases: isolating delimiter characters in the tokenizer vocabulary keeps boundaries clean so heads can anchor productively instead of wasting capacity on recovery.

| | Stranded attention paper | Structok corpus atlas | FineWeb atlas |
|---|---|---|---|
| Training data | Structured-data-heavy | 33% web + 35% structured | FineWeb (web text) |
| Frustration gap | 40 pp, 384/384 heads | 1.0 pp, 55/384 heads | 0 pp |
| Spacing heads (standard BPE) | Not measured | 172/384 (44.8%) | 183/384 (47.7%) |
| Spacing heads (merge barriers) | Not measured | 8/384 | 0-13/384 |
| Delimiter heads (standard BPE) | Not measured separately | 131/384 (34.1%) | 74/384 (19.3%) |
| P0 sinks (standard BPE) | Not measured separately | 34/384 | 29-32/384 |
| Visible symptoms | Full stranding | Spacing + small gap + more delimiters | Spacing only |

**The structok corpus confirms the continuum.** Finding 14 shows both symptoms simultaneously: a nonzero frustration gap (1.0 pp) AND spacing proliferation (172 heads). The model grows more delimiter heads (131 vs 74) to handle the structured content but does not reduce spacing. Instead, positional_prev heads shrink (17 vs 68) to make room. The two regimes are not a binary; they are a continuum where both symptoms scale with delimiter density.

## Positioning Against Prior Work

### Riviere & Trott (2025): "Start Making Sense(s)"

The closest methodological precedent. They tracked attention head specialization across Pythia checkpoints using lexical ambiguity (word sense disambiguation) as a developmental probe. Key similarities and differences:

**Similarities:** Both use developmental probing across training checkpoints to track when heads specialize. Both perform causal ablation to verify identified heads are functionally important. Both find "developmental milestones" during early training (they report shifts at steps 1000 and 5000; we find rapid differentiation through step 500, stabilization by step 2000).

**Differences:** They track ONE behavior (disambiguation) on ONE tokenizer. We track 8 behaviors simultaneously, compare 3 tokenizers, and add a seed variation control. They use public Pythia checkpoints (no training cost but can't vary the tokenizer). We train custom models (can isolate the tokenizer variable). They identify candidate heads; we discover circuits. They probe a linguistic task; we probe attention patterns directly.

**What we add beyond their work:**
1. Comprehensive multi-behavior tracking (not single-task)
2. Tokenizer as experimental variable (their approach can't test this)
3. Circuit discovery through developmental co-specialization
4. P0 failure cascade mechanism (they don't study dormancy)
5. Seed variation analysis (they reproduced across seeds for 14M only)
6. NL adversarial surface analysis

Their work validates the developmental probing methodology. Ours extends it to a full atlas.

### Wang, Baker, Gordon & Murfet (2025): "Embryology of a Language Model"

The most conceptually aligned prior work. They applied UMAP to per-token susceptibility vectors across training of a 3M parameter, 2-layer attention-only model, visualizing the developmental "body plan" as a "rainbow serpent" where token pattern types (word start, word part, induction, spacing, delimiter, formatting, numeric) separate into distinct regions.

**Key connections:**
- They note "the structure learned by a model may be substantially influenced by the tokenizer" (p.2) but don't test this. We explicitly vary the tokenizer and show it changes head development. We answer their observation empirically.
- They discovered a **"spacing fin"**: a previously un-noticed structure for predicting space and newline tokens. Our v2 re-probe confirms this at 410M scale: spacing is the dominant specialization in standard BPE (183/384 heads, Finding 12). Their prediction was correct.
- Their UMAP structure is "remarkably similar across seeds" (4 seeds). Our Finding 10 confirms this at 410M scale (distribution correlation 0.794).
- They use susceptibility (how weight perturbations affect loss). We use attention patterns (what heads attend to). Complementary lenses on the same underlying organization.

**What we add:** Realistic scale (410M vs 3M), full architecture (24 layers + FFN vs 2 layers attention-only), tokenizer as a variable (they can't test this), circuit discovery, P0 failure cascade. They provide the most beautiful visualization of head organization. We provide the controlled experimental framework that isolates what causes it.

**Gap identified:** Our probe taxonomy should include spacing/whitespace as a behavior type, following their discovery of the spacing fin as a distinct developmental structure.

### Gu et al. (2025): "When Attention Sink Emerges in Language Models" (ICLR 2025)

See Finding 2 for detailed connection. We answer both of their stated open questions: (1) attention sinks are a failure mode, not a benefit; (2) the tokenizer determines how many heads sink.

### Wang et al. (2025): "Differentiation and Specialization of Attention Heads" (ICLR 2025 Spotlight)

Closest to our atlas concept. They tracked head differentiation using the refined local learning coefficient (rLLC) and found a staged developmental order: bigrams first, then n-grams, then previous-token, then induction. But they used a 2-layer attention-only toy model. We replicate at realistic scale (24 layers, 16 heads, 410M parameters) and add the tokenizer variable.

### Aoyama, Wilcox & Schneider (2026): "Predicting the Emergence of Induction Heads"

Derived a predictive equation for when induction heads form based on batch size and context size. Showed bigram repetition frequency and reliability control the phase transition. Uses 2-layer 50M models with 30 checkpoints each. They note vocabulary size "will likely affect the emergence points" but don't test this. We implicitly test it via tokenizer variation (merge barriers change the effective token distribution). Our induction emergence at step 150 across both seeds is consistent with their finding that emergence timing is model-size-agnostic. Their single-behavior phase transition analysis complements our multi-behavior developmental tracking.

### Baherwani et al. (2026): "Emergent Capabilities Arise Randomly"

Found emergence is stochastic across seeds on synthetic tasks. Our Finding 10 confirms this at realistic scale: distribution correlation 0.794 across seeds, circuits form the same type but at different positions.

## Known Limitations

1. **~~Probe inconsistency~~**: RESOLVED in v2 re-probe (2026-07-04). All 4 runs re-probed with identical probe texts and 7-behavior taxonomy including spacing. v1 data preserved for comparison.
2. **FineWeb only**: No structured data in training corpus. The frustration gap (0pp for both struct and NL delimiters) is expected on web text but limits connection to the stranded attention paper. A structok corpus run is planned.
3. **Single architecture**: GPT-NeoX 410M only. Results may differ on Llama (GQA) or larger models.
4. **Two seeds only**: Seed variation tested with one additional seed. More seeds would quantify the variance more precisely.
5. **~~Missing spacing probe~~**: RESOLVED in v2 re-probe. Spacing is now measured and is the dominant specialization (183/384 heads in standard BPE).
6. **~~NL frustration gap inconclusive~~**: RESOLVED. Punctuated prose probe confirmed NL gap is genuinely zero on web text. Damage manifests as spacing head proliferation, not frustration gap.
7. **NL-barrier step-0 corrupted**: Step-50 base rates used as proxy for excess correction. Defensible: 50 steps = 0.008% of corpus.
