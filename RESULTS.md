# Atlas Results

## Experiment Summary

Two GPT-NeoX 410M models trained for 20,000 steps on FineWeb (web text corpus), differing only in tokenizer:
- **Baseline**: standard BPE (65,536 vocab)
- **Comparison**: merge-barrier BPE (65,539 vocab, 16 delimiter characters forbidden from merging)

131 checkpoints per run (step 0 through 20,000). Each checkpoint probed across 8 behavior types on 6 probe texts, plus frustration gap measurement and attention entropy. All classifications use excess scores (base-rate corrected) unless stated otherwise.

## Excess Score Methodology

At random initialization, a head's attention is roughly uniform across all positions. If 30% of positions in a probe text are delimiter characters, a random head directs ~30% of attention to delimiters by chance. That's not specialization; it's arithmetic.

Raw scores measure total attention to each behavior type. Excess scores subtract the step-0 base rate (measured from random init) to reveal genuine specialization. A head with 0.30 raw and 0.30 base rate has 0.00 excess (no specialization). A head with 0.30 raw and 0.10 base rate has 0.20 excess (genuine specialist).

Without this correction, the original probe set produced heavily inflated classifications: 172 heads appeared to be "delimiter specialists" in baseline, but only 83 showed genuine excess. The brackets probe (100% delimiter characters) inflated every head's delimiter score to 1.0. The excess methodology, adapted from the merge-barriers paper (Blackwell, 2026), removes this inflation.

## Finding 1: Head Type Distribution at Convergence

| Type | Baseline | Comparison |
|------|----------|------------|
| Positional (prev) | 102 (26.6%) | 99 (25.8%) |
| P0 sink | 96 (25.0%) | 52 (13.5%) |
| Delimiter | 83 (21.6%) | 66 (17.2%) |
| Unclassified | 38 (9.9%) | 57 (14.8%) |
| Induction | 32 (8.3%) | 26 (6.8%) |
| Duplicate | 24 (6.3%) | 37 (9.6%) |
| Bracket | 9 (2.3%) | 47 (12.2%) |

**Positional_prev is the most common genuine specialization** (102 baseline, 99 comparison), not delimiter. These heads attend to the immediately preceding token, the simplest and most universally useful pattern.

**The merge-barrier model develops 5x more bracket specialists** (47 vs 9). Clean delimiter boundaries enable bracket-level structural processing that the standard BPE model cannot develop.

**38-57 heads show no genuine specialization** above base rate (unclassified). These are truly generalist heads, not misclassified specialists.

## Finding 2: Heads Attempt Specialization, Fail, and Collapse into P0 Sinks

Attention heads don't start dormant. They become dormant after failing to specialize. Tracking the 96 baseline P0 heads backward through 131 checkpoints reveals that 35% were delimiter heads that attempted structural specialization before sinking, and 39% were unclassified heads that never found a viable specialization. Only a minority go directly to P0 from random init.

This is the same stranding mechanism described in the companion paper (Blackwell, 2026), operating at lower intensity on web text. The frustration gap is 0pp because the model doesn't develop enough structural capacity to measure a gap, but the damage is still happening: heads are dying because they can't find clean boundaries to anchor on.

P0 sink heads (excess-corrected):
- **Baseline**: 96 heads (25.0%)
- **Comparison**: 52 heads (13.5%)
- **Seed2**: 64 heads (16.7%)

Merge barriers don't just reduce the count. They convert wasted P0 capacity into productive specialization: at the 96 positions where baseline has P0 sinks, comparison has 23 delimiter heads, 22 positional_prev, 9 induction, 8 bracket. Only 17 are P0 in both. 79 heads saved from collapse.

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

## Known Limitations

1. **Probe inconsistency**: Baseline and comparison used old probes (degenerate brackets, short texts). Seed2 used improved probes. Excess correction normalizes base rates but the underlying measurements differ. Full re-probe of baseline with improved probes pending.
2. **FineWeb only**: No structured data in training corpus. The frustration gap (0pp) is expected but limits connection to the stranded attention paper. A structok corpus run is planned.
3. **Single architecture**: GPT-NeoX 410M only. Results may differ on Llama (GQA) or larger models.
4. **Two seeds only**: Seed variation tested with one additional seed. More seeds would quantify the variance more precisely.
