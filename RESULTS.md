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

## Finding 2: Merge Barriers Reduce Dormancy

P0 sink heads (excess-corrected):
- **Baseline**: 96 heads (25.0%)
- **Comparison**: 52 heads (13.5%)

The merge-barrier tokenizer nearly halves the number of heads that collapse into attention sinks. This extends Sandoval-Segura et al. (2025) by showing that dormancy is partially tokenizer-dependent, not just a fixed property of the architecture.

This effect appears even on web-heavy data (FineWeb) with minimal structured content, suggesting merge barriers have a general attention-health benefit beyond structured data processing.

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

## Known Limitations

1. **Probe quality**: The brackets probe used in data collection was degenerate (100% delimiter characters). The excess correction compensates, but improved probes have been written for future runs. Validation on step-20000 pending.
2. **FineWeb only**: No structured data in training corpus. The frustration gap finding (0pp) is expected but limits the connection to the stranded attention paper. A structok corpus run is planned.
3. **No seed variation**: Baseline and comparison use different random inits (different instances). A seed variation run (seed2) is in progress to test whether the developmental order is deterministic.
4. **Single architecture**: GPT-NeoX 410M only. Results may differ on Llama (GQA) or larger models.
