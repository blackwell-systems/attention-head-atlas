# Atlas Results

## Experiment Summary

Two GPT-NeoX 410M models trained for 20,000 steps on FineWeb (web text corpus), differing only in tokenizer:
- **Baseline**: standard BPE (65,536 vocab)
- **Comparison**: merge-barrier BPE (65,539 vocab, 16 delimiter characters forbidden from merging)

131 checkpoints per run (step 0 through 20,000). Each checkpoint probed across 8 behavior types on 6 probe texts, plus frustration gap measurement and attention entropy.

## Finding 1: Developmental Sequence

Heads specialize in a consistent order:

| Behavior | Baseline first appears | Comparison first appears |
|----------|----------------------|------------------------|
| Duplicate | Step 100 (48 heads) | Step 50 (4 heads) |
| P0 sink | Step 200 (7 heads) | Step 200 (1 head) |
| Bracket | Step 150 (1 head) | Step 250 (1 head) |
| Positional (prev) | Step 350 (1 head) | Step 250 (7 heads) |
| Induction | Step 300 (2 heads) | Step 300 (2 heads) |

All 384 heads start classified as "delimiter" at step 0 (random init, uniform attention, delimiter wins by base rate). Differentiation begins by step 50 and is rapid through step 500.

## Finding 2: Merge Barriers Reduce Dormancy 5x

At step 20,000:
- **Baseline**: 27 P0 sink (dormant) heads
- **Comparison**: 5 P0 sink heads

The merge-barrier tokenizer prevents heads from collapsing into attention sinks. This extends Sandoval-Segura et al. (2025) by showing that dormancy is partially tokenizer-dependent, not just a fixed property of the architecture.

This effect appears even on web-heavy data (FineWeb) with minimal structured content, suggesting merge barriers have a general attention-health benefit beyond structured data processing.

## Finding 3: Entropy Divergence

Baseline attention entropy rises from 0.35 back to 0.70 in late training (steps 10,000-20,000). Comparison stays flat at ~0.35. The standard BPE model's attention becomes more diffuse over time; the merge-barrier model maintains focused attention throughout.

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

## Finding 6: Head Type Distribution at Convergence

| Type | Baseline | Comparison |
|------|----------|------------|
| Delimiter | 172 (44.8%) | 163 (42.4%) |
| Duplicate | 146 (38.0%) | 177 (46.1%) |
| Positional (prev) | 24 (6.3%) | 28 (7.3%) |
| P0 sink | 27 (7.0%) | 5 (1.3%) |
| Bracket | 12 (3.1%) | 9 (2.3%) |
| Induction | 3 (0.8%) | 1 (0.3%) |
| Unclassified | 0 | 1 |

Note: "Delimiter" classification is based on dominant behavior score on structured probe text, not the excess-score methodology from the merge-barriers paper. These numbers are not directly comparable to the 105 vs 23 delimiter heads reported there.

## Methodology Notes

- Training corpus: FineWeb (HuggingFaceFW/fineweb, sample-10BT), ~5 GB, pure web text
- Probe texts: 6 fixed texts (prose, code, structured, induction, duplicates, brackets)
- Classification: dominant behavior type from continuous score vector across all 6 probes
- Frustration gap: forced-clean tokenization (segment at 16 barrier chars, tokenize each independently)
- Entropy: per-head attention entropy averaged across all probe texts
- Circuit discovery: pairwise Pearson correlation of flattened score trajectories (131 steps x 6 behaviors = 786 values per head)

## Finding 7: Layer-Depth Specialization

Middle layers specialize the most during training; early and late layers remain relatively stable.

**Baseline**: Layers 8-12 show the largest specialization increase (+0.06 to +0.10 change from early to late training). Layer 11 has the highest final specialization index (0.568). Layer 0 actually decreases (-0.058), suggesting early layers start specialized and become more generalist.

**Comparison**: More distributed. Layers 6, 20, and 22 show the largest increases. The comparison model develops specialization more evenly across depths rather than concentrating in the middle.

This is consistent with the "early=syntax, mid=semantics, late=task" hypothesis: middle layers handle the most complex processing and therefore develop the strongest head specialization.

## Finding 8: Polysemanticity

Most heads are moderate specialists, not pure specialists or generalists:

| Category | Baseline | Comparison |
|----------|----------|------------|
| Specialists (index > 0.7) | 31 (8.1%) | 20 (5.2%) |
| Moderate (0.3-0.7) | 333 (86.7%) | 355 (92.4%) |
| Generalists (< 0.3) | 20 (5.2%) | 9 (2.3%) |

The merge-barrier model has fewer extreme specialists AND fewer extreme generalists. It pushes more heads into the moderate range: capable of multiple behaviors but with clear preferences. The baseline produces more extreme outcomes in both directions.

Specialist-heavy layers in baseline: L3, L10, L11 (concentrated in middle). Comparison: L12 only. The baseline concentrates its specialists; the comparison distributes them.

Generalist-heavy layers in baseline: L16. Comparison: none. The merge-barrier model eliminates generalist clusters entirely.
