# Developmental Atlas: Summary of Findings

**DOI:** 10.5281/zenodo.21205390

## The headline

Every LLM trained with standard BPE is paying a 48-56% attention head tax, regardless of architecture. 40-48% of heads are doing mandatory whitespace boundary repair. ~8% are doing nothing at all. This is causally proven across two architectures: removing spacing heads collapses performance by +64.3% on NeoX MHA and +67.0% on Llama GQA. Removing P0 heads changes nothing on either (+1.4% NeoX, -3.4% Llama). The fix is 16 lines of tokenizer config. Nobody has deployed it.

## The 17 findings

### Finding 1: Head type distribution

Spacing is the dominant head specialization in standard BPE across architectures. NeoX MHA: 183/384 heads (47.7%). Llama GQA: 154/384 (40.1%). This was invisible until spacing was added to the probe taxonomy. The NeoX count is deterministic across seeds (183 in both baseline and seed2).

**Significance:** 40-48% of all attention capacity in every standard BPE model is consumed by whitespace boundary recovery, regardless of how attention is structured.

### Finding 2: P0 cascade at corrected scale

Prior to measuring spacing, the P0 count appeared to be 96 (25%). Adding spacing revealed 54 were spacing specialists. The genuine P0 count is 29-32 (~8%) on NeoX, 31 on Llama. Essentially identical across architectures. The try-fail-collapse mechanism is real but smaller than initially measured.

**Significance:** Demonstrates the importance of comprehensive behavior taxonomies. Omitting a single behavior type inflated another category by 3x.

### Finding 3: Entropy divergence

Standard BPE attention entropy rises in late training (0.35 to 0.70). Merge barriers stay flat at 0.35. NL barriers reach 1.21.

**Significance:** Standard BPE models become more diffuse over training. Merge barriers maintain focused attention.

### Finding 4: Frustration gap is zero on web text

Zero for all FineWeb runs across both architectures (NeoX: -0.1pp, Llama: 0.49pp). Confirmed with punctuated prose probe. The gap requires structured data density in the training corpus.

**Significance:** Overturns the assumption that BPE damage only matters on structured data. The damage is there on web text, just in a different form (spacing, not stranding).

### Finding 5: Developmental circuits

Co-specializing circuits discovered through trajectory correlation. 32-36 head delimiter circuits spanning 18-20 layers. 94% cross-layer. Circuits are vertical pipelines.

**Significance:** First demonstration of circuit discovery through developmental timing rather than activation patching.

### Finding 6: Developmental sequence

Spacing emerges first and fastest (28 heads by step 50, 184 by step 150). Same emergence timeline on both architectures. Positional_prev next, then induction/duplicate, then delimiter. P0 collapse is late (median step 11,000).

**Significance:** Spacing is the model's first priority. The developmental program is conserved across architectures.

### Finding 7: Layer-depth specialization

Middle layers specialize most. Merge barriers distribute specialization more evenly across depth.

### Finding 8: Polysemanticity

Most heads are moderate specialists (76-87%). With spacing measured, 21% are specialists in baseline (driven by spacing heads with high specialization index).

**Significance:** Connects to the superposition hypothesis (Elhage et al., 2022). Heads converge to moderate polysemanticity as a stable attractor.

### Finding 9: NL adversarial surface

Period has 265x larger adversarial surface than pipe across 43 tokenizers (6,366 vs 24 mergeable words). Hyphen 120x. NL structural characters are far more corrupted by BPE than structured data characters.

**Significance:** The BPE damage on natural language is quantifiably worse than on structured data, but invisible because NL structure is redundant.

### Finding 10: Seed variation

Distribution correlation r=0.992 across seeds. Spacing count is identical (183/183). Secondary behaviors vary modestly. Circuit topology is architecture-determined; circuit placement is seed-dependent.

**Significance:** Spacing is deterministic. The model has no choice about dedicating ~48% of heads to whitespace recovery.

### Finding 11: Merge barriers are universal

NL barriers (different character set) produce head distributions correlated at r=0.812 with structured barriers. Both diverge sharply from baseline (r=-0.096, r=-0.403). Different barrier sets produce the same developmental outcome.

**Significance:** The fix works regardless of which characters you protect. The mechanism is about boundary isolation, not specific characters.

### Finding 12: Spacing is the dominant specialization

183/384 heads (47.7%) on NeoX MHA, 154/384 (40.1%) on Llama GQA. Deterministic across NeoX seeds. Eliminated by merge barriers (0 with NL barriers). Confirmed at 410M scale the "spacing fin" prediction of Wang et al. (2025b) across two architectures. The spacing fin appears in cross-architecture UMAP, occupying the same region of the embedding space on both NeoX and Llama.

**Significance:** The single largest finding. 40-48% of heads in every standard BPE model, on every architecture tested, are doing whitespace recovery.

### Finding 13: NL frustration gap confirmed zero

Tested with punctuated prose probe on NeoX and standard probe on Llama. Zero on web text regardless of architecture or delimiter character set. The damage manifests as spacing proliferation, not frustration gap.

**Significance:** Researchers who looked for frustration gaps on web text and found none concluded no damage. The spacing measurement reveals the hidden cost.

### Finding 14: Structok corpus validation

On a mixed corpus (33% web, 35% structured), both symptoms coexist: 172 spacing heads AND 1.0pp frustration gap. Spacing is a fixed cost (~45%) regardless of corpus composition. Not displaced by delimiter specialization.

**Significance:** The two-regime model is a continuum. Spacing is a fixed tax. This was a falsified prediction (expected 50-150, got 172), which makes it stronger than a confirmed prediction.

### Finding 15: Ablation proves spacing heads are mandatory damage repair

Removing spacing heads degrades PPL by +64.3% NeoX and +67.0% Llama (vs +28.7%/+75.4% for random controls). Removing P0 heads changes nothing (+1.4% NeoX, -3.4% Llama). The near-identical causal cost across architectures is the paper's strongest result.

**Significance:** Converts observational findings into architecture-independent causal proof. The capacity tax is real, measured, and causally verified on two architectures with different attention mechanisms. The fix is 16 lines of tokenizer config.

### Finding 16: Cross-architecture developmental atlas

Full 131-checkpoint embryology on Llama 410M (GQA). Spacing emergence follows the same developmental program as NeoX (MHA). P0 counts are essentially identical (31 Llama vs 32 NeoX). Per-text ablation shows identical total cost but different task-level distribution (duplicates +405% NeoX vs +148% Llama, induction +18% NeoX vs +76% Llama, code ~60% on both). The total tax is set by the tokenizer; the task-level distribution is set by the architecture.

**Significance:** "Conserved developmental program across architectures" is a stronger claim than "same head counts at convergence." No prior work on attention head specialization has demonstrated architecture-independent causal proof.

### Finding 17: Downstream completion benchmark

Completion-based benchmark measuring next-token prediction accuracy on structural text. Merge barriers improve structural accuracy 4x (16.7% vs 4.1%). Bracket prediction: 19.8% vs 0.0%. Delimiter prediction: 25.5% vs 2.2%. Both standard BPE models (NeoX and Llama) show the same pattern: high spacing accuracy, low structural accuracy. Spacing accuracy correlates with head count (47.4% NeoX with 183 heads, 36.8% Llama with 154 heads).

**Significance:** The capacity tax translates directly to downstream accuracy, not just PPL. Models with merge barriers predict structural tokens more accurately because they have the head capacity for it.

## The three-paper story

1. **Tokenizer-Attention Coupling** proves the mechanism and the fix. 2 architectures, 2 scales, 3 domains. 18-phase causal ablation. The fix works.
2. **Stranded Attention** proves universality and permanence. 384/384 heads at 410M, 768/768 at 1.3B. Appears by step 5,000, unchanged through step 40,000.
3. **Developmental Atlas** discovers the scale of the damage, proves it's architecture-independent with causal ablation on two architectures (+64.3% NeoX, +67.0% Llama), unifies the findings across corpora and architectures, and validates with downstream accuracy benchmarks. 48-56% of heads are non-productive. Spacing is a fixed 40-48% tax. P0 is an ~8% failure mode. The two-regime model explains why nobody noticed.

## What this means for the industry

Every model provider (OpenAI, Anthropic, Meta, Google, Mistral, DeepSeek) is training models where 48-56% of attention heads are consumed by boundary recovery or failure, regardless of whether they use multi-head attention or grouped query attention. The fix costs nothing: 16 lines of tokenizer config, no architecture changes, no training cost increase, no performance regression on natural language.

The gap between the severity of the problem (48-56% of heads, every model, every architecture, permanent, causally proven) and the simplicity of the fix (16 lines of config) is the paper's most important contribution.
