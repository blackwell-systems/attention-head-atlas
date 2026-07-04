---
title: "Developmental Atlas of Attention Head Specialization: How Transformers Organize During Training and Why Circuits Protect Against Collapse"
author: "Dayna Blackwell, Blackwell Systems"
date: "2026-07-04"
subtitle: "dayna@blackwell-systems.com"
titlepage: true
titlepage-color: "0a0a0a"
titlepage-text-color: "18befc"
titlepage-rule-color: "18befc"
titlepage-rule-height: 2
header-left: "\\hspace{0pt}"
header-right: "Developmental Atlas"
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

## Abstract

<!-- 
WRITE FROM: RESULTS.md (all 11 findings)
KEY CLAIMS:
- First comprehensive developmental atlas at realistic scale (410M, 24 layers, 384 heads)
- 4 runs: no barriers, structured barriers, NL barriers, seed variation
- P0 failure cascade: heads attempt specialization, fail, collapse (extends Gu et al. ICLR 2025)
- Circuits are developmentally protective (100% of P0 heads isolated)
- Merge barriers are universal: NL barriers (r=0.923) produce same effect as structured barriers
- 131 checkpoints per run, 8 behavior types, excess score correction
TONE: Lead with the P0 cascade and universality findings. These are the strongest contributions.
-->

## 1. Introduction

<!--
THE GAP: Every existing study probes ONE behavior type in isolation.
- Olsson et al. 2022: induction heads only
- Gu et al. 2025: dormancy only  
- Wang et al. 2025 (head diff): 2-layer toy model only
- Wang et al. 2025 (embryology): susceptibility vectors, 3M model, one tokenizer
- Riviere & Trott 2025: disambiguation only
- Baherwani et al. 2026: synthetic tasks only
- Aoyama et al. 2026: induction timing only

WHAT WE DO: Track ALL head types simultaneously from step 0 to convergence at realistic scale.
ADD: The tokenizer variable (nobody else controls this).
ADD: Seed variation (confirms Baherwani at scale).
ADD: NL barriers (proves universality).

CONTRIBUTIONS (7):
1. First comprehensive atlas at 410M scale
2. P0 failure cascade mechanism (try-fail-collapse)
3. Developmental circuit discovery (novel methodology)
4. Circuits as developmentally protective
5. Merge barriers reduce dormancy regardless of character set
6. NL adversarial surface (period 265x worse than pipe)
7. Merge barriers are universal (r=0.923 across character sets)

SOURCE: RESULTS.md "Positioning Against Prior Work" section has all the comparisons written.
-->

## 2. Background

### 2.1 Attention Head Specialization

<!--
Brief review: Voita (positional/syntactic/rare), Clark (BERT [SEP]), Olsson (induction), 
Michel (pruning). All descriptive, natural language, one behavior at a time.
SOURCE: references/ directory has all papers. Stranded attention paper Section 7 has good summaries.
-->

### 2.2 Developmental Interpretability

<!--
Wang et al. 2025 (head diff): rLLC on 2-layer model, staged order (bigrams -> n-grams -> prev-token -> induction)
Wang et al. 2025 (embryology): UMAP on susceptibilities, "rainbow serpent", spacing fin discovery
Xu 2026: "When Do Attention Circuits Form?" - induction at ~20-23B tokens, sinks 10-20x later
Aoyama et al. 2026: predictive equation for induction emergence from batch/context size
SOURCE: references/wang-2025-head-differentiation.pdf, wang-2025-embryology-lm.pdf, xu-2026-attention-circuits-form.pdf, predicting-induction-heads.pdf
-->

### 2.3 Attention Sinks and Dormancy

<!--
Xiao et al. 2024: attention sinks as training artifact
Sandoval-Segura et al. 2025: 4-16% dormant, binary framework (active vs dormant)
Gu et al. 2025: "When Attention Sink Emerges" - mechanism emerges after effective optimization
  THEIR OPEN QUESTIONS WE ANSWER:
  1. "unclear whether attention sink benefits LM downstream performance" -> we show it's a failure mode
  2. "how sink tokens are related to pre-training" -> the tokenizer determines it
SOURCE: references/sandoval-segura-2025-dormant-heads.pdf, gu-2025-when-attention-sink-emerges.pdf
-->

### 2.4 Merge Barriers

<!--
Brief reference to companion papers. Don't repeat them.
- Tokenizer-Attention Coupling (DOI: 10.5281/zenodo.20925910): mechanism + proof
- Stranded Attention (DOI: 10.5281/zenodo.21158886): frustration gap, 384/384 stranding
Key concept: excess score methodology (subtract base rate from raw attention scores)
-->

## 3. Method

### 3.1 Architecture and Training

<!--
GPT-NeoX 410M, 24 layers, 16 heads per layer, 384 total heads
FineWeb corpus (HuggingFaceFW/fineweb, sample-10BT, ~5 GB web text)
20,000 steps, batch size 1 (single sequence), context length 2048, bf16, lr 3e-4
131 checkpoints: step 0 (genesis), every 50 steps through 2000, every 200 steps through 20000
SOURCE: EXPERIMENT-DESIGN.md
-->

### 3.2 Runs

<!--
| Run | Tokenizer | Purpose |
|-----|-----------|---------|
| Baseline | standard-64k (no barriers) | Normal development |
| Comparison | structok-64k (16 struct barriers: | @ < > " ' : , ; \t { } [ ] ( )) | Structured data barriers |
| Seed2 | standard-64k (different random init) | Seed variation control |
| NL-barrier | nl-barrier-64k (10 NL barriers: . ' ? ! - " ( ) ; :) | Natural language barriers |
SOURCE: EXPERIMENT-DESIGN.md
-->

### 3.3 Probing

<!--
8 behavior types measured at each checkpoint on 6 probe texts:
1. positional_prev: attention to position n-1
2. positional_p0: attention to position 0 (sink)
3. induction: copy score (attend to token after previous occurrence)
4. delimiter: attention to delimiter token positions
5. bracket: close-bracket to matching open-bracket
6. duplicate: attention to previous occurrences of same token
7. dormant: max attention concentration (HONOR approximation)
8. entropy: per-head attention entropy

Plus frustration gap: normal vs forced-clean tokenization comparison

6 probe texts: prose (no punctuation), code (Go), structured (GCF), induction (repeated sentences), duplicates (repeated words), brackets (real Go with balanced brackets)

SOURCE: eval/probe_heads.py for implementation, probes/ directory for texts
-->

### 3.4 Excess Score Correction

<!--
CRITICAL METHODOLOGY. Raw scores inflated by base rates.
Example: brackets probe is 100% delimiter -> every head scores 1.0 on delimiter
Excess = raw score - step-0 base rate
A head with 0.30 raw and 0.30 base rate has 0.00 excess (no specialization)
Dramatically changes classification: delimiter drops from 172 to 83 in baseline
Without this, the paper's classifications are meaningless.
SOURCE: RESULTS.md "Excess Score Methodology" section, eval/excess_score_correction.py
-->

### 3.5 Circuit Discovery

<!--
Two methods:
1. Position-based: pairwise Pearson correlation of flattened score trajectories (131 steps x 6 behaviors = 786 values per head). Connected components above 0.9 threshold.
2. Velocity-based: correlate derivatives (rate of change between consecutive checkpoints). Finds cross-type developmental links.
SOURCE: eval/analyze_seed2.py (position circuits), eval/analyze_velocity_circuits.py (velocity circuits)
-->

## 4. Results

### 4.1 Head Type Distribution (Finding 1)

<!--
TABLE from RESULTS.md Finding 1:
| Type | Baseline | Struct barriers | NL barriers |
|------|----------|----------------|-------------|
| Positional (prev) | 102 | 99 | 93 |
| P0 sink | 96 | 52 | 57 |
| Delimiter | 83 | 66 | 57 |
| Unclassified | 38 | 57 | 66 |
| Induction | 32 | 26 | 25 |
| Duplicate | 24 | 37 | 26 |
| Bracket | 9 | 47 | 60 |

Key point: positional_prev is dominant (not delimiter). Excess correction reveals this.
CHART: charts/developmental-timeline-excess.png
-->

### 4.2 P0 Failure Cascade (Finding 2)

<!--
THE PAPER'S STRONGEST FINDING. Full narrative in RESULTS.md Finding 2.

Sub-findings:
a) What were P0 heads before they sank: 35% delimiter, 39% unclassified
b) Timing: median step 11000, gradual not phase transition
c) Layer distribution: L23, L17 most vulnerable
d) Merge barriers save 79/96 heads (converted to delimiter/positional/bracket/induction)
e) 100% of P0 heads are isolated from circuits
f) Extends Gu et al.: P0 mechanism available early (step 1-2K), individual heads collapse into it LATE

TABLE: prior types before sinking (from RESULTS.md)
TABLE: what saved heads become in comparison model

CHART: charts/p0-sink-emergence-excess.png
SOURCE: eval/analyze_p0_deep.py
-->

### 4.3 Entropy Divergence (Finding 3)

<!--
Baseline rises from 0.35 to 0.70 in late training. Comparison stays at 0.35.
NL-barrier: 1.21 (highest, because NL barrier chars are very common in web text)
Entropy is seed-independent (baseline and seed2 match).
CHART: charts/entropy-three-way.png
-->

### 4.4 Frustration Gap Is Domain-Dependent (Finding 4)

<!--
0pp on web text for all runs (including NL barriers).
Contrasts with 40pp on structured-data-heavy corpus (stranded attention paper).
The stranding mechanism requires structured data in training, not just clean delimiters.
But the P0 failure cascade (Finding 2) shows the DAMAGE still operates on web text.
-->

### 4.5 Developmental Circuit Discovery (Finding 5)

<!--
32-head delimiter circuit in baseline spanning 20 layers.
36-head in comparison (larger).
21-head positional_prev circuit consistent across all runs.
94% of correlated pairs are cross-layer (vertical pipelines, not horizontal clusters).
Competitive heads: L08H05 anti-correlates with 7 of top 10 pairs.
Velocity circuits: weak (max r=0.48) but cross-type (7/10 top pairs connect different types).
CHART: charts/circuit-comparison.png
SOURCE: eval/analyze_seed2.py, eval/analyze_velocity_circuits.py
-->

### 4.6 Developmental Sequence (Finding 6)

<!--
Differentiation begins by step 50. Rapid through step 500. Stabilizes by step 2000.
At step 0, all heads are "unclassified" (excess-corrected).
CHART: charts/developmental-timeline-excess.png
-->

### 4.7 Layer-Depth Specialization (Finding 7)

<!--
Middle layers (8-12) specialize most in baseline.
Comparison distributes more evenly.
Layer 0 de-specializes over training.
CHART: charts/layer-depth-specialization-excess.png
-->

### 4.8 Polysemanticity (Finding 8)

<!--
87-92% moderate specialists (0.3-0.7 index). Few pure specialists or generalists.
Merge barriers reduce extremes in both directions.
CHART: charts/polysemanticity-excess.png
-->

### 4.9 NL Adversarial Surface (Finding 9)

<!--
Period: 6,366 mergeable words (265x pipe). Hyphen: 2,886 (120x). Paren: 2,353 (98x).
NL structural characters are FAR more corrupted than structured data characters.
TABLE from RESULTS.md Finding 9
SOURCE: results/ascii-adversarial-surface-43-tokenizers-20260625.json
-->

### 4.10 Seed Variation (Finding 10)

<!--
Distribution correlation: r=0.794. Same types, different counts and positions.
Entropy trajectory seed-independent. Circuits same type, different positions (1/21 overlap).
CHART: charts/seed-comparison-excess.png
SOURCE: eval/analyze_seed2.py
-->

### 4.11 Merge Barriers Are Universal (Finding 11)

<!--
THE PAPER'S STRONGEST GENERALIZABILITY CLAIM.
NL barriers (completely different character set) r=0.923 with structured barriers.
60 bracket specialists (most of any run). P0 reduced to 57 (matches struct's 52).
Same developmental outcome from different barrier sets.
The mechanism is not about protecting specific characters. It's about keeping ANY structural delimiter isolated.
TABLE from RESULTS.md Finding 11
SOURCE: eval/analyze_nl_barrier.py
-->

## 5. Discussion

### 5.1 P0 Collapse as a Failure Cascade

<!--
The central mechanistic contribution.
Heads don't start dormant. They become dormant after failing to specialize.
The P0 mechanism (Gu et al.) is infrastructure. Our finding is about who uses it and why.
Merge barriers prevent the failure, converting would-be P0 capacity into productive specialization.
100% circuit isolation means circuits are PROTECTIVE. Heads that wire together survive.
-->

### 5.2 Universality of Merge Barriers

<!--
Two completely different barrier sets -> same developmental effect.
This is NOT a structured-data trick. It's a general principle of BPE tokenization.
Implications: every model trained with standard BPE has wasted capacity from P0 collapse.
The fix is the same regardless of domain: isolate structural delimiters.
-->

### 5.3 Connection to Stranded Attention

<!--
The frustration gap is 0pp on web text but the P0 cascade is the same mechanism at lower intensity.
On structured-data-heavy corpus: full stranding (40pp gap, all heads affected).
On web text: partial stranding (no measurable gap, but 35% of P0 heads were delimiter heads that tried and failed).
The structok corpus atlas run (planned) would bridge these findings.
-->

### 5.4 Implications for Model Providers

<!--
- Merge barriers are a zero-cost improvement (tokenizer config change)
- Reduces wasted attention capacity on any model, any domain
- Period and hyphen barriers alone would address the largest NL adversarial surfaces
- Circuits form the same type regardless of tokenizer -> safe to change tokenizer without disrupting circuit topology
-->

## 6. Limitations

<!--
FROM RESULTS.md Known Limitations:
1. Probe inconsistency across runs (old vs new probes, excess corrects)
2. FineWeb only (no structured data corpus run yet)
3. Single architecture (GPT-NeoX 410M)
4. Two seeds only
5. Missing spacing probe (Wang et al. spacing fin)
6. NL-barrier step-0 corrupted (using step-50 proxy)
-->

## 7. Related Work

<!--
COPY AND EXPAND from RESULTS.md "Positioning Against Prior Work" section.
6 papers with explicit "what we add" for each:
1. Riviere & Trott (2025): single behavior, we do 8
2. Gu et al. (ICLR 2025): we answer both open questions
3. Wang et al. head diff (ICLR Spotlight): 2-layer toy, we do 410M
4. Wang et al. embryology: susceptibility vs attention, complementary lenses
5. Baherwani et al. (2026): synthetic tasks, we confirm at scale
6. Aoyama et al. (2026): single-behavior phase transition, we do multi-behavior
-->

## 8. Conclusion

<!--
Three sentences:
1. First comprehensive developmental atlas reveals P0 collapse is a failure cascade, not a design choice.
2. Circuits protect against collapse; isolated heads die.
3. Merge barriers are universal: different character sets produce the same developmental outcome, suggesting every model trained with standard BPE has unnecessary wasted capacity.
-->

## References

<!--
FULL LIST (compile from RESULTS.md citations + references/ directory):
- Aoyama, Wilcox & Schneider (2026). Predicting induction head emergence. arXiv:2511.16893.
- Baherwani et al. (2026). Emergent capabilities arise randomly. arXiv:2606.25010.
- Blackwell (2026). Tokenizer-attention coupling. DOI: 10.5281/zenodo.20925910.
- Blackwell (2026). Stranded attention. DOI: 10.5281/zenodo.21158886.
- Conmy et al. (2023). Automated circuit discovery. ICML.
- Gu et al. (2025). When attention sink emerges. ICLR.
- Michel, Levy & Neubig (2019). Are sixteen heads really better than one? NeurIPS.
- Musat et al. (2026). Phase transitions in attention. arXiv:2606.12058.
- Olsson et al. (2022). In-context learning and induction heads. Transformer Circuits Thread.
- Riviere & Trott (2025). Start making sense(s). arXiv:2511.21974.
- Sandoval-Segura et al. (2025). Dormant attention heads. arXiv:2410.13835.
- Voita et al. (2019). Specialized heads do the heavy lifting. ACL.
- Wang et al. (2025). Differentiation and specialization of attention heads. ICLR Spotlight. arXiv:2410.02984.
- Wang, Baker, Gordon & Murfet (2025). Embryology of a language model. arXiv:2508.00331.
- Xiao et al. (2024). Efficient streaming language models with attention sinks. arXiv:2309.17453.
- Xu (2026). When do attention circuits form? arXiv:2606.02378.
-->

## Reproducibility

<!--
All experiments require training on a single GPU (A100 or 4090).
Total compute: ~$20 across 4 runs.
Code: github.com/blackwell-systems/attention-head-atlas
Data: Cloudflare R2 (structok-training bucket, atlas/ prefix)
Checkpoints: 524 training checkpoints on R2
Probe results: 524 JSON files on R2 and in repo
Analysis: all post-hoc scripts run locally (no GPU)
-->
