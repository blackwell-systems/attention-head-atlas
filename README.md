# Developmental Atlas of Attention Head Specialization

**Paper:** [Developmental Atlas of Attention Head Specialization: Spacing, Stranding, and the Capacity Tax of BPE Tokenization](paper/developmental-atlas.pdf)

**Author:** Dayna Blackwell, Blackwell Systems

The first comprehensive developmental atlas of attention head specialization at realistic scale. We track 384 heads across 7 behavior types, 131 checkpoints, 6 training runs, and 2 corpora, revealing that standard BPE wastes ~45% of attention heads on whitespace boundary recovery regardless of corpus composition.

## Key Findings

**Spacing is the dominant head specialization in standard BPE.** 183 of 384 heads (47.7%) become spacing specialists, dedicating attention capacity to whitespace boundary recovery. This count is deterministic across seeds (identical in both baseline runs) and is eliminated by merge barriers (0 with NL barriers, 13 with structured barriers). This confirms the "spacing fin" prediction of Wang et al. (2025) at 410M scale.

**Spacing is a fixed cost, not a function of corpus composition.** On a mixed corpus (33% web text, 35% structured data), spacing heads remain at 172/384 (44.8%), nearly identical to the 183 on pure web text. The model grows additional delimiter heads (131 vs 74) to handle structured content, but spacing is untouched. Positional_prev heads shrink instead to make room.

**BPE damage operates in two regimes.** On web text, the frustration gap is zero but 183 heads are wasted on spacing. On structured data, the frustration gap is 40pp (Blackwell, 2026b). On a mixed corpus, both symptoms coexist (1.0pp gap + 172 spacing heads). The regimes are a continuum, not a binary. Merge barriers fix both.

**P0 collapse is a failure cascade at 8.3%, not 25%.** The v1 analysis reported 96 P0 heads; the v2 re-probe with spacing revealed 54 were spacing specialists. 29-32 heads (8.3%) are genuine P0 sinks. The try-fail-collapse mechanism is real but smaller than initially measured. All genuine P0 heads are isolated from co-specializing circuits.

## Relationship to Published Work

This is the third paper in a research program on tokenizer-attention coupling:

1. **Tokenizer-Attention Coupling** (DOI: [10.5281/zenodo.20925910](https://doi.org/10.5281/zenodo.20925910)): Proves BPE merge decisions permanently shape which attention heads develop. 3 domains, 2 architectures (GPT-NeoX, Llama), 2 scales (410M, 1.3B).
2. **Stranded Attention** (DOI: [10.5281/zenodo.21158886](https://doi.org/10.5281/zenodo.21158886)): Characterizes the frustration gap: 40pp of structural attention capacity permanently locked away. 384/384 heads at 410M, 768/768 at 1.3B.
3. **Developmental Atlas** (this paper): Tracks all head types simultaneously from step 0 to convergence. Discovers spacing as dominant specialization, the two-regime model of BPE damage, and circuits as developmentally protective.

## Runs

| Run | Corpus | Tokenizer | Checkpoints | Purpose |
|-----|--------|-----------|-------------|---------|
| baseline | FineWeb (web text) | standard-64k | 131 | Normal BPE development |
| comparison | FineWeb | structok-64k (16 barriers) | 131 | Structured-data barriers |
| seed2 | FineWeb | standard-64k | 131 | Seed variation control |
| nl-barrier | FineWeb | nl-barrier-64k (10 barriers) | 131 | NL barriers |
| structok-baseline | Structok (33% web + 35% structured) | standard-64k | 131 | Corpus effect |
| structok-comparison | Structok | structok-64k (16 barriers) | 131 | Corpus effect + barriers |

All runs use GPT-NeoX 410M (24 layers, 16 heads, 384 total), 20,000 steps, batch size 1, context 2048, bf16, lr 3e-4.

## Repository Structure

```
eval/
  train_atlas.py                   # Training (131 checkpoints, background R2 upload, resume)
  probe_heads.py                   # 7-behavior probing (v2: spacing, hardened, auto-versioning)
  excess_score_correction.py       # Step-0 base-rate subtraction
  prep_data.py                     # FineWeb download + parallel pretokenization
  train_nl_tokenizer.py            # NL-barrier tokenizer training
  measure_nl_frustration_gap.py    # NL frustration gap measurement
  analyze_seed2.py                 # Seed variation analysis
  analyze_p0_deep.py               # P0 failure cascade analysis
  analyze_velocity_circuits.py     # Derivative-based circuit discovery
  analyze_nl_barrier.py            # NL-barrier comparison
  ablate_spacing_heads.py          # Zero-ablation study (spacing, P0, random controls)
probes/                            # 7 probe texts (prose, code, structured, induction, duplicates, brackets, punctuated prose)
results/                           # All probe results (v1, v2, structok, excess-corrected)
charts/
  generate_atlas.py                # Chart generation (6 runs, 10 charts, light/dark themes)
paper/
  developmental-atlas.md           # Paper source (Eisvogel template)
  developmental-atlas.pdf          # Rendered PDF
RESULTS.md                        # 14 findings with full data tables
EXPERIMENT-DESIGN.md              # Experiment design, predictions, validation gates
R2-DATA-MODEL.md                  # R2 storage schema
OUTREACH.md                       # Outreach plan for related authors
```

## Data

All 786 training checkpoints (~1.3 TB) and 1,310 probe results (~260 MB) are on Cloudflare R2. See [R2-DATA-MODEL.md](R2-DATA-MODEL.md) for the full schema.

| Prefix | Contents |
|--------|----------|
| `atlas/runs/{run}/checkpoints/` | 131 checkpoints per run (.pt files, ~1.7 GB each) |
| `atlas/results/{run}/` | v1 probe results (6 behaviors) |
| `atlas/results/{run}-v2/` | v2 probe results (7 behaviors incl. spacing) |
| `atlas/results/structok-{baseline,comparison}/` | Structok corpus probe results |
| `atlas/tokens/` | 3 tokenizer JSONs + 3 FineWeb pretokenized bins |
| `tokens/standard-64k-v2.bin` | Structok corpus pretokenized (standard BPE) |
| `tokens/structok-64k-v2.bin` | Structok corpus pretokenized (merge barriers) |

## Reproducing

All experiments require a single GPU (A100 or RTX 4090). Total compute cost across all 6 runs: ~$18.

```bash
# Train a run
python eval/train_atlas.py \
  --tokenizer tokenizers/standard-64k.json \
  --data corpus.bin \
  --run-name baseline \
  --r2-prefix atlas/runs/baseline \
  --output-dir runs/baseline \
  --steps 20000

# Probe (R2 streaming, resumes from existing)
python eval/probe_heads.py \
  --r2-prefix atlas/runs/baseline \
  --tokenizer tokenizers/standard-64k.json \
  --probe-dir probes/

# Excess correction
python eval/excess_score_correction.py --run baseline

# Generate charts
cd charts && python generate_atlas.py --v2 --use-excess --both-themes
```

## Citation

```
@article{blackwell2026atlas,
  title={Developmental Atlas of Attention Head Specialization: Spacing,
         Stranding, and the Capacity Tax of BPE Tokenization},
  author={Blackwell, Dayna},
  year={2026},
  note={Blackwell Systems}
}
```

## License

MIT
