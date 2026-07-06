# R2 Data Model

All atlas data is stored on Cloudflare R2 in the `structok-training` bucket under the `atlas/` prefix.

## Schema

```
atlas/
├── tokens/                                    # Pretokenized corpora + tokenizer definitions
│   ├── standard-64k.json                      # Standard BPE tokenizer (no barriers)
│   ├── structok-64k.json                      # Structured-data barrier tokenizer (16 chars)
│   ├── nl-barrier-64k.json                    # NL barrier tokenizer (10 chars)
│   ├── atlas-standard-64k.bin                 # FineWeb pretokenized with standard BPE (~2.4 GB)
│   ├── atlas-structok-64k.bin                 # FineWeb pretokenized with structok (~2.6 GB)
│   └── atlas-nl-barrier-64k.bin               # FineWeb pretokenized with NL barriers
│
├── runs/                                      # Training checkpoints (131 per run)
│   ├── baseline/checkpoints/                  # FineWeb, standard BPE
│   │   ├── step-00000.pt                      # Random init (genesis)
│   │   ├── step-00050.pt                      # Every 50 steps through step 2000
│   │   ├── ...
│   │   ├── step-02000.pt
│   │   ├── step-02200.pt                      # Every 200 steps through step 20000
│   │   ├── ...
│   │   └── step-20000.pt
│   │
│   ├── comparison/checkpoints/                # FineWeb, merge-barrier tokenizer
│   │   └── step-00000.pt ... step-20000.pt
│   │
│   ├── seed2/checkpoints/                     # FineWeb, standard BPE, different init
│   │   └── step-00000.pt ... step-20000.pt
│   │
│   ├── nl-barrier/checkpoints/                # FineWeb, NL-barrier tokenizer
│   │   └── step-00000.pt ... step-20000.pt
│   │
│   ├── structok-baseline/checkpoints/         # Structok corpus, standard BPE
│   │   └── step-00000.pt ... step-20000.pt
│   │
│   ├── structok-comparison/checkpoints/       # Structok corpus, merge-barrier tokenizer
│   │   └── step-00000.pt ... step-20000.pt
│   │
│   └── llama-fineweb-baseline/checkpoints/    # FineWeb, standard BPE, Llama 410M (GQA)
│       └── step-00000.pt ... step-20000.pt
│
└── results/                                   # Probe results (1 JSON per checkpoint per run)
    ├── baseline/                              # v1 probes (6 behaviors, no spacing)
    │   └── step-00000.json ... step-20000.json
    ├── comparison/                            # v1
    │   └── ...
    ├── seed2/                                 # v1
    │   └── ...
    ├── nl-barrier/                            # v1
    │   └── ...
    ├── baseline-v2/                           # v2 probes (7 behaviors incl. spacing, consistent probes)
    │   └── step-00000.json ... step-20000.json
    ├── comparison-v2/                         # v2
    │   └── ...
    ├── seed2-v2/                              # v2
    │   └── ...
    ├── nl-barrier-v2/                         # v2
    │   └── ...
    ├── structok-baseline/                     # Structok corpus (probed with v2 probes from the start)
    │   └── step-00000.json ... step-20000.json
    ├── structok-comparison/                   # Structok corpus
    │   └── ...
    └── llama-fineweb-baseline/               # Llama 410M GQA embryology (7 behaviors)
        └── step-00000.json ... step-20000.json
```

The structok corpus pretokenized bins are stored outside the atlas prefix:
```
tokens/
├── standard-64k-v2.bin                        # Structok corpus pretokenized with standard BPE (4.8 GB)
└── structok-64k-v2.bin                        # Structok corpus pretokenized with structok (4.8 GB)
```
Provenance: created by `structok/prep_run002.py` for the merge-barriers paper (run-002).

## File Sizes

| Type | Per file | Per run (131 files) | Total (7 runs) |
|------|----------|---------------------|----------------|
| Checkpoint (.pt) | ~1.6-1.7 GB | ~210-223 GB | ~1,548 GB |
| Probe result (.json) | ~800 KB | ~105 MB | ~1,155 MB |
| Pretokenized bin (FineWeb) | ~2.4 GB | 1 per tokenizer | ~7.4 GB (3 tokenizers) |
| Pretokenized bin (structok) | ~4.8 GB | 1 per tokenizer | ~9.6 GB (2 tokenizers) |
| Tokenizer JSON | ~4 MB | 1 per tokenizer | ~12 MB |

## Naming Conventions

- **Runs**: `atlas/runs/{run-name}/checkpoints/step-{NNNNN}.pt`
- **Results (v1)**: `atlas/results/{run-name}/step-{NNNNN}.json`
- **Results (v2)**: `atlas/results/{run-name}-v2/step-{NNNNN}.json`
- **Results (structok)**: `atlas/results/structok-{baseline|comparison}/step-{NNNNN}.json`
- **Tokens (FineWeb)**: `atlas/tokens/atlas-{tokenizer-name}.bin`
- **Tokens (structok)**: `tokens/{tokenizer-name}-v2.bin` (outside atlas prefix)
- **Tokenizers**: `atlas/tokens/{tokenizer-name}.json`

Run names: `baseline`, `comparison`, `seed2`, `nl-barrier`, `structok-baseline`, `structok-comparison`, `llama-fineweb-baseline`

## v1 vs v2 Probe Data

**v1**: 6 behavior types (positional_prev, positional_p0, induction, delimiter, bracket, duplicate). Baseline and comparison used original probe texts; seed2 and NL-barrier used improved probes. Results in `atlas/results/{run}/`.

**v2**: 7 behavior types (adds spacing). All 4 FineWeb runs re-probed with identical improved probe texts on a single RTX 4090. Results in `atlas/results/{run}-v2/`. v1 data preserved.

**Structok**: Probed with v2 probes from the start (no separate v1). Results in `atlas/results/structok-{baseline|comparison}/`.

**Llama**: Probed with v2 probes (7 behaviors). Results in `atlas/results/llama-fineweb-baseline/`. Same probe texts and classification logic as NeoX v2 runs.

## Non-Clobbering

Each run and version writes to its own prefix. No two runs share a prefix. The v2 re-probe uses `-v2` suffix to avoid overwriting v1 data. Training and probing can run in parallel on different runs without conflict.

## Checkpoint Contents

Each `.pt` file contains:
```python
{
    "step": int,                    # Training step number
    "model_state_dict": OrderedDict,  # Model weights only (no optimizer state)
    "loss": float,                  # Training loss at this step (NaN for step 0)
}
```

## Architecture Notes

The first 6 runs use GPT-NeoX 410M (MHA, 24 layers, 16 heads, 384 total). The `llama-fineweb-baseline` run uses Llama 410M (GQA, 24 layers, 16 query heads, 4 KV heads, 384 total). Checkpoint size is ~1.6 GB for Llama vs ~1.7 GB for NeoX (smaller intermediate_size: 2816 vs 4096). Probe results have the same schema; the `--size 410m-llama` flag tells `probe_heads.py` to load the correct architecture.

## Probe Result Contents

### v1 (6 behaviors)

```python
{
    "checkpoint": str,              # Source checkpoint path or R2 key
    "step": str,                    # e.g. "step-00050"
    "timestamp": str,               # ISO timestamp
    "classifications": [            # 384 entries (24 layers x 16 heads)
        {
            "layer": int,
            "head": int,
            "dominant": str,        # Dominant behavior type
            "confidence": float,
            "specialization_index": float,
            "entropy": float,
            "unclassified": bool,
            "scores": {             # Raw scores per behavior
                "positional_prev": float,
                "positional_p0": float,
                "induction": float,
                "delimiter": float,
                "bracket": float,
                "duplicate": float,
            },
        },
    ],
    "raw_scores": {                 # Per-probe raw measurements
        "prose": { "seq_len": int, "positional_prev": [[float]], ... },
        "code": { ... },
        "structured": { ... },
        "induction": { ... },
        "duplicates": { ... },
        "brackets": { ... },
        "frustration_gap": {        # Normal vs forced-clean comparison
            "normal_mean": float,
            "clean_mean": float,
            "gap": float,
            "min_delta": float,
            "max_delta": float,
            "heads_woke_up": int,
            "total_heads": int,
            "per_layer_deltas": [[float]],
        },
    },
}
```

### v2 (7 behaviors, adds spacing)

Same structure as v1 with the addition of:
- `"spacing": float` in each classification's `scores` dict
- `"spacing": [[float]]` in each probe's raw scores

## Excess-Corrected Results

Excess-corrected results are stored locally in the repo under `results/{run}-excess/` (v1) and `results/{run}-v2-excess/` (v2) but NOT on R2. They are derived from the raw results by `eval/excess_score_correction.py` and can be regenerated at any time. The excess results add `excess_scores` and `dominant_raw` fields to each classification.

## Relationship to structok R2 Data

The atlas data coexists with the merge-barriers training data in the same `structok-training` bucket. The `atlas/` prefix separates atlas data from the main experiment data stored under `checkpoints/`, `tokens/`, `logs/`, `archive/`, `corpus/`, etc. The structok corpus pretokenized bins are at `tokens/standard-64k-v2.bin` and `tokens/structok-64k-v2.bin` (outside the atlas prefix, shared with the merge-barriers experiments).
