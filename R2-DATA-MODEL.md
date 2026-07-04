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
│   ├── baseline/checkpoints/
│   │   ├── step-00000.pt                      # Random init (genesis)
│   │   ├── step-00050.pt                      # Every 50 steps through step 2000
│   │   ├── ...
│   │   ├── step-02000.pt
│   │   ├── step-02200.pt                      # Every 200 steps through step 20000
│   │   ├── ...
│   │   └── step-20000.pt
│   │
│   ├── comparison/checkpoints/                # Same structure, merge-barrier tokenizer
│   │   └── step-00000.pt ... step-20000.pt
│   │
│   ├── seed2/checkpoints/                     # Same structure, standard BPE, different init
│   │   └── step-00000.pt ... step-20000.pt
│   │
│   └── nl-barrier/checkpoints/                # Same structure, NL-barrier tokenizer
│       └── step-00000.pt ... step-20000.pt
│
└── results/                                   # Probe results (1 JSON per checkpoint per run)
    ├── baseline/
    │   ├── step-00000.json ... step-20000.json  # 131 probe results
    │
    ├── comparison/
    │   └── step-00000.json ... step-20000.json
    │
    ├── seed2/
    │   └── step-00000.json ... step-20000.json
    │
    └── nl-barrier/
        └── step-00000.json ... step-20000.json
```

## File Sizes

| Type | Per file | Per run (131 files) | Total (4 runs) |
|------|----------|---------------------|----------------|
| Checkpoint (.pt) | ~1.7 GB | ~223 GB | ~892 GB |
| Probe result (.json) | ~200 KB | ~26 MB | ~104 MB |
| Pretokenized bin | ~2.4 GB | 1 per tokenizer | ~7.4 GB (3 tokenizers) |
| Tokenizer JSON | ~4 MB | 1 per tokenizer | ~12 MB |

## Naming Conventions

- **Runs**: `atlas/runs/{run-name}/checkpoints/step-{NNNNN}.pt`
- **Results**: `atlas/results/{run-name}/step-{NNNNN}.json`
- **Tokens**: `atlas/tokens/atlas-{tokenizer-name}.bin`
- **Tokenizers**: `atlas/tokens/{tokenizer-name}.json`

Run names: `baseline`, `comparison`, `seed2`, `nl-barrier`

## Non-Clobbering

Each run writes to its own prefix. No two runs share a prefix. Training and probing can run in parallel on different runs without conflict.

## Checkpoint Contents

Each `.pt` file contains:
```python
{
    "step": int,                    # Training step number
    "model_state_dict": OrderedDict,  # Model weights only (no optimizer state)
    "loss": float,                  # Training loss at this step (NaN for step 0)
}
```

## Probe Result Contents

Each `.json` file contains:
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
        ...
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

## Excess-Corrected Results

Excess-corrected results are stored locally in the repo under `results/{run}-excess/` but NOT on R2. They are derived from the raw results by `eval/excess_score_correction.py` and can be regenerated at any time. The excess results add `excess_scores` and `dominant_raw` fields to each classification.

## Relationship to structok R2 Data

The atlas data coexists with the merge-barriers training data in the same `structok-training` bucket. The `atlas/` prefix separates atlas data from the main experiment data stored under `checkpoints/`, `tokens/`, `logs/`, `archive/`, etc.
