# Runs

Each run gets its own directory with checkpoints, probe results, and analysis.

## Structure

```
runs/
  baseline/                    # Standard BPE tokenizer, standard corpus
    checkpoints/               # step-00050.pt through step-20000.pt (~130 files)
    training_log.json          # Step/loss/time for each checkpoint
    probe-results/             # One JSON per checkpoint from probe_heads.py
    analysis.json              # Aggregated developmental timeline
    
  merge-barriers/              # Merge-barrier tokenizer, same corpus (comparison)
    checkpoints/
    training_log.json
    probe-results/
    analysis.json
```

## Naming convention

- Checkpoints: `step-XXXXX.pt` (zero-padded to 5 digits)
- Probe results: `step-XXXXX.json` (same naming, in probe-results/)
- Charts: generated in `../charts/` from analysis.json
