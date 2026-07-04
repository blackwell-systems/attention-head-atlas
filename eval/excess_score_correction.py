#!/usr/bin/env python3
"""
Apply excess score correction to atlas probe results.

The raw probe scores are inflated by base rates (e.g., delimiter score is high
when many positions are delimiters). The excess score methodology subtracts the
base rate (measured from step-0 random init) to reveal genuine specialization.

Reads existing probe result JSONs, computes corrected classifications,
writes corrected results alongside originals.

Usage:
  python excess_score_correction.py --run baseline
  python excess_score_correction.py --run comparison
  python excess_score_correction.py --run baseline --run comparison
"""

import argparse
import json
import numpy as np
from pathlib import Path


BEHAVIORS = ['positional_prev', 'positional_p0', 'induction', 'delimiter', 'bracket', 'duplicate']


def compute_base_rates(step0_path):
    """Compute per-probe per-behavior base rates from step-0 (random init)."""
    with open(step0_path) as f:
        d = json.load(f)

    probes = [p for p in d['raw_scores'] if p != 'frustration_gap']
    base_rates = {}
    for probe_name in probes:
        probe_data = d['raw_scores'][probe_name]
        base_rates[probe_name] = {}
        for b in BEHAVIORS:
            if b in probe_data:
                all_vals = [v for layer in probe_data[b] for v in layer]
                base_rates[probe_name][b] = float(np.mean(all_vals))
    return base_rates, probes


def classify_with_excess(result, base_rates, probes, num_layers=24, num_heads=16):
    """Reclassify heads using excess scores."""
    classifications = []

    for layer in range(num_layers):
        for head in range(num_heads):
            excess = {}
            for b in BEHAVIORS:
                raw_vals = []
                for probe_name in probes:
                    probe_data = result['raw_scores'].get(probe_name, {})
                    if b in probe_data:
                        raw = probe_data[b][layer][head]
                        base = base_rates[probe_name].get(b, 0)
                        raw_vals.append(max(0, raw - base))
                excess[b] = float(np.mean(raw_vals)) if raw_vals else 0

            max_excess = max(excess.values())
            dominant = max(excess, key=excess.get)
            total = sum(excess.values()) + 1e-10
            spec_index = max_excess / total
            confidence = excess[dominant] / total
            is_unclassified = max_excess < 0.02

            # Get entropy from original classification
            orig_class = None
            for c in result.get('classifications', []):
                if c['layer'] == layer and c['head'] == head:
                    orig_class = c
                    break

            classifications.append({
                "layer": layer,
                "head": head,
                "dominant": "unclassified" if is_unclassified else dominant,
                "dominant_raw": orig_class['dominant'] if orig_class else "?",
                "confidence": round(confidence, 4),
                "specialization_index": round(spec_index, 4),
                "entropy": orig_class.get('entropy', 0) if orig_class else 0,
                "unclassified": is_unclassified,
                "excess_scores": {k: round(v, 4) for k, v in excess.items()},
                "raw_scores": orig_class.get('scores', {}) if orig_class else {},
            })

    return classifications


def process_run(run_name, results_dir):
    """Process all checkpoints for a run."""
    run_dir = results_dir / run_name
    files = sorted(run_dir.glob('step-*.json'))
    print("Processing %s: %d checkpoints" % (run_name, len(files)))

    # Get base rates from step 0
    step0_path = run_dir / 'step-00000.json'
    if not step0_path.exists():
        print("  ERROR: step-00000.json not found")
        return
    base_rates, probes = compute_base_rates(step0_path)

    print("  Base rates computed from step 0:")
    for probe_name in probes:
        top = sorted(base_rates[probe_name].items(), key=lambda x: -x[1])[:3]
        top_str = ', '.join(['%s:%.3f' % (b, v) for b, v in top])
        print("    %s: %s" % (probe_name, top_str))

    # Process each checkpoint
    corrected_dir = results_dir / ('%s-excess' % run_name)
    corrected_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        with open(f) as fh:
            result = json.load(fh)

        corrected = classify_with_excess(result, base_rates, probes)

        output = {
            "checkpoint": result.get("checkpoint", ""),
            "step": result.get("step", f.stem),
            "timestamp": result.get("timestamp", ""),
            "classifications": corrected,
            "raw_scores": result.get("raw_scores", {}),
        }

        out_path = corrected_dir / f.name
        with open(out_path, 'w') as fh:
            json.dump(output, fh, indent=2)

    # Summary: compare raw vs excess at step 20000
    step20k_path = run_dir / 'step-20000.json'
    if step20k_path.exists():
        with open(step20k_path) as fh:
            raw_result = json.load(fh)

        raw_types = {}
        for c in raw_result['classifications']:
            raw_types[c['dominant']] = raw_types.get(c['dominant'], 0) + 1

        corrected_20k = corrected_dir / 'step-20000.json'
        with open(corrected_20k) as fh:
            exc_result = json.load(fh)

        exc_types = {}
        for c in exc_result['classifications']:
            exc_types[c['dominant']] = exc_types.get(c['dominant'], 0) + 1

        print("\n  Step 20000 classification (raw vs excess):")
        print("  %-18s  %-6s  %-6s" % ("Type", "Raw", "Excess"))
        all_types = sorted(set(list(raw_types.keys()) + list(exc_types.keys())))
        for t in all_types:
            print("  %-18s  %-6d  %-6d" % (t, raw_types.get(t, 0), exc_types.get(t, 0)))

    print("  Corrected results written to %s/" % corrected_dir)
    print()


def main():
    parser = argparse.ArgumentParser(description="Apply excess score correction")
    parser.add_argument("--run", action="append", required=True, help="Run name(s) to process")
    parser.add_argument("--results-dir", default=None, help="Results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).parent.parent / 'results'

    for run_name in args.run:
        process_run(run_name, results_dir)


if __name__ == "__main__":
    main()
