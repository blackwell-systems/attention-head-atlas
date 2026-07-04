#!/usr/bin/env python3
"""
Analyze NL-barrier results: compare to baseline and comparison.

Tests whether natural-language-optimized merge barriers (. ' ? ! - " ( ) ; :)
produce different head development than structured-data barriers or no barriers.

Usage:
    python analyze_nl_barrier.py
"""

import json
import numpy as np
from pathlib import Path
from collections import deque

RESULTS_DIR = Path(__file__).parent.parent / 'results'
BEHAVIORS = ['delimiter', 'duplicate', 'bracket', 'positional_prev', 'positional_p0', 'induction', 'unclassified']


def get_type_counts(result):
    counts = {b: 0 for b in BEHAVIORS}
    for c in result['classifications']:
        dom = c['dominant']
        if dom in counts:
            counts[dom] += 1
        else:
            counts['unclassified'] += 1
    return counts


def load_step(run, step_name):
    with open(RESULTS_DIR / run / ('%s.json' % step_name)) as f:
        return json.load(f)


def load_timeline(run):
    run_dir = RESULTS_DIR / run
    files = sorted(run_dir.glob('step-*.json'))
    steps = []
    all_counts = []
    entropies = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        steps.append(int(f.stem.replace('step-', '')))
        all_counts.append(get_type_counts(d))
        ent_sum = sum(c.get('entropy', 0) for c in d['classifications'])
        entropies.append(ent_sum / 384)
    return steps, all_counts, entropies


def find_circuits(run):
    run_dir = RESULTS_DIR / run
    files = sorted(run_dir.glob('step-*.json'))
    score_keys = ['positional_prev', 'positional_p0', 'induction', 'delimiter', 'bracket', 'duplicate']
    num_heads = 384
    trajectories = np.zeros((num_heads, len(files), len(score_keys)))
    for t, f in enumerate(files):
        with open(f) as fh:
            d = json.load(fh)
        for c in d['classifications']:
            h_idx = c['layer'] * 16 + c['head']
            scores = c.get('excess_scores', c.get('scores', {}))
            for b_idx, b in enumerate(score_keys):
                trajectories[h_idx, t, b_idx] = scores.get(b, 0)
    flat = trajectories.reshape(num_heads, -1)
    corr = np.corrcoef(flat)
    np.fill_diagonal(corr, 0)
    adj = corr > 0.9
    visited = set()
    circuits = []
    for start in range(num_heads):
        if start in visited or not adj[start].any():
            continue
        cluster = []
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            for neighbor in np.where(adj[node])[0]:
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(cluster) >= 3:
            circuits.append(sorted(cluster))
    return circuits


def main():
    print("=" * 60)
    print("NL-BARRIER ANALYSIS")
    print("=" * 60)

    # 1. Head type distribution comparison at step 20000
    print("\n=== 1. HEAD TYPE DISTRIBUTION (excess-corrected, step 20000) ===\n")

    runs = {
        'baseline-excess': 'Baseline (no barriers)',
        'comparison-excess': 'Comparison (struct barriers)',
        'nl-barrier-excess': 'NL-barrier (. \' ? ! - etc)',
    }

    all_counts = {}
    for run, label in runs.items():
        try:
            result = load_step(run, 'step-20000')
            all_counts[run] = get_type_counts(result)
        except FileNotFoundError:
            print("  %s: not found" % run)

    print("%-18s  %-10s  %-10s  %-10s" % ("Type", "Baseline", "Struct", "NL"))
    for b in BEHAVIORS:
        vals = [all_counts.get(r, {}).get(b, 0) for r in runs]
        print("%-18s  %-10d  %-10d  %-10d" % (b, *vals))

    # Correlations
    print()
    run_list = list(runs.keys())
    for i in range(len(run_list)):
        for j in range(i + 1, len(run_list)):
            v1 = [all_counts[run_list[i]].get(b, 0) for b in BEHAVIORS]
            v2 = [all_counts[run_list[j]].get(b, 0) for b in BEHAVIORS]
            r = np.corrcoef(v1, v2)[0, 1]
            print("  %s vs %s: r=%.3f" % (run_list[i].replace('-excess', ''),
                                            run_list[j].replace('-excess', ''), r))

    # 2. P0 comparison
    print("\n=== 2. P0 SINK COMPARISON ===\n")
    for run, label in runs.items():
        if run in all_counts:
            p0 = all_counts[run].get('positional_p0', 0)
            print("  %-30s  %d P0 heads (%.1f%%)" % (label, p0, p0 / 384 * 100))

    # 3. Entropy comparison
    print("\n=== 3. ENTROPY AT KEY STEPS ===\n")
    print("%-8s  %-12s  %-12s  %-12s" % ("Step", "Baseline", "Struct", "NL"))
    checkpoints = [50, 1000, 5000, 10000, 20000]
    timelines = {}
    for run in runs:
        try:
            timelines[run] = load_timeline(run)
        except Exception:
            pass

    for step in checkpoints:
        vals = []
        for run in runs:
            if run in timelines:
                steps, _, ents = timelines[run]
                idx = steps.index(step) if step in steps else None
                vals.append(ents[idx] if idx is not None else float('nan'))
            else:
                vals.append(float('nan'))
        print("%-8d  %-12.4f  %-12.4f  %-12.4f" % (step, *vals))

    # 4. Bracket comparison (NL barriers include ( ) which should help)
    print("\n=== 4. BRACKET SPECIALISTS ===\n")
    for run, label in runs.items():
        if run in all_counts:
            br = all_counts[run].get('bracket', 0)
            print("  %-30s  %d bracket heads" % (label, br))

    # 5. Circuit discovery
    print("\n=== 5. CIRCUIT DISCOVERY ===\n")
    for run, label in runs.items():
        try:
            circuits = find_circuits(run)
            large = [c for c in circuits if len(c) >= 5]
            total_in = sum(len(c) for c in large)
            print("  %s:" % label)
            print("    Circuits (>=5 heads): %d" % len(large))
            if large:
                largest = max(large, key=len)
                # Get types
                result = load_step(run, 'step-20000')
                final_types = {}
                for c in result['classifications']:
                    h_idx = c['layer'] * 16 + c['head']
                    final_types[h_idx] = c['dominant']
                type_counts = {}
                for h in largest:
                    t = final_types.get(h, '?')
                    type_counts[t] = type_counts.get(t, 0) + 1
                layers = len(set(h // 16 for h in largest))
                print("    Largest: %d heads, %d layers, types: %s" % (
                    len(largest), layers,
                    dict(sorted(type_counts.items(), key=lambda x: -x[1]))))
        except Exception as e:
            print("  %s: error: %s" % (label, e))

    # 6. Does NL barrier change frustration gap?
    print("\n=== 6. FRUSTRATION GAP ===\n")
    for run, label in runs.items():
        try:
            result = load_step(run, 'step-20000')
            fg = result.get('raw_scores', {}).get('frustration_gap', {})
            if fg:
                print("  %-30s  gap: %.3f (normal: %.3f, clean: %.3f)" % (
                    label, fg.get('gap', 0), fg.get('normal_mean', 0), fg.get('clean_mean', 0)))
            else:
                print("  %-30s  no frustration gap data" % label)
        except Exception as e:
            print("  %s: %s" % (label, e))

    # 7. Key question: does NL barrier look more like baseline or comparison?
    print("\n=== 7. IS NL-BARRIER CLOSER TO BASELINE OR COMPARISON? ===\n")
    if all(r in all_counts for r in runs):
        nl_vals = [all_counts['nl-barrier-excess'].get(b, 0) for b in BEHAVIORS]
        b_vals = [all_counts['baseline-excess'].get(b, 0) for b in BEHAVIORS]
        c_vals = [all_counts['comparison-excess'].get(b, 0) for b in BEHAVIORS]

        r_baseline = np.corrcoef(nl_vals, b_vals)[0, 1]
        r_comparison = np.corrcoef(nl_vals, c_vals)[0, 1]
        print("  NL vs Baseline:    r=%.3f" % r_baseline)
        print("  NL vs Comparison:  r=%.3f" % r_comparison)

        if r_comparison > r_baseline:
            print("  CONCLUSION: NL barriers produce head development more similar to structured barriers.")
        else:
            print("  CONCLUSION: NL barriers produce head development more similar to no barriers.")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
