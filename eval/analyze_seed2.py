#!/usr/bin/env python3
"""
Analyze seed2 results: compare to baseline for seed-dependence of developmental order.

1. Head type distribution comparison (seed2-excess vs baseline-excess at step 20000)
2. Developmental timeline (when each type emerges)
3. Circuit discovery (does the same circuit form?)
4. Entropy trajectory comparison

Usage:
    python analyze_seed2.py
"""

import json
import numpy as np
from pathlib import Path
from collections import deque

RESULTS_DIR = Path(__file__).parent.parent / 'results'
BEHAVIORS = ['delimiter', 'duplicate', 'bracket', 'positional_prev', 'positional_p0', 'induction', 'unclassified']


def load_step(run, step_name):
    """Load a single step result."""
    with open(RESULTS_DIR / run / ('%s.json' % step_name)) as f:
        return json.load(f)


def get_type_counts(result):
    """Count head types from classifications."""
    counts = {b: 0 for b in BEHAVIORS}
    for c in result['classifications']:
        dom = c['dominant']
        if dom in counts:
            counts[dom] += 1
        else:
            counts['unclassified'] += 1
    return counts


def load_timeline(run):
    """Load full developmental timeline."""
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


def find_circuits(run, threshold=0.9):
    """Find co-specializing circuits via trajectory correlation."""
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

    adj = corr > threshold
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
    print("SEED VARIATION ANALYSIS")
    print("=" * 60)

    # 1. Head type distribution comparison at step 20000
    print("\n=== 1. HEAD TYPE DISTRIBUTION AT STEP 20000 ===\n")

    baseline = load_step('baseline-excess', 'step-20000')
    seed2 = load_step('seed2-excess', 'step-20000')

    b_counts = get_type_counts(baseline)
    s_counts = get_type_counts(seed2)

    print("%-18s  %-10s  %-10s  %-s" % ("Type", "Baseline", "Seed2", "Diff"))
    for b in BEHAVIORS:
        diff = s_counts[b] - b_counts[b]
        print("%-18s  %-10d  %-10d  %+d" % (b, b_counts[b], s_counts[b], diff))

    # Correlation between distributions
    b_vals = [b_counts[b] for b in BEHAVIORS]
    s_vals = [s_counts[b] for b in BEHAVIORS]
    corr = np.corrcoef(b_vals, s_vals)[0, 1]
    print("\nDistribution correlation: %.3f" % corr)
    if corr > 0.9:
        print("CONCLUSION: Highly similar. Developmental outcome is largely deterministic.")
    elif corr > 0.7:
        print("CONCLUSION: Similar overall pattern with some seed-dependent variation.")
    else:
        print("CONCLUSION: Substantially different. Emergence is stochastic.")

    # 2. Developmental timeline comparison
    print("\n=== 2. DEVELOPMENTAL TIMELINE ===\n")

    # Note: seed2 used different probes, so raw comparison is approximate
    # Excess correction normalizes across probe sets
    b_steps, b_counts_list, b_ent = load_timeline('baseline-excess')
    s_steps, s_counts_list, s_ent = load_timeline('seed2-excess')

    print("First emergence (excess-corrected):")
    print("%-18s  %-20s  %-20s" % ("Type", "Baseline", "Seed2"))
    for b in BEHAVIORS:
        if b == 'unclassified':
            continue
        b_first = None
        s_first = None
        for i, counts in enumerate(b_counts_list):
            if counts[b] > 0 and b_first is None:
                b_first = b_steps[i]
        for i, counts in enumerate(s_counts_list):
            if counts[b] > 0 and s_first is None:
                s_first = s_steps[i]
        print("%-18s  step %-15s  step %-15s" % (
            b,
            str(b_first) if b_first else "never",
            str(s_first) if s_first else "never"))

    # 3. Entropy comparison
    print("\n=== 3. ENTROPY TRAJECTORY ===\n")

    # Compare at key checkpoints
    checkpoints = [0, 1000, 5000, 10000, 20000]
    print("%-8s  %-12s  %-12s" % ("Step", "Baseline", "Seed2"))
    for step in checkpoints:
        b_idx = b_steps.index(step) if step in b_steps else None
        s_idx = s_steps.index(step) if step in s_steps else None
        b_e = b_ent[b_idx] if b_idx is not None else float('nan')
        s_e = s_ent[s_idx] if s_idx is not None else float('nan')
        print("%-8d  %-12.4f  %-12.4f" % (step, b_e, s_e))

    # 4. Circuit discovery
    print("\n=== 4. CIRCUIT DISCOVERY ===\n")

    # Get final types for labeling
    s_final_types = {}
    for c in seed2['classifications']:
        h_idx = c['layer'] * 16 + c['head']
        s_final_types[h_idx] = c['dominant']

    b_final_types = {}
    for c in baseline['classifications']:
        h_idx = c['layer'] * 16 + c['head']
        b_final_types[h_idx] = c['dominant']

    print("Baseline circuits (threshold 0.9):")
    b_circuits = find_circuits('baseline-excess')
    for i, cluster in enumerate(b_circuits):
        if len(cluster) >= 5:
            type_counts = {}
            for h in cluster:
                t = b_final_types.get(h, '?')
                type_counts[t] = type_counts.get(t, 0) + 1
            layers = len(set(h // 16 for h in cluster))
            print("  Circuit %d: %d heads, %d layers, types: %s" % (
                i + 1, len(cluster), layers,
                dict(sorted(type_counts.items(), key=lambda x: -x[1]))))

    print("\nSeed2 circuits (threshold 0.9):")
    s_circuits = find_circuits('seed2-excess')
    for i, cluster in enumerate(s_circuits):
        if len(cluster) >= 5:
            type_counts = {}
            for h in cluster:
                t = s_final_types.get(h, '?')
                type_counts[t] = type_counts.get(t, 0) + 1
            layers = len(set(h // 16 for h in cluster))
            print("  Circuit %d: %d heads, %d layers, types: %s" % (
                i + 1, len(cluster), layers,
                dict(sorted(type_counts.items(), key=lambda x: -x[1]))))

    # Compare circuit sizes
    b_largest = max(len(c) for c in b_circuits) if b_circuits else 0
    s_largest = max(len(c) for c in s_circuits) if s_circuits else 0
    print("\nLargest circuit: baseline %d heads, seed2 %d heads" % (b_largest, s_largest))

    # Position overlap
    if b_circuits and s_circuits:
        b_set = set(b_circuits[np.argmax([len(c) for c in b_circuits])])
        s_set = set(s_circuits[np.argmax([len(c) for c in s_circuits])])
        overlap = len(b_set & s_set)
        print("Position overlap in largest circuits: %d heads (%d baseline, %d seed2)" % (
            overlap, len(b_set), len(s_set)))
        if overlap > 0:
            print("Shared positions: %s" % ', '.join(
                'L%02dH%02d' % (h // 16, h % 16) for h in sorted(b_set & s_set)))

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
