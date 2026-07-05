#!/usr/bin/env python3
"""
Velocity-based circuit discovery: find heads that change together.

Instead of correlating absolute score trajectories (which finds heads with
similar final states), this correlates the DERIVATIVES (rate of change between
consecutive checkpoints). Two heads that jump at the same training steps are
developmentally linked, even if they specialize on different behavior types.

This finds cross-type circuits: a delimiter head and a bracket head that wire
up together because they respond to the same training signal simultaneously.

Usage:
    python analyze_velocity_circuits.py --run baseline-excess
    python analyze_velocity_circuits.py --run baseline-excess --run comparison-excess --run seed2-excess
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import deque

RESULTS_DIR = Path(__file__).parent.parent / 'results'
BEHAVIORS = ['positional_prev', 'positional_p0', 'induction', 'delimiter', 'bracket', 'duplicate', 'spacing']


def load_trajectories(run_dir):
    """Load score trajectories for all heads."""
    files = sorted(run_dir.glob('step-*.json'))
    num_heads = 384
    trajectories = np.zeros((num_heads, len(files), len(BEHAVIORS)))
    steps = []

    for t, f in enumerate(files):
        with open(f) as fh:
            d = json.load(fh)
        steps.append(int(f.stem.replace('step-', '')))
        for c in d['classifications']:
            h_idx = c['layer'] * 16 + c['head']
            scores = c.get('excess_scores', c.get('scores', {}))
            for b_idx, b in enumerate(BEHAVIORS):
                trajectories[h_idx, t, b_idx] = scores.get(b, 0)

    return trajectories, steps


def compute_velocity(trajectories):
    """Compute per-step derivatives (velocity) of score trajectories."""
    # diff along time axis: (num_heads, num_steps-1, num_behaviors)
    return np.diff(trajectories, axis=1)


def find_circuits(corr_matrix, threshold=0.7, min_size=3):
    """Find connected components above correlation threshold."""
    num_heads = corr_matrix.shape[0]
    adj = corr_matrix > threshold
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
        if len(cluster) >= min_size:
            circuits.append(sorted(cluster))

    return circuits


def get_head_type(result, h_idx):
    """Get dominant type for a head from a result file."""
    for c in result['classifications']:
        if c['layer'] * 16 + c['head'] == h_idx:
            return c['dominant']
    return '?'


def analyze_run(run_name):
    """Full velocity circuit analysis for one run."""
    run_dir = RESULTS_DIR / run_name
    if not run_dir.exists():
        print("  %s: not found, skipping" % run_name)
        return

    print("=== %s ===" % run_name.upper())

    trajectories, steps = load_trajectories(run_dir)
    velocity = compute_velocity(trajectories)

    # Flatten velocity: (num_heads, (num_steps-1) * num_behaviors)
    num_heads = velocity.shape[0]
    flat_vel = velocity.reshape(num_heads, -1)

    # Pearson correlation on velocities
    corr = np.corrcoef(flat_vel)
    np.fill_diagonal(corr, 0)

    # Load final types
    final_file = run_dir / 'step-20000.json'
    final_types = {}
    if final_file.exists():
        with open(final_file) as f:
            final = json.load(f)
        for c in final['classifications']:
            h_idx = c['layer'] * 16 + c['head']
            final_types[h_idx] = c['dominant']

    # Find velocity circuits
    vel_circuits = find_circuits(corr, threshold=0.7, min_size=3)
    print("  Velocity circuits (threshold 0.7): %d found" % len(vel_circuits))

    for i, cluster in enumerate(vel_circuits):
        type_counts = {}
        for h in cluster:
            t = final_types.get(h, '?')
            type_counts[t] = type_counts.get(t, 0) + 1
        layers = sorted(set(h // 16 for h in cluster))
        is_cross_type = len(type_counts) > 1

        print("  Circuit %d: %d heads, %d layers, %s%s" % (
            i + 1, len(cluster), len(layers),
            dict(sorted(type_counts.items(), key=lambda x: -x[1])),
            ' [CROSS-TYPE]' if is_cross_type else ''))

        if len(cluster) <= 10:
            members = ['L%02dH%02d(%s)' % (h // 16, h % 16, final_types.get(h, '?'))
                       for h in cluster]
            print("    Members: %s" % ', '.join(members))

    # Compare velocity circuits to position circuits
    pos_flat = trajectories.reshape(num_heads, -1)
    pos_corr = np.corrcoef(pos_flat)
    np.fill_diagonal(pos_corr, 0)
    pos_circuits = find_circuits(pos_corr, threshold=0.9, min_size=3)

    print("\n  Position circuits (threshold 0.9): %d" % len(pos_circuits))
    print("  Velocity circuits (threshold 0.7): %d" % len(vel_circuits))

    # How many velocity circuits are cross-type?
    cross_type = sum(1 for c in vel_circuits
                    if len(set(final_types.get(h, '?') for h in c)) > 1)
    single_type = len(vel_circuits) - cross_type
    print("  Cross-type velocity circuits: %d" % cross_type)
    print("  Single-type velocity circuits: %d" % single_type)

    # Find the most volatile steps (where most heads change the most)
    total_velocity = np.abs(velocity).sum(axis=(0, 2))  # sum across heads and behaviors
    top_steps = np.argsort(total_velocity)[-5:][::-1]
    print("\n  Most volatile steps (highest total change):")
    for idx in top_steps:
        step = steps[idx + 1]  # +1 because diff shifts by 1
        mag = total_velocity[idx]
        print("    step %d: total velocity %.2f" % (step, mag))

    # Top correlated velocity pairs
    pairs = []
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            pairs.append((corr[i, j], i, j))
    pairs.sort(reverse=True)

    print("\n  Top 10 velocity-correlated head pairs:")
    print("  %-8s  %-8s  %-6s  %-12s  %-12s  %-s" % (
        'Head A', 'Head B', 'Corr', 'Type A', 'Type B', 'Cross?'))
    for corr_val, i, j in pairs[:10]:
        li, hi = i // 16, i % 16
        lj, hj = j // 16, j % 16
        ta = final_types.get(i, '?')
        tb = final_types.get(j, '?')
        cross = 'YES' if ta != tb else ''
        print("  L%02dH%02d   L%02dH%02d   %.3f   %-12s  %-12s  %s" % (
            li, hi, lj, hj, corr_val, ta, tb, cross))

    print()


def main():
    parser = argparse.ArgumentParser(description='Velocity-based circuit discovery')
    parser.add_argument('--run', action='append',
                       default=None,
                       help='Run name(s) to analyze')
    args = parser.parse_args()

    runs = args.run or ['baseline-excess', 'comparison-excess', 'seed2-excess']

    print("=" * 60)
    print("VELOCITY CIRCUIT DISCOVERY")
    print("Correlating derivatives (rate of change), not absolute scores")
    print("=" * 60)
    print()

    for run in runs:
        analyze_run(run)

    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
