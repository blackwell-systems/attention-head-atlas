#!/usr/bin/env python3
"""
Deep analysis of P0 sink (dormant) heads.

1. What were P0 heads before they sank? Track backward through checkpoints.
2. When do they sink? Gradual drift or sudden collapse?
3. Layer distribution of P0 sinks.
4. What do the "saved" heads do instead in the comparison model?
5. P0 count comparison across seeds.
6. Are P0 heads in circuits or isolated?

Usage:
    python analyze_p0_deep.py
"""

import json
import numpy as np
from pathlib import Path
from collections import deque

RESULTS_DIR = Path(__file__).parent.parent / 'results'
BEHAVIORS = ['delimiter', 'duplicate', 'bracket', 'positional_prev', 'positional_p0', 'induction', 'spacing', 'unclassified']


def load_all_classifications(run):
    """Load classifications at every checkpoint for a run."""
    run_dir = RESULTS_DIR / run
    files = sorted(run_dir.glob('step-*.json'))
    timeline = {}

    for f in files:
        step = int(f.stem.replace('step-', ''))
        with open(f) as fh:
            d = json.load(fh)
        head_types = {}
        for c in d['classifications']:
            h_idx = c['layer'] * 16 + c['head']
            head_types[h_idx] = c['dominant']
        timeline[step] = head_types

    return timeline


def main():
    print("=" * 60)
    print("P0 SINK DEEP ANALYSIS")
    print("=" * 60)

    # 1. What were P0 heads before they sank?
    print("\n=== 1. WHAT WERE P0 HEADS BEFORE THEY SANK? ===\n")

    for run in ['baseline-excess', 'seed2-excess']:
        timeline = load_all_classifications(run)
        steps = sorted(timeline.keys())

        # Find P0 heads at step 20000
        p0_heads = [h for h, t in timeline[20000].items() if t == 'positional_p0']
        print("%s: %d P0 heads at step 20000" % (run, len(p0_heads)))

        # Track backward: what was each P0 head at earlier steps?
        prior_types = {}
        for h in p0_heads:
            history = []
            for step in steps:
                t = timeline[step].get(h, '?')
                if t != 'positional_p0':
                    history.append(t)
            if history:
                # Most common non-P0 type = what it "tried" before sinking
                from collections import Counter
                prior = Counter(history).most_common(1)[0][0]
                prior_types[prior] = prior_types.get(prior, 0) + 1

        print("  Prior specializations (most common non-P0 type):")
        for t, count in sorted(prior_types.items(), key=lambda x: -x[1]):
            print("    %-18s %d heads (%.0f%%)" % (t, count, count / len(p0_heads) * 100))
        print()

    # 2. When do they sink?
    print("=== 2. WHEN DO THEY SINK? ===\n")

    for run in ['baseline-excess']:
        timeline = load_all_classifications(run)
        steps = sorted(timeline.keys())
        p0_heads = [h for h, t in timeline[20000].items() if t == 'positional_p0']

        sink_steps = []
        for h in p0_heads:
            # Find first step where head becomes P0 and stays P0
            first_p0 = None
            for i, step in enumerate(steps):
                if timeline[step].get(h) == 'positional_p0':
                    # Check if it stays P0 for the rest
                    remaining = [timeline[s].get(h) for s in steps[i:]]
                    p0_frac = sum(1 for t in remaining if t == 'positional_p0') / len(remaining)
                    if p0_frac > 0.7:  # mostly P0 from here on
                        first_p0 = step
                        break
            if first_p0 is not None:
                sink_steps.append(first_p0)

        print("%s:" % run)
        if sink_steps:
            print("  Earliest sink: step %d" % min(sink_steps))
            print("  Latest sink: step %d" % max(sink_steps))
            print("  Median sink: step %d" % int(np.median(sink_steps)))

            # Distribution by training phase
            early = sum(1 for s in sink_steps if s <= 500)
            mid = sum(1 for s in sink_steps if 500 < s <= 2000)
            late = sum(1 for s in sink_steps if s > 2000)
            print("  Early (0-500): %d heads" % early)
            print("  Mid (500-2000): %d heads" % mid)
            print("  Late (2000+): %d heads" % late)
        print()

    # 3. Layer distribution
    print("=== 3. LAYER DISTRIBUTION OF P0 SINKS ===\n")

    for run in ['baseline-excess', 'comparison-excess', 'seed2-excess']:
        run_dir = RESULTS_DIR / run
        if not run_dir.exists():
            continue
        with open(run_dir / 'step-20000.json') as f:
            d = json.load(f)

        layer_p0 = np.zeros(24)
        total_p0 = 0
        for c in d['classifications']:
            if c['dominant'] == 'positional_p0':
                layer_p0[c['layer']] += 1
                total_p0 += 1

        print("%s: %d P0 heads" % (run, total_p0))
        if total_p0 > 0:
            top_layers = np.argsort(layer_p0)[::-1][:5]
            for l in top_layers:
                if layer_p0[l] > 0:
                    print("  L%02d: %d heads" % (l, layer_p0[l]))
        print()

    # 4. What do the "saved" heads do instead?
    print("=== 4. WHAT DO SAVED HEADS DO INSTEAD? ===\n")

    b_timeline = load_all_classifications('baseline-excess')
    c_dir = RESULTS_DIR / 'comparison-excess'
    if c_dir.exists():
        with open(c_dir / 'step-20000.json') as f:
            c_data = json.load(f)

        b_p0_heads = set(h for h, t in b_timeline[20000].items() if t == 'positional_p0')

        with open(RESULTS_DIR / 'comparison-excess' / 'step-20000.json') as f:
            c_data = json.load(f)

        c_types_at_b_p0_positions = {}
        for c in c_data['classifications']:
            h_idx = c['layer'] * 16 + c['head']
            if h_idx in b_p0_heads:
                t = c['dominant']
                c_types_at_b_p0_positions[t] = c_types_at_b_p0_positions.get(t, 0) + 1

        print("Baseline has %d P0 heads. At those same positions, comparison has:" % len(b_p0_heads))
        for t, count in sorted(c_types_at_b_p0_positions.items(), key=lambda x: -x[1]):
            print("  %-18s %d heads" % (t, count))

        # How many are still P0 in comparison?
        still_p0 = c_types_at_b_p0_positions.get('positional_p0', 0)
        saved = len(b_p0_heads) - still_p0
        print("\n  %d heads saved from P0 sink by merge barriers" % saved)
        print("  %d heads still P0 in both" % still_p0)
    print()

    # 5. Seed comparison
    print("=== 5. P0 COUNT ACROSS SEEDS ===\n")

    for run in ['baseline-excess', 'seed2-excess']:
        run_dir = RESULTS_DIR / run
        if not run_dir.exists():
            continue
        with open(run_dir / 'step-20000.json') as f:
            d = json.load(f)
        p0_count = sum(1 for c in d['classifications'] if c['dominant'] == 'positional_p0')
        print("  %s: %d P0 heads" % (run, p0_count))

    print()

    # 6. Are P0 heads in circuits?
    print("=== 6. P0 HEADS IN CIRCUITS ===\n")

    for run in ['baseline-excess']:
        run_dir = RESULTS_DIR / run
        files = sorted(run_dir.glob('step-*.json'))

        score_keys = ['positional_prev', 'positional_p0', 'induction', 'delimiter', 'bracket', 'duplicate', 'spacing']
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
        in_circuit = set()
        for start in range(num_heads):
            if not adj[start].any():
                continue
            cluster = []
            visited_local = set()
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited_local:
                    continue
                visited_local.add(node)
                cluster.append(node)
                for neighbor in np.where(adj[node])[0]:
                    if neighbor not in visited_local:
                        queue.append(neighbor)
            if len(cluster) >= 3:
                in_circuit.update(cluster)

        with open(run_dir / 'step-20000.json') as f:
            d = json.load(f)
        p0_heads = set()
        for c in d['classifications']:
            if c['dominant'] == 'positional_p0':
                p0_heads.add(c['layer'] * 16 + c['head'])

        p0_in_circuit = p0_heads & in_circuit
        p0_isolated = p0_heads - in_circuit
        print("%s:" % run)
        print("  P0 heads in circuits: %d" % len(p0_in_circuit))
        print("  P0 heads isolated: %d" % len(p0_isolated))
        print("  (%.0f%% isolated)" % (len(p0_isolated) / len(p0_heads) * 100 if p0_heads else 0))

    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
