#!/usr/bin/env python3
"""
Generate atlas visualization charts from probe results.

Produces publication-quality charts from the developmental atlas data.
Supports raw and excess-corrected results, three runs (baseline, comparison, seed2).

Usage:
    python generate_atlas.py                    # light theme, excess scores
    python generate_atlas.py --dark             # dark theme
    python generate_atlas.py --both-themes      # generate both variants
    python generate_atlas.py --both-scores      # generate raw and excess
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

RESULTS_DIR = Path(__file__).parent.parent / 'results'
CHARTS_DIR = Path(__file__).parent

BEHAVIORS = ['delimiter', 'duplicate', 'bracket', 'positional_prev', 'positional_p0', 'induction', 'unclassified']
COLORS = {
    'delimiter': '#18befc',
    'duplicate': '#ff9944',
    'bracket': '#22c55e',
    'positional_prev': '#a78bfa',
    'positional_p0': '#ff4444',
    'induction': '#f59e0b',
    'unclassified': '#888888',
}

RUN_COLORS = {
    'baseline': '#ff4444',
    'comparison': '#18befc',
    'seed2': '#22c55e',
    'nl-barrier': '#a78bfa',
}
RUN_LABELS = {
    'baseline': 'Baseline (standard BPE)',
    'comparison': 'Comparison (struct barriers)',
    'seed2': 'Seed2 (standard BPE, diff init)',
    'nl-barrier': 'NL barriers (. \' ? ! - etc)',
}

# Theme globals
BG = 'white'
TEXT = '#1a1a1a'
GRID = '#cccccc'
LEGEND_BG = '#f0f0f0'
LIGHT = True


def set_theme(light=True):
    global BG, TEXT, GRID, LEGEND_BG, LIGHT
    LIGHT = light
    if light:
        BG, TEXT, GRID, LEGEND_BG = 'white', '#1a1a1a', '#cccccc', '#f0f0f0'
        plt.style.use('default')
    else:
        BG, TEXT, GRID, LEGEND_BG = '#0a0a0a', 'white', '#333333', '#1a1a1a'
        plt.style.use('dark_background')


def suffix():
    return '' if LIGHT else '-dark'


def setup_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight='bold', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=11)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.spines['bottom'].set_color(GRID)
    ax.spines['left'].set_color(GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.4 if LIGHT else 0.2, color=GRID)


def save(fig, name):
    name = '%s%s.png' % (name, suffix())
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / name, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  %s' % name)


def get_run_dir(run, use_excess):
    """Get results directory for a run."""
    if use_excess:
        return RESULTS_DIR / ('%s-excess' % run)
    return RESULTS_DIR / run


def run_exists(run, use_excess):
    """Check if a run's results exist."""
    d = get_run_dir(run, use_excess)
    return d.exists() and len(list(d.glob('step-*.json'))) > 0


def load_timeline(run_dir):
    """Load all probe results for a run, return structured timeline data."""
    files = sorted(run_dir.glob('step-*.json'))
    steps = []
    type_counts = {b: [] for b in BEHAVIORS}
    avg_spec = []
    avg_entropy = []
    layer_spec = []

    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        steps.append(int(f.stem.replace('step-', '')))

        counts = {b: 0 for b in BEHAVIORS}
        spec_sum = 0
        ent_sum = 0
        l_specs = np.zeros(24)
        l_counts = np.zeros(24)

        for c in d['classifications']:
            dom = c['dominant']
            if dom not in counts:
                counts['unclassified'] += 1
            else:
                counts[dom] += 1
            spec_sum += c.get('specialization_index', 0)
            ent_sum += c.get('entropy', 0)
            l_specs[c['layer']] += c.get('specialization_index', 0)
            l_counts[c['layer']] += 1

        for b in BEHAVIORS:
            type_counts[b].append(counts[b])
        avg_spec.append(spec_sum / 384)
        avg_entropy.append(ent_sum / 384)
        layer_spec.append(l_specs / np.maximum(l_counts, 1))

    return steps, type_counts, avg_spec, avg_entropy, np.array(layer_spec)


def get_available_runs(use_excess):
    """Return list of runs that have data."""
    runs = []
    for run in ['baseline', 'comparison', 'seed2', 'nl-barrier']:
        if run_exists(run, use_excess):
            runs.append(run)
    return runs


# ── Charts ──

def chart_developmental_timeline(use_excess=False):
    """Stacked area chart of head type distribution over training."""
    label = 'excess' if use_excess else 'raw'
    runs = get_available_runs(use_excess)
    ncols = len(runs)

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        run_dir = get_run_dir(run, use_excess)
        steps, tc, _, _, _ = load_timeline(run_dir)
        setup_ax(ax, RUN_LABELS.get(run, run), xlabel='Training step', ylabel='Number of heads')
        bottom = np.zeros(len(steps))
        for b in BEHAVIORS:
            vals = np.array(tc[b])
            ax.fill_between(steps, bottom, bottom + vals, alpha=0.7,
                           color=COLORS[b], label=b)
            bottom += vals
        ax.set_ylim(0, 384)
        ax.legend(loc='center right', fontsize=7, facecolor=LEGEND_BG,
                 edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle('Attention Head Atlas: Developmental Timeline (%s scores)' % label,
                fontsize=14, fontweight='bold', color=TEXT)
    save(fig, 'developmental-timeline-%s' % label)


def chart_entropy_three_way(use_excess=False):
    """Attention entropy over training for all runs."""
    runs = get_available_runs(use_excess)

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_ax(ax, 'Attention Entropy Over Training\nEntropy trajectory is seed-independent; merge barriers stay focused',
             xlabel='Training step', ylabel='Mean attention entropy')

    for run in runs:
        run_dir = get_run_dir(run, use_excess)
        steps, _, _, ent, _ = load_timeline(run_dir)
        ax.plot(steps, ent, color=RUN_COLORS[run], linewidth=2,
               label=RUN_LABELS.get(run, run))

    ax.legend(fontsize=9, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)
    save(fig, 'entropy-three-way')


def chart_p0_emergence(use_excess=False):
    """P0 sink head count over training."""
    label = 'excess' if use_excess else 'raw'
    runs = get_available_runs(use_excess)

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_ax(ax, 'Dormant Head Emergence',
             xlabel='Training step', ylabel='P0 sink heads')

    for run in runs:
        run_dir = get_run_dir(run, use_excess)
        steps, tc, _, _, _ = load_timeline(run_dir)
        ax.plot(steps, tc['positional_p0'], color=RUN_COLORS[run], linewidth=2,
               label=RUN_LABELS.get(run, run))

    ax.legend(fontsize=9, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)
    save(fig, 'p0-sink-emergence-%s' % label)


def chart_specialization_index(use_excess=False):
    """Mean specialization index over training."""
    label = 'excess' if use_excess else 'raw'
    runs = get_available_runs(use_excess)

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_ax(ax, 'Head Specialization Over Training',
             xlabel='Training step', ylabel='Mean specialization index')

    for run in runs:
        run_dir = get_run_dir(run, use_excess)
        steps, _, spec, _, _ = load_timeline(run_dir)
        ax.plot(steps, spec, color=RUN_COLORS[run], linewidth=2,
               label=RUN_LABELS.get(run, run))

    ax.legend(fontsize=9, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)
    save(fig, 'specialization-index-%s' % label)


def chart_layer_depth(use_excess=False):
    """Heatmap of per-layer specialization index over training."""
    label = 'excess' if use_excess else 'raw'
    runs = get_available_runs(use_excess)
    ncols = len(runs)

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 8))
    if ncols == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        run_dir = get_run_dir(run, use_excess)
        steps, _, _, _, layer_spec = load_timeline(run_dir)
        im = ax.imshow(layer_spec.T, aspect='auto', cmap='viridis',
                       extent=[steps[0], steps[-1], 23.5, -0.5],
                       interpolation='nearest')
        setup_ax(ax, RUN_LABELS.get(run, run).split('(')[0].strip(),
                xlabel='Training step', ylabel='Layer')
        plt.colorbar(im, ax=ax, label='Spec. index')

    fig.suptitle('Layer-Depth Specialization (%s scores)' % label,
                fontsize=14, fontweight='bold', color=TEXT)
    save(fig, 'layer-depth-specialization-%s' % label)


def chart_polysemanticity(use_excess=False):
    """Specialist vs generalist head counts over training."""
    label = 'excess' if use_excess else 'raw'
    runs = get_available_runs(use_excess)
    ncols = len(runs)

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        run_dir = get_run_dir(run, use_excess)
        files = sorted(run_dir.glob('*.json'))
        steps = []
        specialists = []
        generalists = []

        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            steps.append(int(f.stem.replace('step-', '')))
            spec_count = sum(1 for c in d['classifications']
                           if c.get('specialization_index', 0) > 0.7)
            gen_count = sum(1 for c in d['classifications']
                          if c.get('specialization_index', 0) < 0.3)
            specialists.append(spec_count)
            generalists.append(gen_count)

        short_label = RUN_LABELS.get(run, run).split('(')[0].strip()
        setup_ax(ax, short_label, xlabel='Training step', ylabel='Number of heads')
        ax.plot(steps, specialists, color='#18befc', linewidth=2, label='Specialists (>0.7)')
        ax.plot(steps, generalists, color='#ff9944', linewidth=2, label='Generalists (<0.3)')
        ax.fill_between(steps, 0, specialists, alpha=0.1, color='#18befc')
        ax.fill_between(steps, 0, generalists, alpha=0.1, color='#ff9944')
        ax.legend(fontsize=8, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle('Polysemanticity: Specialists vs Generalists (%s scores)' % label,
                fontsize=14, fontweight='bold', color=TEXT)
    save(fig, 'polysemanticity-%s' % label)


def chart_seed_comparison(use_excess=False):
    """Bar chart comparing baseline vs seed2 head type distribution at step 20000."""
    label = 'excess' if use_excess else 'raw'
    b_dir = get_run_dir('baseline', use_excess)
    s_dir = get_run_dir('seed2', use_excess)

    if not b_dir.exists() or not s_dir.exists():
        return

    with open(b_dir / 'step-20000.json') as f:
        b_data = json.load(f)
    with open(s_dir / 'step-20000.json') as f:
        s_data = json.load(f)

    b_counts = {b: 0 for b in BEHAVIORS}
    s_counts = {b: 0 for b in BEHAVIORS}
    for c in b_data['classifications']:
        dom = c['dominant']
        if dom in b_counts:
            b_counts[dom] += 1
        else:
            b_counts['unclassified'] += 1
    for c in s_data['classifications']:
        dom = c['dominant']
        if dom in s_counts:
            s_counts[dom] += 1
        else:
            s_counts['unclassified'] += 1

    # Correlation
    b_vals = [b_counts[b] for b in BEHAVIORS]
    s_vals = [s_counts[b] for b in BEHAVIORS]
    corr = np.corrcoef(b_vals, s_vals)[0, 1]

    fig, ax = plt.subplots(figsize=(12, 6))
    setup_ax(ax, 'Seed Variation: Same Tokenizer, Different Init (r=%.3f)\n%s scores at step 20000' % (corr, label),
             ylabel='Number of heads')

    x = np.arange(len(BEHAVIORS))
    width = 0.35
    edge = GRID if LIGHT else BG
    ax.bar(x - width/2, b_vals, width, label='Baseline', color='#ff4444', edgecolor=edge, linewidth=0.5)
    ax.bar(x + width/2, s_vals, width, label='Seed2', color='#22c55e', edgecolor=edge, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(BEHAVIORS, fontsize=9, rotation=15, ha='right')
    ax.legend(fontsize=10, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)

    save(fig, 'seed-comparison-%s' % label)


def find_largest_circuit(run_dir):
    """Find the largest co-specializing circuit in a run."""
    files = sorted(run_dir.glob('step-*.json'))
    if not files:
        return []

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
        if len(cluster) >= 5:
            circuits.append(sorted(cluster))

    return max(circuits, key=len) if circuits else []


def chart_circuit_comparison():
    """Visualize circuit positions for all available runs."""
    runs_to_compare = []
    for run in ['baseline', 'comparison', 'seed2', 'nl-barrier']:
        run_dir = RESULTS_DIR / ('%s-excess' % run)
        if not run_dir.exists():
            continue
        largest = find_largest_circuit(run_dir)
        if largest:
            runs_to_compare.append((run, largest))

    if len(runs_to_compare) < 2:
        return

    ncols = len(runs_to_compare)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    # Collect all circuits for overlap detection
    all_circuits = {name: set(circuit) for name, circuit in runs_to_compare}

    for ax, (run_name, circuit) in zip(axes, runs_to_compare):
        grid = np.zeros((24, 16))
        for h in circuit:
            grid[h // 16, h % 16] = 1.0

        # Mark positions shared with any other run
        other_heads = set()
        for other_name, other_circuit in all_circuits.items():
            if other_name != run_name:
                other_heads |= other_circuit
        overlap = set(circuit) & other_heads
        for h in overlap:
            grid[h // 16, h % 16] = 2.0

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['white' if LIGHT else '#1a1a1a',
                               RUN_COLORS.get(run_name, '#18befc'),
                               '#f59e0b'])
        ax.imshow(grid, aspect='auto', cmap=cmap, interpolation='nearest')
        setup_ax(ax, '%s: %d heads' % (run_name, len(circuit)),
                xlabel='Head', ylabel='Layer')
        ax.set_xticks(range(0, 16, 2))
        ax.set_yticks(range(0, 24, 2))

    # Count total pairwise overlaps
    total_overlap = 0
    for i in range(len(runs_to_compare)):
        for j in range(i + 1, len(runs_to_compare)):
            total_overlap += len(set(runs_to_compare[i][1]) & set(runs_to_compare[j][1]))

    fig.suptitle('Circuit Topology Across Runs\n'
                 'Yellow = position shared with another run (%d total overlaps)' % total_overlap,
                fontsize=13, fontweight='bold', color=TEXT)
    save(fig, 'circuit-comparison')


def chart_emergence_order(use_excess=False):
    """When each behavior type first exceeds a threshold of heads."""
    label = 'excess' if use_excess else 'raw'
    runs = get_available_runs(use_excess)
    threshold = 5  # first step where type has >= 5 heads

    active_behaviors = [b for b in BEHAVIORS if b != 'unclassified']

    fig, ax = plt.subplots(figsize=(12, 7))
    setup_ax(ax, 'Emergence Order: When Each Specialization First Appears\n'
             'First step with >= %d heads of each type (%s scores)' % (threshold, label),
             xlabel='Training step', ylabel='')

    y_positions = np.arange(len(active_behaviors))
    bar_height = 0.25
    offsets = np.linspace(-bar_height * (len(runs) - 1) / 2,
                          bar_height * (len(runs) - 1) / 2, len(runs))

    for run_idx, run in enumerate(runs):
        run_dir = get_run_dir(run, use_excess)
        steps, tc, _, _, _ = load_timeline(run_dir)

        emergence_steps = []
        for b in active_behaviors:
            first = None
            for i, count in enumerate(tc[b]):
                if count >= threshold:
                    first = steps[i]
                    break
            emergence_steps.append(first if first is not None else 20000)

        y = y_positions + offsets[run_idx]
        bars = ax.barh(y, emergence_steps, height=bar_height,
                      color=RUN_COLORS.get(run, '#888888'), alpha=0.8,
                      label=RUN_LABELS.get(run, run))

        for bar, step in zip(bars, emergence_steps):
            if step < 20000:
                ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                       str(step), va='center', fontsize=8, color=TEXT)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(active_behaviors, fontsize=10)
    ax.set_xlim(0, 2500)
    ax.legend(fontsize=9, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT,
             loc='lower right')
    ax.invert_yaxis()

    save(fig, 'emergence-order-%s' % label)


ALL_CHARTS = [
    chart_developmental_timeline,
    chart_entropy_three_way,
    chart_p0_emergence,
    chart_specialization_index,
    chart_layer_depth,
    chart_polysemanticity,
    chart_seed_comparison,
    chart_emergence_order,
]

STANDALONE_CHARTS = [
    chart_circuit_comparison,
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate atlas charts')
    parser.add_argument('--dark', action='store_true', help='Dark theme')
    parser.add_argument('--both-themes', action='store_true', help='Generate both themes')
    parser.add_argument('--use-excess', action='store_true', help='Use excess-corrected data')
    parser.add_argument('--both-scores', action='store_true', help='Generate raw and excess charts')
    args = parser.parse_args()

    themes = [True, False] if args.both_themes else [not args.dark]
    score_modes = [False, True] if args.both_scores else [args.use_excess]

    for light in themes:
        set_theme(light)
        tag = 'light' if light else 'dark'
        for use_excess in score_modes:
            label = 'excess' if use_excess else 'raw'
            print('Generating charts (%s, %s scores)...' % (tag, label))
            for fn in ALL_CHARTS:
                fn(use_excess=use_excess)

        # Standalone charts (not score-dependent)
        for fn in STANDALONE_CHARTS:
            fn()

    print('Done.')
