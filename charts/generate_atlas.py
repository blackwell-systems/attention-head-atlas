#!/usr/bin/env python3
"""
Generate atlas visualization charts from probe results.

Produces publication-quality charts from the developmental atlas data.
Supports raw and excess-corrected results, four runs (baseline, comparison, seed2, nl-barrier).

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

BEHAVIORS = ['delimiter', 'duplicate', 'bracket', 'positional_prev', 'positional_p0', 'induction', 'spacing', 'unclassified']
COLORS = {
    'delimiter': '#18befc',
    'duplicate': '#ff9944',
    'bracket': '#22c55e',
    'positional_prev': '#a78bfa',
    'positional_p0': '#ff4444',
    'induction': '#f59e0b',
    'spacing': '#ec4899',
    'unclassified': '#888888',
}

RUN_COLORS = {
    'baseline': '#ff4444',
    'comparison': '#18befc',
    'seed2': '#22c55e',
    'nl-barrier': '#a78bfa',
    'structok-baseline': '#f97316',
    'structok-comparison': '#06b6d4',
    'llama-fineweb-baseline': '#facc15',
}
RUN_LABELS = {
    'baseline': 'Baseline (standard BPE)',
    'comparison': 'Comparison (struct barriers)',
    'seed2': 'Seed2 (standard BPE, diff init)',
    'nl-barrier': 'NL barriers (. \' ? ! - etc)',
    'structok-baseline': 'Structok corpus (standard BPE)',
    'structok-comparison': 'Structok corpus (struct barriers)',
    'llama-fineweb-baseline': 'Llama 410M (GQA, standard BPE)',
}
# Runs probed with v2 probes from the start (no -v2 suffix needed)
NO_V2_SUFFIX_RUNS = {'structok-baseline', 'structok-comparison', 'llama-fineweb-baseline'}

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


USE_V2 = False  # Set by --v2 flag


def get_run_dir(run, use_excess):
    """Get results directory for a run.
    Structok runs have no -v2 variant (probed with v2 probes from the start)."""
    if USE_V2 and run not in NO_V2_SUFFIX_RUNS:
        if use_excess:
            return RESULTS_DIR / ('%s-v2-excess' % run)
        return RESULTS_DIR / ('%s-v2' % run)
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


ALL_RUNS = ['baseline', 'comparison', 'seed2', 'nl-barrier', 'structok-baseline', 'structok-comparison', 'llama-fineweb-baseline']


def get_available_runs(use_excess):
    """Return list of runs that have data."""
    runs = []
    for run in ALL_RUNS:
        if run_exists(run, use_excess):
            runs.append(run)
    return runs


# ── Charts ──

def make_grid(n):
    """Return (nrows, ncols) for arranging n panels. 2x3 for 5-6, 1xN for fewer."""
    if n <= 4:
        return 1, n
    return 2, (n + 1) // 2


def chart_developmental_timeline(use_excess=False):
    """Stacked area chart of head type distribution over training."""
    label = 'excess' if use_excess else 'raw'
    runs = get_available_runs(use_excess)
    nrows, ncols = make_grid(len(runs))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    # Hide unused axes
    for i in range(len(runs), len(axes)):
        axes[i].set_visible(False)

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
        ax.legend(loc='center right', fontsize=6, facecolor=LEGEND_BG,
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
    save(fig, 'entropy-all-runs')


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
    nrows, ncols = make_grid(len(runs))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6 * nrows))
    axes = np.array(axes).flatten()
    for i in range(len(runs), len(axes)):
        axes[i].set_visible(False)

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
    nrows, ncols = make_grid(len(runs))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    for i in range(len(runs), len(axes)):
        axes[i].set_visible(False)

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
    for run in ALL_RUNS:
        run_dir = get_run_dir(run, use_excess=True)
        if not run_dir.exists():
            continue
        largest = find_largest_circuit(run_dir)
        if largest:
            runs_to_compare.append((run, largest))

    if len(runs_to_compare) < 2:
        return

    nrows, ncols = make_grid(len(runs_to_compare))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6 * nrows))
    axes = np.array(axes).flatten()
    for i in range(len(runs_to_compare), len(axes)):
        axes[i].set_visible(False)

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


def chart_frustration_gap_emergence(use_excess=False):
    """Frustration gap over training steps for structok corpus runs."""
    structok_runs = ['structok-baseline', 'structok-comparison']
    available = [r for r in structok_runs if run_exists(r, use_excess)]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_ax(ax, 'Frustration Gap Emergence (Structok Corpus)\n'
             'Gap appears on structured data, merge barriers prevent it',
             xlabel='Training step', ylabel='Frustration gap (pp)')

    for run in available:
        run_dir = get_run_dir(run, use_excess)
        files = sorted(run_dir.glob('step-*.json'))
        steps = []
        gaps = []
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            steps.append(int(f.stem.replace('step-', '')))
            fg = d.get('raw_scores', {}).get('frustration_gap', {})
            gaps.append(fg.get('gap', 0) * 100)  # convert to pp

        ax.plot(steps, gaps, color=RUN_COLORS.get(run, '#888888'), linewidth=2,
               label=RUN_LABELS.get(run, run))

    ax.axhline(y=0, color=GRID, linewidth=0.5, linestyle='--')
    ax.legend(fontsize=9, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)
    save(fig, 'frustration-gap-emergence')


def chart_ablation_comparison(use_excess=False):
    """Bar chart comparing ablation deltas across three models."""
    ablation_dir = RESULTS_DIR / 'ablation'
    if not ablation_dir.exists():
        return

    models = []
    for name, fname in [('NeoX 410M\nFineWeb', 'ablation-baseline.json'),
                         ('Llama 410M\nFineWeb', 'ablation-llama-fineweb-410m.json'),
                         ('NeoX 410M\nStructok', 'ablation-structok-baseline.json'),
                         ('NeoX 410M\nBarriers', 'ablation-comparison.json')]:
        path = ablation_dir / fname
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        models.append((name, d))

    if not models:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_ax(ax, 'Spacing Head Ablation: Mandatory Damage Repair\n'
             'Removing spacing heads hurts more than removing random heads',
             ylabel='PPL change (%)')

    x = np.arange(len(models))
    width = 0.25
    edge = GRID if LIGHT else BG

    spacing_deltas = [m[1]['spacing_ablation']['delta_pct'] for m in models]
    random_deltas = [m[1]['random_controls']['mean_delta_pct'] for m in models]
    p0_deltas = [m[1].get('p0_ablation', {}).get('delta_pct', 0) for m in models]

    ax.bar(x - width, spacing_deltas, width, label='Spacing heads removed',
           color='#ec4899', edgecolor=edge, linewidth=0.5)
    ax.bar(x, random_deltas, width, label='Random heads removed (control)',
           color='#888888', edgecolor=edge, linewidth=0.5)
    ax.bar(x + width, p0_deltas, width, label='P0 heads removed',
           color='#ff4444', edgecolor=edge, linewidth=0.5)

    ax.axhline(y=0, color=GRID, linewidth=0.5, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], fontsize=10)
    ax.legend(fontsize=9, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)

    # Add value labels
    for i, (s, r, p) in enumerate(zip(spacing_deltas, random_deltas, p0_deltas)):
        ax.text(i - width, s + 1, '%+.1f%%' % s, ha='center', va='bottom', fontsize=8, color=TEXT)
        ax.text(i, r + 1 if r >= 0 else r - 3, '%+.1f%%' % r, ha='center',
                va='bottom' if r >= 0 else 'top', fontsize=8, color=TEXT)
        ax.text(i + width, p + 1, '%+.1f%%' % p, ha='center', va='bottom', fontsize=8, color=TEXT)

    save(fig, 'ablation-comparison')


def chart_ablation_per_text(use_excess=False):
    """Horizontal bar chart of per-text degradation from spacing ablation."""
    path = RESULTS_DIR / 'ablation' / 'ablation-baseline.json'
    if not path.exists():
        return

    with open(path) as f:
        d = json.load(f)

    baseline_texts = d['baseline']['per_text']
    spacing_texts = d['spacing_ablation']['per_text']

    texts = []
    deltas = []
    for name in sorted(baseline_texts.keys()):
        base = baseline_texts[name]
        ablated = spacing_texts[name]
        delta = (ablated - base) / base * 100
        texts.append(name)
        deltas.append(delta)

    # Sort by delta descending
    pairs = sorted(zip(texts, deltas), key=lambda x: -x[1])
    texts = [p[0] for p in pairs]
    deltas = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(10, 5))
    setup_ax(ax, 'Task Dependency on Spacing Recovery\n'
             'PPL degradation when 183 spacing heads are removed (FineWeb baseline)',
             xlabel='PPL change (%)')

    edge = GRID if LIGHT else BG
    bars = ax.barh(range(len(texts)), deltas, color='#ec4899', edgecolor=edge, linewidth=0.5)

    ax.set_yticks(range(len(texts)))
    ax.set_yticklabels(texts, fontsize=10)
    ax.invert_yaxis()

    for bar, delta in zip(bars, deltas):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
               '%+.1f%%' % delta, va='center', fontsize=9, color=TEXT)

    save(fig, 'ablation-per-text')


def chart_capacity_tax(use_excess=False):
    """Stacked bar showing head allocation: spacing, P0, productive."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    categories = ['Spacing\n(damage repair)', 'P0 sinks\n(doing nothing)', 'Productive']
    colors_list = ['#ec4899', '#ff4444', '#22c55e']
    edge = GRID if LIGHT else BG

    # NeoX Baseline
    ax = axes[0]
    setup_ax(ax, 'NeoX 410M (MHA)\nStandard BPE', ylabel='Heads')
    counts = [183, 32, 384 - 183 - 32]
    ax.bar(categories, counts, color=colors_list, edgecolor=edge, linewidth=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c + 5, '%d\n(%.0f%%)' % (c, c / 384 * 100), ha='center', fontsize=9, color=TEXT)
    ax.set_ylim(0, 250)

    # Llama Baseline
    ax = axes[1]
    setup_ax(ax, 'Llama 410M (GQA)\nStandard BPE', ylabel='Heads')
    counts = [154, 31, 384 - 154 - 31]
    ax.bar(categories, counts, color=colors_list, edgecolor=edge, linewidth=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c + 5, '%d\n(%.0f%%)' % (c, c / 384 * 100), ha='center', fontsize=9, color=TEXT)
    ax.set_ylim(0, 250)

    # Comparison (merge barriers)
    ax = axes[2]
    setup_ax(ax, 'NeoX 410M (MHA)\nMerge Barriers', ylabel='Heads')
    counts = [13, 40, 384 - 13 - 40]
    ax.bar(categories, counts, color=colors_list, edgecolor=edge, linewidth=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c + 5, '%d\n(%.0f%%)' % (c, c / 384 * 100), ha='center', fontsize=9, color=TEXT)
    ax.set_ylim(0, 400)

    fig.suptitle('The Capacity Tax Across Architectures',
                fontsize=13, fontweight='bold', color=TEXT)
    save(fig, 'capacity-tax')


def chart_cross_architecture_emergence(use_excess=False):
    """Side-by-side spacing/P0/delimiter emergence curves for NeoX vs Llama."""
    label = 'excess' if use_excess else 'raw'
    neox_dir = get_run_dir('baseline', use_excess)
    llama_dir = get_run_dir('llama-fineweb-baseline', use_excess)

    if not neox_dir.exists() or not llama_dir.exists():
        return

    n_steps, n_tc, _, _, _ = load_timeline(neox_dir)
    l_steps, l_tc, _, _, _ = load_timeline(llama_dir)

    behaviors_to_plot = ['spacing', 'positional_p0', 'delimiter', 'positional_prev']
    behavior_labels = {
        'spacing': 'Spacing',
        'positional_p0': 'P0 sinks',
        'delimiter': 'Delimiter',
        'positional_prev': 'Positional (prev)',
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, b in zip(axes, behaviors_to_plot):
        setup_ax(ax, behavior_labels[b], xlabel='Training step', ylabel='Head count')
        ax.plot(n_steps, n_tc[b], color=RUN_COLORS['baseline'], linewidth=2,
                label='NeoX 410M (MHA)')
        ax.plot(l_steps, l_tc[b], color=RUN_COLORS['llama-fineweb-baseline'], linewidth=2,
                label='Llama 410M (GQA)')
        ax.legend(fontsize=9, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle('Cross-Architecture Embryology: NeoX vs Llama (%s scores)\n'
                 'Same corpus, same tokenizer, different architectures' % label,
                fontsize=13, fontweight='bold', color=TEXT)
    save(fig, 'cross-architecture-emergence-%s' % label)


def chart_ablation_cross_architecture(use_excess=False):
    """Side-by-side per-text ablation for NeoX vs Llama."""
    neox_path = RESULTS_DIR / 'ablation' / 'ablation-baseline.json'
    llama_path = RESULTS_DIR / 'ablation' / 'ablation-llama-fineweb-410m.json'

    if not neox_path.exists() or not llama_path.exists():
        return

    with open(neox_path) as f:
        neox = json.load(f)
    with open(llama_path) as f:
        llama = json.load(f)

    texts = sorted(neox['baseline']['per_text'].keys())
    neox_deltas = []
    llama_deltas = []
    for t in texts:
        nb = neox['baseline']['per_text'][t]
        ns = neox['spacing_ablation']['per_text'][t]
        neox_deltas.append((ns - nb) / nb * 100)
        lb = llama['baseline']['per_text'][t]
        ls = llama['spacing_ablation']['per_text'].get(t, lb)
        llama_deltas.append((ls - lb) / lb * 100)

    fig, ax = plt.subplots(figsize=(12, 6))
    setup_ax(ax, 'Spacing Ablation: Per-Text Degradation Across Architectures\n'
             'NeoX (183 heads removed) vs Llama (154 heads removed)',
             xlabel='PPL change (%)')

    x = np.arange(len(texts))
    width = 0.35
    edge = GRID if LIGHT else BG
    ax.barh(x - width/2, neox_deltas, width, label='NeoX 410M (MHA)',
            color=RUN_COLORS['baseline'], edgecolor=edge, linewidth=0.5)
    ax.barh(x + width/2, llama_deltas, width, label='Llama 410M (GQA)',
            color=RUN_COLORS['llama-fineweb-baseline'], edgecolor=edge, linewidth=0.5)

    ax.set_yticks(x)
    ax.set_yticklabels(texts, fontsize=10)
    ax.legend(fontsize=9, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)

    save(fig, 'ablation-cross-architecture')


def chart_benchmark_structural_accuracy(use_excess=False):
    """Per-token-type structural accuracy across three models."""
    benchmark_dir = RESULTS_DIR / 'benchmark'
    if not benchmark_dir.exists():
        return

    models = []
    for name, fname in [('NeoX Baseline\n(standard BPE)', 'completion-baseline.json'),
                         ('Llama Baseline\n(standard BPE)', 'completion-llama.json'),
                         ('NeoX Comparison\n(merge barriers)', 'completion-comparison.json')]:
        path = benchmark_dir / fname
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        by_type = d['results']['structural_accuracy']['by_type']
        models.append((name, by_type))

    if len(models) < 2:
        return

    token_types = ['bracket', 'delimiter', 'spacing']
    type_labels = {'bracket': 'Bracket', 'delimiter': 'Delimiter', 'spacing': 'Spacing'}
    model_colors = ['#ff4444', '#facc15', '#18befc']

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_ax(ax, 'Structural Token Prediction Accuracy\n'
             'The capacity tax translates directly to downstream accuracy',
             ylabel='Top-1 accuracy (%)')

    x = np.arange(len(token_types))
    width = 0.25
    edge = GRID if LIGHT else BG

    for i, (name, by_type) in enumerate(models):
        vals = [by_type.get(t, {}).get('accuracy', 0) * 100 for t in token_types]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name.replace('\n', ' '),
                      color=model_colors[i], edgecolor=edge, linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                       '%.1f%%' % v, ha='center', va='bottom', fontsize=8, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels([type_labels[t] for t in token_types], fontsize=11)
    ax.legend(fontsize=9, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)
    ax.set_ylim(0, max(55, ax.get_ylim()[1]))

    save(fig, 'benchmark-structural-accuracy')


def chart_benchmark_task_comparison(use_excess=False):
    """Task-level accuracy across three models."""
    benchmark_dir = RESULTS_DIR / 'benchmark'
    if not benchmark_dir.exists():
        return

    models = []
    for name, fname in [('NeoX Baseline', 'completion-baseline.json'),
                         ('Llama Baseline', 'completion-llama.json'),
                         ('NeoX Comparison', 'completion-comparison.json')]:
        path = benchmark_dir / fname
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        models.append((name, d['results']))

    if len(models) < 2:
        return

    tasks = ['bracket_closing', 'json_structure', 'structural_accuracy']
    task_labels = {
        'bracket_closing': 'Bracket\nclosing',
        'json_structure': 'JSON\nstructure',
        'structural_accuracy': 'Overall\nstructural',
    }
    model_colors = ['#ff4444', '#facc15', '#18befc']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: task accuracy
    ax = axes[0]
    setup_ax(ax, 'Task Accuracy', ylabel='Accuracy (%)')
    x = np.arange(len(tasks))
    width = 0.25
    edge = GRID if LIGHT else BG

    for i, (name, results) in enumerate(models):
        vals = [results[t]['accuracy'] * 100 for t in tasks]
        ax.bar(x + (i - 1) * width, vals, width, label=name,
               color=model_colors[i], edgecolor=edge, linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([task_labels[t] for t in tasks], fontsize=10)
    ax.legend(fontsize=9, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)

    # Right: whitespace prediction (space vs word accuracy)
    ax = axes[1]
    setup_ax(ax, 'Whitespace Prediction', ylabel='Accuracy (%)')

    ws_metrics = ['space_accuracy', 'word_accuracy']
    ws_labels = ['Predicts\nspace token', 'Predicts\ncorrect word']
    x = np.arange(len(ws_metrics))

    for i, (name, results) in enumerate(models):
        ws = results['whitespace_prediction']
        vals = [ws.get(m, 0) * 100 for m in ws_metrics]
        ax.bar(x + (i - 1) * width, vals, width, label=name,
               color=model_colors[i], edgecolor=edge, linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ws_labels, fontsize=10)
    ax.legend(fontsize=9, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle('Downstream Completion Benchmark: Capacity Tax in Action',
                fontsize=13, fontweight='bold', color=TEXT)
    save(fig, 'benchmark-task-comparison')


ALL_CHARTS = [
    chart_developmental_timeline,
    chart_entropy_three_way,
    chart_p0_emergence,
    chart_specialization_index,
    chart_layer_depth,
    chart_polysemanticity,
    chart_seed_comparison,
    chart_emergence_order,
    chart_frustration_gap_emergence,
    chart_ablation_comparison,
    chart_ablation_per_text,
    chart_capacity_tax,
    chart_cross_architecture_emergence,
    chart_ablation_cross_architecture,
    chart_benchmark_structural_accuracy,
    chart_benchmark_task_comparison,
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
    parser.add_argument('--v2', action='store_true', help='Use v2 probe data (with spacing)')
    args = parser.parse_args()

    if args.v2:
        import generate_atlas as _self
        _self.USE_V2 = True

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
