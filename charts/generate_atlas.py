#!/usr/bin/env python3
"""
Generate atlas visualization charts from probe results.

Produces publication-quality charts from the developmental atlas data.
Supports both raw and excess-corrected results.

Usage:
    python generate_atlas.py                    # light theme (default)
    python generate_atlas.py --dark             # dark theme
    python generate_atlas.py --use-excess       # use excess-corrected data
    python generate_atlas.py --both-themes      # generate both variants
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

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


def chart_developmental_timeline(use_excess=False):
    """Stacked area chart of head type distribution over training."""
    label = 'excess' if use_excess else 'raw'
    b_dir = RESULTS_DIR / ('baseline-excess' if use_excess else 'baseline')
    c_dir = RESULTS_DIR / ('comparison-excess' if use_excess else 'comparison')

    b_steps, b_types, _, _, _ = load_timeline(b_dir)
    c_steps, c_types, _, _, _ = load_timeline(c_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, steps, tc, title in [
        (ax1, b_steps, b_types, 'Baseline (standard BPE)'),
        (ax2, c_steps, c_types, 'Comparison (merge barriers)'),
    ]:
        setup_ax(ax, title, xlabel='Training step', ylabel='Number of heads')
        bottom = np.zeros(len(steps))
        for b in BEHAVIORS:
            vals = np.array(tc[b])
            ax.fill_between(steps, bottom, bottom + vals, alpha=0.7,
                           color=COLORS[b], label=b)
            bottom += vals
        ax.set_ylim(0, 384)
        ax.legend(loc='center right', fontsize=8, facecolor=LEGEND_BG,
                 edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle('Attention Head Atlas: Developmental Timeline (%s scores)' % label,
                fontsize=14, fontweight='bold', color=TEXT)
    save(fig, 'developmental-timeline-%s' % label)


def chart_entropy_divergence(use_excess=False):
    """Attention entropy over training for both runs."""
    b_dir = RESULTS_DIR / ('baseline-excess' if use_excess else 'baseline')
    c_dir = RESULTS_DIR / ('comparison-excess' if use_excess else 'comparison')

    b_steps, _, _, b_ent, _ = load_timeline(b_dir)
    c_steps, _, _, c_ent, _ = load_timeline(c_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_ax(ax, 'Attention Entropy Over Training\nBaseline becomes diffuse; merge barriers stay focused',
             xlabel='Training step', ylabel='Mean attention entropy')
    ax.plot(b_steps, b_ent, color='#ff4444', linewidth=2, label='Baseline (standard BPE)')
    ax.plot(c_steps, c_ent, color='#18befc', linewidth=2, label='Comparison (merge barriers)')
    ax.legend(fontsize=10, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)
    save(fig, 'entropy-divergence')


def chart_p0_emergence(use_excess=False):
    """P0 sink head count over training."""
    b_dir = RESULTS_DIR / ('baseline-excess' if use_excess else 'baseline')
    c_dir = RESULTS_DIR / ('comparison-excess' if use_excess else 'comparison')

    b_steps, b_types, _, _, _ = load_timeline(b_dir)
    c_steps, c_types, _, _, _ = load_timeline(c_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_ax(ax, 'Dormant Head Emergence: Merge Barriers Reduce Dormancy',
             xlabel='Training step', ylabel='P0 sink heads')
    ax.plot(b_steps, b_types['positional_p0'], color='#ff4444', linewidth=2,
           label='Baseline (standard BPE)')
    ax.plot(c_steps, c_types['positional_p0'], color='#18befc', linewidth=2,
           label='Comparison (merge barriers)')
    ax.legend(fontsize=10, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)

    label = 'excess' if use_excess else 'raw'
    save(fig, 'p0-sink-emergence-%s' % label)


def chart_specialization_index(use_excess=False):
    """Mean specialization index over training."""
    b_dir = RESULTS_DIR / ('baseline-excess' if use_excess else 'baseline')
    c_dir = RESULTS_DIR / ('comparison-excess' if use_excess else 'comparison')

    b_steps, _, b_spec, _, _ = load_timeline(b_dir)
    c_steps, _, c_spec, _, _ = load_timeline(c_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_ax(ax, 'Head Specialization Over Training',
             xlabel='Training step', ylabel='Mean specialization index')
    ax.plot(b_steps, b_spec, color='#ff4444', linewidth=2, label='Baseline (standard BPE)')
    ax.plot(c_steps, c_spec, color='#18befc', linewidth=2, label='Comparison (merge barriers)')
    ax.legend(fontsize=10, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)

    label = 'excess' if use_excess else 'raw'
    save(fig, 'specialization-index-%s' % label)


def chart_layer_depth(use_excess=False):
    """Heatmap of per-layer specialization index over training."""
    b_dir = RESULTS_DIR / ('baseline-excess' if use_excess else 'baseline')
    c_dir = RESULTS_DIR / ('comparison-excess' if use_excess else 'comparison')

    b_steps, _, _, _, b_layer = load_timeline(b_dir)
    c_steps, _, _, _, c_layer = load_timeline(c_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    for ax, layer_spec, steps, title in [
        (ax1, b_layer, b_steps, 'Baseline'),
        (ax2, c_layer, c_steps, 'Comparison'),
    ]:
        im = ax.imshow(layer_spec.T, aspect='auto', cmap='viridis',
                       extent=[steps[0], steps[-1], 23.5, -0.5],
                       interpolation='nearest')
        setup_ax(ax, '%s: Specialization by Layer' % title,
                xlabel='Training step', ylabel='Layer')
        plt.colorbar(im, ax=ax, label='Mean specialization index')

    label = 'excess' if use_excess else 'raw'
    fig.suptitle('Layer-Depth Specialization Over Training (%s scores)' % label,
                fontsize=14, fontweight='bold', color=TEXT)
    save(fig, 'layer-depth-specialization-%s' % label)


def chart_polysemanticity(use_excess=False):
    """Specialist vs generalist head counts over training."""
    label = 'excess' if use_excess else 'raw'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, run, title in [
        (ax1, 'baseline-excess' if use_excess else 'baseline', 'Baseline'),
        (ax2, 'comparison-excess' if use_excess else 'comparison', 'Comparison'),
    ]:
        run_dir = RESULTS_DIR / run
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

        setup_ax(ax, title, xlabel='Training step', ylabel='Number of heads')
        ax.plot(steps, specialists, color='#18befc', linewidth=2, label='Specialists (>0.7)')
        ax.plot(steps, generalists, color='#ff9944', linewidth=2, label='Generalists (<0.3)')
        ax.fill_between(steps, 0, specialists, alpha=0.1, color='#18befc')
        ax.fill_between(steps, 0, generalists, alpha=0.1, color='#ff9944')
        ax.legend(fontsize=10, facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle('Polysemanticity: Specialists vs Generalists (%s scores)' % label,
                fontsize=14, fontweight='bold', color=TEXT)
    save(fig, 'polysemanticity-%s' % label)


ALL_CHARTS = [
    chart_developmental_timeline,
    chart_entropy_divergence,
    chart_p0_emergence,
    chart_specialization_index,
    chart_layer_depth,
    chart_polysemanticity,
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
                if fn == chart_entropy_divergence:
                    fn()  # entropy doesn't change with excess correction
                else:
                    fn(use_excess=use_excess)

    print('Done.')
