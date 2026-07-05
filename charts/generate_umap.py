#!/usr/bin/env python3
"""
UMAP projection of head behavior vectors at step 20000.

Extends Wang et al. (2025b) "Embryology of a Language Model" to 410M scale.
Their Figure 1 shows a "rainbow serpent" UMAP of 16-dimensional susceptibility
vectors from a 3M model. We project 7-dimensional behavior score vectors from
384 heads across 6 runs, colored by dominant behavior type.

Provenance: written 2026-07-05 for the developmental atlas project.
Motivated by Wang et al. (2025b) who discovered the "spacing fin" in their
3M UMAP and our confirmation that spacing is 47.7% of heads at 410M.

Usage:
    python generate_umap.py                    # light theme
    python generate_umap.py --dark             # dark theme
    python generate_umap.py --both-themes      # both
    python generate_umap.py --developmental     # animate across checkpoints
"""

import argparse
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import umap
except ImportError:
    print("Install umap-learn: pip install umap-learn")
    exit(1)

RESULTS_DIR = Path(__file__).parent.parent / 'results'
CHARTS_DIR = Path(__file__).parent

BEHAVIORS = ['positional_prev', 'positional_p0', 'induction', 'delimiter', 'bracket', 'duplicate', 'spacing']

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

RUN_NAMES = {
    'baseline-v2-excess': 'Baseline (standard BPE)',
    'comparison-v2-excess': 'Comparison (struct barriers)',
    'seed2-v2-excess': 'Seed2 (standard BPE, diff init)',
    'nl-barrier-v2-excess': 'NL barriers',
    'structok-baseline-excess': 'Structok corpus (standard BPE)',
    'structok-comparison-excess': 'Structok corpus (struct barriers)',
}

# Theme
BG = 'white'
TEXT = '#1a1a1a'
GRID = '#cccccc'


def set_theme(light=True):
    global BG, TEXT, GRID
    if light:
        BG, TEXT, GRID = 'white', '#1a1a1a', '#cccccc'
        plt.style.use('default')
    else:
        BG, TEXT, GRID = '#0a0a0a', 'white', '#333333'
        plt.style.use('dark_background')


def load_head_vectors(run_dir, step='step-20000'):
    """Load 7-behavior score vectors for all 384 heads."""
    path = run_dir / ('%s.json' % step)
    if not path.exists():
        return None, None

    with open(path) as f:
        d = json.load(f)

    vectors = []
    types = []
    for c in d['classifications']:
        scores = c.get('excess_scores', c.get('scores', {}))
        vec = [scores.get(b, 0) for b in BEHAVIORS]
        vectors.append(vec)
        types.append(c['dominant'])

    return np.array(vectors), types


def plot_umap_single(ax, embedding, types, title):
    """Plot a single UMAP projection on an axis."""
    for btype in BEHAVIORS + ['unclassified']:
        mask = [t == btype for t in types]
        if not any(mask):
            continue
        pts = embedding[mask]
        ax.scatter(pts[:, 0], pts[:, 1],
                  c=COLORS.get(btype, '#888888'),
                  s=15, alpha=0.7, edgecolors='none',
                  label='%s (%d)' % (btype, sum(mask)))

    ax.set_title(title, fontsize=10, fontweight='bold', color=TEXT)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def generate_comparison_umap(light=True):
    """Generate 2x3 grid of UMAP projections for all 6 runs."""
    set_theme(light)
    suffix = '' if light else '-dark'

    runs = [
        'baseline-v2-excess',
        'comparison-v2-excess',
        'seed2-v2-excess',
        'nl-barrier-v2-excess',
        'structok-baseline-excess',
        'structok-comparison-excess',
    ]

    # Collect all vectors for a joint UMAP (same embedding space for comparison)
    all_vectors = []
    all_types = []
    all_runs = []
    run_slices = {}

    for run in runs:
        run_dir = RESULTS_DIR / run
        vectors, types = load_head_vectors(run_dir)
        if vectors is None:
            print('  Skipping %s (not found)' % run)
            continue
        start = len(all_vectors)
        all_vectors.extend(vectors)
        all_types.extend(types)
        all_runs.extend([run] * len(vectors))
        run_slices[run] = (start, start + len(vectors))

    if not all_vectors:
        print('No data found')
        return

    all_vectors = np.array(all_vectors)

    print('  Computing joint UMAP (%d points, %d dimensions)...' % all_vectors.shape)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_vectors)

    # Plot 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, run in enumerate(runs):
        if run not in run_slices:
            axes[i].set_visible(False)
            continue
        start, end = run_slices[run]
        run_embedding = embedding[start:end]
        run_types = all_types[start:end]
        plot_umap_single(axes[i], run_embedding, run_types, RUN_NAMES.get(run, run))

    # Shared legend
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[b],
                       markersize=8, label=b) for b in BEHAVIORS + ['unclassified']]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
              frameon=True, facecolor=BG, edgecolor=GRID,
              labelcolor=TEXT)

    fig.suptitle('Head Behavior UMAP: 384 Heads Across 6 Runs\n'
                 'Joint embedding (same coordinate space for direct comparison)',
                fontsize=13, fontweight='bold', color=TEXT)
    fig.patch.set_facecolor(BG)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    out = CHARTS_DIR / ('umap-comparison%s.png' % suffix)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  Saved %s' % out.name)


def generate_developmental_umap(light=True):
    """Generate UMAP at multiple training steps to show the serpent forming."""
    set_theme(light)
    suffix = '' if light else '-dark'

    steps = ['step-00000', 'step-00050', 'step-00150', 'step-00500',
             'step-02000', 'step-05000', 'step-10000', 'step-20000']

    run_dir = RESULTS_DIR / 'baseline-v2-excess'

    # Collect all vectors across all steps for joint embedding
    all_vectors = []
    all_types = []
    step_slices = {}

    for step in steps:
        vectors, types = load_head_vectors(run_dir, step)
        if vectors is None:
            print('  Skipping %s (not found)' % step)
            continue
        start = len(all_vectors)
        all_vectors.extend(vectors)
        all_types.extend(types)
        step_slices[step] = (start, start + len(vectors))

    if not all_vectors:
        print('No data found')
        return

    all_vectors = np.array(all_vectors)

    print('  Computing developmental UMAP (%d points)...' % len(all_vectors))
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_vectors)

    # Plot 2x4 grid
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
    axes = axes.flatten()

    for i, step in enumerate(steps):
        if step not in step_slices:
            if i < len(axes):
                axes[i].set_visible(False)
            continue
        start, end = step_slices[step]
        step_embedding = embedding[start:end]
        step_types = all_types[start:end]
        step_num = int(step.replace('step-', ''))
        plot_umap_single(axes[i], step_embedding, step_types, 'Step %d' % step_num)

    for i in range(len(steps), len(axes)):
        axes[i].set_visible(False)

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[b],
                       markersize=8, label=b) for b in BEHAVIORS + ['unclassified']]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
              frameon=True, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle('Developmental UMAP: Baseline Head Organization Over Training\n'
                 'Spacing (pink) dominates by step 150; the "spacing fin" is the largest structure',
                fontsize=13, fontweight='bold', color=TEXT)
    fig.patch.set_facecolor(BG)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    out = CHARTS_DIR / ('umap-developmental%s.png' % suffix)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  Saved %s' % out.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate UMAP visualizations')
    parser.add_argument('--dark', action='store_true')
    parser.add_argument('--both-themes', action='store_true')
    parser.add_argument('--developmental', action='store_true', help='Also generate developmental UMAP')
    args = parser.parse_args()

    themes = [True, False] if args.both_themes else [not args.dark]

    for light in themes:
        tag = 'light' if light else 'dark'
        print('Generating UMAPs (%s)...' % tag)
        generate_comparison_umap(light)
        if args.developmental:
            generate_developmental_umap(light)

    print('Done.')
