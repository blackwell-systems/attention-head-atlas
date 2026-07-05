#!/usr/bin/env python3
"""
Token-level UMAP: extends Wang et al. (2025b) "rainbow serpent" to 410M scale.

Their Figure 1 projects tokens by 16-dimensional susceptibility vectors from
a 3M model, revealing a serpent with a "spacing fin" appendage.

We project tokens by 384-dimensional attention vectors from our 410M models,
showing that the "fin" is the dominant structure, not an appendage.

Usage:
    python generate_token_umap.py                    # light theme
    python generate_token_umap.py --dark             # dark theme
    python generate_token_umap.py --both-themes      # both
"""

import argparse
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

RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'umap'
CHARTS_DIR = Path(__file__).parent

COLORS = {
    'spacing': '#ec4899',
    'delimiter': '#18befc',
    'word_start': '#a78bfa',
    'word_part': '#c084fc',
    'numeric': '#f59e0b',
    'bracket': '#22c55e',
    'other': '#888888',
}

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


def generate_token_umap(light=True):
    set_theme(light)
    suffix = '' if light else '-dark'

    # Load both runs
    bl_path = RESULTS_DIR / 'attention-baseline.npz'
    comp_path = RESULTS_DIR / 'attention-comparison.npz'

    if not bl_path.exists() or not comp_path.exists():
        print('Missing .npz files in results/umap/')
        return

    bl = np.load(bl_path, allow_pickle=True)
    comp = np.load(comp_path, allow_pickle=True)

    # Joint UMAP: embed both in the same space
    all_vectors = np.concatenate([bl['vectors'], comp['vectors']], axis=0)
    all_types = np.concatenate([bl['types'], comp['types']])
    split = len(bl['vectors'])

    print('  Computing joint token UMAP (%d tokens, %d dimensions)...' % all_vectors.shape)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_vectors)

    bl_emb = embedding[:split]
    bl_types = all_types[:split]
    comp_emb = embedding[split:]
    comp_types = all_types[split:]

    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    type_order = ['spacing', 'delimiter', 'word_start', 'word_part', 'numeric', 'other']

    for ax, emb, types, title in [
        (ax1, bl_emb, bl_types, 'Standard BPE (baseline)\nThe "spacing fin" is the body'),
        (ax2, comp_emb, comp_types, 'Merge Barriers (comparison)\nNo spacing cluster'),
    ]:
        # Plot each type, spacing last so it's on top
        for btype in reversed(type_order):
            mask = types == btype
            if not mask.any():
                continue
            pts = emb[mask]
            count = mask.sum()
            ax.scatter(pts[:, 0], pts[:, 1],
                      c=COLORS.get(btype, '#888888'),
                      s=8, alpha=0.6, edgecolors='none')

        ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    # Token count annotations
    bl_spacing = (bl_types == 'spacing').sum()
    comp_spacing = (comp_types == 'spacing').sum()
    ax1.annotate('Spacing: %d tokens\n(%.0f%% of input)' % (bl_spacing, 100*bl_spacing/len(bl_types)),
                xy=(0.02, 0.02), xycoords='axes fraction', fontsize=9, color='#ec4899',
                fontweight='bold', va='bottom')
    ax2.annotate('Spacing: %d tokens\n(%.0f%% of input)' % (comp_spacing, 100*comp_spacing/len(comp_types)),
                xy=(0.02, 0.02), xycoords='axes fraction', fontsize=9, color='#ec4899',
                fontweight='bold', va='bottom')

    # Shared legend
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[t],
                       markersize=8, label=t) for t in type_order]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9,
              frameon=True, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle('Token-Level UMAP: 384-Dimensional Attention Profiles at 410M Scale\n'
                 'Extending Wang et al. (2025b) from 3M/16 heads to 410M/384 heads',
                fontsize=13, fontweight='bold', color=TEXT)
    fig.patch.set_facecolor(BG)
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])

    out = CHARTS_DIR / ('umap-token-comparison%s.png' % suffix)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  Saved %s' % out.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate token-level UMAP')
    parser.add_argument('--dark', action='store_true')
    parser.add_argument('--both-themes', action='store_true')
    args = parser.parse_args()

    themes = [True, False] if args.both_themes else [not args.dark]
    for light in themes:
        tag = 'light' if light else 'dark'
        print('Generating token UMAP (%s)...' % tag)
        generate_token_umap(light)

    print('Done.')
