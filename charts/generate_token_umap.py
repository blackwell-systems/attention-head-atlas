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


UMAP_DATA_DIR = Path(__file__).parent / 'umap-data'


def generate_cross_architecture_umap(light=True):
    """3-panel UMAP: NeoX baseline, Llama baseline, NeoX comparison."""
    set_theme(light)
    suffix = '' if light else '-dark'

    bl_path = RESULTS_DIR / 'attention-baseline.npz'
    comp_path = RESULTS_DIR / 'attention-comparison.npz'
    llama_path = UMAP_DATA_DIR / 'attention-llama-fineweb.npz'

    if not all(p.exists() for p in [bl_path, comp_path, llama_path]):
        print('Missing .npz files for cross-architecture UMAP')
        return

    bl = np.load(bl_path, allow_pickle=True)
    comp = np.load(comp_path, allow_pickle=True)
    llama = np.load(llama_path, allow_pickle=True)

    # Joint UMAP across all three
    all_vectors = np.concatenate([bl['vectors'], llama['vectors'], comp['vectors']], axis=0)
    all_types = np.concatenate([bl['types'], llama['types'], comp['types']])
    split1 = len(bl['vectors'])
    split2 = split1 + len(llama['vectors'])

    print('  Computing joint 3-way token UMAP (%d tokens, %d dimensions)...' % all_vectors.shape)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_vectors)

    panels = [
        (embedding[:split1], all_types[:split1],
         'NeoX 410M (MHA)\nStandard BPE'),
        (embedding[split1:split2], all_types[split1:split2],
         'Llama 410M (GQA)\nStandard BPE'),
        (embedding[split2:], all_types[split2:],
         'NeoX 410M (MHA)\nMerge Barriers'),
    ]

    type_order = ['spacing', 'delimiter', 'word_start', 'word_part', 'numeric', 'other']

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (emb, types, title) in zip(axes, panels):
        for btype in reversed(type_order):
            mask = types == btype
            if not mask.any():
                continue
            pts = emb[mask]
            ax.scatter(pts[:, 0], pts[:, 1],
                      c=COLORS.get(btype, '#888888'),
                      s=8, alpha=0.6, edgecolors='none')

        ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(GRID)

        spacing_count = (types == 'spacing').sum()
        ax.annotate('Spacing: %d tokens (%.0f%%)' % (spacing_count, 100*spacing_count/len(types)),
                    xy=(0.02, 0.02), xycoords='axes fraction', fontsize=9, color='#ec4899',
                    fontweight='bold', va='bottom')

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[t],
                       markersize=8, label=t) for t in type_order]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9,
              frameon=True, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle('Cross-Architecture Token UMAP: Spacing Fin Across MHA and GQA\n'
                 '384-dimensional attention profiles, joint embedding',
                fontsize=13, fontweight='bold', color=TEXT)
    fig.patch.set_facecolor(BG)
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])

    out = CHARTS_DIR / ('umap-cross-architecture%s.png' % suffix)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  Saved %s' % out.name)


def generate_llama_developmental_umap(light=True):
    """Developmental UMAP: Llama at steps 0, 150, 500, 2000, 20000."""
    set_theme(light)
    suffix = '' if light else '-dark'

    steps = ['00000', '00150', '00500', '02000']
    step_labels = ['Step 0\n(random init)', 'Step 150', 'Step 500', 'Step 2000']
    paths = [UMAP_DATA_DIR / ('attention-llama-dev-%s.npz' % s) for s in steps]
    paths.append(UMAP_DATA_DIR / 'attention-llama-fineweb.npz')
    step_labels.append('Step 20000\n(converged)')

    missing = [p for p in paths if not p.exists()]
    if missing:
        print('Missing developmental UMAP files: %s' % [p.name for p in missing])
        return

    datasets = [np.load(p, allow_pickle=True) for p in paths]

    # Joint UMAP
    all_vectors = np.concatenate([d['vectors'] for d in datasets], axis=0)
    all_types = np.concatenate([d['types'] for d in datasets])
    splits = np.cumsum([len(d['vectors']) for d in datasets])

    print('  Computing developmental UMAP for Llama (%d tokens, %d dims)...' % all_vectors.shape)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_vectors)

    type_order = ['spacing', 'delimiter', 'word_start', 'word_part', 'numeric', 'other']

    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 6))

    start = 0
    for i, (ax, label) in enumerate(zip(axes, step_labels)):
        end = splits[i]
        emb = embedding[start:end]
        types = all_types[start:end]
        start = end

        for btype in reversed(type_order):
            mask = types == btype
            if not mask.any():
                continue
            pts = emb[mask]
            ax.scatter(pts[:, 0], pts[:, 1],
                      c=COLORS.get(btype, '#888888'),
                      s=8, alpha=0.6, edgecolors='none')

        ax.set_title(label, fontsize=10, fontweight='bold', color=TEXT)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[t],
                       markersize=8, label=t) for t in type_order]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9,
              frameon=True, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle('Llama 410M (GQA): Spacing Fin Emergence\n'
                 'Developmental UMAP across training',
                fontsize=13, fontweight='bold', color=TEXT)
    fig.patch.set_facecolor(BG)
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])

    out = CHARTS_DIR / ('umap-llama-developmental%s.png' % suffix)
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
        generate_cross_architecture_umap(light)
        generate_llama_developmental_umap(light)

    print('Done.')
