#!/usr/bin/env python3
"""
Generate the developmental atlas visualization from probe results.

Reads all step-XXXXX.json files from a probe-results directory and produces:
1. developmental-timeline.png: heatmap (x=step, y=head, color=dominant type)
2. type-counts-over-time.png: stacked area chart of head type distribution
3. emergence-order.png: when each behavior type first exceeds threshold
4. layer-depth.png: which layers specialize first

Usage:
  python generate_atlas.py --results-dir ../runs/baseline/probe-results/ --output-dir ./
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.rcParams.update({
    'font.size': 11,
    'figure.facecolor': '#0a0a0a',
    'axes.facecolor': '#0a0a0a',
    'savefig.facecolor': '#0a0a0a',
})

BEHAVIOR_COLORS = {
    'positional_prev': '#ff4444',
    'positional_p0': '#ff8800',
    'induction': '#ffcc00',
    'delimiter': '#18befc',
    'bracket': '#22c55e',
    'content': '#a855f7',
    'duplicate': '#ec4899',
    'dormant': '#666666',
}


def load_results(results_dir):
    """Load all probe results sorted by step."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("step-*.json"))

    data = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
            step = int(f.stem.split("-")[1])
            d["step"] = step
            data.append(d)

    return sorted(data, key=lambda x: x["step"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", default="./")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        print("No results found in %s" % args.results_dir)
        return

    steps = [r["step"] for r in results]
    num_steps = len(steps)
    num_heads = 384  # 24 layers * 16 heads

    print("Loaded %d checkpoints (steps %d to %d)" % (num_steps, steps[0], steps[-1]))

    # Build classification matrix [steps x heads]
    behavior_types = list(BEHAVIOR_COLORS.keys())
    type_to_idx = {t: i for i, t in enumerate(behavior_types)}

    classification_matrix = np.zeros((num_steps, num_heads), dtype=int)
    for step_idx, r in enumerate(results):
        for c in r["classifications"]:
            head_idx = c["layer"] * 16 + c["head"]
            classification_matrix[step_idx, head_idx] = type_to_idx.get(c["dominant"], 0)

    # Chart 1: Developmental timeline heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    cmap = plt.cm.colors.ListedColormap([BEHAVIOR_COLORS[t] for t in behavior_types])
    im = ax.imshow(classification_matrix.T, aspect='auto', cmap=cmap,
                   interpolation='nearest', origin='lower')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Head Index (layer*16 + head)')
    ax.set_title('Attention Head Developmental Atlas\nColor = dominant behavior type at each training step')

    # X-axis labels
    tick_positions = np.linspace(0, num_steps-1, min(10, num_steps), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(steps[i]) for i in tick_positions])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=BEHAVIOR_COLORS[t], label=t) for t in behavior_types]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'developmental-timeline.png', dpi=150)
    plt.close()
    print("  Chart 1: developmental-timeline.png")

    # Chart 2: Type counts over time (stacked area)
    fig, ax = plt.subplots(figsize=(12, 6))
    type_counts_over_time = np.zeros((num_steps, len(behavior_types)))
    for step_idx in range(num_steps):
        for type_idx in range(len(behavior_types)):
            type_counts_over_time[step_idx, type_idx] = np.sum(
                classification_matrix[step_idx] == type_idx)

    ax.stackplot(steps, type_counts_over_time.T,
                 labels=behavior_types,
                 colors=[BEHAVIOR_COLORS[t] for t in behavior_types])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Number of Heads')
    ax.set_title('Head Type Distribution Over Training')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(steps[0], steps[-1])

    plt.tight_layout()
    plt.savefig(output_dir / 'type-counts-over-time.png', dpi=150)
    plt.close()
    print("  Chart 2: type-counts-over-time.png")

    print("\nDone. Charts saved to %s" % output_dir)


if __name__ == "__main__":
    main()
