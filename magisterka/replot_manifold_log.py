#!/usr/bin/env python3
"""Regenerate manifold distance plots in log scale from saved CSVs.

Reads `manifold_distances.csv` (per-concept) produced by manifold_analysis.py
and writes log-scale variants of the heatmaps, gap plot, and same-vs-cross
boxplot.  Does NOT re-run any model forward passes.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load_distances(csv_path: str) -> Tuple[Dict, List[int], List[str]]:
    """Load per-concept distances.

    Returns (data, sample_layers, test_words) where
      data[(q, m)][metric] = np.ndarray of values aligned with test_words order.
    """
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    layers = sorted({int(r['query_layer']) for r in rows}
                    | {int(r['manifold_layer']) for r in rows})
    # Preserve first-seen order of test concepts
    seen = set()
    test_words: List[str] = []
    for r in rows:
        c = r['test_concept']
        if c not in seen:
            seen.add(c)
            test_words.append(c)

    metric_names = ('mahalanobis', 'pca_residual', 'nearest_eucl')
    buckets: Dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        q = int(r['query_layer'])
        m = int(r['manifold_layer'])
        for name in metric_names:
            buckets[(q, m)][name][r['test_concept']] = float(r[name])

    data: Dict = {}
    for key, by_name in buckets.items():
        data[key] = {
            name: np.array([by_name[name][c] for c in test_words], dtype=np.float64)
            for name in metric_names
        }
    return data, layers, test_words


def _safe_log_norm(mat: np.ndarray) -> LogNorm:
    """LogNorm with positive floor to handle zeros/negatives gracefully."""
    pos = mat[mat > 0]
    if pos.size == 0:
        return LogNorm(vmin=1e-12, vmax=1.0)
    vmin = float(pos.min())
    vmax = float(pos.max())
    if vmax <= vmin:
        vmax = vmin * 10
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_heatmaps_log(data: Dict, layers: List[int], model_label: str, out_dir: str):
    metric_names = ['mahalanobis', 'pca_residual', 'nearest_eucl']
    nL = len(layers)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, name in zip(axes, metric_names):
        mat = np.zeros((nL, nL))
        for i, q in enumerate(layers):
            for j, m in enumerate(layers):
                mat[i, j] = data[(q, m)][name].mean()
        im = ax.imshow(
            np.maximum(mat, 0.0),
            aspect='auto', cmap='viridis', origin='lower',
            extent=[layers[0] - 0.5, layers[-1] + 0.5,
                    layers[0] - 0.5, layers[-1] + 0.5],
            norm=_safe_log_norm(mat),
        )
        ax.set_xlabel('Manifold Layer')
        ax.set_ylabel('Query (Test-vector) Layer')
        ax.set_title(f'{name}  (mean, log color)')
        ax.plot([layers[0], layers[-1]], [layers[0], layers[-1]],
                'r--', alpha=0.5, linewidth=1)
        plt.colorbar(im, ax=ax)
    fig.suptitle(f'Manifold distance heatmaps (log color)  ({model_label})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'distance_heatmaps_log.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_gap_log(data: Dict, model_label: str, out_dir: str):
    metric_names = ['mahalanobis', 'pca_residual', 'nearest_eucl']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, name in zip(axes, metric_names):
        gaps: Dict[int, List[float]] = {}
        for (q, m), cell in data.items():
            g = abs(m - q)
            gaps.setdefault(g, []).extend(cell[name].tolist())
        sorted_g = sorted(gaps.keys())
        means = np.array([np.mean(gaps[g]) for g in sorted_g])
        stds = np.array([np.std(gaps[g]) for g in sorted_g])
        # Clip lower error bar for log axis
        lower = np.maximum(means - stds, np.where(means > 0, means * 1e-3, 1e-12))
        yerr = np.vstack([means - lower, stds])
        ax.errorbar(sorted_g, np.maximum(means, 1e-12), yerr=yerr,
                    fmt='o-', capsize=3, color='steelblue')
        ax.set_xlabel('|manifold_layer − query_layer|')
        ax.set_ylabel(f'{name} (log)')
        ax.set_yscale('log')
        ax.set_title(f'{name} vs layer gap (log)')
        ax.grid(True, alpha=0.3, which='both')
    fig.suptitle(f'Distance vs layer gap (log y)  ({model_label})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'distance_vs_gap_log.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_same_vs_cross_log(data: Dict, model_label: str, out_dir: str):
    metric_names = ['mahalanobis', 'pca_residual', 'nearest_eucl']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, name in zip(axes, metric_names):
        same, cross = [], []
        for (q, m), cell in data.items():
            (same if q == m else cross).extend(cell[name].tolist())
        # Filter non-positive values for log boxplot
        same_pos = [v for v in same if v > 0]
        cross_pos = [v for v in cross if v > 0]
        ax.boxplot([same_pos, cross_pos],
                   labels=['same layer', 'cross layer'], showmeans=True)
        ax.set_ylabel(f'{name} (log)')
        ax.set_yscale('log')
        ax.set_title(f'{name}: same vs cross layer (log)')
        ax.grid(True, alpha=0.3, axis='y', which='both')
    fig.suptitle(f'Same vs cross-layer distance (log y)  ({model_label})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'same_vs_cross_layer_log.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    p = argparse.ArgumentParser(description="Re-plot manifold metrics in log scale.")
    p.add_argument('--results-dir', required=True,
                   help='Directory containing manifold_distances.csv')
    p.add_argument('--csv', default=None,
                   help='Optional explicit path to the per-concept CSV')
    args = p.parse_args()

    csv_path = args.csv or os.path.join(args.results_dir, 'manifold_distances.csv')
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")

    meta_path = os.path.join(args.results_dir, 'manifold_metadata.json')
    model_label = 'manifold'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        model_label = meta.get('model_name', model_label).split('/')[-1]

    data, layers, _ = load_distances(csv_path)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    plot_heatmaps_log(data, layers, model_label, args.results_dir)
    plot_gap_log(data, model_label, args.results_dir)
    plot_same_vs_cross_log(data, model_label, args.results_dir)


if __name__ == '__main__':
    main()
