"""Generate additional heatmaps from a saved detection sweep pickle.

Produces two extra heatmaps next to the existing introspection/control pair:

  1. `<base>_diff.png`            intro - control                       (per-cell)
  2. `<base>_gain_ratio.png`      intro_gain / control_gain             (per-cell)

where `*_gain = mean - baseline` and the baselines are read from
`metadata.baseline_intro` / `metadata.baseline_control`. Per-cell values are
averaged over concepts for each (sample_layer, injection_layer) pair.

Usage
-----
    python replot_detection_extras.py path/to/detection_results_*.pkl [--output-dir DIR]
"""

from __future__ import annotations
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _aggregate(per_cell, sample_layers, injection_layers, key):
    """Average `per_cell[(S, I, concept)][key]` over concepts -> (|S|, |I|)."""
    s_idx = {s: i for i, s in enumerate(sample_layers)}
    i_idx = {l: i for i, l in enumerate(injection_layers)}
    sums = np.zeros((len(sample_layers), len(injection_layers)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for (s, l, _concept), cell in per_cell.items():
        if s not in s_idx or l not in i_idx:
            continue
        v = cell.get(key)
        if v is None:
            continue
        sums[s_idx[s], i_idx[l]] += float(v)
        counts[s_idx[s], i_idx[l]] += 1.0
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(counts > 0, sums / counts, np.nan)


def _save_heatmap(matrix, sample_layers, injection_layers, title, cbar_label,
                  filename, cmap='RdBu_r', symmetric=True, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    if symmetric and vmin is None and vmax is None:
        finite = matrix[np.isfinite(matrix)]
        if finite.size:
            m = float(np.nanmax(np.abs(finite)))
            vmin, vmax = -m, m
    im = ax.imshow(
        matrix, aspect='auto', cmap=cmap, origin='lower',
        extent=[injection_layers[0], injection_layers[-1],
                sample_layers[0], sample_layers[-1]],
        vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel('Injection Layer Index', fontsize=12)
    ax.set_ylabel('Sample Layer Index', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl', help='Path to detection_results_*.pkl')
    ap.add_argument('--output-dir', default=None,
                    help='Defaults to the pkl directory')
    ap.add_argument('--ratio-clip', type=float, default=5.0,
                    help='Clip |intro_gain/control_gain| to this value for plotting (default 5).')
    ap.add_argument('--ratio-eps', type=float, default=1e-3,
                    help='Mask cells where |control_gain| < eps to avoid noise blow-up (default 1e-3).')
    args = ap.parse_args()

    pkl_path = Path(args.pkl).resolve()
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    meta = results['metadata']
    sample_layers = list(meta['sample_layers'])
    injection_layers = list(meta['injection_layers'])
    baseline_intro = float(meta.get('baseline_intro', 0.0))
    baseline_control = float(meta.get('baseline_control', 0.0))
    mode_tag = meta.get('vector_mode', 'concept')
    per_cell = results['per_cell']

    intro = _aggregate(per_cell, sample_layers, injection_layers, 'intro_mean')
    control = _aggregate(per_cell, sample_layers, injection_layers, 'control_mean')

    diff = intro - control
    intro_gain = intro - baseline_intro
    control_gain = control - baseline_control
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = intro_gain / control_gain
    # Mask near-zero denominators (otherwise the heatmap is dominated by noise)
    ratio = np.where(np.abs(control_gain) < args.ratio_eps, np.nan, ratio)
    if args.ratio_clip is not None and args.ratio_clip > 0:
        ratio = np.clip(ratio, -args.ratio_clip, args.ratio_clip)

    out_dir = Path(args.output_dir) if args.output_dir else pkl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = pkl_path.stem  # detection_results_<mode>_<ts>

    model_display = str(meta.get('model_name', '')).split('/')[-1]

    _save_heatmap(
        diff, sample_layers, injection_layers,
        title=f'Introspection \u2212 Control ({mode_tag})\n{model_display}',
        cbar_label='intro_mean \u2212 control_mean  [logit(Yes) \u2212 logit(No)]',
        filename=str(out_dir / f"{base}_diff.png"),
    )

    _save_heatmap(
        ratio, sample_layers, injection_layers,
        title=(f'Introspection gain / Control gain ({mode_tag})\n'
               f'{model_display}  '
               f'(baseline_intro={baseline_intro:.2f}, baseline_control={baseline_control:.2f}; '
               f'masked |control_gain|<{args.ratio_eps}; clipped \u00b1{args.ratio_clip})'),
        cbar_label='(intro \u2212 baseline_intro) / (control \u2212 baseline_control)',
        filename=str(out_dir / f"{base}_gain_ratio.png"),
    )

    # Also dump the raw matrices alongside the plots for downstream analysis.
    np.savez(
        out_dir / f"{base}_extras.npz",
        sample_layers=np.array(sample_layers),
        injection_layers=np.array(injection_layers),
        intro=intro, control=control,
        intro_gain=intro_gain, control_gain=control_gain,
        diff=diff, ratio_raw=intro_gain / np.where(control_gain == 0, np.nan, control_gain),
        baseline_intro=baseline_intro, baseline_control=baseline_control,
    )
    print(f"saved {out_dir / (base + '_extras.npz')}")


if __name__ == '__main__':
    main()
