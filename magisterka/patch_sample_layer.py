"""Splice a single-sample-layer rerun into an existing detection sweep pickle.

Use this when one sample-layer vector file was wrong / misplaced and you have
rerun just that row with the harness (single `--sample-layers <S>` and the
correct `--sample-layer-files`). This script:

  * loads the original full-sweep pickle and the single-row patch pickle
  * sanity-checks that injection_layers, vector_mode and the patched sample
    layer match
  * replaces all `per_cell[(S, *, *)]`, `summary_per_sample_layer[S]` and
    `sample_layer_aggregates[S]` entries for the patched layer
  * writes a new pickle/json/csv/peaks csv with a `_patched` tag
  * regenerates the standard intro/control heatmap and (optionally) the
    diff / gain-ratio extras

Usage
-----
    python patch_sample_layer.py ORIGINAL.pkl PATCH.pkl [--output-dir DIR]
                                 [--no-extras] [--no-heatmap]
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _load(p: str | Path):
    with open(p, 'rb') as f:
        return pickle.load(f)


def _validate(original: dict, patch: dict) -> int:
    o_meta = original['metadata']
    p_meta = patch['metadata']

    o_il = list(o_meta['injection_layers'])
    p_il = list(p_meta['injection_layers'])
    if o_il != p_il:
        raise ValueError(f"injection_layers differ: original={o_il} patch={p_il}")

    o_mode = o_meta.get('vector_mode')
    p_mode = p_meta.get('vector_mode')
    if o_mode != p_mode:
        raise ValueError(f"vector_mode differs: original={o_mode!r} patch={p_mode!r}")

    p_sl = list(p_meta['sample_layers'])
    if len(p_sl) != 1:
        raise ValueError(f"patch pickle must contain exactly one sample layer, got {p_sl}")
    patched_layer = p_sl[0]
    if patched_layer not in o_meta['sample_layers']:
        raise ValueError(
            f"patched sample layer {patched_layer} is not in original sample_layers "
            f"{list(o_meta['sample_layers'])}"
        )

    o_concepts = set(original['per_cell'].keys())
    o_concepts_for_S = {c for (s, _l, c) in o_concepts if s == patched_layer}
    p_concepts_for_S = {c for (s, _l, c) in patch['per_cell'].keys() if s == patched_layer}
    missing = o_concepts_for_S - p_concepts_for_S
    extra = p_concepts_for_S - o_concepts_for_S
    if missing:
        print(f"[warn] patch is missing {len(missing)} concepts that the original "
              f"had for layer {patched_layer}; those cells will be left as-is.")
    if extra:
        print(f"[warn] patch adds {len(extra)} new concepts for layer "
              f"{patched_layer}; they will be inserted.")
    return patched_layer


def _splice(original: dict, patch: dict, S: int) -> dict:
    merged = {
        'metadata': dict(original['metadata']),
        'per_cell': dict(original['per_cell']),
        'summary_per_sample_layer': dict(original['summary_per_sample_layer']),
        'sample_layer_aggregates': dict(original['sample_layer_aggregates']),
    }

    # Drop only the cells for the patched layer, then merge in patch cells for S.
    merged['per_cell'] = {k: v for k, v in merged['per_cell'].items() if k[0] != S}
    for k, v in patch['per_cell'].items():
        if k[0] != S:
            continue
        merged['per_cell'][k] = v

    if S in patch['summary_per_sample_layer']:
        merged['summary_per_sample_layer'][S] = patch['summary_per_sample_layer'][S]
    if S in patch['sample_layer_aggregates']:
        merged['sample_layer_aggregates'][S] = patch['sample_layer_aggregates'][S]

    # Note the patch in metadata so it shows up in JSON / future loads.
    merged['metadata'].setdefault('patches', [])
    merged['metadata']['patches'].append({
        'patched_sample_layer': S,
        'patch_timestamp': patch['metadata'].get('timestamp'),
        'patch_model_name': patch['metadata'].get('model_name'),
        'applied_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
    })
    return merged


def _save(results: dict, output_dir: Path, base_stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{base_stem}_patched_{ts}"

    pkl_path = output_dir / f"{stem}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"saved {pkl_path}")

    json_path = output_dir / f"{stem}.json"
    json_data = {
        'metadata': results['metadata'],
        'summary_per_sample_layer': results['summary_per_sample_layer'],
        'sample_layer_aggregates': {str(k): v for k, v in results['sample_layer_aggregates'].items()},
        'per_cell': {str(k): v for k, v in results['per_cell'].items()},
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"saved {json_path}")

    csv_path = output_dir / f"{stem}.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'sample_layer', 'injection_layer', 'concept',
            'intro_mean', 'intro_std',
            'control_mean', 'control_std',
            'noise_mean', 'noise_std',
            'effective_norm', 'vector_mode',
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (S, I, c), cell in results['per_cell'].items():
            w.writerow({
                'sample_layer': S, 'injection_layer': I, 'concept': c,
                'intro_mean': cell['intro_mean'], 'intro_std': cell['intro_std'],
                'control_mean': cell['control_mean'], 'control_std': cell['control_std'],
                'noise_mean': cell['noise_mean'], 'noise_std': cell['noise_std'],
                'effective_norm': cell.get('effective_norm'),
                'vector_mode': cell.get('vector_mode',
                                        results['metadata'].get('vector_mode')),
            })
    print(f"saved {csv_path}")

    return pkl_path


def _replot_heatmap(merged: dict, output_dir: Path, stem: str):
    """Regenerate the same intro/control heatmap that the harness produces."""
    import matplotlib.pyplot as plt
    import numpy as np

    meta = merged['metadata']
    sample_layers = list(meta['sample_layers'])
    injection_layers = list(meta['injection_layers'])
    intro = np.zeros((len(sample_layers), len(injection_layers)))
    ctrl = np.zeros_like(intro)
    for i, S in enumerate(sample_layers):
        agg = merged['sample_layer_aggregates'].get(S, {})
        intro[i, :] = np.array(agg.get('intro_diffs',  [np.nan] * len(injection_layers)))
        ctrl[i, :]  = np.array(agg.get('control_diffs', [np.nan] * len(injection_layers)))

    vmin = float(np.nanmin([intro.min(), ctrl.min()]))
    vmax = float(np.nanmax([intro.max(), ctrl.max()]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    extent = [injection_layers[0], injection_layers[-1],
              sample_layers[0], sample_layers[-1]]
    im1 = ax1.imshow(intro, aspect='auto', cmap='RdBu_r', origin='lower',
                     extent=extent, vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(ctrl, aspect='auto', cmap='RdBu_r', origin='lower',
                     extent=extent, vmin=vmin, vmax=vmax)
    ax1.set_xlabel('Injection Layer Index'); ax1.set_ylabel('Sample Layer Index')
    ax2.set_xlabel('Injection Layer Index'); ax2.set_ylabel('Sample Layer Index')
    ax1.set_title('Introspection (Yes-No)', fontweight='bold')
    ax2.set_title('Control (Yes-No)', fontweight='bold')
    fig.suptitle(f"Detection Heatmaps (patched): {meta.get('model_name','').split('/')[-1]}",
                 fontweight='bold')
    fig.subplots_adjust(right=0.90, wspace=0.25)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.72])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Mean Logit(Yes) - Logit(No)')

    out = output_dir / f"{stem}_heatmap.png"
    try:
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"saved {out}")
    except OSError as e:
        print(f"[warn] failed to save {out}: {e!r}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('original', help='Full-sweep detection_results_*.pkl')
    ap.add_argument('patch', help='Single-sample-layer detection_results_*.pkl with the corrected run')
    ap.add_argument('--output-dir', default=None,
                    help='Defaults to the original pickle directory')
    ap.add_argument('--no-heatmap', action='store_true',
                    help='Skip regenerating the intro/control heatmap')
    ap.add_argument('--no-extras', action='store_true',
                    help='Skip regenerating the diff/gain_ratio extras (replot_detection_extras.py)')
    args = ap.parse_args()

    original_path = Path(args.original).resolve()
    patch_path = Path(args.patch).resolve()
    original = _load(original_path)
    patch = _load(patch_path)
    patched_layer = _validate(original, patch)
    print(f"patching sample layer {patched_layer}")

    merged = _splice(original, patch, patched_layer)
    out_dir = Path(args.output_dir) if args.output_dir else original_path.parent

    mode_tag = merged['metadata'].get('vector_mode', 'run')
    base_stem = f"detection_results_{mode_tag}"
    pkl_path = _save(merged, out_dir, base_stem)
    stem = pkl_path.stem

    if not args.no_heatmap:
        _replot_heatmap(merged, out_dir, stem)

    if not args.no_extras:
        script = Path(__file__).with_name('replot_detection_extras.py')
        if script.exists():
            print(f"running {script.name} on {pkl_path.name}")
            subprocess.run([sys.executable, str(script), str(pkl_path),
                            '--output-dir', str(out_dir)], check=False)
        else:
            print(f"[warn] {script} not found; skipping extras")


if __name__ == '__main__':
    main()
