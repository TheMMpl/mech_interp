#!/usr/bin/env python3
"""
Manifold analysis for steering vectors.

Question: does steering with a vector sampled at layer L_s but injected at
layer L_i degrade introspection because the injection pushes the residual
stream off the natural concept manifold at L_i?

Approach:
  1. For each requested sample layer, collect last-token activations from
     "Tell me about <word>." prompts for:
        - BASELINE_WORDS  (used to build the baseline mean)
        - TRAIN_CONCEPTS  (manifold)
        - TRIPLET_SPECIFICS (manifold)
        - TEST_CONCEPTS   (query points)
  2. Build steering vectors:
        - non-baseline:  v_c = act_c - mean(BASELINE_WORDS)
        - baseline LOO:  v_i = act_i - mean({act_j : j != i})
     The manifold at each layer = {train + triplet + baseline_LOO}.
     Query points = {test}.
  3. For every (query_layer, manifold_layer) pair compute three distance
     metrics for each test vector against the manifold:
        - Mahalanobis (computed in the manifold's PCA subspace, k components,
          to avoid the rank-deficient hidden-dim covariance)
        - PCA reconstruction residual norm (top-k components)
        - Euclidean distance to the nearest manifold vector
  4. Track vector L2 norms per layer per group.

Outputs: per-(query_layer, manifold_layer) CSVs, aggregate distance-vs-layer-gap
plots, and a norms-vs-layer plot.
"""

import os
import gc
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from introspection_gemma import IntrospectionExperiment
import concept_vectors as cv


# ---------------------------------------------------------------------------
# Metric helpers (numpy)
# ---------------------------------------------------------------------------

def fit_pca(manifold: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, components [k, D], explained_variance [k])."""
    n, d = manifold.shape
    k = min(k, n - 1, d)
    mean = manifold.mean(axis=0)
    centered = manifold - mean
    # Use SVD on centered data: centered = U S V^T, components = V^T rows
    # For d possibly huge, do thin SVD on (n, d) — fine when n is small.
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:k]
    explained_var = (S[:k] ** 2) / max(n - 1, 1)
    return mean, components, explained_var


def pca_residual(query: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """L2 norm of the residual after projecting onto top-k PCA components."""
    centered = query - mean
    proj = centered @ components.T          # (Q, k)
    recon = proj @ components               # (Q, D)
    return np.linalg.norm(centered - recon, axis=1)


def mahalanobis_in_pca(
    query: np.ndarray,
    manifold: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    explained_var: np.ndarray,
    shrinkage: float = 1e-4,
) -> np.ndarray:
    """Mahalanobis distance computed in the manifold's PCA subspace.

    Uses the diagonal of explained variance (PCA basis diagonalises the sample
    covariance) plus a small shrinkage term for numerical stability.
    """
    centered_q = (query - mean) @ components.T          # (Q, k)
    var = explained_var + shrinkage * explained_var.max()
    inv = 1.0 / var
    dist_sq = (centered_q ** 2 * inv).sum(axis=1)
    return np.sqrt(np.maximum(dist_sq, 0.0))


def nearest_euclidean(query: np.ndarray, manifold: np.ndarray) -> np.ndarray:
    """Distance from each query row to its nearest manifold row."""
    # (Q, N) pairwise distances via broadcasting; fine for our sizes.
    diff = query[:, None, :] - manifold[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    return d.min(axis=1)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation (safe against zero rows)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _rescale_to_norm(x: np.ndarray, target_norm: float, eps: float = 1e-12) -> np.ndarray:
    """Row-wise rescale so each row has the given L2 norm."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x * (target_norm / np.maximum(norms, eps))


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ManifoldAnalyzer(IntrospectionExperiment):
    """Collect activations and run manifold-distance analyses on steering vectors."""

    def collect_activations(
        self,
        sample_layers: List[int],
        words: List[str],
        prompt_template: str = "Tell me about {word}.",
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Return {layer_idx: {word: last-token hidden state on CPU (float32)}}.

        Runs one forward pass per word with output_hidden_states=True and
        extracts the requested layers.
        """
        out: Dict[int, Dict[str, torch.Tensor]] = {l: {} for l in sample_layers}
        n = len(words)
        for i, word in enumerate(words):
            messages = [{"role": "user", "content": prompt_template.format(word=word)}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.input_device)
            with torch.no_grad():
                res = self.model(**inputs, output_hidden_states=True)
            hs = res.hidden_states  # tuple length num_layers+1
            for layer_idx in sample_layers:
                act = hs[layer_idx][0, -1, :].detach().to(torch.float32).cpu()
                out[layer_idx][word] = act
            del res, inputs
            if self.verbose and ((i + 1) % 25 == 0 or i + 1 == n):
                print(f"  collected activations: {i + 1}/{n}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return out

    @staticmethod
    def build_steering_vectors(
        layer_acts: Dict[str, torch.Tensor],
        baseline_words: List[str],
        non_baseline_words: List[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Build steering vectors for one layer.

        Returns (baseline_loo_vectors, non_baseline_vectors), each {word: np.ndarray}.

        - Non-baseline: act - mean(BASELINE_WORDS)
        - Baseline LOO: act_i - mean(BASELINE_WORDS \\ {i})
        """
        baseline_stack = torch.stack([layer_acts[w] for w in baseline_words])  # (B, D)
        baseline_sum = baseline_stack.sum(dim=0)
        baseline_mean = baseline_stack.mean(dim=0)
        B = baseline_stack.shape[0]

        baseline_loo = {}
        for i, w in enumerate(baseline_words):
            loo_mean = (baseline_sum - baseline_stack[i]) / (B - 1)
            baseline_loo[w] = (baseline_stack[i] - loo_mean).numpy()

        non_baseline = {}
        for w in non_baseline_words:
            non_baseline[w] = (layer_acts[w] - baseline_mean).numpy()

        return baseline_loo, non_baseline

    def run(
        self,
        sample_layers: List[int],
        output_dir: str,
        pca_k: int = 64,
        include_triplets: bool = True,
        include_baseline_loo: bool = True,
        match_manifold_norm: bool = False,
        prompt_template: str = "Tell me about {word}.",
    ) -> Dict:
        """Run the full manifold analysis sweep.

        Args:
            include_baseline_loo: If True, add leave-one-out baseline-word
                vectors to the manifold. Their behaviour differs from concept
                vectors (different prompt distribution post-mean-subtraction);
                set False to drop them from manifold/PCA while still using them
                for the norm-comparison plot and as the source of baseline mean.
            match_manifold_norm: If True, when comparing a test vector from
                query layer q against the manifold at layer m, rescale the test
                vector so its L2 norm equals the mean L2 norm of the manifold
                at m. The manifold itself is left untouched. This isolates
                direction effects from cross-layer magnitude drift.
                Norm-plot statistics are always computed on the raw vectors.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        sample_layers = sorted(set(sample_layers))

        baseline_words = list(cv.BASELINE_WORDS)
        # Deduplicate (BASELINE_WORDS has some repeats)
        baseline_words = list(dict.fromkeys(baseline_words))
        train_words = list(cv.TRAIN_CONCEPTS)
        test_words = list(cv.TEST_CONCEPTS)
        triplet_words = list(cv.TRIPLET_SPECIFICS) if include_triplets else []

        all_words = list(dict.fromkeys(
            baseline_words + train_words + test_words + triplet_words
        ))

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Manifold analysis: {len(sample_layers)} sample layers, "
                  f"{len(all_words)} unique words, pca_k={pca_k}")
            print(f"  baseline={len(baseline_words)}  train={len(train_words)}  "
                  f"test={len(test_words)}  triplets={len(triplet_words)}")
            print(f"  include_baseline_loo={include_baseline_loo}  match_manifold_norm={match_manifold_norm}")
            print(f"{'='*70}")

        # 1. Collect activations
        acts = self.collect_activations(sample_layers, all_words, prompt_template)

        # 2. Per-layer steering vectors (always build raw, then optionally drop/normalize)
        manifold_words = train_words + triplet_words  # non-baseline manifold members
        per_layer = {}
        for L in sample_layers:
            baseline_loo, non_baseline = self.build_steering_vectors(
                acts[L], baseline_words, non_baseline_words=manifold_words + test_words
            )
            baseline_loo_keys = list(baseline_loo.keys())
            baseline_loo_mat = np.stack(
                [baseline_loo[w] for w in baseline_loo_keys]
            ).astype(np.float64)
            train_triplet_mat = np.stack(
                [non_baseline[w] for w in manifold_words]
            ).astype(np.float64)
            test_mat = np.stack([non_baseline[w] for w in test_words]).astype(np.float64)

            if include_baseline_loo:
                manifold_mat = np.concatenate([baseline_loo_mat, train_triplet_mat], axis=0)
                manifold_keys = baseline_loo_keys + manifold_words
            else:
                manifold_mat = train_triplet_mat
                manifold_keys = list(manifold_words)

            per_layer[L] = {
                'manifold_keys': manifold_keys,
                'manifold': manifold_mat,
                'test_keys': test_words,
                'test': test_mat,
                'manifold_mean_norm': float(np.linalg.norm(manifold_mat, axis=1).mean()),
                'baseline_loo_raw': baseline_loo_mat,
                'train_triplet_raw': train_triplet_mat,
                'test_raw': np.stack([non_baseline[w] for w in test_words]).astype(np.float64),
            }

        # 3. Vector norms per layer per group (always on raw, un-normalized vectors)
        norms = self._compute_norms_raw(per_layer)
        self._save_norms(norms, output_dir)
        self._plot_norms(norms, output_dir)

        # 4. Cross-layer distance metrics
        metrics = self._compute_cross_layer_metrics(
            per_layer, sample_layers, pca_k, match_manifold_norm=match_manifold_norm,
        )
        self._save_metrics(metrics, sample_layers, test_words, output_dir, pca_k)
        self._plot_metrics(metrics, sample_layers, output_dir, pca_k)

        # 5. 3D PCA visualisations (one per sample layer)
        self._plot_pca_3d(per_layer, manifold_words, test_words, output_dir,
                          include_baseline_loo=include_baseline_loo,
                          n_baseline=len(baseline_words))

        meta = {
            'model_name': self.model_name,
            'sample_layers': sample_layers,
            'pca_k': pca_k,
            'n_baseline': len(baseline_words),
            'n_train': len(train_words),
            'n_test': len(test_words),
            'n_triplets': len(triplet_words),
            'include_baseline_loo': include_baseline_loo,
            'match_manifold_norm': match_manifold_norm,
            'manifold_size': per_layer[sample_layers[0]]['manifold'].shape[0],
            'prompt_template': prompt_template,
            'timestamp': datetime.now().isoformat(),
        }
        with open(os.path.join(output_dir, 'manifold_metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        if self.verbose:
            print(f"\nDone. Outputs in {output_dir}")
        return {'metadata': meta, 'norms': norms, 'metrics': metrics}

    # ------------------------------------------------------------------
    # Norms
    # ------------------------------------------------------------------

    def _compute_norms_raw(self, per_layer: Dict) -> Dict:
        """Per-layer norms split by group {baseline_loo, train_triplet, test}.

        Always computed on the *raw* (un-normalized) vectors so the norm plot
        remains meaningful even when distance metrics use normalized inputs.
        """
        result = {}
        for L, d in per_layer.items():
            result[L] = {
                'baseline_loo': np.linalg.norm(d['baseline_loo_raw'], axis=1),
                'train_triplet': np.linalg.norm(d['train_triplet_raw'], axis=1),
                'test': np.linalg.norm(d['test_raw'], axis=1),
            }
        return result

    def _save_norms(self, norms: Dict, output_dir: str):
        import csv
        path = os.path.join(output_dir, 'vector_norms.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['layer', 'group', 'mean', 'std', 'min', 'max', 'n'])
            for L in sorted(norms.keys()):
                for grp, vals in norms[L].items():
                    w.writerow([L, grp, float(vals.mean()), float(vals.std()),
                                float(vals.min()), float(vals.max()), len(vals)])
        if self.verbose:
            print(f"Saved norms CSV: {path}")

    def _plot_norms(self, norms: Dict, output_dir: str):
        layers = sorted(norms.keys())
        groups = ['baseline_loo', 'train_triplet', 'test']
        colors = {'baseline_loo': 'gray', 'train_triplet': 'steelblue', 'test': 'firebrick'}

        def _draw(ax, log_y: bool):
            for g in groups:
                means = np.array([norms[L][g].mean() for L in layers])
                stds = np.array([norms[L][g].std() for L in layers])
                if log_y:
                    # Clip lower error bar to keep it positive on log axis.
                    lower = np.maximum(means - stds, means * 1e-3)
                    yerr = np.vstack([means - lower, stds])
                else:
                    yerr = stds
                ax.errorbar(layers, means, yerr=yerr, fmt='o-', capsize=3,
                            label=g, color=colors[g], alpha=0.85)
            ax.set_xlabel('Sample Layer')
            ax.set_ylabel('Steering-Vector L2 Norm' + (' (log)' if log_y else ''))
            ax.grid(True, alpha=0.3, which='both' if log_y else 'major')
            ax.legend()
            if log_y:
                ax.set_yscale('log')

        # Linear-scale plot
        fig, ax = plt.subplots(figsize=(10, 6))
        _draw(ax, log_y=False)
        ax.set_title(f'Vector norms by layer  ({self.model_name.split("/")[-1]})')
        plt.tight_layout()
        path = os.path.join(output_dir, 'vector_norms.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if self.verbose:
            print(f"Saved norms plot: {path}")

        # Log-scale plot
        fig, ax = plt.subplots(figsize=(10, 6))
        _draw(ax, log_y=True)
        ax.set_title(f'Vector norms by layer (log scale)  ({self.model_name.split("/")[-1]})')
        plt.tight_layout()
        path_log = os.path.join(output_dir, 'vector_norms_log.png')
        plt.savefig(path_log, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if self.verbose:
            print(f"Saved norms plot (log): {path_log}")

    # ------------------------------------------------------------------
    # Cross-layer distance metrics
    # ------------------------------------------------------------------

    def _compute_cross_layer_metrics(
        self,
        per_layer: Dict,
        sample_layers: List[int],
        pca_k: int,
        match_manifold_norm: bool = False,
    ) -> Dict:
        """For each (query_layer, manifold_layer), compute three metrics for every test concept.

        When ``match_manifold_norm`` is True the test vectors are per-row rescaled
        so their L2 norm equals the mean L2 norm of the manifold at the manifold
        layer; this isolates direction effects from magnitude drift.
        Returns dict keyed by (q_layer, m_layer):
            {
              'mahalanobis': np.ndarray (n_test,),
              'pca_residual': np.ndarray (n_test,),
              'nearest_eucl': np.ndarray (n_test,),
            }
        """
        # Precompute PCA per manifold layer
        pca_per_layer = {}
        for L in sample_layers:
            manifold = per_layer[L]['manifold']
            mean, comp, var = fit_pca(manifold, pca_k)
            pca_per_layer[L] = (mean, comp, var)

        results = {}
        for q in sample_layers:
            test_q_base = per_layer[q]['test']  # (T, D)
            for m in sample_layers:
                manifold = per_layer[m]['manifold']
                mean, comp, var = pca_per_layer[m]
                if match_manifold_norm:
                    target = per_layer[m]['manifold_mean_norm']
                    test_q = _rescale_to_norm(test_q_base, target)
                else:
                    test_q = test_q_base
                results[(q, m)] = {
                    'mahalanobis': mahalanobis_in_pca(test_q, manifold, mean, comp, var),
                    'pca_residual': pca_residual(test_q, mean, comp),
                'nearest_eucl': nearest_euclidean(test_q, manifold),
                }
            if self.verbose:
                print(f"  computed metrics for query layer {q}")
        return results

    def _save_metrics(
        self,
        metrics: Dict,
        sample_layers: List[int],
        test_words: List[str],
        output_dir: str,
        pca_k: int,
    ):
        import csv
        path = os.path.join(output_dir, 'manifold_distances.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['query_layer', 'manifold_layer', 'test_concept',
                        'mahalanobis', 'pca_residual', 'nearest_eucl'])
            for q in sample_layers:
                for m in sample_layers:
                    cell = metrics[(q, m)]
                    for i, tw in enumerate(test_words):
                        w.writerow([q, m, tw,
                                    float(cell['mahalanobis'][i]),
                                    float(cell['pca_residual'][i]),
                                    float(cell['nearest_eucl'][i])])
        if self.verbose:
            print(f"Saved per-concept metrics CSV: {path}")

        # Summary aggregates per pair
        sum_path = os.path.join(output_dir, 'manifold_distances_summary.csv')
        with open(sum_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['query_layer', 'manifold_layer', 'layer_gap',
                        'metric', 'mean', 'std', 'median'])
            for q in sample_layers:
                for m in sample_layers:
                    cell = metrics[(q, m)]
                    for name in ('mahalanobis', 'pca_residual', 'nearest_eucl'):
                        vals = cell[name]
                        w.writerow([q, m, m - q, name,
                                    float(vals.mean()), float(vals.std()),
                                    float(np.median(vals))])
        if self.verbose:
            print(f"Saved summary CSV: {sum_path}")

    def _plot_metrics(
        self,
        metrics: Dict,
        sample_layers: List[int],
        output_dir: str,
        pca_k: int,
    ):
        metric_names = ['mahalanobis', 'pca_residual', 'nearest_eucl']
        nL = len(sample_layers)

        # 1. Heatmaps: query_layer × manifold_layer (mean over test concepts)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, name in zip(axes, metric_names):
            mat = np.zeros((nL, nL))
            for i, q in enumerate(sample_layers):
                for j, m in enumerate(sample_layers):
                    mat[i, j] = metrics[(q, m)][name].mean()
            im = ax.imshow(mat, aspect='auto', cmap='viridis', origin='lower',
                           extent=[sample_layers[0] - 0.5, sample_layers[-1] + 0.5,
                                   sample_layers[0] - 0.5, sample_layers[-1] + 0.5])
            ax.set_xlabel('Manifold Layer')
            ax.set_ylabel('Query (Test-vector) Layer')
            ax.set_title(f'{name}  (mean over test concepts)')
            # Diagonal reference
            ax.plot([sample_layers[0], sample_layers[-1]],
                    [sample_layers[0], sample_layers[-1]],
                    'r--', alpha=0.5, linewidth=1)
            plt.colorbar(im, ax=ax)
        fig.suptitle(f'Manifold distance heatmaps  ({self.model_name.split("/")[-1]}, pca_k={pca_k})',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, 'distance_heatmaps.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if self.verbose:
            print(f"Saved heatmaps: {path}")

        # 2. Distance vs |layer gap| (collapsed across (q, m) pairs)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, name in zip(axes, metric_names):
            gaps: Dict[int, List[float]] = {}
            for (q, m), cell in metrics.items():
                g = abs(m - q)
                gaps.setdefault(g, []).extend(cell[name].tolist())
            sorted_g = sorted(gaps.keys())
            means = [np.mean(gaps[g]) for g in sorted_g]
            stds = [np.std(gaps[g]) for g in sorted_g]
            ax.errorbar(sorted_g, means, yerr=stds, fmt='o-', capsize=3, color='steelblue')
            ax.set_xlabel('|manifold_layer − query_layer|')
            ax.set_ylabel(f'{name}')
            ax.set_title(f'{name} vs layer gap')
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'Distance vs layer gap  ({self.model_name.split("/")[-1]}, pca_k={pca_k})',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, 'distance_vs_gap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if self.verbose:
            print(f"Saved gap plot: {path}")

        # 3. Same-layer vs cross-layer comparison (one bar plot per metric)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, name in zip(axes, metric_names):
            same = []
            cross = []
            for (q, m), cell in metrics.items():
                if q == m:
                    same.extend(cell[name].tolist())
                else:
                    cross.extend(cell[name].tolist())
            data = [same, cross]
            ax.boxplot(data, labels=['same layer', 'cross layer'], showmeans=True)
            ax.set_ylabel(name)
            ax.set_title(f'{name}: same vs cross layer')
            ax.grid(True, alpha=0.3, axis='y')
        fig.suptitle(f'Same vs cross-layer distance  ({self.model_name.split("/")[-1]}, pca_k={pca_k})',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, 'same_vs_cross_layer.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if self.verbose:
            print(f"Saved same-vs-cross plot: {path}")

    # ------------------------------------------------------------------
    # 3D PCA visualisation
    # ------------------------------------------------------------------

    def _plot_pca_3d(
        self,
        per_layer: Dict,
        manifold_words: List[str],
        test_words: List[str],
        output_dir: str,
        include_baseline_loo: bool = True,
        n_baseline: int = 0,
        annotate_test: bool = True,
        max_test_labels: int = 25,
    ):
        """Per-layer 3D PCA scatter showing baseline_loo / train+triplet / test groups.

        PCA is fit on the combined manifold (baseline_loo + train + triplet) so
        the basis reflects the natural concept-vector spread; test vectors are
        projected into the same basis.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D)

        pca_dir = os.path.join(output_dir, 'pca_3d')
        Path(pca_dir).mkdir(parents=True, exist_ok=True)

        for L, d in per_layer.items():
            manifold = d['manifold']
            test = d['test']
            mean, comp, var = fit_pca(manifold, k=3)
            if comp.shape[0] < 3:
                if self.verbose:
                    print(f"  layer {L}: not enough components for 3D PCA, skipping")
                continue
            total_var = float(var.sum())
            # Guard against numerical zero by using manifold's centered total variance.
            centered_manifold = manifold - mean
            denom = float((centered_manifold ** 2).sum() / max(manifold.shape[0] - 1, 1))
            ratios = (var / denom) if denom > 0 else np.zeros_like(var)

            proj_manifold = (manifold - mean) @ comp.T   # (N, 3)
            proj_test = (test - mean) @ comp.T           # (T, 3)

            split = n_baseline if include_baseline_loo else 0
            base_proj = proj_manifold[:split]
            train_proj = proj_manifold[split:]

            fig = plt.figure(figsize=(11, 9))
            ax = fig.add_subplot(111, projection='3d')

            if len(base_proj) > 0:
                ax.scatter(base_proj[:, 0], base_proj[:, 1], base_proj[:, 2],
                           c='gray', alpha=0.45, s=18, label=f'baseline_loo (n={len(base_proj)})')
            ax.scatter(train_proj[:, 0], train_proj[:, 1], train_proj[:, 2],
                       c='steelblue', alpha=0.75, s=28, label=f'train+triplet (n={len(train_proj)})')
            ax.scatter(proj_test[:, 0], proj_test[:, 1], proj_test[:, 2],
                       c='firebrick', alpha=0.9, s=36, marker='^',
                       label=f'test (n={len(proj_test)})')

            if annotate_test:
                # Avoid label clutter; cap number of annotations.
                step = max(1, len(test_words) // max_test_labels)
                for i in range(0, len(test_words), step):
                    ax.text(proj_test[i, 0], proj_test[i, 1], proj_test[i, 2],
                            test_words[i], fontsize=7, color='firebrick', alpha=0.85)

            ax.set_xlabel(f'PC1 ({ratios[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({ratios[1]*100:.1f}%)')
            ax.set_zlabel(f'PC3 ({ratios[2]*100:.1f}%)')
            ax.set_title(
                f'PCA-3D manifold @ layer {L}  ({self.model_name.split("/")[-1]})\n'
                f'top-3 var ratio sum = {ratios.sum()*100:.1f}%',
                fontsize=11, fontweight='bold'
            )
            ax.legend(loc='upper left', fontsize=9)
            plt.tight_layout()
            path = os.path.join(pca_dir, f'pca3d_layer_{L:03d}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            if self.verbose:
                print(f"Saved 3D PCA plot: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_layers(spec: str) -> List[int]:
    """Parse '10,20,30' or '10-40:5' (start-end:step) into a list of ints."""
    out: List[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            range_part, _, step_part = part.partition(':')
            a, b = range_part.split('-')
            step = int(step_part) if step_part else 1
            out.extend(range(int(a), int(b) + 1, step))
        else:
            out.append(int(part))
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser(description="Steering-vector manifold analysis.")
    p.add_argument('--model', default='google/gemma-3-27b-it')
    p.add_argument('--sample-layers', required=True,
                   help="Comma list and/or ranges, e.g. '10,20,30' or '5-55:5'")
    p.add_argument('--output-dir', default='manifold_results')
    p.add_argument('--pca-k', type=int, default=64)
    p.add_argument('--no-triplets', action='store_true',
                   help='Exclude TRIPLET_SPECIFICS from the manifold')
    p.add_argument('--no-baseline-loo', action='store_true',
                   help='Exclude baseline LOO vectors from the manifold (still '
                        'plotted in the norms figure for comparison)')
    p.add_argument('--normalize', dest='match_manifold_norm', action='store_true',
                   help='Rescale each test vector to match the mean L2 norm of '
                        'the manifold at the manifold layer before computing '
                        'distances. Isolates direction effects from magnitude '
                        'drift across layers. Manifold vectors are left as-is.')
    p.add_argument('--no-fine-tune', action='store_true',
                   help='Disable PEFT adapter loading')
    p.add_argument('--device-map', default='auto')
    p.add_argument('--max-gpu-memory', default=None)
    p.add_argument('--max-cpu-memory', default=None)
    p.add_argument('--offload-folder', default=None)
    p.add_argument('--prompt-template', default='Tell me about {word}.')
    args = p.parse_args()

    sample_layers = parse_layers(args.sample_layers)
    analyzer = ManifoldAnalyzer(
        model_name=args.model,
        verbose=True,
        fine_tune=not args.no_fine_tune,
        model_device_map=args.device_map,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=args.offload_folder,
    )
    analyzer.run(
        sample_layers=sample_layers,
        output_dir=args.output_dir,
        pca_k=args.pca_k,
        include_triplets=not args.no_triplets,
        include_baseline_loo=not args.no_baseline_loo,
        match_manifold_norm=args.match_manifold_norm,
        prompt_template=args.prompt_template,
    )


if __name__ == '__main__':
    main()
