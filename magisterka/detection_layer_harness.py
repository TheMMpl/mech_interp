#!/usr/bin/env python3
"""
Detection Layer Harness

Compare where detection happens vs where vectors were sampled.
2D sweep: sample layer × injection layer, compute dual-peak detection metric.
"""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import gc
import pickle
import random
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from datetime import datetime
from introspection_gemma import IntrospectionExperiment
import concept_vectors


# Supported vector modes for run_detection_sweep
VECTOR_MODES = (
    "concept",
    "concept_sample_norm",
    "concept_injection_norm",
    "random_sample_norm",
    "random_injection_norm",
    "random_fixed",
)

# Backwards-compatibility aliases: the original "layer_norm" naming meant
# "norm targeted to the sample-layer norm". The new explicit name is
# "*_sample_norm" — the old names map straight to it.
VECTOR_MODE_ALIASES = {
    "concept_layer_norm": "concept_sample_norm",
    "random_layer_norm":  "random_sample_norm",
}


def _canonical_vector_mode(mode: str) -> str:
    return VECTOR_MODE_ALIASES.get(mode, mode)


# Fallback hidden-size defaults keyed by case-insensitive substring of the
# HuggingFace model id. Used by run_detection_sweep when the model config does
# not expose hidden_size directly (e.g. multimodal Gemma 3 wraps it inside
# `config.text_config`). Verified against the cached vector files in
# backup/adapter_full/manual/.
HIDDEN_SIZE_DEFAULTS = {
    "gemma-3-27b": 5376,
    "gemma-4-31b": 5376,  # alias used in our CLI examples
    "gemma-3-12b": 3840,
    "gemma-3-4b":  2560,
    "qwen2.5-14b": 5120,
    "qwen2.5-7b":  3584,
    "qwen2.5-1.5b": 1536,
    "qwen2.5-0.5b": 896,
}


def lookup_default_hidden_size(model_name: str) -> Optional[int]:
    """Return a hidden-size default for the model id, or None if not known."""
    lname = (model_name or "").lower()
    for key, hs in HIDDEN_SIZE_DEFAULTS.items():
        if key in lname:
            return hs
    return None


def infer_hidden_size(model, model_name: Optional[str] = None) -> Optional[int]:
    """Best-effort hidden-size lookup.

    Walks the usual config locations (config.hidden_size, config.text_config,
    config.llm_config) and falls back to a model-id substring map.
    """
    cfg = getattr(model, "config", None)
    candidates = [cfg, getattr(cfg, "text_config", None), getattr(cfg, "llm_config", None)]
    for c in candidates:
        if c is None:
            continue
        hs = getattr(c, "hidden_size", None)
        if isinstance(hs, int) and hs > 0:
            return hs
    return lookup_default_hidden_size(model_name or getattr(cfg, "_name_or_path", "") or "")


def load_layer_norms_csv(csv_path: str, group: str = "train_triplet",
                         stat: str = "mean") -> Dict[int, float]:
    """Parse a manifold-analysis vector_norms.csv and return {layer_idx: norm}.

    Expects columns: layer, group, mean, std, min, max, n.
    Selects rows matching `group` and returns the value of `stat` per layer.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Layer-norms CSV not found: {csv_path}")
    out: Dict[int, float] = {}
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("group") != group:
                continue
            try:
                layer = int(row["layer"])
                val = float(row[stat])
            except (KeyError, ValueError):
                continue
            out[layer] = val
    if not out:
        raise ValueError(
            f"No rows matched group={group!r} in {csv_path}. "
            f"Available groups can be checked with `cut -d, -f2 {csv_path} | sort -u`."
        )
    return out


def resolve_layer_norm(layer_norms: Dict[int, float], layer_idx: int) -> float:
    """Look up the per-layer norm; fall back to nearest layer if not exact."""
    if layer_idx in layer_norms:
        return layer_norms[layer_idx]
    if not layer_norms:
        raise ValueError("Empty layer_norms dict")
    nearest = min(layer_norms.keys(), key=lambda k: abs(k - layer_idx))
    return layer_norms[nearest]


def _unit_random_like(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random Gaussian vector with the same shape/dtype as reference, L2-normalized."""
    noise = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    n = noise.norm()
    if n > 0:
        noise = noise / n
    return noise.to(dtype=reference.dtype)

# Adapter path for fine-tuned models
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/workspace/project/adapter_bias_corrected")

# =============================================================================
# Per-region refusal ablation weights (OPTUNA-optimized for Gemma 3 27B, 62 layers)
# Copied from 03d_refusal_abliteration.py – Trial 1584 (20-region config).
# Use these when --refusal-directions is provided with a per-layer .pt file.
# =============================================================================
GEMMA3_27B_REGION_WEIGHTS = {
    "very_early_a": 0.010190365613071925,
    "very_early_b": 0.09976487098474057,
    "very_early_c": 0.009846349798252014,
    "very_early_d": 0.010714741304450688,
    "early_a":      0.023812035217103455,
    "early_b":      0.006873821994170306,
    "early_c":      0.0023568060724657135,
    "early_d":      0.11762696391562547,
    "pre_key_a":    0.024324361266584712,
    "pre_key_b":    0.009936585603088419,
    "key_a":        0.000533052460819306,
    "key_b":        0.0057508808893361974,
    "mid_a":        0.020646470409482434,
    "mid_b":        0.02205567035624907,
    "mid_c":        0.004716948598867072,
    "mid_d":        0.003251529189292551,
    "late_a":       0.07694211978232157,
    "late_b":       0.03330589279564281,
    "final_a":      2.358688691270255e-05,
    "final_b":      0.003955462234418926,
}

GEMMA3_27B_REGION_BOUNDARIES = {
    "very_early_a_end": 2,
    "very_early_b_end": 5,
    "very_early_c_end": 8,
    "very_early_d_end": 10,
    "early_a_end":      13,
    "early_b_end":      15,
    "early_c_end":      18,
    "early_d_end":      20,
    "pre_key_a_end":    24,
    "pre_key_b_end":    28,
    "key_a_end":        32,
    "key_b_end":        35,
    "mid_a_end":        38,
    "mid_b_end":        41,
    "mid_c_end":        44,
    "mid_d_end":        47,
    "late_a_end":       51,
    "late_b_end":       55,
    "final_a_end":      58,
    # final_b: 59-61
}


def build_region_weight_fn(region_weights: dict, boundaries: dict):
    """Return a callable (layer_idx: int) -> float using the 20-region boundary scheme.

    Mirrors get_layer_weight() from 03d_refusal_abliteration.py.
    """
    b = boundaries

    def weight_fn(layer_idx: int) -> float:
        if layer_idx <= b["very_early_a_end"]:  return region_weights["very_early_a"]
        if layer_idx <= b["very_early_b_end"]:  return region_weights["very_early_b"]
        if layer_idx <= b["very_early_c_end"]:  return region_weights["very_early_c"]
        if layer_idx <= b["very_early_d_end"]:  return region_weights["very_early_d"]
        if layer_idx <= b["early_a_end"]:        return region_weights["early_a"]
        if layer_idx <= b["early_b_end"]:        return region_weights["early_b"]
        if layer_idx <= b["early_c_end"]:        return region_weights["early_c"]
        if layer_idx <= b["early_d_end"]:        return region_weights["early_d"]
        if layer_idx <= b["pre_key_a_end"]:      return region_weights["pre_key_a"]
        if layer_idx <= b["pre_key_b_end"]:      return region_weights["pre_key_b"]
        if layer_idx <= b["key_a_end"]:          return region_weights["key_a"]
        if layer_idx <= b["key_b_end"]:          return region_weights["key_b"]
        if layer_idx <= b["mid_a_end"]:          return region_weights["mid_a"]
        if layer_idx <= b["mid_b_end"]:          return region_weights["mid_b"]
        if layer_idx <= b["mid_c_end"]:          return region_weights["mid_c"]
        if layer_idx <= b["mid_d_end"]:          return region_weights["mid_d"]
        if layer_idx <= b["late_a_end"]:         return region_weights["late_a"]
        if layer_idx <= b["late_b_end"]:         return region_weights["late_b"]
        if layer_idx <= b["final_a_end"]:        return region_weights["final_a"]
        return region_weights["final_b"]

    return weight_fn


class DetectionAnalyzer(IntrospectionExperiment):
    """Automated analyzer for detection layer vs sample layer comparison."""

    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 verbose: bool = True,
                 fine_tune: bool = True,
                 model_device_map: str = "auto",
                 max_gpu_memory: Optional[str] = None,
                 max_cpu_memory: Optional[str] = None,
                 offload_folder: Optional[str] = None):
        """Initialize DetectionAnalyzer (inherits from IntrospectionExperiment).
        
        Args:
            model_name: HuggingFace model identifier
            verbose: Verbose output during processing
            fine_tune: Whether to use fine-tuned adapter
        """
        super().__init__(
            model_name=model_name,
            verbose=verbose,
            fine_tune=fine_tune,
            model_device_map=model_device_map,
            max_gpu_memory=max_gpu_memory,
            max_cpu_memory=max_cpu_memory,
            offload_folder=offload_folder,
        )
        self.vector_cache = {}  # Cache loaded vectors: {sample_layer: {concept: tensor}}

    def _ensure_vectors_for_layer(self, layer_idx: int, vector_path: str) -> Dict[str, torch.Tensor]:
        """Ensure vectors exist for a specific sample layer, generating file if missing."""
        if os.path.exists(vector_path):
            return torch.load(vector_path, map_location='cpu')

        if self.verbose:
            print(f"  Vector file not found: {vector_path}")
            print(f"  Auto-generating vectors for sample layer {layer_idx}...")

        Path(vector_path).parent.mkdir(parents=True, exist_ok=True)

        # Generate vectors with the currently loaded model to avoid loading another large model.
        baseline_acts = []
        for word in concept_vectors.BASELINE_WORDS:
            messages = [{"role": "user", "content": f"Tell me about {word}."}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
            baseline_acts.append(out.hidden_states[layer_idx][0, -1, :].detach().cpu())

        baseline_mean = torch.stack(baseline_acts).mean(dim=0)
        vectors = {}
        all_concepts = concept_vectors.TRAIN_CONCEPTS + concept_vectors.TEST_CONCEPTS + concept_vectors.TRIPLET_SPECIFICS
        for concept in all_concepts:
            messages = [{"role": "user", "content": f"Tell me about {concept}."}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
            act = out.hidden_states[layer_idx][0, -1, :].detach().cpu()
            vectors[concept] = act - baseline_mean

        torch.save(vectors, vector_path)
        if self.verbose:
            print(f"  Saved generated vectors to {vector_path}")

        self._release_cuda()

        return vectors

    def _release_cuda(self):
        """Best-effort memory cleanup between long-running sweep steps."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _unload_model(self):
        """Delete self.model from GPU memory to free space for a second model load.

        Two-stage release:
        1. Replace every CUDA parameter/buffer tensor with an empty CPU tensor so
           the GPU allocation is returned to the allocator *immediately*, without
           waiting for Python's cyclic GC to finalise the model object tree.
        2. Drop the Python reference and flush the CUDA cache.
        """
        if self.model is None:
            return
        if self.verbose:
            if torch.cuda.is_available():
                used_before = sum(
                    torch.cuda.memory_allocated(i)
                    for i in range(torch.cuda.device_count())
                ) / 1024 ** 3
                print(f"Unloading original model "
                      f"(GPU allocated before: {used_before:.1f} GB)...")
            else:
                print("Unloading original model...")

        # Stage 1: wipe each CUDA tensor so its storage is freed immediately.
        try:
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.data.device.type == 'cuda':
                        param.data = torch.empty(0)
                for buf in self.model.buffers():
                    if buf.device.type == 'cuda':
                        buf.data = torch.empty(0)
        except Exception as exc:
            if self.verbose:
                print(f"  Warning during parameter wipe: {exc}")

        # Stage 2: drop Python references and reclaim the allocator cache.
        del self.model
        self.model = None
        # Two collector passes: first breaks reference cycles, second frees them.
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if self.verbose:
                used_after = sum(
                    torch.cuda.memory_allocated(i)
                    for i in range(torch.cuda.device_count())
                ) / 1024 ** 3
                print(f"  GPU allocated after unload: {used_after:.1f} GB")

    def _measure_control_score(self) -> float:
        """Measure control-question Yes-No logit difference without steering."""
        prompt_control = self.format_prompt(self.control_question)
        inputs_control = self.tokenizer(prompt_control, return_tensors="pt").to(self.device)
        _, control_diff = self.get_top_logits(inputs_control, top_k=10)
        return control_diff

    def _measure_question_score(
        self,
        question,
        layer_idx: int,
        vector: Optional[torch.Tensor],
        token_pos: int,
        steer_all_tokens: bool = False,
    ) -> float:
        """Measure Yes-No logit difference for an arbitrary question with optional steering."""

        def steering_hook(module, input, output):
            if vector is None:
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
                local_vector = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
                modified = hidden_states.clone()
                if steer_all_tokens:
                    modified = modified + local_vector.unsqueeze(0).unsqueeze(0)
                else:
                    seq_len = modified.shape[1]
                    token_idx = token_pos if token_pos >= 0 else seq_len + token_pos
                    if 0 <= token_idx < seq_len:
                        modified[:, token_idx, :] = modified[:, token_idx, :] + local_vector
                return (modified,) + output[1:]

            hidden_states = output
            local_vector = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
            modified = hidden_states.clone()
            if steer_all_tokens:
                modified = modified + local_vector.unsqueeze(0).unsqueeze(0)
            else:
                seq_len = modified.shape[1]
                token_idx = token_pos if token_pos >= 0 else seq_len + token_pos
                if 0 <= token_idx < seq_len:
                    modified[:, token_idx, :] = modified[:, token_idx, :] + local_vector
            return modified

        hook = None
        if vector is not None:
            hook = self.layer_modules[layer_idx].register_forward_hook(steering_hook)

        try:
            prompt = self.format_prompt(question)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            _, diff = self.get_top_logits(inputs, top_k=10)
            return diff
        finally:
            if hook is not None:
                hook.remove()

    def _build_concept_selection(
        self,
        all_concepts: List[str],
        concept_filter: Optional[List[str]] = None,
        plot_concepts: Optional[List[str]] = None,
        plot_random_sample: int = 0,
        selection_seed: int = 42,
        run_selected_only: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Build run concept list and plot concept list from explicit and random selections."""
        all_set = set(all_concepts)
        explicit_run = [c for c in (concept_filter or []) if c in all_set]
        explicit_plot = [c for c in (plot_concepts or []) if c in all_set]

        selected_for_plots = list(dict.fromkeys(explicit_plot + explicit_run))
        if plot_random_sample > 0:
            rng = random.Random(selection_seed)
            remaining = [c for c in all_concepts if c not in selected_for_plots]
            sampled = rng.sample(remaining, min(plot_random_sample, len(remaining)))
            selected_for_plots.extend(sampled)

        if run_selected_only:
            run_concepts = selected_for_plots
        elif explicit_run:
            run_concepts = explicit_run
        else:
            run_concepts = all_concepts

        run_concepts = sorted(list(dict.fromkeys(run_concepts)))
        selected_for_plots = sorted(list(dict.fromkeys(selected_for_plots)))
        return run_concepts, selected_for_plots

    def _plot_single_concept_curve(
        self,
        sample_layer_idx: int,
        concept: str,
        injection_layers: List[int],
        intro_diffs: List[float],
        control_diffs: List[float],
        intro_stds: List[float],
        control_stds: List[float],
        baseline_intro: float,
        baseline_control: float,
        peak_metric: Dict,
        output_dir: str,
        mode_tag: str = "concept",
    ):
        """Plot one concept curve in the same style as _plot_layer_effects."""
        fig = plt.figure(figsize=(14, 9))
        ax = plt.subplot(111)

        plt.axhline(y=baseline_intro, color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Baseline introspection: {baseline_intro:+.2f}')
        plt.axhline(y=baseline_control, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Baseline control: {baseline_control:+.2f}')

        plt.plot(injection_layers, intro_diffs, 'o-', linewidth=2, markersize=5, color='blue',
                label='Introspection question')
        plt.plot(injection_layers, control_diffs, 's-', linewidth=2, markersize=5, color='red',
                label='Control question')

        if np.any(np.array(intro_stds) > 0):
            plt.errorbar(injection_layers, intro_diffs, yerr=intro_stds, fmt='none', ecolor='blue',
                        capsize=4, alpha=0.4)
        if np.any(np.array(control_stds) > 0):
            plt.errorbar(injection_layers, control_diffs, yerr=control_stds, fmt='none', ecolor='red',
                        capsize=4, alpha=0.4)

        if peak_metric['early_peak_layer'] is not None:
            plt.plot(peak_metric['early_peak_layer'], peak_metric['early_peak_magnitude'],
                    'bv', markersize=10, label=f"Early peak (L{peak_metric['early_peak_layer']})")
        if peak_metric['late_peak_layer'] is not None:
            plt.plot(peak_metric['late_peak_layer'], peak_metric['late_peak_magnitude'],
                    'r^', markersize=10, label=f"Late peak (L{peak_metric['late_peak_layer']})")

        plt.axvline(x=sample_layer_idx, color='green', linestyle=':', linewidth=2, alpha=0.6,
                   label=f'Sample layer L{sample_layer_idx}')

        plt.xlabel('Injection Layer Index', fontsize=13)
        plt.ylabel('Logit(Yes) - Logit(No)', fontsize=13)
        model_display = self.model_name.split("/")[-1]
        plt.title(f'Detection Analysis: {model_display} (Sampled @ L{sample_layer_idx}, concept={concept})',
                 fontsize=13, fontweight='bold', pad=20)

        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='best')
        plt.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)

        y_min, y_max = plt.ylim()
        if y_max > 0:
            plt.axhspan(0, y_max, alpha=0.05, color='green')
        if y_min < 0:
            plt.axhspan(y_min, 0, alpha=0.05, color='orange')

        textstr = f'Concept: {concept}\nSample Layer: {sample_layer_idx}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')

        plt.tight_layout()
        safe_concept = concept.replace(' ', '_').replace('/', '_')
        filename = os.path.join(output_dir, f"single_sample_l{sample_layer_idx}_{safe_concept}_{mode_tag}.png")
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        except OSError as e:
            print(f"[warn] failed to save {filename}: {e!r} — continuing sweep")
        plt.close(fig)

    def load_vectors_for_layer(self, layer_idx: int, vector_path: str) -> Dict[str, torch.Tensor]:
        """Load or auto-generate vectors for a sample layer.
        
        Args:
            layer_idx: Sample layer index (used in error messages)
            vector_path: Path to .pt file containing vectors dict
            
        Returns:
            Dictionary mapping concept names to vectors (on CPU)
        """
        # Check cache first
        if layer_idx in self.vector_cache:
            if self.verbose:
                print(f"  Using cached vectors for layer {layer_idx}")
            return self.vector_cache[layer_idx]

        vectors = self._ensure_vectors_for_layer(layer_idx, vector_path)

        if self.verbose:
            print(f"  Loaded vectors for sample layer {layer_idx} from {vector_path}")

        # Validate: should be dict of concept -> tensor
        if not isinstance(vectors, dict):
            raise ValueError(f"Vector file must contain a dict, got {type(vectors)}")

        for concept, vec in vectors.items():
            if not torch.is_tensor(vec):
                raise ValueError(f"Vector for concept '{concept}' is not a tensor")
        
        if self.verbose:
            print(f"  Loaded {len(vectors)} concept vectors")

        # Cache it
        self.vector_cache[layer_idx] = vectors
        return vectors

    def get_aligned_concepts(self, 
                            sample_layer_files: Dict[int, str],
                            intersection_mode: bool = True) -> List[str]:
        """Load vectors for all sample layers and find aligned concept keys.
        
        Args:
            sample_layer_files: Mapping of sample_layer_idx -> vector_file_path
            intersection_mode: If True, keep only concepts present in all layers.
                              If False, use union of all concepts (pad with None).
        
        Returns:
            List of concept keys (keys) common across all sample layers (or union)
        """
        all_concepts = []
        for layer_idx, vector_path in sample_layer_files.items():
            vectors = self.load_vectors_for_layer(layer_idx, vector_path)
            all_concepts.append(set(vectors.keys()))

        if intersection_mode:
            # Keep only concepts present in all layers
            aligned = set.intersection(*all_concepts) if all_concepts else set()
            if self.verbose:
                print(f"  Intersection mode: {len(aligned)} concepts aligned across all layers")
        else:
            # Use union (optional, not recommended for fair comparison)
            aligned = set.union(*all_concepts) if all_concepts else set()
            if self.verbose:
                print(f"  Union mode: {len(aligned)} concepts across all layers")

        return sorted(list(aligned))

    def compute_dual_peak_detection(self, 
                                   intro_diffs: List[float],
                                   injection_layers: List[int],
                                   sample_layer_idx: int,
                                   threshold: float = 0.0) -> Dict:
        """Compute dual-peak detection metric for one sample layer.
        
        Args:
            intro_diffs: List of introspection logit diffs, indexed by injection layer
            injection_layers: List of injection layer indices
            sample_layer_idx: The sample (source) layer for this vector
            threshold: Threshold for first crossing (secondary metric)
            
        Returns:
            Dict with keys:
                - early_peak_layer: injection layer for early peak (if exists)
                - early_peak_magnitude: logit diff at early peak
                - early_peak_distance: sample_layer - early_peak_layer (positive layers before sample)
                - late_peak_layer: injection layer for late peak (if exists)
                - late_peak_magnitude: logit diff at late peak
                - late_peak_distance: late_peak_layer - sample_layer (positive = after)
                - threshold_crossing_layer: first layer where intro_diff > threshold
        """
        result = {
            'early_peak_layer': None,
            'early_peak_magnitude': None,
            'early_peak_distance': None,
            'late_peak_layer': None,
            'late_peak_magnitude': None,
            'late_peak_distance': None,
            'threshold_crossing_layer': None,
        }

        if not intro_diffs or not injection_layers:
            return result

        # Convert to numpy for easier indexing
        intro_diffs_np = np.array(intro_diffs)
        injection_layers_np = np.array(injection_layers)

        # Find early peak (argmax before sample layer)
        early_mask = injection_layers_np < sample_layer_idx
        if np.any(early_mask):
            early_indices = np.where(early_mask)[0]
            early_argmax = early_indices[np.argmax(intro_diffs_np[early_indices])]
            result['early_peak_layer'] = int(injection_layers_np[early_argmax])
            result['early_peak_magnitude'] = float(intro_diffs_np[early_argmax])
            result['early_peak_distance'] = int(sample_layer_idx - result['early_peak_layer'])

        # Find late peak (argmax after sample layer)
        late_mask = injection_layers_np > sample_layer_idx
        if np.any(late_mask):
            late_indices = np.where(late_mask)[0]
            late_argmax = late_indices[np.argmax(intro_diffs_np[late_indices])]
            result['late_peak_layer'] = int(injection_layers_np[late_argmax])
            result['late_peak_magnitude'] = float(intro_diffs_np[late_argmax])
            result['late_peak_distance'] = int(result['late_peak_layer'] - sample_layer_idx)

        # Find threshold crossing
        crossing_mask = intro_diffs_np > threshold
        if np.any(crossing_mask):
            crossing_indices = np.where(crossing_mask)[0]
            result['threshold_crossing_layer'] = int(injection_layers_np[crossing_indices[0]])

        return result

    def run_detection_sweep(self,
                           sample_layer_files: Optional[Dict[int, str]] = None,
                           injection_layers: Optional[List[int]] = None,
                           magnitude: float = 1.0,
                           threshold: float = 0.0,
                           num_trials: int = 1,
                           token_pos: int = -1,
                           concept_filter: Optional[List[str]] = None,
                           output_dir: Optional[str] = None,
                           plot_concepts: Optional[List[str]] = None,
                           plot_random_sample: int = 0,
                           selection_seed: int = 42,
                           run_selected_only: bool = False,
                           plot_single_during_run: bool = False,
                           keep_trial_history: bool = False,
                           vector_mode: str = "concept",
                           layer_norms: Optional[Dict[int, float]] = None,
                           fixed_norm: Optional[float] = None,
                           random_seed: int = 12345,
                           sample_layers: Optional[List[int]] = None,
                           num_random_vectors: int = 20,
                           hidden_size_override: Optional[int] = None) -> Dict:
        """Run 2D sweep: sample layer × injection layer.

        Args:
            sample_layer_files: Dict mapping sample_layer_idx -> vector_file_path.
                Required for 'concept' and 'concept_layer_norm' modes. Optional
                for the 'random_*' modes — when omitted the sample layers come
                from `sample_layers` and the per-sample-layer norm from
                `layer_norms` (no vector files are loaded).
            injection_layers: List of injection layer indices to test
            magnitude: Global scaling factor applied AFTER the vector is built
            threshold: Threshold for detection crossing (secondary metric)
            num_trials: Number of trials per cell
            token_pos: Token position for injection (-1 for last)
            concept_filter: Optional list of concepts to test (subset of aligned)
            output_dir: Directory to save outputs (default: cwd)
            vector_mode: One of:
                - 'concept'             : inject the concept vector (existing behavior).
                                          Requires sample_layer_files.
                - 'random_fixed'        : random unit vector scaled to a single scalar
                                          target norm used for every sample layer.
                                          The scalar is `fixed_norm` if provided,
                                          otherwise the mean of `layer_norms.values()`.
                                          Does NOT require sample_layer_files.
                - 'random_layer_norm'   : random unit vector scaled to
                                          `layer_norms[sample_layer]`.
                                          Does NOT require sample_layer_files.
                - 'concept_layer_norm'  : concept vector renormalized so its norm
                                          equals `layer_norms[sample_layer]`.
                                          Requires sample_layer_files.
                For all modes the steering injection used is `vector * magnitude`.
            layer_norms: {sample_layer: target_norm} from the manifold-analysis CSV.
            fixed_norm: Optional override for 'random_fixed' mode. When None and
                        layer_norms is supplied, the mean across all CSV layer
                        norms is used.
            random_seed: Seed for the random-vector generator
            sample_layers: Explicit sample-layer list used when
                           sample_layer_files is omitted (random_* modes).
            num_random_vectors: Number of synthetic random "concepts" sampled
                                per sample layer when running a random_* mode
                                without sample_layer_files (default: 20).

        Returns:
            Results dict with metadata, per-cell metrics, and summaries
        """
        if vector_mode not in VECTOR_MODES and vector_mode not in VECTOR_MODE_ALIASES:
            raise ValueError(f"vector_mode must be one of {VECTOR_MODES} (or alias in "
                             f"{list(VECTOR_MODE_ALIASES)}), got {vector_mode!r}")
        vector_mode = _canonical_vector_mode(vector_mode)
        if injection_layers is None:
            raise ValueError("injection_layers is required")

        random_only = vector_mode in ("random_fixed", "random_sample_norm", "random_injection_norm")
        needs_files = vector_mode in ("concept", "concept_sample_norm", "concept_injection_norm")
        needs_layer_norms = vector_mode in (
            "concept_sample_norm", "concept_injection_norm",
            "random_sample_norm", "random_injection_norm",
        )

        # random_* modes require some norm source
        if random_only and not layer_norms and vector_mode != "random_fixed":
            raise ValueError(f"vector_mode={vector_mode!r} requires layer_norms")
        if vector_mode == "random_fixed" and not layer_norms and fixed_norm is None and not sample_layer_files:
            raise ValueError(
                "vector_mode='random_fixed' needs either --layer-norms-csv (mean is used), "
                "an explicit --fixed-norm, or --sample-layer-files (mean concept norm per layer)."
            )
        if needs_layer_norms and not layer_norms:
            raise ValueError(f"vector_mode={vector_mode!r} requires layer_norms")

        # Resolve sample-layer set + decide whether to load vector files
        if needs_files:
            if not sample_layer_files:
                raise ValueError(f"vector_mode={vector_mode!r} requires sample_layer_files")
            sample_layers_resolved = sorted(sample_layer_files.keys())
            load_files = True
        else:
            if sample_layer_files:
                sample_layers_resolved = sorted(sample_layer_files.keys())
                load_files = True  # honour them so concepts align with the concept run
            else:
                if not sample_layers:
                    raise ValueError(
                        f"vector_mode={vector_mode!r} without sample_layer_files "
                        "requires `sample_layers`"
                    )
                sample_layers_resolved = sorted(sample_layers)
                load_files = False

        if output_dir is None:
            output_dir = '.'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        injection_layers = sorted(injection_layers)

        # Resolve concept list
        if load_files:
            all_concepts = self.get_aligned_concepts(
                {sl: sample_layer_files[sl] for sl in sample_layers_resolved},
                intersection_mode=True,
            )
        else:
            # Synthetic labels: one "random_i" per random vector draw
            all_concepts = [f"random_{i}" for i in range(num_random_vectors)]

        concepts, selected_for_plots = self._build_concept_selection(
            all_concepts=all_concepts,
            concept_filter=concept_filter,
            plot_concepts=plot_concepts,
            plot_random_sample=plot_random_sample,
            selection_seed=selection_seed,
            run_selected_only=run_selected_only,
        )
        if not concepts:
            raise ValueError("No concepts selected to run. Check concept names or selection flags.")

        # Resolve hidden size for synthetic random vectors
        if hidden_size_override is not None:
            hidden_size = int(hidden_size_override)
        else:
            hidden_size = infer_hidden_size(self.model, self.model_name)

        # For random_fixed, pre-resolve the single global target norm
        global_random_norm = None
        if vector_mode == "random_fixed":
            if fixed_norm is not None:
                global_random_norm = float(fixed_norm)
            elif layer_norms:
                global_random_norm = float(np.mean(list(layer_norms.values())))
            # else: when sample_layer_files is provided we fall back to the
            # per-sample-layer mean concept norm computed below.

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Detection Layer Sweep: {len(sample_layers_resolved)} sample layers × {len(injection_layers)} injection layers")
            print(f"Vector mode: {vector_mode} (load_files={load_files})")
            print(f"Concepts: {len(concepts)} (magnitude={magnitude}, threshold={threshold})")
            if selected_for_plots:
                print(f"Selected for single plots: {len(selected_for_plots)}")
            print(f"Trials per cell: {num_trials}")
            if vector_mode == "random_fixed" and global_random_norm is not None:
                print(f"random_fixed global target norm: {global_random_norm:.4f}")
            print(f"{'='*70}\n")

        # Initialize result structure
        results = {
            'metadata': {
                'model_name': self.model_name,
                'sample_layers': sample_layers_resolved,
                'injection_layers': sorted(injection_layers),
                'magnitude': magnitude,
                'threshold': threshold,
                'num_trials': num_trials,
                'token_pos': token_pos,
                'concepts_tested': len(concepts),
                'selected_plot_concepts': selected_for_plots,
                'timestamp': datetime.now().isoformat(),
                'vector_mode': vector_mode,
                'fixed_norm': fixed_norm,
                'random_seed': random_seed,
                'layer_norms': {int(k): float(v) for k, v in (layer_norms or {}).items()},
                'global_random_norm': global_random_norm,
                'num_random_vectors': num_random_vectors if not load_files else None,
                'hidden_size': hidden_size,
            },
            'per_cell': {},  # {(sample_layer, injection_layer, concept): {...}}
            'summary_per_sample_layer': {},  # {sample_layer: {early_peak, late_peak, ...}}
            'sample_layer_aggregates': {},  # {sample_layer: {intro_diffs, control_diffs, intro_stds, control_stds}}
        }

        # Extra baseline measurement
        if self.verbose:
            print("Computing baseline...")
        baseline_intro, baseline_control = self.run_baseline()
        results['metadata']['baseline_intro'] = float(baseline_intro)
        results['metadata']['baseline_control'] = float(baseline_control)

        # Main loop: sample layer × injection layer × concept × trials
        total_cells = len(sample_layers_resolved) * len(injection_layers) * len(concepts)
        completed = 0

        for sample_layer_idx in sample_layers_resolved:
            if self.verbose:
                print(f"\nSample Layer {sample_layer_idx}:")

            # Load vectors for this sample layer if needed; otherwise use a
            # placeholder shape (hidden_size,) for random vector generation.
            if load_files:
                vectors_dict = self.load_vectors_for_layer(
                    sample_layer_idx, sample_layer_files[sample_layer_idx]
                )
                concept_norms_here = [float(vectors_dict[c].norm().item())
                                      for c in concepts if c in vectors_dict]
                mean_concept_norm = float(np.mean(concept_norms_here)) if concept_norms_here else 0.0
                shape_ref_dtype = next(iter(vectors_dict.values())).dtype
                shape_ref_shape = next(iter(vectors_dict.values())).shape
            else:
                vectors_dict = None
                mean_concept_norm = 0.0
                if hidden_size is None:
                    raise RuntimeError(
                        "Could not determine hidden_size from model config. "
                        "Pass --hidden-size explicitly, or add a default for this "
                        "model id to HIDDEN_SIZE_DEFAULTS in detection_layer_harness.py "
                        "(see e.g. backup/adapter_full/manual/ for tensor shapes)."
                    )
                shape_ref_dtype = torch.float32
                shape_ref_shape = (hidden_size,)

            if self.verbose and vector_mode == "random_fixed":
                used_norm = (global_random_norm if global_random_norm is not None
                             else mean_concept_norm)
                print(f"  random_fixed target norm @ L{sample_layer_idx}: {used_norm:.4f}")
            if self.verbose and vector_mode == "random_layer_norm":
                target_norm = float(resolve_layer_norm(layer_norms, sample_layer_idx))
                print(f"  random_layer_norm target norm @ L{sample_layer_idx}: {target_norm:.4f}")

            sum_intro = np.zeros(len(injection_layers), dtype=float)
            sum_control = np.zeros(len(injection_layers), dtype=float)
            sumsq_intro = np.zeros(len(injection_layers), dtype=float)
            sumsq_control = np.zeros(len(injection_layers), dtype=float)

            for concept in concepts:
                # Reference tensor (real or zero placeholder) used only for shape/dtype
                if vectors_dict is not None and concept in vectors_dict:
                    concept_vec = vectors_dict[concept].cpu()
                else:
                    concept_vec = torch.zeros(shape_ref_shape, dtype=shape_ref_dtype)

                # Build the direction once per (sample_layer, concept).
                # Concept modes use the actual concept vector; random modes use a
                # deterministic random direction (unit-norm) so that scaling happens
                # purely through the per-cell target norm.
                if vector_mode in ("concept", "concept_sample_norm", "concept_injection_norm"):
                    direction_vec = concept_vec
                    concept_dir_norm = float(direction_vec.norm().item())
                else:
                    g = torch.Generator()
                    g.manual_seed(random_seed + hash((sample_layer_idx, concept)) % (2**31))
                    direction_vec = _unit_random_like(concept_vec, g)
                    concept_dir_norm = 1.0  # unit by construction

                # Per-mode helper that produces (vector, effective_norm) for a
                # given injection_layer. Direction is fixed; only the norm varies.
                def _build_for_injection(inj_layer: int):
                    if vector_mode == "concept":
                        return direction_vec, concept_dir_norm
                    if vector_mode == "concept_sample_norm":
                        target_ = float(resolve_layer_norm(layer_norms, sample_layer_idx))
                        if concept_dir_norm > 0:
                            return direction_vec * (target_ / concept_dir_norm), target_
                        return direction_vec, target_
                    if vector_mode == "concept_injection_norm":
                        target_ = float(resolve_layer_norm(layer_norms, inj_layer))
                        if concept_dir_norm > 0:
                            return direction_vec * (target_ / concept_dir_norm), target_
                        return direction_vec, target_
                    if vector_mode == "random_sample_norm":
                        target_ = float(resolve_layer_norm(layer_norms, sample_layer_idx))
                        return direction_vec * target_, target_
                    if vector_mode == "random_injection_norm":
                        target_ = float(resolve_layer_norm(layer_norms, inj_layer))
                        return direction_vec * target_, target_
                    # random_fixed: single global scalar across the whole sweep
                    target_ = (global_random_norm
                               if global_random_norm is not None
                               else mean_concept_norm)
                    return direction_vec * target_, target_

                concept_intro_diffs = []
                concept_control_diffs = []
                concept_intro_stds = []
                concept_control_stds = []

                for injection_layer in injection_layers:
                    intro_trial_diffs = []
                    control_trial_diffs = []

                    # Build per-cell steering vector (direction fixed for this
                    # (S, concept); target norm may vary per injection layer
                    # depending on vector_mode).
                    vector, effective_norm = _build_for_injection(injection_layer)

                    for trial in range(num_trials):
                        # Measure introspection score with this vector at this layer
                        intro_diff = self._measure_injection_score(
                            layer_idx=injection_layer,
                            vector=vector * magnitude,
                            token_pos=token_pos,
                            steer_all_tokens=False
                        )

                        # Measure control question with the same steering applied, matching prior experiments.
                        control_diff = self._measure_question_score(
                            question=self.control_question,
                            layer_idx=injection_layer,
                            vector=vector * magnitude,
                            token_pos=token_pos,
                            steer_all_tokens=False,
                        )

                        intro_trial_diffs.append(intro_diff)
                        control_trial_diffs.append(control_diff)

                    # Compute mean/std across trials
                    intro_mean = float(np.mean(intro_trial_diffs))
                    control_mean = float(np.mean(control_trial_diffs))
                    intro_std = float(np.std(intro_trial_diffs)) if num_trials > 1 else 0.0
                    control_std = float(np.std(control_trial_diffs)) if num_trials > 1 else 0.0

                    # Generate matched noise for comparison
                    noise_vector = self.generate_matched_noise_vector(vector)
                    noise_trial_diffs = []
                    for trial in range(num_trials):
                        noise_diff = self._measure_injection_score(
                            layer_idx=injection_layer,
                            vector=noise_vector * magnitude,
                            token_pos=token_pos,
                            steer_all_tokens=False
                        )
                        noise_trial_diffs.append(noise_diff)
                    noise_mean = float(np.mean(noise_trial_diffs))
                    noise_std = float(np.std(noise_trial_diffs)) if num_trials > 1 else 0.0

                    # Store cell result
                    cell_key = (sample_layer_idx, injection_layer, concept)
                    cell_data = {
                        'intro_mean': intro_mean,
                        'intro_std': intro_std,
                        'control_mean': control_mean,
                        'control_std': control_std,
                        'noise_mean': noise_mean,
                        'noise_std': noise_std,
                        'effective_norm': effective_norm,
                        'vector_mode': vector_mode,
                    }
                    if keep_trial_history:
                        cell_data['intro_trials'] = intro_trial_diffs
                        cell_data['control_trials'] = control_trial_diffs
                        cell_data['noise_trials'] = noise_trial_diffs
                    results['per_cell'][cell_key] = cell_data

                    concept_intro_diffs.append(intro_mean)
                    concept_control_diffs.append(control_mean)
                    concept_intro_stds.append(intro_std)
                    concept_control_stds.append(control_std)

                    inj_pos = injection_layers.index(injection_layer)
                    sum_intro[inj_pos] += intro_mean
                    sum_control[inj_pos] += control_mean
                    sumsq_intro[inj_pos] += intro_mean * intro_mean
                    sumsq_control[inj_pos] += control_mean * control_mean

                    completed += 1
                    if completed % 10 == 0:
                        print(f"  Progress: {completed}/{total_cells} cells")

                    self._release_cuda()

                if plot_single_during_run and concept in selected_for_plots:
                    concept_peak_metric = self.compute_dual_peak_detection(
                        intro_diffs=concept_intro_diffs,
                        injection_layers=injection_layers,
                        sample_layer_idx=sample_layer_idx,
                        threshold=threshold,
                    )
                    self._plot_single_concept_curve(
                        sample_layer_idx=sample_layer_idx,
                        concept=concept,
                        injection_layers=injection_layers,
                        intro_diffs=concept_intro_diffs,
                        control_diffs=concept_control_diffs,
                        intro_stds=concept_intro_stds,
                        control_stds=concept_control_stds,
                        baseline_intro=baseline_intro,
                        baseline_control=baseline_control,
                        peak_metric=concept_peak_metric,
                        output_dir=output_dir,
                        mode_tag=vector_mode,
                    )

                del direction_vec
                self._release_cuda()

            # Compute dual-peak detection for this sample layer (averaged over concepts)
            n_concepts = float(len(concepts))
            avg_intro_per_injection = (sum_intro / n_concepts).tolist()
            avg_control_per_injection = (sum_control / n_concepts).tolist()
            var_intro = np.maximum((sumsq_intro / n_concepts) - (sum_intro / n_concepts) ** 2, 0.0)
            var_control = np.maximum((sumsq_control / n_concepts) - (sum_control / n_concepts) ** 2, 0.0)
            std_intro = np.sqrt(var_intro).tolist()
            std_control = np.sqrt(var_control).tolist()

            peak_metric = self.compute_dual_peak_detection(
                intro_diffs=avg_intro_per_injection,
                injection_layers=injection_layers,
                sample_layer_idx=sample_layer_idx,
                threshold=threshold
            )

            results['summary_per_sample_layer'][sample_layer_idx] = peak_metric
            results['sample_layer_aggregates'][sample_layer_idx] = {
                'intro_diffs': avg_intro_per_injection,
                'control_diffs': avg_control_per_injection,
                'intro_stds': std_intro,
                'control_stds': std_control,
            }

            self._release_cuda()

        if self.verbose:
            print(f"\n{'='*70}")
            print("Sweep complete. Processing results...")

        return results

    def save_results(self, results: Dict, output_dir: str = '.', tag: Optional[str] = None):
        """Save results to CSV, JSON and a pickle dump (for full-fidelity replotting).

        Args:
            results: Results dict from run_detection_sweep
            output_dir: Directory to save to
            tag: Optional filename tag (e.g. the vector mode). When None, the
                 metadata['vector_mode'] is used if present.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if tag is None:
            tag = results.get('metadata', {}).get('vector_mode', 'run')
        tag_suffix = f"_{tag}" if tag else ""

        # Save JSON with full metadata and summaries
        json_path = os.path.join(output_dir, f"detection_results{tag_suffix}_{timestamp}.json")
        json_data = {
            'metadata': results['metadata'],
            'summary_per_sample_layer': results['summary_per_sample_layer'],
            'sample_layer_aggregates': {
                str(k): v for k, v in results['sample_layer_aggregates'].items()
            },
            'per_cell': {str(k): v for k, v in results['per_cell'].items()},
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        if self.verbose:
            print(f"Saved JSON: {json_path}")

        # Save a pickle of the raw results dict (preserves tuple keys, full fidelity)
        pkl_path = os.path.join(output_dir, f"detection_results{tag_suffix}_{timestamp}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)
        if self.verbose:
            print(f"Saved pickle: {pkl_path}")

        # Save CSV with one row per cell
        csv_path = os.path.join(output_dir, f"detection_results{tag_suffix}_{timestamp}.csv")
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'sample_layer', 'injection_layer', 'concept',
                'intro_mean', 'intro_std',
                'control_mean', 'control_std',
                'noise_mean', 'noise_std',
                'effective_norm', 'vector_mode',
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for (sample_layer, injection_layer, concept), cell_data in results['per_cell'].items():
                writer.writerow({
                    'sample_layer': sample_layer,
                    'injection_layer': injection_layer,
                    'concept': concept,
                    'intro_mean': cell_data['intro_mean'],
                    'intro_std': cell_data['intro_std'],
                    'control_mean': cell_data['control_mean'],
                    'control_std': cell_data['control_std'],
                    'noise_mean': cell_data['noise_mean'],
                    'noise_std': cell_data['noise_std'],
                    'effective_norm': cell_data.get('effective_norm'),
                    'vector_mode': cell_data.get('vector_mode',
                                                 results.get('metadata', {}).get('vector_mode')),
                })
        if self.verbose:
            print(f"Saved CSV: {csv_path}")

        # Save peak summary CSV with positions/distances per sample layer
        peaks_csv_path = os.path.join(output_dir, f"detection_peaks{tag_suffix}_{timestamp}.csv")
        with open(peaks_csv_path, 'w', newline='') as f:
            fieldnames = [
                'sample_layer',
                'early_peak_layer', 'early_peak_magnitude', 'early_peak_distance',
                'late_peak_layer', 'late_peak_magnitude', 'late_peak_distance',
                'threshold_crossing_layer',
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample_layer, peak_data in results['summary_per_sample_layer'].items():
                writer.writerow({
                    'sample_layer': sample_layer,
                    'early_peak_layer': peak_data['early_peak_layer'],
                    'early_peak_magnitude': peak_data['early_peak_magnitude'],
                    'early_peak_distance': peak_data['early_peak_distance'],
                    'late_peak_layer': peak_data['late_peak_layer'],
                    'late_peak_magnitude': peak_data['late_peak_magnitude'],
                    'late_peak_distance': peak_data['late_peak_distance'],
                    'threshold_crossing_layer': peak_data['threshold_crossing_layer'],
                })
        if self.verbose:
            print(f"Saved peak summary CSV: {peaks_csv_path}")

        return json_path, csv_path, peaks_csv_path

    def plot_per_sample_layer(self,
                             results: Dict,
                             sample_layer_idx: int,
                             output_dir: str = '.'):
        """Plot introspection curves for one sample layer (matching _plot_layer_effects style).
        
        Args:
            results: Results dict from run_detection_sweep
            sample_layer_idx: Which sample layer to plot
            output_dir: Directory to save plot
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get aggregate data for this sample layer
        aggregate = results['sample_layer_aggregates'][sample_layer_idx]
        injection_layers = results['metadata']['injection_layers']
        baseline_intro = results['metadata']['baseline_intro']
        baseline_control = results['metadata']['baseline_control']
        peak_metric = results['summary_per_sample_layer'][sample_layer_idx]

        intro_diffs = aggregate['intro_diffs']
        control_diffs = aggregate['control_diffs']
        intro_stds = aggregate['intro_stds']
        control_stds = aggregate['control_stds']

        # Create figure matching _plot_layer_effects style
        fig = plt.figure(figsize=(14, 9))
        ax = plt.subplot(111)

        # Plot baselines (dashed lines)
        plt.axhline(y=baseline_intro, color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Baseline introspection: {baseline_intro:+.2f}')
        plt.axhline(y=baseline_control, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Baseline control: {baseline_control:+.2f}')

        # Plot introspection (blue circles)
        plt.plot(injection_layers, intro_diffs, 'o-', linewidth=2, markersize=5, color='blue',
                label='Introspection question')

        # Plot control (red squares)
        plt.plot(injection_layers, control_diffs, 's-', linewidth=2, markersize=5, color='red',
                label='Control question')

        # Error bars if multi-trial
        if np.any(np.array(intro_stds) > 0):
            plt.errorbar(injection_layers, intro_diffs, yerr=intro_stds, fmt='none', ecolor='blue',
                        capsize=4, alpha=0.4)
        if np.any(np.array(control_stds) > 0):
            plt.errorbar(injection_layers, control_diffs, yerr=control_stds, fmt='none', ecolor='red',
                        capsize=4, alpha=0.4)

        # Mark peak positions
        if peak_metric['early_peak_layer'] is not None:
            early_layer = peak_metric['early_peak_layer']
            early_mag = peak_metric['early_peak_magnitude']
            plt.plot(early_layer, early_mag, 'bv', markersize=10, label=f'Early peak (L{early_layer})')

        if peak_metric['late_peak_layer'] is not None:
            late_layer = peak_metric['late_peak_layer']
            late_mag = peak_metric['late_peak_magnitude']
            plt.plot(late_layer, late_mag, 'r^', markersize=10, label=f'Late peak (L{late_layer})')

        # Mark sample layer as vertical line
        plt.axvline(x=sample_layer_idx, color='green', linestyle=':', linewidth=2, alpha=0.6,
                   label=f'Sample layer L{sample_layer_idx}')

        # Styling
        plt.xlabel('Injection Layer Index', fontsize=13)
        plt.ylabel('Logit(Yes) - Logit(No)', fontsize=13)

        model_display = self.model_name.split("/")[-1]
        plt.title(f'Detection Analysis: {model_display} (Sampled @ L{sample_layer_idx})',
                 fontsize=13, fontweight='bold', pad=20)

        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='best')
        plt.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)

        # Shaded region
        y_min, y_max = plt.ylim()
        if y_max > 0:
            plt.axhspan(0, y_max, alpha=0.05, color='green')
        if y_min < 0:
            plt.axhspan(y_min, 0, alpha=0.05, color='orange')

        # Info box
        textstr = f'Model: {model_display}\nSample Layer: {sample_layer_idx}\n'
        textstr += f'Early peak: L{peak_metric["early_peak_layer"]} (dist={peak_metric["early_peak_distance"]})' if peak_metric['early_peak_layer'] is not None else 'Early peak: None'
        textstr += f'\nLate peak: L{peak_metric["late_peak_layer"]} (dist={peak_metric["late_peak_distance"]})' if peak_metric['late_peak_layer'] is not None else '\nLate peak: None'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')

        plt.tight_layout()

        mode_tag = results.get('metadata', {}).get('vector_mode', 'concept')
        # Save
        filename = os.path.join(output_dir, f"detection_sample_l{sample_layer_idx}_{mode_tag}.png")
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Saved plot: {filename}")
        except OSError as e:
            print(f"[warn] failed to save {filename}: {e!r} — continuing sweep")
        plt.close()

    def plot_heatmap(self,
                    results: Dict,
                    output_dir: str = '.'):
        """Plot heatmaps: introspection and control, sample layer × injection layer.
        
        Args:
            results: Results dict from run_detection_sweep
            output_dir: Directory to save plot
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sample_layers = results['metadata']['sample_layers']
        injection_layers = results['metadata']['injection_layers']

        # Build matrices: sample_layer × injection_layer (already aggregated in run)
        intro_matrix = np.zeros((len(sample_layers), len(injection_layers)))
        control_matrix = np.zeros((len(sample_layers), len(injection_layers)))

        for i, sample_layer in enumerate(sample_layers):
            intro_matrix[i, :] = np.array(results['sample_layer_aggregates'][sample_layer]['intro_diffs'])
            control_matrix[i, :] = np.array(results['sample_layer_aggregates'][sample_layer]['control_diffs'])

        vmin = min(float(np.min(intro_matrix)), float(np.min(control_matrix)))
        vmax = max(float(np.max(intro_matrix)), float(np.max(control_matrix)))

        # Create side-by-side heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        im1 = ax1.imshow(intro_matrix, aspect='auto', cmap='RdBu_r', origin='lower',
                         extent=[injection_layers[0], injection_layers[-1],
                                 sample_layers[0], sample_layers[-1]],
                         vmin=vmin, vmax=vmax)
        im2 = ax2.imshow(control_matrix, aspect='auto', cmap='RdBu_r', origin='lower',
                         extent=[injection_layers[0], injection_layers[-1],
                                 sample_layers[0], sample_layers[-1]],
                         vmin=vmin, vmax=vmax)

        ax1.set_xlabel('Injection Layer Index', fontsize=12)
        ax1.set_ylabel('Sample Layer Index', fontsize=12)
        ax1.set_title('Introspection (Yes-No)', fontsize=12, fontweight='bold')

        ax2.set_xlabel('Injection Layer Index', fontsize=12)
        ax2.set_ylabel('Sample Layer Index', fontsize=12)
        ax2.set_title('Control (Yes-No)', fontsize=12, fontweight='bold')

        model_display = self.model_name.split("/")[-1]
        fig.suptitle(f'Detection Heatmaps: {model_display}\n(Sample Layer × Injection Layer)',
                     fontsize=13, fontweight='bold', y=0.98)

        # Reserve room on the right and place a dedicated colorbar axis.
        fig.subplots_adjust(right=0.90, wspace=0.25)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.72])
        cbar = fig.colorbar(im2, cax=cbar_ax)
        cbar.set_label('Mean Logit(Yes) - Logit(No)', fontsize=11)

        plt.tight_layout(rect=[0, 0, 0.90, 0.95])

        mode_tag = results.get('metadata', {}).get('vector_mode', 'concept')
        # Save
        filename = os.path.join(output_dir, f"detection_heatmap_{mode_tag}.png")
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Saved heatmap: {filename}")
        except OSError as e:
            print(f"[warn] failed to save {filename}: {e!r} — continuing sweep")
        plt.close()

    def plot_detection_summary(self,
                              results: Dict,
                              output_dir: str = '.'):
        """Plot detection summary with clearer geometry and distance views.
        
        Args:
            results: Results dict from run_detection_sweep
            output_dir: Directory to save plot
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sample_layers = results['metadata']['sample_layers']
        early_peaks = []
        late_peaks = []
        early_distances = []
        late_distances = []

        for sample_layer in sample_layers:
            peak = results['summary_per_sample_layer'][sample_layer]
            early_peaks.append(peak['early_peak_layer'])
            late_peaks.append(peak['late_peak_layer'])
            early_distances.append(peak['early_peak_distance'])
            late_distances.append(peak['late_peak_distance'])

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Plot 1: sample layer vs detected peak layer with y=x reference
        ax = axes[0, 0]
        valid_early_pairs = [(s, p) for s, p in zip(sample_layers, early_peaks) if p is not None]
        valid_late_pairs = [(s, p) for s, p in zip(sample_layers, late_peaks) if p is not None]
        if valid_early_pairs:
            ax.scatter([x for x, _ in valid_early_pairs], [y for _, y in valid_early_pairs],
                       color='blue', alpha=0.8, label='Early peak', marker='v')
        if valid_late_pairs:
            ax.scatter([x for x, _ in valid_late_pairs], [y for _, y in valid_late_pairs],
                       color='red', alpha=0.8, label='Late peak', marker='^')
        min_layer = min(sample_layers)
        max_layer = max(sample_layers)
        ax.plot([min_layer, max_layer], [min_layer, max_layer], 'k--', alpha=0.5, label='y = x')
        ax.set_xlabel('Sample Layer', fontsize=11)
        ax.set_ylabel('Peak Injection Layer', fontsize=11)
        ax.set_title('Peak Position vs Sample Layer', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: distances as a function of sample layer
        ax = axes[0, 1]
        early_dist_pairs = [(s, d) for s, d in zip(sample_layers, early_distances) if d is not None]
        late_dist_pairs = [(s, d) for s, d in zip(sample_layers, late_distances) if d is not None]
        if early_dist_pairs:
            ax.plot([s for s, _ in early_dist_pairs], [d for _, d in early_dist_pairs],
                    'o-', color='blue', alpha=0.8, label='Early distance (sample - peak)')
        if late_dist_pairs:
            ax.plot([s for s, _ in late_dist_pairs], [d for _, d in late_dist_pairs],
                    's-', color='red', alpha=0.8, label='Late distance (peak - sample)')
        ax.set_xlabel('Sample Layer', fontsize=11)
        ax.set_ylabel('Distance (layers)', fontsize=11)
        ax.set_title('Peak Distance by Sample Layer', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: distance distributions
        ax = axes[1, 0]
        valid_early_dist = [d for d in early_distances if d is not None]
        valid_late_dist = [d for d in late_distances if d is not None]
        if valid_early_dist:
            ax.hist(valid_early_dist, bins=min(10, max(3, len(valid_early_dist))), alpha=0.5,
                    label='Early distance', color='blue')
        if valid_late_dist:
            ax.hist(valid_late_dist, bins=min(10, max(3, len(valid_late_dist))), alpha=0.5,
                    label='Late distance', color='red')
        ax.set_xlabel('Distance (layers)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distance Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: compact summary text
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = "Summary Statistics\n" + "=" * 40 + "\n\n"
        summary_text += f"Early peaks found: {len(valid_early_pairs)}/{len(early_peaks)}\n"
        summary_text += f"Late peaks found: {len(valid_late_pairs)}/{len(late_peaks)}\n"
        if valid_early_dist:
            summary_text += f"Early peak mean distance: {np.mean(valid_early_dist):.1f}\n"
            summary_text += f"Early peak std distance: {np.std(valid_early_dist):.1f}\n"
            summary_text += f"Early peak median distance: {np.median(valid_early_dist):.1f}\n"
        if valid_late_dist:
            summary_text += f"Late peak mean distance: {np.mean(valid_late_dist):.1f}\n"
            summary_text += f"Late peak std distance: {np.std(valid_late_dist):.1f}\n"
            summary_text += f"Late peak median distance: {np.median(valid_late_dist):.1f}\n"

        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        mode_tag = results.get('metadata', {}).get('vector_mode', 'concept')
        # Save
        filename = os.path.join(output_dir, f"detection_summary_{mode_tag}.png")
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Saved summary plot: {filename}")
        except OSError as e:
            print(f"[warn] failed to save {filename}: {e!r} — continuing sweep")
        plt.close()

    # ------------------------------------------------------------------
    # Ablation comparison: injection vs injection + refusal ablation
    # ------------------------------------------------------------------

    def _measure_question_score_with_ablation(
        self,
        question,
        layer_idx: int,
        vector: Optional[torch.Tensor],
        token_pos: int,
        direction: Optional[torch.Tensor] = None,
        per_layer_directions: Optional[torch.Tensor] = None,
        ablation_strength: float = 1.0,
        ablation_scope: str = "all_layers",
        steer_all_tokens: bool = False,
        layer_weight_fn=None,
    ) -> float:
        """Like _measure_question_score but with simultaneous refusal-direction ablation.

        Supports two ablation modes (exactly one must be provided):
          - ``direction``: a single unit-norm tensor applied identically to all
            ablated layers (original behaviour, backwards compatible).
          - ``per_layer_directions``: a (num_layers, hidden_dim) tensor where
            row i is the unit-norm refusal direction for layer i (new mode
            matching 03d_refusal_abliteration.py).  In this mode the hook is
            applied to ALL token positions and ``token_pos``/``steer_all_tokens``
            are ignored for the ablation (they still govern the steering injection).

        Args:
            question: Chat-template message list
            layer_idx: Layer index for steering injection
            vector: Steering vector (None = no injection)
            token_pos: Token position for injection (-1 = last)
            direction: Single unit-norm refusal direction (old single-direction mode)
            per_layer_directions: Per-layer directions tensor (new per-layer mode)
            ablation_strength: Uniform projection weight (per-layer mode: used when
                layer_weight_fn is None; single-direction mode: always used)
            ablation_scope: Which layers receive ablation hooks
            steer_all_tokens: Apply steering at every token position
            layer_weight_fn: Optional callable (layer_idx)->float for per-layer weights
                (per-layer mode only). When provided, overrides ablation_strength.
        """
        if per_layer_directions is None and direction is None:
            raise ValueError(
                "Either 'direction' or 'per_layer_directions' must be provided to "
                "_measure_question_score_with_ablation"
            )

        layer_indices = self._resolve_ablation_layer_indices(layer_idx, ablation_scope)

        if per_layer_directions is not None:
            ablation_handles = self._register_per_layer_ablation_hooks(
                layer_indices=layer_indices,
                refusal_dirs=per_layer_directions,
                ablation_strength=ablation_strength,
                token_pos=token_pos,
                steer_all_tokens=steer_all_tokens,
                weight_fn=layer_weight_fn,
            )
        else:
            ablation_handles = self._register_refusal_ablation_hooks(
                layer_indices=layer_indices,
                direction=direction,
                ablation_strength=ablation_strength,
                token_pos=token_pos,
                steer_all_tokens=steer_all_tokens,
            )

        try:
            return self._measure_question_score(
                question=question,
                layer_idx=layer_idx,
                vector=vector,
                token_pos=token_pos,
                steer_all_tokens=steer_all_tokens,
            )
        finally:
            for handle in ablation_handles:
                handle.remove()

    def run_ablation_comparison_sweep(
        self,
        sample_layer_idx: int,
        vector_path: str,
        injection_layers: List[int],
        refusal_direction_path: Optional[str] = None,
        per_layer_directions_path: Optional[str] = None,
        magnitude: float = 1.0,
        ablation_strength: float = 1.0,
        ablation_scope: str = "all_layers",
        per_region_weights: Optional[Dict] = None,
        region_boundaries: Optional[Dict] = None,
        threshold: float = 0.0,
        num_trials: int = 1,
        token_pos: int = -1,
        concept_filter: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        plot_concepts: Optional[List[str]] = None,
        plot_random_sample: int = 0,
        selection_seed: int = 42,
        run_selected_only: bool = False,
        plot_single_during_run: bool = False,
    ) -> Dict:
        """Compare steering vs steering+refusal-ablation for one sample layer.

        For each injection layer × concept, measures logit(Yes) − logit(No) for:
          1. Plain steering (intro and control questions)
          2. Steering + refusal projection ablation (intro and control questions)

        The result supports testing the hypothesis: ablation selectively lifts the
        introspection signal where the plain-steering conditions are already separated
        (non-overlapping error bars), while shifting both conditions together in
        regions where they overlap.

        Args:
            sample_layer_idx: Layer from which concept vectors were sampled
            vector_path: Path to .pt file containing concept-vector dict
            injection_layers: Layer indices to sweep over
            refusal_direction_path: Path to .pt single refusal direction tensor
                (original mode – same direction applied to all ablated layers).
                Mutually exclusive with per_layer_directions_path.
            per_layer_directions_path: Path to .pt tensor of shape (num_layers,
                hidden_dim) where row i is the refusal direction for layer i
                (new per-layer mode, as in 03d_refusal_abliteration.py).
                Mutually exclusive with refusal_direction_path.
            magnitude: Scaling factor for vectors
            ablation_strength: Uniform projection weight for per-layer mode when
                per_region_weights is None; always used for single-direction mode.
            ablation_scope: Which layers receive ablation hooks
                ('all_layers', 'from_layer_onward', 'steering_layer')
            per_region_weights: Dict of 20-region weights (per-layer mode only).
                When provided, overrides ablation_strength per layer using the
                same boundary scheme as 03d_refusal_abliteration.py.
                Use GEMMA3_27B_REGION_WEIGHTS for the Gemma 3 27B OPTUNA values.
            region_boundaries: Dict of layer-index boundaries for the 20 regions.
                Defaults to GEMMA3_27B_REGION_BOUNDARIES when per_region_weights
                is provided.
            threshold: Threshold for secondary detection crossing metric
            num_trials: Trials per (injection_layer, concept) cell
            token_pos: Token position for injection (-1 = last)
            concept_filter: Explicit list of concepts to include
            output_dir: Output directory for plots and results
            plot_concepts: Concepts to plot individually
            plot_random_sample: Add N random concepts to per-concept plot set
            selection_seed: Seed for random concept sampling
            run_selected_only: Restrict sweep to selected/plot concepts only
            plot_single_during_run: Save per-concept plots as they complete

        Returns:
            Dict with keys:
              'metadata'  – run configuration and baseline values
              'per_cell'  – {(concept, injection_layer): {intro/control mean+std × 2 conditions}}
              'aggregate' – concept-averaged diffs and stds for all four curves
        """
        if output_dir is None:
            output_dir = '.'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        injection_layers = sorted(injection_layers)

        # Validate and load refusal direction(s)
        if refusal_direction_path is not None and per_layer_directions_path is not None:
            raise ValueError(
                "Provide at most one of 'refusal_direction_path' (single direction) or "
                "'per_layer_directions_path' (per-layer directions), not both."
            )
        if refusal_direction_path is None and per_layer_directions_path is None:
            raise ValueError(
                "One of 'refusal_direction_path' or 'per_layer_directions_path' must be provided."
            )

        if per_layer_directions_path is not None:
            per_layer_dirs = self._load_per_layer_refusal_directions(per_layer_directions_path)
            single_direction = None
            ablation_mode = "per_layer"
            ablation_path = per_layer_directions_path
            # Build per-layer weight function
            if per_region_weights is not None:
                boundaries = region_boundaries if region_boundaries is not None \
                    else GEMMA3_27B_REGION_BOUNDARIES
                layer_weight_fn = build_region_weight_fn(per_region_weights, boundaries)
                ablation_mode = "per_layer_region_weights"
            else:
                layer_weight_fn = None  # use uniform ablation_strength
        else:
            single_direction = self._load_refusal_direction(refusal_direction_path)
            per_layer_dirs = None
            layer_weight_fn = None
            ablation_mode = "single"
            ablation_path = refusal_direction_path

        # Load vectors for this sample layer
        vectors_dict = self.load_vectors_for_layer(sample_layer_idx, vector_path)
        all_concepts = sorted(list(vectors_dict.keys()))

        # Build concept selection
        concepts, selected_for_plots = self._build_concept_selection(
            all_concepts=all_concepts,
            concept_filter=concept_filter,
            plot_concepts=plot_concepts,
            plot_random_sample=plot_random_sample,
            selection_seed=selection_seed,
            run_selected_only=run_selected_only,
        )
        if not concepts:
            raise ValueError("No concepts selected. Check concept names or selection flags.")

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Ablation Comparison Sweep: Sample L{sample_layer_idx} × {len(injection_layers)} injection layers")
            print(f"Concepts: {len(concepts)}, Magnitude: {magnitude}, "
                  f"Ablation: strength={ablation_strength} scope={ablation_scope} mode={ablation_mode}"
                  + (" [region-weights]" if layer_weight_fn is not None else ""))
            if selected_for_plots:
                print(f"Selected for single plots: {len(selected_for_plots)}")
            print(f"Trials per cell: {num_trials}")
            print(f"{'='*70}\n")

        # Baseline (no steering, no ablation)
        if self.verbose:
            print("Computing baseline...")
        baseline_intro, baseline_control = self.run_baseline()

        results = {
            'metadata': {
                'model_name': self.model_name,
                'sample_layer_idx': sample_layer_idx,
                'injection_layers': injection_layers,
                'magnitude': magnitude,
                'ablation_strength': ablation_strength,
                'ablation_scope': ablation_scope,
                'ablation_mode': ablation_mode,
                'use_region_weights': layer_weight_fn is not None,
                'threshold': threshold,
                'num_trials': num_trials,
                'token_pos': token_pos,
                'concepts_tested': len(concepts),
                'selected_plot_concepts': selected_for_plots,
                'refusal_direction_path': ablation_path,
                'baseline_intro': float(baseline_intro),
                'baseline_control': float(baseline_control),
                'timestamp': datetime.now().isoformat(),
            },
            'per_cell': {},
            'aggregate': {},
        }

        n_layers = len(injection_layers)
        sum_intro = np.zeros(n_layers)
        sum_ctrl = np.zeros(n_layers)
        sum_intro_abl = np.zeros(n_layers)
        sum_ctrl_abl = np.zeros(n_layers)
        sumsq_intro = np.zeros(n_layers)
        sumsq_ctrl = np.zeros(n_layers)
        sumsq_intro_abl = np.zeros(n_layers)
        sumsq_ctrl_abl = np.zeros(n_layers)

        total_cells = len(injection_layers) * len(concepts)
        completed = 0

        for concept in concepts:
            vector = vectors_dict[concept].cpu()
            scaled = vector * magnitude

            c_intro, c_ctrl, c_intro_abl, c_ctrl_abl = [], [], [], []
            c_intro_s, c_ctrl_s, c_intro_abl_s, c_ctrl_abl_s = [], [], [], []

            for inj_pos, injection_layer in enumerate(injection_layers):
                t_intro, t_ctrl, t_intro_abl, t_ctrl_abl = [], [], [], []

                for _ in range(num_trials):
                    # --- Plain steering ---
                    t_intro.append(self._measure_question_score(
                        question=self.introspection_question,
                        layer_idx=injection_layer,
                        vector=scaled,
                        token_pos=token_pos,
                    ))
                    t_ctrl.append(self._measure_question_score(
                        question=self.control_question,
                        layer_idx=injection_layer,
                        vector=scaled,
                        token_pos=token_pos,
                    ))
                    # --- Steering + refusal ablation ---
                    t_intro_abl.append(self._measure_question_score_with_ablation(
                        question=self.introspection_question,
                        layer_idx=injection_layer,
                        vector=scaled,
                        token_pos=token_pos,
                        direction=single_direction,
                        per_layer_directions=per_layer_dirs,
                        ablation_strength=ablation_strength,
                        ablation_scope=ablation_scope,
                        layer_weight_fn=layer_weight_fn,
                    ))
                    t_ctrl_abl.append(self._measure_question_score_with_ablation(
                        question=self.control_question,
                        layer_idx=injection_layer,
                        vector=scaled,
                        token_pos=token_pos,
                        direction=single_direction,
                        per_layer_directions=per_layer_dirs,
                        ablation_strength=ablation_strength,
                        ablation_scope=ablation_scope,
                        layer_weight_fn=layer_weight_fn,
                    ))

                def _ms(trials):
                    m = float(np.mean(trials))
                    s = float(np.std(trials)) if num_trials > 1 else 0.0
                    return m, s

                im, is_ = _ms(t_intro)
                cm, cs = _ms(t_ctrl)
                iam, ias = _ms(t_intro_abl)
                cam, cas = _ms(t_ctrl_abl)

                c_intro.append(im); c_intro_s.append(is_)
                c_ctrl.append(cm);  c_ctrl_s.append(cs)
                c_intro_abl.append(iam); c_intro_abl_s.append(ias)
                c_ctrl_abl.append(cam); c_ctrl_abl_s.append(cas)

                results['per_cell'][(concept, injection_layer)] = {
                    'intro_mean': im, 'intro_std': is_,
                    'control_mean': cm, 'control_std': cs,
                    'intro_ablated_mean': iam, 'intro_ablated_std': ias,
                    'control_ablated_mean': cam, 'control_ablated_std': cas,
                }

                sum_intro[inj_pos] += im
                sum_ctrl[inj_pos] += cm
                sum_intro_abl[inj_pos] += iam
                sum_ctrl_abl[inj_pos] += cam
                sumsq_intro[inj_pos] += im * im
                sumsq_ctrl[inj_pos] += cm * cm
                sumsq_intro_abl[inj_pos] += iam * iam
                sumsq_ctrl_abl[inj_pos] += cam * cam

                completed += 1
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{total_cells} cells "
                          f"(concept={concept}, inj_layer={injection_layer})")

                self._release_cuda()

            if plot_single_during_run and concept in selected_for_plots:
                self._plot_single_concept_ablation_comparison(
                    sample_layer_idx=sample_layer_idx,
                    concept=concept,
                    injection_layers=injection_layers,
                    intro_diffs=c_intro,
                    control_diffs=c_ctrl,
                    intro_abl_diffs=c_intro_abl,
                    control_abl_diffs=c_ctrl_abl,
                    intro_stds=c_intro_s,
                    control_stds=c_ctrl_s,
                    intro_abl_stds=c_intro_abl_s,
                    control_abl_stds=c_ctrl_abl_s,
                    baseline_intro=float(baseline_intro),
                    baseline_control=float(baseline_control),
                    ablation_scope=ablation_scope,
                    ablation_strength=ablation_strength,
                    output_dir=output_dir,
                )

            del vector, scaled
            self._release_cuda()

        # Build aggregate across concepts
        n = float(len(concepts))

        def _agg_std(sumsq, total, n):
            var = np.maximum((sumsq / n) - (total / n) ** 2, 0.0)
            return np.sqrt(var).tolist()

        results['aggregate'] = {
            'intro_diffs': (sum_intro / n).tolist(),
            'control_diffs': (sum_ctrl / n).tolist(),
            'intro_ablated_diffs': (sum_intro_abl / n).tolist(),
            'control_ablated_diffs': (sum_ctrl_abl / n).tolist(),
            'intro_stds': _agg_std(sumsq_intro, sum_intro, n),
            'control_stds': _agg_std(sumsq_ctrl, sum_ctrl, n),
            'intro_ablated_stds': _agg_std(sumsq_intro_abl, sum_intro_abl, n),
            'control_ablated_stds': _agg_std(sumsq_ctrl_abl, sum_ctrl_abl, n),
        }

        if self.verbose:
            print(f"\n{'='*70}")
            print("Ablation comparison sweep complete.")

        return results

    def _plot_single_concept_ablation_comparison(
        self,
        sample_layer_idx: int,
        concept: str,
        injection_layers: List[int],
        intro_diffs: List[float],
        control_diffs: List[float],
        intro_abl_diffs: List[float],
        control_abl_diffs: List[float],
        intro_stds: List[float],
        control_stds: List[float],
        intro_abl_stds: List[float],
        control_abl_stds: List[float],
        baseline_intro: float,
        baseline_control: float,
        ablation_scope: str,
        ablation_strength: float,
        output_dir: str,
        abl_label: Optional[str] = None,
    ):
        """Per-concept overlay: plain steering vs steering+ablation."""
        fig, ax = plt.subplots(figsize=(14, 9))

        inj = np.array(injection_layers)

        # Baselines
        ax.axhline(y=baseline_intro, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Baseline intro: {baseline_intro:+.2f}')
        ax.axhline(y=baseline_control, color='firebrick', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Baseline control: {baseline_control:+.2f}')

        # Plain steering (solid)
        ax.plot(inj, intro_diffs, 'o-', linewidth=2, markersize=5, color='steelblue',
                label='Intro (plain)')
        ax.plot(inj, control_diffs, 's-', linewidth=2, markersize=5, color='firebrick',
                label='Control (plain)')

        # Steering + ablation (dashed)
        _abl_tag = abl_label if abl_label is not None else f'+ablation α={ablation_strength}'
        ax.plot(inj, intro_abl_diffs, 'o--', linewidth=2, markersize=5, color='steelblue',
                alpha=0.65, label=f'Intro ({_abl_tag})')
        ax.plot(inj, control_abl_diffs, 's--', linewidth=2, markersize=5, color='firebrick',
                alpha=0.65, label=f'Control ({_abl_tag})')

        # Shaded error bands
        for vals, stds, color, alpha in [
            (intro_diffs,     intro_stds,     'steelblue', 0.18),
            (control_diffs,   control_stds,   'firebrick', 0.18),
            (intro_abl_diffs, intro_abl_stds, 'steelblue', 0.10),
            (control_abl_diffs, control_abl_stds, 'firebrick', 0.10),
        ]:
            arr_v = np.array(vals)
            arr_s = np.array(stds)
            if np.any(arr_s > 0):
                ax.fill_between(inj, arr_v - arr_s, arr_v + arr_s, color=color, alpha=alpha)

        # Sample layer marker and zero line
        ax.axvline(x=sample_layer_idx, color='green', linestyle=':', linewidth=2, alpha=0.65,
                   label=f'Sample layer L{sample_layer_idx}')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.4)

        y_min, y_max = ax.get_ylim()
        if y_max > 0:
            ax.axhspan(0, y_max, alpha=0.04, color='green')
        if y_min < 0:
            ax.axhspan(y_min, 0, alpha=0.04, color='orange')

        ax.set_xlabel('Injection Layer Index', fontsize=13)
        ax.set_ylabel('Logit(Yes) − Logit(No)', fontsize=13)
        model_display = self.model_name.split("/")[-1]
        ax.set_title(
            f'Ablation Comparison: {model_display}  |  Sample L{sample_layer_idx}  |  concept={concept}',
            fontsize=12, fontweight='bold', pad=20,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

        textstr = (f'Concept: {concept}\nSample Layer: {sample_layer_idx}\n'
                   f'Ablation scope: {ablation_scope}\n'
                   + (f'Ablation: {abl_label}' if abl_label else f'Ablation strength: {ablation_strength}'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')

        plt.tight_layout()
        safe_concept = concept.replace(' ', '_').replace('/', '_')
        filename = os.path.join(
            output_dir, f"ablation_cmp_sample_l{sample_layer_idx}_{safe_concept}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if self.verbose:
            print(f"  Saved per-concept ablation plot: {filename}")

    def plot_ablation_comparison(
        self,
        results: Dict,
        output_dir: str = '.',
    ):
        """Plot concept-averaged ablation comparison (analogous to plot_per_sample_layer).

        Shows four curves on the same axes:
          • solid blue  – introspection, plain steering
          • solid red   – control, plain steering
          • dashed blue – introspection, steering + refusal ablation
          • dashed red  – control, steering + refusal ablation

        Shaded ±1σ bands (across concepts) are drawn around each curve.
        A vertical green line marks the sample layer.

        The relative movement of dashed vs solid curves tests the hypothesis:
          – in regions where solid intro and control are separated, ablation lifts
            intro without moving control (selective improvement);
          – where they overlap, both dashed curves shift together.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        meta = results['metadata']
        agg = results['aggregate']
        injection_layers = meta['injection_layers']
        sample_layer_idx = meta['sample_layer_idx']
        baseline_intro = meta['baseline_intro']
        baseline_control = meta['baseline_control']

        inj = np.array(injection_layers)
        intro = np.array(agg['intro_diffs'])
        ctrl = np.array(agg['control_diffs'])
        intro_abl = np.array(agg['intro_ablated_diffs'])
        ctrl_abl = np.array(agg['control_ablated_diffs'])
        intro_s = np.array(agg['intro_stds'])
        ctrl_s = np.array(agg['control_stds'])
        intro_abl_s = np.array(agg['intro_ablated_stds'])
        ctrl_abl_s = np.array(agg['control_ablated_stds'])

        fig, ax = plt.subplots(figsize=(14, 9))

        # Baselines
        ax.axhline(y=baseline_intro, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.4,
                   label=f'Baseline intro: {baseline_intro:+.2f}')
        ax.axhline(y=baseline_control, color='firebrick', linestyle='--', linewidth=1.5, alpha=0.4,
                   label=f'Baseline control: {baseline_control:+.2f}')

        # Plain steering (solid)
        ax.plot(inj, intro, 'o-', linewidth=2.5, markersize=6, color='steelblue',
                label='Intro (plain steering)')
        ax.plot(inj, ctrl, 's-', linewidth=2.5, markersize=6, color='firebrick',
                label='Control (plain steering)')

        # Steering + ablation (dashed)
        if meta.get('ablation_mode') == 'pretrained_model':
            _abl_short = meta.get('abliterated_model_name', 'abliterated').split('/')[-1]
            _abl_tag = f'abliterated: {_abl_short}'
        else:
            _abl_tag = f'+ablation α={meta["ablation_strength"]}'
        ax.plot(inj, intro_abl, 'o--', linewidth=2.5, markersize=6, color='steelblue', alpha=0.65,
                label=f'Intro ({_abl_tag})')
        ax.plot(inj, ctrl_abl, 's--', linewidth=2.5, markersize=6, color='firebrick', alpha=0.65,
                label=f'Control ({_abl_tag})')

        # Shaded ±1σ bands
        for vals, stds, color, alpha in [
            (intro,     intro_s,     'steelblue', 0.18),
            (ctrl,      ctrl_s,      'firebrick', 0.18),
            (intro_abl, intro_abl_s, 'steelblue', 0.10),
            (ctrl_abl,  ctrl_abl_s,  'firebrick', 0.10),
        ]:
            if np.any(stds > 0):
                ax.fill_between(inj, vals - stds, vals + stds, color=color, alpha=alpha)

        # Sample layer marker and zero line
        ax.axvline(x=sample_layer_idx, color='green', linestyle=':', linewidth=2.5, alpha=0.7,
                   label=f'Sample layer L{sample_layer_idx}')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.4)

        y_min, y_max = ax.get_ylim()
        if y_max > 0:
            ax.axhspan(0, y_max, alpha=0.04, color='green')
        if y_min < 0:
            ax.axhspan(y_min, 0, alpha=0.04, color='orange')

        ax.set_xlabel('Injection Layer Index', fontsize=13)
        ax.set_ylabel('Logit(Yes) − Logit(No)', fontsize=13)
        model_display = meta['model_name'].split("/")[-1]
        ax.set_title(
            f'Ablation Comparison (avg {meta["concepts_tested"]} concepts): '
            f'{model_display}  |  Sample L{sample_layer_idx}',
            fontsize=12, fontweight='bold', pad=20,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

        if meta.get('ablation_mode') == 'pretrained_model':
            _abl_info = f'Abliterated: {meta.get("abliterated_model_name", "?").split("/")[-1]}'
        else:
            _abl_info = f'Ablation strength: {meta["ablation_strength"]}'
        textstr = (
            f'Model: {model_display}\n'
            f'Sample Layer: {sample_layer_idx}\n'
            f'Ablation scope: {meta["ablation_scope"]}\n'
            f'{_abl_info}\n'
            f'Concepts: {meta["concepts_tested"]}'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')

        plt.tight_layout()
        filename = os.path.join(output_dir, f"ablation_comparison_sample_l{sample_layer_idx}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        if self.verbose:
            print(f"Saved ablation comparison plot: {filename}")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Pretrained-abliterated-model comparison mode
    # ------------------------------------------------------------------

    def run_pretrained_ablation_comparison_sweep(
        self,
        sample_layer_idx: int,
        vector_path: str,
        injection_layers: List[int],
        abliterated_model_name: str,
        magnitude: float = 1.0,
        threshold: float = 0.0,
        num_trials: int = 1,
        token_pos: int = -1,
        concept_filter: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        plot_concepts: Optional[List[str]] = None,
        plot_random_sample: int = 0,
        selection_seed: int = 42,
        run_selected_only: bool = False,
        plot_single_during_run: bool = False,
        abliterated_fine_tune: bool = False,
        abliterated_device_map: str = "auto",
        abliterated_max_gpu_memory: Optional[str] = None,
        abliterated_max_cpu_memory: Optional[str] = None,
        abliterated_offload_folder: Optional[str] = None,
        unload_original: bool = True,
    ) -> Dict:
        """Compare steering on the original model vs. the same steering on a
        pre-abliterated model (e.g. ``uzaymacar/gemma-3-27b-abliterated``).

        Runs in two sequential phases to keep GPU memory manageable:

        Phase 1 – Original model (``self``):
            For each concept × injection layer, measure logit(Yes)−logit(No) with
            plain steering.  These become the "plain steering" curves.

        Phase 2 – Abliterated model:
            Load *abliterated_model_name*, apply the same steering vectors, and
            measure the same quantities.  These become the "ablated" curves.

        The results dict is structurally identical to the one returned by
        ``run_ablation_comparison_sweep`` so all downstream plot helpers work
        without modification.

        Args:
            sample_layer_idx: Layer from which concept vectors were sampled.
            vector_path: Path to .pt concept-vector dict.
            injection_layers: Layer indices to sweep.
            abliterated_model_name: HuggingFace name of the pre-abliterated model.
            magnitude: Steering-vector scale factor.
            threshold: Threshold for secondary detection-crossing metric.
            num_trials: Trials per (concept, injection_layer) cell.
            token_pos: Token position for injection (−1 = last).
            concept_filter: Explicit concept subset (default: all in vector file).
            output_dir: Directory for plots / JSON.
            plot_concepts: Concepts to generate individual plots for.
            plot_random_sample: Add N randomly sampled concepts to the plot set.
            selection_seed: Seed for random concept sampling.
            run_selected_only: Restrict sweep to plot/selected concepts only.
            plot_single_during_run: Generate per-concept plots after phase 2.
            abliterated_fine_tune: Whether to load a fine-tune adapter on the
                abliterated model (almost always False).
            abliterated_device_map: Transformers device_map for the second model.
            abliterated_max_gpu_memory: Per-GPU memory budget for the second model.
            abliterated_max_cpu_memory: CPU RAM budget for the second model.
            abliterated_offload_folder: Disk offload folder for the second model.
            unload_original: If True, delete self.model from GPU before loading the
                abliterated model.  Strongly recommended for large models.

        Returns:
            Dict with the same schema as ``run_ablation_comparison_sweep``:
            ``metadata``, ``per_cell``, ``aggregate``.
            Extra metadata keys: ``abliterated_model_name``,
            ``baseline_intro_abliterated``, ``baseline_control_abliterated``.
        """
        if output_dir is None:
            output_dir = '.'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        injection_layers = sorted(injection_layers)

        # --- Load vectors and build concept selection ---
        vectors_dict = self.load_vectors_for_layer(sample_layer_idx, vector_path)
        all_concepts = sorted(list(vectors_dict.keys()))

        concepts, selected_for_plots = self._build_concept_selection(
            all_concepts=all_concepts,
            concept_filter=concept_filter,
            plot_concepts=plot_concepts,
            plot_random_sample=plot_random_sample,
            selection_seed=selection_seed,
            run_selected_only=run_selected_only,
        )
        if not concepts:
            raise ValueError("No concepts selected. Check concept names or selection flags.")

        n_layers = len(injection_layers)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Pretrained Ablation Comparison: Sample L{sample_layer_idx} × "
                  f"{n_layers} injection layers")
            print(f"Original model:     {self.model_name}")
            print(f"Abliterated model:  {abliterated_model_name}")
            print(f"Concepts: {len(concepts)}, Magnitude: {magnitude}, Trials: {num_trials}")
            if selected_for_plots:
                print(f"Selected for single plots: {len(selected_for_plots)}")
            print(f"{'='*70}\n")

        # ----------------------------------------------------------------
        # Phase 1: original model
        # ----------------------------------------------------------------
        if self.verbose:
            print("Phase 1/2 – Computing baseline on original model...")
        baseline_intro, baseline_control = self.run_baseline()

        sum_intro = np.zeros(n_layers)
        sum_ctrl = np.zeros(n_layers)
        sumsq_intro = np.zeros(n_layers)
        sumsq_ctrl = np.zeros(n_layers)
        phase1_cells: Dict[Tuple, Dict] = {}

        total_cells = n_layers * len(concepts)
        completed = 0

        for concept in concepts:
            vector = vectors_dict[concept].cpu()
            scaled = vector * magnitude

            for inj_pos, injection_layer in enumerate(injection_layers):
                t_intro, t_ctrl = [], []
                for _ in range(num_trials):
                    t_intro.append(self._measure_question_score(
                        question=self.introspection_question,
                        layer_idx=injection_layer,
                        vector=scaled,
                        token_pos=token_pos,
                    ))
                    t_ctrl.append(self._measure_question_score(
                        question=self.control_question,
                        layer_idx=injection_layer,
                        vector=scaled,
                        token_pos=token_pos,
                    ))

                im = float(np.mean(t_intro))
                cm = float(np.mean(t_ctrl))
                is_ = float(np.std(t_intro)) if num_trials > 1 else 0.0
                cs = float(np.std(t_ctrl)) if num_trials > 1 else 0.0

                phase1_cells[(concept, injection_layer)] = {
                    'intro_mean': im, 'intro_std': is_,
                    'control_mean': cm, 'control_std': cs,
                }
                sum_intro[inj_pos] += im
                sum_ctrl[inj_pos] += cm
                sumsq_intro[inj_pos] += im * im
                sumsq_ctrl[inj_pos] += cm * cm

                completed += 1
                if completed % 10 == 0:
                    print(f"  Phase 1 progress: {completed}/{total_cells} cells "
                          f"(concept={concept}, inj_layer={injection_layer})")
                self._release_cuda()

            del vector, scaled
            self._release_cuda()

        # ----------------------------------------------------------------
        # Optionally free GPU before loading the second model
        # ----------------------------------------------------------------
        if unload_original:
            self._unload_model()

        # ----------------------------------------------------------------
        # Phase 2: abliterated model
        # ----------------------------------------------------------------
        if self.verbose:
            print(f"\nPhase 2/2 – Loading abliterated model: {abliterated_model_name}")

        abl_analyzer = DetectionAnalyzer(
            model_name=abliterated_model_name,
            verbose=self.verbose,
            fine_tune=abliterated_fine_tune,
            model_device_map=abliterated_device_map,
            max_gpu_memory=abliterated_max_gpu_memory,
            max_cpu_memory=abliterated_max_cpu_memory,
            offload_folder=abliterated_offload_folder,
        )

        # uzaymacar/gemma-3-27b-abliterated (and similar repos) ship a custom
        # modeling file whose hooks are NOT activated until init_ablation() is
        # called explicitly.  Without this the model runs as the plain base model.
        if hasattr(abl_analyzer.model, 'init_ablation'):
            if self.verbose:
                print("Calling model.init_ablation() to register refusal-ablation hooks...")
            abl_analyzer.model.init_ablation()

        if self.verbose:
            print("Computing baseline on abliterated model...")
        baseline_intro_abl, baseline_control_abl = abl_analyzer.run_baseline()

        sum_intro_abl = np.zeros(n_layers)
        sum_ctrl_abl = np.zeros(n_layers)
        sumsq_intro_abl = np.zeros(n_layers)
        sumsq_ctrl_abl = np.zeros(n_layers)
        phase2_cells: Dict[Tuple, Dict] = {}

        completed = 0
        for concept in concepts:
            vector = vectors_dict[concept].cpu()
            scaled = vector * magnitude

            for inj_pos, injection_layer in enumerate(injection_layers):
                t_intro_abl, t_ctrl_abl = [], []
                for _ in range(num_trials):
                    t_intro_abl.append(abl_analyzer._measure_question_score(
                        question=abl_analyzer.introspection_question,
                        layer_idx=injection_layer,
                        vector=scaled,
                        token_pos=token_pos,
                    ))
                    t_ctrl_abl.append(abl_analyzer._measure_question_score(
                        question=abl_analyzer.control_question,
                        layer_idx=injection_layer,
                        vector=scaled,
                        token_pos=token_pos,
                    ))

                iam = float(np.mean(t_intro_abl))
                cam = float(np.mean(t_ctrl_abl))
                ias = float(np.std(t_intro_abl)) if num_trials > 1 else 0.0
                cas = float(np.std(t_ctrl_abl)) if num_trials > 1 else 0.0

                phase2_cells[(concept, injection_layer)] = {
                    'intro_ablated_mean': iam, 'intro_ablated_std': ias,
                    'control_ablated_mean': cam, 'control_ablated_std': cas,
                }
                sum_intro_abl[inj_pos] += iam
                sum_ctrl_abl[inj_pos] += cam
                sumsq_intro_abl[inj_pos] += iam * iam
                sumsq_ctrl_abl[inj_pos] += cam * cam

                completed += 1
                if completed % 10 == 0:
                    print(f"  Phase 2 progress: {completed}/{total_cells} cells "
                          f"(concept={concept}, inj_layer={injection_layer})")
                abl_analyzer._release_cuda()

            del vector, scaled
            abl_analyzer._release_cuda()

        # ----------------------------------------------------------------
        # Merge phase 1 + phase 2 into a unified result dict
        # ----------------------------------------------------------------
        def _agg_std(sumsq, total, n):
            var = np.maximum((sumsq / n) - (total / n) ** 2, 0.0)
            return np.sqrt(var).tolist()

        n = float(len(concepts))
        per_cell: Dict = {}
        for key in phase1_cells:
            per_cell[key] = {**phase1_cells[key], **phase2_cells[key]}

        results = {
            'metadata': {
                'model_name': self.model_name,
                'sample_layer_idx': sample_layer_idx,
                'injection_layers': injection_layers,
                'magnitude': magnitude,
                'ablation_strength': None,
                'ablation_scope': 'pretrained_model',
                'ablation_mode': 'pretrained_model',
                'abliterated_model_name': abliterated_model_name,
                'use_region_weights': False,
                'threshold': threshold,
                'num_trials': num_trials,
                'token_pos': token_pos,
                'concepts_tested': len(concepts),
                'selected_plot_concepts': selected_for_plots,
                'refusal_direction_path': None,
                'baseline_intro': float(baseline_intro),
                'baseline_control': float(baseline_control),
                'baseline_intro_abliterated': float(baseline_intro_abl),
                'baseline_control_abliterated': float(baseline_control_abl),
                'timestamp': datetime.now().isoformat(),
            },
            'per_cell': per_cell,
            'aggregate': {
                'intro_diffs': (sum_intro / n).tolist(),
                'control_diffs': (sum_ctrl / n).tolist(),
                'intro_ablated_diffs': (sum_intro_abl / n).tolist(),
                'control_ablated_diffs': (sum_ctrl_abl / n).tolist(),
                'intro_stds': _agg_std(sumsq_intro, sum_intro, n),
                'control_stds': _agg_std(sumsq_ctrl, sum_ctrl, n),
                'intro_ablated_stds': _agg_std(sumsq_intro_abl, sum_intro_abl, n),
                'control_ablated_stds': _agg_std(sumsq_ctrl_abl, sum_ctrl_abl, n),
            },
        }

        # Per-concept plots (generated after both phases complete)
        if plot_single_during_run and selected_for_plots:
            abl_short = abliterated_model_name.split('/')[-1]
            for concept in selected_for_plots:
                if concept not in {c for c, _ in per_cell}:
                    continue
                c_intro     = [per_cell[(concept, l)]['intro_mean']          for l in injection_layers]
                c_ctrl      = [per_cell[(concept, l)]['control_mean']        for l in injection_layers]
                c_intro_abl = [per_cell[(concept, l)]['intro_ablated_mean']  for l in injection_layers]
                c_ctrl_abl  = [per_cell[(concept, l)]['control_ablated_mean']for l in injection_layers]
                c_intro_s   = [per_cell[(concept, l)]['intro_std']           for l in injection_layers]
                c_ctrl_s    = [per_cell[(concept, l)]['control_std']         for l in injection_layers]
                c_intro_abl_s = [per_cell[(concept, l)]['intro_ablated_std'] for l in injection_layers]
                c_ctrl_abl_s  = [per_cell[(concept, l)]['control_ablated_std']for l in injection_layers]
                self._plot_single_concept_ablation_comparison(
                    sample_layer_idx=sample_layer_idx,
                    concept=concept,
                    injection_layers=injection_layers,
                    intro_diffs=c_intro,
                    control_diffs=c_ctrl,
                    intro_abl_diffs=c_intro_abl,
                    control_abl_diffs=c_ctrl_abl,
                    intro_stds=c_intro_s,
                    control_stds=c_ctrl_s,
                    intro_abl_stds=c_intro_abl_s,
                    control_abl_stds=c_ctrl_abl_s,
                    baseline_intro=float(baseline_intro),
                    baseline_control=float(baseline_control),
                    ablation_scope='pretrained_model',
                    ablation_strength=0.0,
                    output_dir=output_dir,
                    abl_label=f'abliterated: {abl_short}',
                )

        if self.verbose:
            print(f"\n{'='*70}")
            print("Pretrained ablation comparison sweep complete.")

        return results
