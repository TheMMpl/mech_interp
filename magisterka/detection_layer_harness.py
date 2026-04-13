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
import random
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from datetime import datetime
from introspection_gemma import IntrospectionExperiment
import concept_vectors

# Adapter path for fine-tuned models
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/workspace/project/adapter_bias_corrected")


class DetectionAnalyzer(IntrospectionExperiment):
    """Automated analyzer for detection layer vs sample layer comparison."""

    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 verbose: bool = True,
                 fine_tune: bool = True):
        """Initialize DetectionAnalyzer (inherits from IntrospectionExperiment).
        
        Args:
            model_name: HuggingFace model identifier
            verbose: Verbose output during processing
            fine_tune: Whether to use fine-tuned adapter
        """
        super().__init__(model_name=model_name, verbose=verbose, fine_tune=fine_tune)
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
                modified = hidden_states.clone()
                if steer_all_tokens:
                    modified = modified + vector.unsqueeze(0).unsqueeze(0)
                else:
                    seq_len = modified.shape[1]
                    token_idx = token_pos if token_pos >= 0 else seq_len + token_pos
                    if 0 <= token_idx < seq_len:
                        modified[:, token_idx, :] = modified[:, token_idx, :] + vector
                return (modified,) + output[1:]

            hidden_states = output
            modified = hidden_states.clone()
            if steer_all_tokens:
                modified = modified + vector.unsqueeze(0).unsqueeze(0)
            else:
                seq_len = modified.shape[1]
                token_idx = token_pos if token_pos >= 0 else seq_len + token_pos
                if 0 <= token_idx < seq_len:
                    modified[:, token_idx, :] = modified[:, token_idx, :] + vector
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
        filename = os.path.join(output_dir, f"single_sample_l{sample_layer_idx}_{safe_concept}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
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
                           sample_layer_files: Dict[int, str],
                           injection_layers: List[int],
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
                           keep_trial_history: bool = False) -> Dict:
        """Run 2D sweep: sample layer × injection layer.
        
        Args:
            sample_layer_files: Dict mapping sample_layer_idx -> vector_file_path
            injection_layers: List of injection layer indices to test
            magnitude: Scaling factor for vectors
            threshold: Threshold for detection crossing (secondary metric)
            num_trials: Number of trials per cell
            token_pos: Token position for injection (-1 for last)
            concept_filter: Optional list of concepts to test (subset of aligned)
            output_dir: Directory to save outputs (default: cwd)
            
        Returns:
            Results dict with metadata, per-cell metrics, and summaries
        """
        if output_dir is None:
            output_dir = '.'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        injection_layers = sorted(injection_layers)

        # Get aligned concepts
        all_concepts = self.get_aligned_concepts(sample_layer_files, intersection_mode=True)
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

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Detection Layer Sweep: {len(sample_layer_files)} sample layers × {len(injection_layers)} injection layers")
            print(f"Concepts: {len(concepts)} (magnitude={magnitude}, threshold={threshold})")
            if selected_for_plots:
                print(f"Selected for single plots: {len(selected_for_plots)}")
            print(f"Trials per cell: {num_trials}")
            print(f"{'='*70}\n")

        # Initialize result structure
        results = {
            'metadata': {
                'model_name': self.model_name,
                'sample_layers': sorted(list(sample_layer_files.keys())),
                'injection_layers': sorted(injection_layers),
                'magnitude': magnitude,
                'threshold': threshold,
                'num_trials': num_trials,
                'token_pos': token_pos,
                'concepts_tested': len(concepts),
                'selected_plot_concepts': selected_for_plots,
                'timestamp': datetime.now().isoformat(),
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
        total_cells = len(sample_layer_files) * len(injection_layers) * len(concepts)
        completed = 0

        for sample_layer_idx, vector_path in sorted(sample_layer_files.items()):
            if self.verbose:
                print(f"\nSample Layer {sample_layer_idx}:")

            # Load vectors for this sample layer
            vectors_dict = self.load_vectors_for_layer(sample_layer_idx, vector_path)

            sum_intro = np.zeros(len(injection_layers), dtype=float)
            sum_control = np.zeros(len(injection_layers), dtype=float)
            sumsq_intro = np.zeros(len(injection_layers), dtype=float)
            sumsq_control = np.zeros(len(injection_layers), dtype=float)

            for concept in concepts:
                vector = vectors_dict[concept].to(self.device)
                concept_intro_diffs = []
                concept_control_diffs = []
                concept_intro_stds = []
                concept_control_stds = []

                for injection_layer in injection_layers:
                    intro_trial_diffs = []
                    control_trial_diffs = []

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
                    )

                del vector
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

    def save_results(self, results: Dict, output_dir: str = '.'):
        """Save results to CSV and JSON.
        
        Args:
            results: Results dict from run_detection_sweep
            output_dir: Directory to save to
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON with full metadata and summaries
        json_path = os.path.join(output_dir, f"detection_results_{timestamp}.json")
        json_data = {
            'metadata': results['metadata'],
            'summary_per_sample_layer': results['summary_per_sample_layer'],
            'per_cell': {str(k): v for k, v in results['per_cell'].items()},  # Convert tuple keys to strings
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        if self.verbose:
            print(f"Saved JSON: {json_path}")

        # Save CSV with one row per cell
        csv_path = os.path.join(output_dir, f"detection_results_{timestamp}.csv")
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'sample_layer', 'injection_layer', 'concept',
                'intro_mean', 'intro_std',
                'control_mean', 'control_std',
                'noise_mean', 'noise_std',
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
                })
        if self.verbose:
            print(f"Saved CSV: {csv_path}")

        # Save peak summary CSV with positions/distances per sample layer
        peaks_csv_path = os.path.join(output_dir, f"detection_peaks_{timestamp}.csv")
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

        # Save
        filename = os.path.join(output_dir, f"detection_sample_l{sample_layer_idx}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        if self.verbose:
            print(f"Saved plot: {filename}")
        plt.close()

    def plot_heatmap(self,
                    results: Dict,
                    output_dir: str = '.'):
        """Plot heatmap: sample layer × injection layer.
        
        Args:
            results: Results dict from run_detection_sweep
            output_dir: Directory to save plot
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sample_layers = results['metadata']['sample_layers']
        injection_layers = results['metadata']['injection_layers']

        # Build matrix: sample_layer × injection_layer (already aggregated in run)
        matrix = np.zeros((len(sample_layers), len(injection_layers)))

        for i, sample_layer in enumerate(sample_layers):
            matrix[i, :] = np.array(results['sample_layer_aggregates'][sample_layer]['intro_diffs'])

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', origin='lower',
                      extent=[injection_layers[0], injection_layers[-1],
                             sample_layers[0], sample_layers[-1]])

        ax.set_xlabel('Injection Layer Index', fontsize=12)
        ax.set_ylabel('Sample Layer Index', fontsize=12)

        model_display = self.model_name.split("/")[-1]
        ax.set_title(f'Detection Heatmap: {model_display}\n(Sample Layer × Injection Layer)',
                    fontsize=13, fontweight='bold', pad=15)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Mean Intro Logit Diff', fontsize=11)

        plt.tight_layout()

        # Save
        filename = os.path.join(output_dir, "detection_heatmap.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        if self.verbose:
            print(f"Saved heatmap: {filename}")
        plt.close()

    def plot_detection_summary(self,
                              results: Dict,
                              output_dir: str = '.'):
        """Plot detection summary: early/late peak positions and distances.
        
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

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Early peak positions
        ax = axes[0, 0]
        valid_early = [l for l in early_peaks if l is not None]
        ax.bar(range(len(valid_early)), valid_early, color='blue', alpha=0.7)
        ax.set_xlabel('Sample Layer (index)', fontsize=11)
        ax.set_ylabel('Early Peak Injection Layer', fontsize=11)
        ax.set_title('Early Peak Position', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 2: Late peak positions
        ax = axes[0, 1]
        valid_late = [l for l in late_peaks if l is not None]
        ax.bar(range(len(valid_late)), valid_late, color='red', alpha=0.7)
        ax.set_xlabel('Sample Layer (index)', fontsize=11)
        ax.set_ylabel('Late Peak Injection Layer', fontsize=11)
        ax.set_title('Late Peak Position', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 3: Distance distributions
        ax = axes[1, 0]
        valid_early_dist = [d for d in early_distances if d is not None]
        valid_late_dist = [d for d in late_distances if d is not None]
        ax.hist(valid_early_dist, bins=10, alpha=0.5, label='Early distance', color='blue')
        ax.hist(valid_late_dist, bins=10, alpha=0.5, label='Late distance', color='red')
        ax.set_xlabel('Distance (layers)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Peak Distance Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Summary table (as text)
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = "Summary Statistics\n" + "=" * 40 + "\n\n"
        summary_text += f"Early peaks found: {len(valid_early)}/{len(early_peaks)}\n"
        summary_text += f"Late peaks found: {len(valid_late)}/{len(late_peaks)}\n"
        if valid_early_dist:
            summary_text += f"Early peak mean distance: {np.mean(valid_early_dist):.1f}\n"
            summary_text += f"Early peak std distance: {np.std(valid_early_dist):.1f}\n"
        if valid_late_dist:
            summary_text += f"Late peak mean distance: {np.mean(valid_late_dist):.1f}\n"
            summary_text += f"Late peak std distance: {np.std(valid_late_dist):.1f}\n"

        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save
        filename = os.path.join(output_dir, "detection_summary.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        if self.verbose:
            print(f"Saved summary plot: {filename}")
        plt.close()
