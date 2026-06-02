#!/usr/bin/env python3
"""
Ablation Comparison CLI

Test whether refusal-direction ablation selectively improves introspection
detection at injection layers where the plain-steering signal is already clean
(non-overlapping error bars between control and introspection questions),
without compromising control performance.

Two ablation modes are supported:

  hook (default)
    Apply refusal-projection hooks at inference time on the original model.
    Requires --refusal-direction or --refusal-directions.

  pretrained
    Load a pre-abliterated model (default: uzaymacar/gemma-3-27b-abliterated)
    and run the same steering experiment on both the original and the
    abliterated model.  No refusal direction file is needed.
    The original model is unloaded before loading the abliterated one to
    keep GPU memory manageable (override with --no-unload-original).

Where signals are separated: ablation should lift the introspection curve
without shifting the control curve.

Where signals overlap: ablation shifts both curves together.

Example usage (hook mode):

  python ablation_comparison_cli.py \\
    --model "Qwen/Qwen2.5-7B-Instruct" \\
    --sample-layer 20 \\
    --vector-file vectors_20.pt \\
    --injection-layers 0 5 10 15 20 25 30 \\
    --refusal-direction result/qwen-7b/direction.pt \\
    --magnitude 3.0 \\
    --ablation-strength 1.0 \\
    --ablation-scope all_layers \\
    --trials 3 \\
    --plot-concepts sandwich bomb love logic \\
    --plot-random-sample 3 \\
    --output-dir ./ablation_results \\
    --plot \\
    --no-fine-tune

Example usage (pretrained mode):

  python ablation_comparison_cli.py \\
    --model "google/gemma-3-27b-it" \\
    --ablation-mode pretrained \\
    --abliterated-model "uzaymacar/gemma-3-27b-abliterated" \\
    --sample-layer 30 \\
    --vector-file adapter_full/vectors_30.pt \\
    --injection-layers 5 10 15 20 25 30 35 40 45 50 55 \\
    --magnitude 3.0 \\
    --trials 1 \\
    --plot-concepts sandwich bomb love logic \\
    --output-dir ./ablation_results \\
    --plot \\
    --no-fine-tune
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from detection_layer_harness import (
    DetectionAnalyzer,
    GEMMA3_27B_REGION_WEIGHTS,
    GEMMA3_27B_REGION_BOUNDARIES,
)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation Comparison: steering vs steering+refusal-ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Required ---
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name")
    parser.add_argument("--sample-layer", type=int, required=True,
                        help="Layer from which concept vectors were sampled")
    parser.add_argument("--vector-file", required=True,
                        help="Path to .pt file with concept vectors dict")
    parser.add_argument("--injection-layers", type=int, nargs="+", required=True,
                        help="Injection layer indices to sweep")
    # Refusal direction: provide exactly one of the two options below
    # (not required when --ablation-mode pretrained)
    dir_group = parser.add_mutually_exclusive_group(required=False)
    dir_group.add_argument("--refusal-direction",
                        help="Path to .pt single refusal direction tensor (same direction "
                             "applied to all ablated layers). Required for --ablation-mode hook.")
    dir_group.add_argument("--refusal-directions",
                        help="Path to .pt tensor of shape (num_layers, hidden_dim) with "
                             "per-layer refusal directions (as produced by "
                             "03d_refusal_abliteration.py). Required for --ablation-mode hook.")

    # --- Sweep parameters ---
    parser.add_argument("--magnitude", type=float, default=1.0,
                        help="Steering vector scale factor (default: 1.0)")
    parser.add_argument("--ablation-strength", type=float, nargs="+", default=[1.0],
                        help="Projection ablation strength(s); if multiple, runs one sweep per value "
                             "(default: 1.0). For per-layer mode with --use-region-weights this is "
                             "ignored in favour of the OPTUNA per-region weights.")
    parser.add_argument("--ablation-scope",
                        choices=["all_layers", "from_layer_onward", "steering_layer"],
                        default="all_layers",
                        help="Which layers receive ablation hooks (default: all_layers). "
                             "Ignored in --ablation-mode pretrained.")
    parser.add_argument("--use-region-weights", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="When using --refusal-directions, apply OPTUNA-optimized per-region "
                             "weights (Gemma 3 27B Trial 1584) instead of a uniform "
                             "--ablation-strength. Enabled by default. Pass --no-use-region-weights "
                             "to use a uniform weight instead.")
    parser.add_argument("--per-region-weights", default=None,
                        help="JSON string or path to JSON file with 20-region weight dict. "
                             "Overrides the built-in GEMMA3_27B_REGION_WEIGHTS when "
                             "--use-region-weights is set.")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Detection crossing threshold (default: 0.0)")
    parser.add_argument("--trials", type=int, default=1,
                        help="Trials per (injection_layer, concept) cell (default: 1)")
    parser.add_argument("--token-pos", type=int, default=-1,
                        help="Token position for injection, -1 = last (default: -1)")

    # --- Concept selection ---
    parser.add_argument("--concepts", nargs="+", default=None,
                        help="Explicit concept keys to test (default: all in vector file)")
    parser.add_argument("--plot-concepts", nargs="+", default=None,
                        help="Concepts to generate individual per-concept plots for")
    parser.add_argument("--plot-random-sample", type=int, default=0,
                        help="Add N randomly sampled concepts to the per-concept plot set")
    parser.add_argument("--selection-seed", type=int, default=42,
                        help="Seed for random concept sampling (default: 42)")
    parser.add_argument("--run-selected-only", action="store_true",
                        help="Restrict sweep to plot/selected concepts only")

    # --- Output ---
    parser.add_argument("--output-dir", default="./ablation_results",
                        help="Output directory (default: ./ablation_results)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots (aggregate + per-concept during run)")
    parser.add_argument("--save-json", action="store_true",
                        help="Save full results to JSON")

    # --- Model loading ---
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--no-fine-tune", action="store_true",
                        help="Skip loading the fine-tuned adapter")
    parser.add_argument("--model-device-map", default="auto",
                        help="Transformers device_map (default: auto)")
    parser.add_argument("--max-gpu-memory", default=None,
                        help="Per-GPU memory budget, e.g. '40GiB'")
    parser.add_argument("--max-cpu-memory", default=None,
                        help="CPU RAM budget, e.g. '220GiB'")
    parser.add_argument("--offload-folder", default=None,
                        help="Folder for CPU/disk offload tensors")

    # --- Ablation mode ---
    parser.add_argument("--ablation-mode",
                        choices=["hook", "pretrained"],
                        default="hook",
                        help="'hook' (default): apply refusal-projection hooks at inference time. "
                             "'pretrained': compare against a pre-abliterated model loaded from HF "
                             "(no hook needed; --refusal-direction/--refusal-directions ignored).")
    parser.add_argument("--abliterated-model",
                        default="uzaymacar/gemma-3-27b-abliterated",
                        help="HuggingFace name of the pre-abliterated model used when "
                             "--ablation-mode pretrained (default: uzaymacar/gemma-3-27b-abliterated).")
    parser.add_argument("--abliterated-fine-tune", action="store_true",
                        help="Load the fine-tune adapter on the abliterated model too "
                             "(almost always leave off).")
    parser.add_argument("--abliterated-device-map", default="auto",
                        help="Transformers device_map for the abliterated model (default: auto).")
    parser.add_argument("--abliterated-max-gpu-memory", default=None,
                        help="Per-GPU memory budget for the abliterated model, e.g. '40GiB'.")
    parser.add_argument("--abliterated-max-cpu-memory", default=None,
                        help="CPU RAM budget for the abliterated model, e.g. '220GiB'.")
    parser.add_argument("--abliterated-offload-folder", default=None,
                        help="Disk offload folder for the abliterated model.")
    parser.add_argument("--no-unload-original", action="store_true",
                        help="Do NOT unload the original model before loading the abliterated one "
                             "(risks OOM on large models; --ablation-mode pretrained only).")

    args = parser.parse_args()

    # Automatically enable plotting when per-concept plot arguments are given,
    # so that --plot-concepts / --plot-random-sample work without requiring --plot.
    if args.plot_concepts or args.plot_random_sample > 0:
        args.plot = True

    # --- Validate mode-specific requirements ---
    if args.ablation_mode == "hook":
        if args.refusal_direction is None and args.refusal_directions is None:
            parser.error(
                "--ablation-mode hook requires either --refusal-direction or "
                "--refusal-directions."
            )

    # Resolve per-region weights (only meaningful for --refusal-directions in hook mode)
    resolved_region_weights = None
    if args.ablation_mode == "hook" and args.refusal_directions and args.use_region_weights:
        if args.per_region_weights is not None:
            # Accept JSON string or path
            try:
                resolved_region_weights = json.loads(args.per_region_weights)
            except (json.JSONDecodeError, TypeError):
                with open(args.per_region_weights) as f:
                    resolved_region_weights = json.load(f)
        else:
            resolved_region_weights = GEMMA3_27B_REGION_WEIGHTS

    print(f"{'='*70}")
    print("Ablation Comparison Sweep")
    print(f"{'='*70}")
    print(f"Mode:             {args.ablation_mode}")
    print(f"Model:            {args.model}")
    print(f"Sample layer:     {args.sample_layer}")
    print(f"Vector file:      {args.vector_file}")
    print(f"Injection layers: {args.injection_layers}")
    if args.ablation_mode == "hook":
        if args.refusal_direction:
            print(f"Refusal dir:      {args.refusal_direction}  [single]")
        else:
            rw_label = "OPTUNA region weights" if resolved_region_weights else f"uniform {args.ablation_strength}"
            print(f"Refusal dirs:     {args.refusal_directions}  [per-layer, {rw_label}]")
        print(f"Ablation scope:   {args.ablation_scope}")
        if resolved_region_weights is None:
            print(f"Ablation strengths: {args.ablation_strength}")
        else:
            print(f"Ablation weights: per-region (20 sub-regions)")
    else:
        print(f"Abliterated model: {args.abliterated_model}")
        print(f"Unload original:  {not args.no_unload_original}")
    print(f"Magnitude:        {args.magnitude}")
    print(f"Trials:           {args.trials}")
    print(f"Output:           {args.output_dir}")
    print(f"{'='*70}\n")

    # Build original model (shared across all runs)
    analyzer = DetectionAnalyzer(
        model_name=args.model,
        verbose=args.verbose,
        fine_tune=not args.no_fine_tune,
        model_device_map=args.model_device_map,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=args.offload_folder,
    )

    all_results = {}

    # ----------------------------------------------------------------
    # Pretrained mode: single run, no strength sweep
    # ----------------------------------------------------------------
    if args.ablation_mode == "pretrained":
        run_dir = args.output_dir
        print("\n--- Pretrained abliterated model comparison ---")
        results = analyzer.run_pretrained_ablation_comparison_sweep(
            sample_layer_idx=args.sample_layer,
            vector_path=args.vector_file,
            injection_layers=args.injection_layers,
            abliterated_model_name=args.abliterated_model,
            magnitude=args.magnitude,
            threshold=args.threshold,
            num_trials=args.trials,
            token_pos=args.token_pos,
            concept_filter=args.concepts,
            output_dir=run_dir,
            plot_concepts=args.plot_concepts,
            plot_random_sample=args.plot_random_sample,
            selection_seed=args.selection_seed,
            run_selected_only=args.run_selected_only,
            plot_single_during_run=args.plot,
            abliterated_fine_tune=args.abliterated_fine_tune,
            abliterated_device_map=args.abliterated_device_map,
            abliterated_max_gpu_memory=args.abliterated_max_gpu_memory,
            abliterated_max_cpu_memory=args.abliterated_max_cpu_memory,
            abliterated_offload_folder=args.abliterated_offload_folder,
            unload_original=not args.no_unload_original,
        )

        all_results['pretrained'] = results

        if args.plot:
            print("  Generating aggregate comparison plot...")
            analyzer.plot_ablation_comparison(results, output_dir=run_dir)

        if args.save_json:
            import json as _json
            json_path = str(Path(run_dir) / f"ablation_results_l{args.sample_layer}.json")
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            serialisable = {
                'metadata': results['metadata'],
                'aggregate': results['aggregate'],
                'per_cell': {
                    f"{concept}__L{inj}": cell
                    for (concept, inj), cell in results['per_cell'].items()
                },
            }
            with open(json_path, 'w') as f:
                _json.dump(serialisable, f, indent=2)
            print(f"  Saved JSON: {json_path}")

        agg = results['aggregate']
        avg_intro_gain = float(np.mean(
            np.array(agg['intro_ablated_diffs']) - np.array(agg['intro_diffs'])
        ))
        avg_ctrl_shift = float(np.mean(np.abs(
            np.array(agg['control_ablated_diffs']) - np.array(agg['control_diffs'])
        )))
        print(f"  Avg intro gain (abliterated vs original): {avg_intro_gain:+.3f}")
        print(f"  Avg |control shift| (abliterated vs original): {avg_ctrl_shift:.3f}")

    # ----------------------------------------------------------------
    # Hook mode: sweep over ablation strengths (original behaviour)
    # ----------------------------------------------------------------
    else:
        for strength in args.ablation_strength:
            # Each strength gets its own subdirectory when sweeping multiple values
            if len(args.ablation_strength) > 1:
                run_dir = str(Path(args.output_dir) / f"abl_{strength:.2f}")
            else:
                run_dir = args.output_dir

            print(f"\n--- Ablation strength {strength} ---")
            results = analyzer.run_ablation_comparison_sweep(
                sample_layer_idx=args.sample_layer,
                vector_path=args.vector_file,
                injection_layers=args.injection_layers,
                refusal_direction_path=args.refusal_direction,
                per_layer_directions_path=args.refusal_directions,
                magnitude=args.magnitude,
                ablation_strength=strength,
                ablation_scope=args.ablation_scope,
                per_region_weights=resolved_region_weights,
                threshold=args.threshold,
                num_trials=args.trials,
                token_pos=args.token_pos,
                concept_filter=args.concepts,
                output_dir=run_dir,
                plot_concepts=args.plot_concepts,
                plot_random_sample=args.plot_random_sample,
                selection_seed=args.selection_seed,
                run_selected_only=args.run_selected_only,
                plot_single_during_run=args.plot,
            )

            all_results[strength] = results

            if args.plot:
                print("  Generating aggregate comparison plot...")
                analyzer.plot_ablation_comparison(results, output_dir=run_dir)

            if args.save_json:
                import json as _json
                json_path = str(Path(run_dir) / f"ablation_results_l{args.sample_layer}.json")
                Path(run_dir).mkdir(parents=True, exist_ok=True)
                serialisable = {
                    'metadata': results['metadata'],
                    'aggregate': results['aggregate'],
                    'per_cell': {
                        f"{concept}__L{inj}": cell
                        for (concept, inj), cell in results['per_cell'].items()
                    },
                }
                with open(json_path, 'w') as f:
                    _json.dump(serialisable, f, indent=2)
                print(f"  Saved JSON: {json_path}")

            # Brief summary per strength
            agg = results['aggregate']
            avg_intro_gain = float(np.mean(
                np.array(agg['intro_ablated_diffs']) - np.array(agg['intro_diffs'])
            ))
            avg_ctrl_shift = float(np.mean(np.abs(
                np.array(agg['control_ablated_diffs']) - np.array(agg['control_diffs'])
            )))
            print(f"  Avg intro gain from ablation : {avg_intro_gain:+.3f}")
            print(f"  Avg |control shift| from ablation: {avg_ctrl_shift:.3f}")

    print(f"\n{'='*70}")
    print("Complete!")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
