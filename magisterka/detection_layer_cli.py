#!/usr/bin/env python3
"""
Entry point for Detection Layer Harness

Example usage:
python detection_layer_cli.py \
    --model "google/gemma-4-31b-it" \
    --sample-layers 15 20 25 30 35 40 45 50 55\
    --sample-layer-files adapter_full/vectors_15.pt adapter_full/vectors_20.pt adapter_full/vectors_25.pt adapter_full/vectors_30.pt adapter_full/vectors_35.pt adapter_full/vectors_40.pt adapter_full/vectors_45.pt adapter_full/vectors_50.pt adapter_full/vectors_55.pt \
    --injection-layers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 \
    --magnitude 3.0 \
    --threshold 0.0 \
    --trials 1 \
    --plot-concepts sandwich bomb love logic\
    --output-dir ./detection_results \
    --plot \
    --plot-random-sample 5 \
    --selection-seed 7 \
    --run-selected-only \
    --no-fine-tune 



python detection_layer_cli.py \
    --model "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" \
    --sample-layers 10 15 20 25 30 35 40 45 50 55 60 65 70\
    --sample-layer-files qwen/vectors_10.pt qwen/vectors_15.pt qwen/vectors_20.pt qwen/vectors_25.pt qwen/vectors_30.pt qwen/vectors_35.pt qwen/vectors_40.pt qwen/vectors_45.pt qwen/vectors_50.pt qwen/vectors_55.pt qwen/vectors_60.pt qwen/vectors_65.pt qwen/vectors_70.pt \
    --injection-layers 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
    --magnitude 3.0 \
    --threshold 0.0 \
    --trials 1 \
    --plot-concepts sandwich bomb love logic\
    --output-dir ./qwen_detection \
    --plot \
    --plot-random-sample 5 \
    --selection-seed 7 \
    --run-selected-only \
    --no-fine-tune \
    --model-device-map auto --max-gpu-memory 90GiB --max-cpu-memory 160GiB --offload-folder /workspace/offload_qwen235
import sys
from detection_layer_harness import DetectionAnalyzer
"""

def main():
    parser = argparse.ArgumentParser(
        description="Detection Layer Harness: Compare where detection happens vs sample layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 3 sample layers
  python detection_layer_cli.py \\
    --model "Qwen/Qwen2.5-0.5B-Instruct" \\
    --sample-layers 10 20 30 \\
    --sample-layer-files vectors_10.pt vectors_20.pt vectors_30.pt \\
    --injection-layers 0 5 10 15 20 25 \\
    --magnitude 2.0 \\
    --output-dir ./detection_results \\
    --plot

  # With custom threshold and multiple trials
  python detection_layer_cli.py \\
    --model "Qwen/Qwen2.5-1.5B-Instruct" \\
    --sample-layers 5 15 25 \\
    --sample-layer-files vec_5.pt vec_15.pt vec_25.pt \\
    --injection-layers 0 10 20 \\
    --magnitude 1.5 \\
    --threshold 0.5 \\
    --trials 3 \\
    --output-dir ./results \\
    --plot \\
    --verbose
        """
    )
    
    # Required args
    parser.add_argument("--model", required=True,
                       help="HuggingFace model name")
    parser.add_argument("--sample-layers", type=int, nargs="+", required=True,
                       help="Sample layer indices")
    parser.add_argument("--sample-layer-files", nargs="+", required=True,
                       help="Vector file paths for each sample layer (same order as --sample-layers)")
    parser.add_argument("--injection-layers", type=int, nargs="+", required=True,
                       help="Injection layer indices to test")
    
    # Optional args
    parser.add_argument("--magnitude", type=float, default=1.0,
                       help="Scaling factor for vectors (default: 1.0)")
    parser.add_argument("--threshold", type=float, default=0.0,
                       help="Threshold for detection crossing (default: 0.0)")
    parser.add_argument("--trials", type=int, default=1,
                       help="Number of trials per cell (default: 1)")
    parser.add_argument("--token-pos", type=int, default=-1,
                       help="Token position for injection (default: -1 for last)")
    parser.add_argument("--concepts", nargs="+", default=None,
                       help="Specific concepts to test (default: all aligned)")
    parser.add_argument("--plot-concepts", nargs="+", default=None,
                       help="Explicit concepts for per-concept single plots")
    parser.add_argument("--plot-random-sample", type=int, default=0,
                       help="Add N random concepts to the plot set")
    parser.add_argument("--selection-seed", type=int, default=42,
                       help="Random seed used for concept sampling")
    parser.add_argument("--run-selected-only", action="store_true",
                       help="Run full experiment only on selected concepts (explicit + random)")
    parser.add_argument("--output-dir", default="./detection_results",
                       help="Output directory (default: ./detection_results)")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--no-fine-tune", action="store_true",
                       help="Don't use fine-tuned adapter")
    parser.add_argument("--keep-trial-history", action="store_true",
                       help="Keep per-trial arrays in output (higher memory usage)")
    parser.add_argument("--model-device-map", default="auto",
                       help="Transformers device_map (default: auto, use 'none' for single-device)")
    parser.add_argument("--max-gpu-memory", default=None,
                       help="Per-GPU memory budget, e.g. '40GiB'")
    parser.add_argument("--max-cpu-memory", default=None,
                       help="CPU RAM budget, e.g. '220GiB' for model offload")
    parser.add_argument("--offload-folder", default=None,
                       help="Folder for CPU/disk offload tensors when using device_map")

    args = parser.parse_args()

    # Validate sample layers and files match
    if len(args.sample_layers) != len(args.sample_layer_files):
        print(f"Error: {len(args.sample_layers)} sample layers but {len(args.sample_layer_files)} files")
        sys.exit(1)

    # Build mapping
    sample_layer_files = dict(zip(args.sample_layers, args.sample_layer_files))

    print(f"{'='*70}")
    print(f"Detection Layer Harness")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Sample layers: {args.sample_layers}")
    print(f"Injection layers: {args.injection_layers}")
    print(f"Magnitude: {args.magnitude}, Threshold: {args.threshold}, Trials: {args.trials}")
    if args.plot_concepts:
        print(f"Explicit plot concepts: {len(args.plot_concepts)}")
    if args.plot_random_sample > 0:
        print(f"Random plot sample: {args.plot_random_sample} (seed={args.selection_seed})")
    if args.run_selected_only:
        print("Run mode: selected concepts only")
    print(f"Device map: {args.model_device_map}")
    if args.max_gpu_memory:
        print(f"Max GPU memory: {args.max_gpu_memory}")
    if args.max_cpu_memory:
        print(f"Max CPU memory: {args.max_cpu_memory}")
    if args.offload_folder:
        print(f"Offload folder: {args.offload_folder}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}\n")

    # Create analyzer
    analyzer = DetectionAnalyzer(
        model_name=args.model,
        verbose=args.verbose,
        fine_tune=not args.no_fine_tune,
        model_device_map=args.model_device_map,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=args.offload_folder,
    )

    # Run sweep
    print("Starting detection sweep...")
    results = analyzer.run_detection_sweep(
        sample_layer_files=sample_layer_files,
        injection_layers=args.injection_layers,
        magnitude=args.magnitude,
        threshold=args.threshold,
        num_trials=args.trials,
        token_pos=args.token_pos,
        concept_filter=args.concepts,
        output_dir=args.output_dir,
        plot_concepts=args.plot_concepts,
        plot_random_sample=args.plot_random_sample,
        selection_seed=args.selection_seed,
        run_selected_only=args.run_selected_only,
        plot_single_during_run=args.plot,
        keep_trial_history=args.keep_trial_history,
    )

    # Save results
    print("\nSaving results...")
    json_path, csv_path, peaks_csv_path = analyzer.save_results(results, output_dir=args.output_dir)

    # Generate plots
    if args.plot:
        print("\nGenerating plots...")
        
        # Per-sample-layer curves
        for sample_layer in args.sample_layers:
            analyzer.plot_per_sample_layer(results, sample_layer, output_dir=args.output_dir)
        
        # Heatmap
        analyzer.plot_heatmap(results, output_dir=args.output_dir)
        
        # Summary
        analyzer.plot_detection_summary(results, output_dir=args.output_dir)

    print(f"\n{'='*70}")
    print("Complete!")
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  Peaks CSV: {peaks_csv_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
