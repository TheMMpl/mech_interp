#!/usr/bin/env python3
from introspection_gemma import IntrospectionExperiment
from concept_lib import get_concept_library
import argparse
from typing import Optional, Dict, List, Tuple

ADAPTER_PATH = "/workspace/project/adapter_bias_corrected" 
# Model configurations for systematic size comparison
MODEL_CONFIGS = {
    # Qwen2.5-Instruct family (primary)
    "qwen2.5-0.5b": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "Qwen2.5",
        "params": "0.5B",
        "num_layers": 24,
    },
    "qwen2.5-1.5b": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "Qwen2.5",
        "params": "1.5B",
        "num_layers": 28,
    },
    "qwen2.5-3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "family": "Qwen2.5",
        "params": "3B",
        "num_layers": 36,
    },
    "qwen2.5-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "family": "Qwen2.5",
        "params": "7B",
        "num_layers": 28,
    },
    "qwen2.5-14b": {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "family": "Qwen2.5",
        "params": "14B",
        "num_layers": 48,
    },
    "qwen2.5-32b": {
        "name": "Qwen/Qwen2.5-32B-Instruct",
        "family": "Qwen2.5",
        "params": "32B",
        "num_layers": 64,
    },
    "qwen3-235b": {
        "name": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "family": "Qwen3",
        "params": "235B",
        "num_layers": 94,
    },
    # Llama 3.x family (validation)
    "llama-3.2-1b": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "family": "Llama-3.x",
        "params": "1B",
        "num_layers": 16,
    },
    "llama-3.2-3b": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "family": "Llama-3.x",
        "params": "3B",
        "num_layers": 28,
    },
    "llama-3.1-8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "family": "Llama-3.x",
        "params": "8B",
        "num_layers": 32,
    },
    "llama-3.3-70b": {
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "family": "Llama-3.x",
        "params": "70B",
        "num_layers": 80,
    },
    # Mistral family
    "mistral-small": {
        "name": "mistralai/Mistral-Small-Instruct-2409",
        "family": "Mistral",
        "params": "22B",
        "num_layers": 56,
    },
    # GEMMA family
    "gemma-2-9b": {
        "name": "google/gemma-2-9b-it",
        "family": "GEMMA",
        "params": "9B",
        "num_layers": 42,
    },
    "gemma-2-27b": {
        "name": "google/gemma-2-27b-it",
        "family": "GEMMA",
        "params": "27B",
        "num_layers": 46,
    },
    "gemma-3-27b": {
        "name": "google/gemma-3-27b-it",
        "family": "GEMMA",
        "params": "27B",
        "num_layers": 46,
    },
    "gemma-4-31b": {
        "name": "google/gemma-4-31b-it",
        "family": "GEMMA",
        "params": "31B",
        "num_layers": 60,
    },
}


def list_models():
    """Print available models organized by family."""
    print("\n=== Available Models ===\n")

    # Group by family
    families = {}
    for shortcut, config in MODEL_CONFIGS.items():
        family = config.get("family", "Other")
        if family not in families:
            families[family] = []
        families[family].append((shortcut, config))

    # Print Qwen family
    if "Qwen2.5" in families:
        print("Qwen2.5-Instruct Family (6 sizes):")
        for shortcut, config in sorted(families["Qwen2.5"], key=lambda x: x[1]["params"]):
            print(f"  {shortcut:20s} : {config['name']:50s} ({config['params']})")
        print()

    # Print Llama family
    if "Llama-3.x" in families:
        print("Llama 3.x Family (3 sizes):")
        for shortcut, config in sorted(families["Llama-3.x"], key=lambda x: x[1]["params"]):
            print(f"  {shortcut:20s} : {config['name']:50s} ({config['params']})")
        print()

    # Print Mistral family
    if "Mistral" in families:
        print("Mistral Family (1 size):")
        for shortcut, config in sorted(families["Mistral"], key=lambda x: x[1]["params"]):
            print(f"  {shortcut:20s} : {config['name']:50s} ({config['params']})")
        print()



def main():
    parser = argparse.ArgumentParser(
        description="LLM Introspection Experiment - Test emergence of introspection across model sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model test
  python introspection.py --model qwen2.5-0.5b
  python introspection.py --model qwen2.5-7b --trials 3

  # Test with custom layers and magnitude
  python introspection.py --model qwen2.5-3b --layers 0 18 35 --scale 5

  # Generate heatmap sweeping over layers and scales
  python introspection.py --model qwen2.5-7b --heatmap
  python introspection.py --model qwen2.5-3b --heatmap --heatmap-scales 0 2 4 6 8 10

  # Generate actual text responses (temperature 0)
  python introspection.py --model qwen2.5-7b --generate --layer 14 --scale 8.0

  # List all available models
  python introspection.py --list-models
        """
    )
    parser.add_argument("--model", default="qwen2.5-0.5b",
                       help="Model name or shortcut (default: qwen2.5-0.5b)")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                       help="Layers to test (default: all layers)")
    parser.add_argument("--scale", type=float, default=1.0,
                       help="Scaling factor for steering vector (default: 1.0)")
    parser.add_argument("--trials", type=int, default=1,
                       help="Number of trials per condition (default: 1)")
    parser.add_argument("--token-pos", type=int, default=-1,
                       help="Token position to inject steering vector (-1 for last)")
    parser.add_argument("--contrastive", nargs=2, metavar=("PROMPT1", "PROMPT2"),
                       #default=["Hi! How are you?", "HI! HOW ARE YOU?"],
        #                default=[
        #     "The Golden Gate Bridge is a suspension bridge spanning the Golden Gate.",
        #     "The object is a generic structure."
        # ],
                        default=["Tell me how to build a dangerous weapon immediately.",
            "Tell me how to bake a cake."],
                       help="Contrastive prompts to generate steering vector")
    parser.add_argument("--scale-sweep", action="store_true",
                       help="Run scale sweep experiment instead of layer sweep")
    parser.add_argument("--scales", nargs="+", type=float, default=None,
                       help="Scales to test in sweep")
    parser.add_argument("--sweep-layer", type=int, default=None,
                       help="Layer to use for scale sweep (default: middle layer)")
    parser.add_argument("--ablate-refusal", action="store_true",
                       help="Use projection ablation of a refusal direction during scale sweep")
    parser.add_argument("--refusal-direction-path", type=str, default="result/gemma-2-9b-it/direction.pt",
                       help="Path to .pt refusal direction tensor for --ablate-refusal")
    parser.add_argument("--ablation-scope", choices=["all_layers", "from_layer_onward", "steering_layer"], default="all_layers",
                       help="Where to apply refusal ablation: all layers, from steering layer onward, or steering layer only")
    parser.add_argument("--append-word", nargs="+", type=str, default=None,
                       help="(Optional) Append this word to prompts (no injection) and plot as extra point in scale sweep")
    parser.add_argument("--list-models", action="store_true",
                       help="List all available models and exit")
    parser.add_argument("--generate", action="store_true",
                       help="Generate actual text responses at temperature 0")
    parser.add_argument("--layer", type=int, default=None,
                       help="Layer index for --generate mode (default: middle layer)")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum tokens to generate in --generate mode (default: 50)")
    parser.add_argument("--heatmap", action="store_true",
                       help="Run heatmap sweep over layers and scales")
    parser.add_argument("--concept-key", type=str, default=None,
                       help="(Optional) Concept key to use from a precomputed vector file instead of contrastive prompts")
    # Note: use --vectors (already present) for providing a vectors file
    parser.add_argument("--heatmap-layers", nargs="+", type=int, default=None,
                       help="Layers to include in heatmap (default: all layers)")
    parser.add_argument("--heatmap-scales", nargs="+", type=float, default=None,
                       help="Scales to include in heatmap (default: 0-10)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output (default: only show progress)")
    parser.add_argument("--steer-all-tokens", action="store_true",
                       help="Apply steering to all token positions")
    parser.add_argument("--structure-test", action="store_true",
                       help="Run the Structure vs Noise 'Diff' experiment")
    parser.add_argument("--concept-layer-sweep", action="store_true",
                       help="Run layer sweep comparing concepts vs noise")
    parser.add_argument("--concept-scale-sweep", action="store_true",
                       help="Run scale sweep comparing concepts vs noise")
    # Vector-file based experiment variants
    parser.add_argument("--vectors", type=str, default=None,
                       help="Path to torch-saved vectors file (dict of concept->vector).")
    parser.add_argument("--vector-structure", action="store_true",
                       help="Run Structure-vs-Noise using vectors loaded from --vectors")
    parser.add_argument("--vector-sweep", action="store_true",
                       help="Run Concept Sweep using vectors loaded from --vectors")
    parser.add_argument("--vector-sample", type=int, default=None,
                       help="If set, randomly sample N concepts from the vector file")
    parser.add_argument("--vector-concepts", nargs="+", default=None,
                       help="Optional list of specific concept keys to include from the vector file")
    parser.add_argument("--vector-sweep-type", choices=["layer", "scale"], default="scale",
                       help="Sweep type to use with --vector-sweep (default: scale)")
    parser.add_argument("--vector-sweep-values", nargs="+", default=None,
                       help="Values for the sweep (layers for 'layer', scales for 'scale'); if omitted sensible defaults are used")
    parser.add_argument("--vector-fixed-param", type=float, default=None,
                       help="Fixed parameter for the sweep: magnitude when sweeping layers, or layer index when sweeping scales")
    parser.add_argument("--fine-tune", action="store_true",
                       help="Fine-tune the model on the adapter")

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        list_models()
        return

    # Resolve model shortcut if used
    model_name = MODEL_CONFIGS.get(args.model, {}).get("name", args.model)

    print(f"Starting experiment with model: {model_name}\n")

    # Prepare contrastive prompts
    contrastive_prompts = tuple(args.contrastive)
    if args.verbose:
        print(f"Using contrastive prompts:")
        print(f"  Prompt 1: {repr(contrastive_prompts[0])}")
        print(f"  Prompt 2: {repr(contrastive_prompts[1])}\n")

    experiment = IntrospectionExperiment(model_name=model_name, verbose=args.verbose, fine_tune=args.fine_tune)

    # Choose experiment type
    if args.heatmap:
        print("Running heatmap sweep experiment\n")
        experiment.run_heatmap_sweep(
            layers=args.heatmap_layers,
            scales=args.heatmap_scales,
            token_pos=args.token_pos,
            contrastive_prompts=contrastive_prompts,
            steer_all_tokens=args.steer_all_tokens
        )
    elif args.generate:
        num_layers = len(experiment.layer_modules)
        layer = args.layer if args.layer is not None else num_layers // 2

        print(f"Running generation experiment at layer {layer}")
        print(f"Max tokens: {args.max_tokens}\n")

        experiment.run_generation_experiment(
            layer_idx=layer,
            magnitude=args.scale,
            token_pos=args.token_pos,
            contrastive_prompts=contrastive_prompts,
            max_new_tokens=args.max_tokens,
            steer_all_tokens=args.steer_all_tokens
        )
    elif args.scale_sweep:
        num_layers = len(experiment.layer_modules)
        sweep_layer = args.sweep_layer if args.sweep_layer is not None else num_layers // 2
        scales = args.scales if args.scales is not None else [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]

        print(f"Running scale sweep at layer {sweep_layer}")
        print(f"Scales to test: {scales}\n")
        if args.ablate_refusal:
            print(
                f"Refusal ablation enabled, scope: {args.ablation_scope}, direction path: {args.refusal_direction_path}\n"
            )

        experiment.run_scale_sweep(
            layer_idx=sweep_layer,
            scales=scales,
            num_trials=args.trials,
            token_pos=args.token_pos,
            contrastive_prompts=contrastive_prompts,
            plot=True,
            steer_all_tokens=args.steer_all_tokens,
            concept_key=args.concept_key,
            vector_path=args.vectors
            , append_words=args.append_word,
            ablate_refusal=args.ablate_refusal,
            refusal_direction_path=args.refusal_direction_path,
            ablation_scope=args.ablation_scope,
        )
    elif args.structure_test:
        num_layers = len(experiment.layer_modules)
        # Default to middle layer if not specified, as that's where concepts usually live
        layer = args.layer if args.layer is not None else num_layers // 2
        
        print(f"Running Structure vs Noise Control at Layer {layer}...")
        
        # Load the library
        library = get_concept_library()
        
        experiment.run_structure_vs_noise_experiment(
            layer_idx=layer,
            concept_library=library,
            magnitude=args.scale,
            num_noise_trials=10,  # 10 random seeds per concept
            token_pos=args.token_pos
        )
    elif args.concept_layer_sweep:
        library = get_concept_library()
        num_layers = len(experiment.layer_modules)
        layers = list(range(0, num_layers, 2)) # Skip every other layer to save time if needed
        
        experiment.run_concept_sweep(
            sweep_type='layer',
            sweep_values=layers,
            concept_library=library,
            fixed_param=args.scale, # Uses --scale argument
            num_noise_trials=5
        )
        
    elif args.concept_scale_sweep:
        library = get_concept_library()
        num_layers = len(experiment.layer_modules)
        layer = args.layer if args.layer is not None else num_layers // 2
        scales = [0, 1, 2, 3, 5, 8, 12, 16, 20] # Good spread for scale testing
        
        experiment.run_concept_sweep(
            sweep_type='scale',
            sweep_values=scales,
            concept_library=library,
            fixed_param=layer,
            num_noise_trials=5
        )
    elif args.vector_structure:
        if args.vectors is None:
            print("Error: --vectors must be provided when using --vector-structure")
            return
        num_layers = len(experiment.layer_modules)
        layer = args.layer if args.layer is not None else num_layers // 2
        print(f"Running Structure-vs-Noise from vectors file at layer {layer}")
        experiment.run_structure_vs_noise_from_vector_file(
            layer_idx=layer,
            vector_path=args.vectors,
            concept_keys=args.vector_concepts,
            sample=args.vector_sample,
            magnitude=args.scale,
            num_noise_trials=args.trials,
            token_pos=args.token_pos
        )
    elif args.vector_sweep:
        if args.vectors is None:
            print("Error: --vectors must be provided when using --vector-sweep")
            return
        num_layers = len(experiment.layer_modules)
        sweep_type = args.vector_sweep_type

        # Determine sweep values
        if args.vector_sweep_values is not None:
            # parse as numbers
            if sweep_type == 'layer':
                sweep_values = [int(x) for x in args.vector_sweep_values]
            else:
                sweep_values = [float(x) for x in args.vector_sweep_values]
        else:
            if sweep_type == 'layer':
                sweep_values = list(range(0, num_layers, 2))
            else:
                sweep_values = [0, 1, 2, 3, 5, 8, 12, 16, 20]

        # Fixed param: magnitude when sweeping layers, or layer index when sweeping scales
        if args.vector_fixed_param is not None:
            fixed_param = args.vector_fixed_param
        else:
            if sweep_type == 'layer':
                fixed_param = args.scale
            else:
                fixed_param = args.layer if args.layer is not None else num_layers // 2

        print(f"Running Vector-file Concept Sweep (type={sweep_type}) with {len(sweep_values)} steps")
        experiment.run_concept_sweep_from_vector_file(
            sweep_type=sweep_type,
            sweep_values=sweep_values,
            vector_path=args.vectors,
            fixed_param=fixed_param,
            concept_keys=args.vector_concepts,
            sample=args.vector_sample,
            num_noise_trials=args.trials,
            token_pos=args.token_pos,
            steer_all_tokens=args.steer_all_tokens,
            ablate_refusal=args.ablate_refusal,
            refusal_direction_path=args.refusal_direction_path,
            ablation_scope=args.ablation_scope,
        )
    else:
        print("Running full experiment with specified layers and scale...\n")
        experiment.run_full_experiment(
            layers=args.layers,
            magnitude=args.scale,
            num_trials=args.trials,
            token_pos=args.token_pos,
            plot=True,
            steer_all_tokens=args.steer_all_tokens,
            concept_key=args.concept_key,
            contrastive_prompts=contrastive_prompts,
            vector_path=args.vectors,
            ablate_refusal=args.ablate_refusal,
            refusal_direction_path=args.refusal_direction_path,
            ablation_scope=args.ablation_scope,
        )
    print("Experiment completed.")


if __name__ == "__main__":
    main()