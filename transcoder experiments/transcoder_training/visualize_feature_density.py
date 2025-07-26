import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from utils import compute_feature_density

def plot_density_histogram(densities, feature_label, bins):
    """
    Plot a histogram of feature densities.
    Args:
        densities: np.ndarray of feature densities
        feature_label: str, label for the plot
        bins: int, number of bins
    """
    plt.figure(figsize=(8, 4))
    plt.hist(densities, bins=bins, color='royalblue', edgecolor='black')
    plt.xlabel('Feature Density')
    plt.ylabel('Number of Features')
    plt.title(f'Distribution of Feature Densities\n{feature_label}')
    plt.tight_layout()
    plt.show()

def plot_sparsity_histogram(activations, threshold, bins):
    """
    Plot a histogram of the number of active features per data point.
    Args:
        activations: np.ndarray, shape (num_data_points, num_features)
        threshold: float, activation threshold
        bins: int, number of bins
    """
    sparsity = (activations > threshold).sum(axis=1)
    plt.figure(figsize=(8, 4))
    plt.hist(sparsity, bins=bins, color='orange', edgecolor='black')
    plt.xlabel('Number of Active Features per Data Point')
    plt.ylabel('Count')
    plt.title('Sparsity Distribution')
    plt.tight_layout()
    plt.show()

def print_top_activations_and_tokens(experiment_results, feature_indices, topk):
    """
    Print the top-k activations and corresponding tokens for each feature.
    Args:
        experiment_results: str, path to directory with acts_and_tokens files
        feature_indices: list of int, feature indices to show
        topk: int, number of top activations to print
    """
    for feat in feature_indices:
        acts_file = os.path.join(experiment_results, f'feature{feat}_acts_and_tokens.txt')
        if not os.path.exists(acts_file):
            print(f"[Warning] File not found: {acts_file}")
            continue
        print(f"\nTop {topk} activations for feature {feat}:")
        activations = []
        tokens = []
        with open(acts_file, 'r') as f:
            lines = f.readlines()
        # Each activation/token pair is separated by '-----------\n'
        for line in lines:
            if line.startswith('tensor('):
                # Format: tensor(value)  token
                parts = line.strip().split(')')
                if len(parts) < 2:
                    continue
                act_str = parts[0].replace('tensor(', '')
                try:
                    act = float(act_str)
                except ValueError:
                    continue
                token = parts[1].strip()
                activations.append(act)
                tokens.append(token)
        # Show top-k
        for i in range(min(topk, len(activations))):
            print(f"{activations[i]:.4f}\t{tokens[i]}")

def main():
    """
    CLI for legacy feature density and sparsity visualization. For new analyses, use feature_analysis.py.
    """
    parser = argparse.ArgumentParser(description="Visualize feature density and sparsity, and show top activations/tokens. (Legacy: see feature_analysis.py for new visualizations)")
    parser.add_argument('--activations', type=str, required=True, help='Path to activations .npy file (shape: [num_data_points, num_features])')
    parser.add_argument('--threshold', type=float, default=0.0, help='Activation threshold for density/sparsity (default: 0.0)')
    parser.add_argument('--features', type=str, default=None, help='Comma-separated list of feature indices to visualize (default: all)')
    parser.add_argument('--bins', type=int, default=50, help='Number of bins for histogram (default: 50)')
    parser.add_argument('--experiment_results', type=str, default=None, help='Path to experiment_results folder for acts_and_tokens files')
    parser.add_argument('--topk', type=int, default=10, help='Number of top activations/tokens to show per feature (default: 10)')
    args = parser.parse_args()

    # Load activations
    activations = np.load(args.activations)
    if activations.ndim != 2:
        raise ValueError(f"Activations should be 2D, got shape {activations.shape}")

    # Feature selection
    if args.features is not None:
        feature_indices = [int(idx) for idx in args.features.split(',')]
        activations = activations[:, feature_indices]
        feature_label = f"Selected features ({len(feature_indices)})"
    else:
        feature_indices = list(range(activations.shape[1]))
        feature_label = f"All features ({activations.shape[1]})"

    # Compute densities
    densities = compute_feature_density(activations, threshold=args.threshold)
    plot_density_histogram(densities, feature_label, args.bins)
    plot_sparsity_histogram(activations, args.threshold, args.bins)
    if args.experiment_results is not None:
        print_top_activations_and_tokens(args.experiment_results, feature_indices, args.topk)

if __name__ == '__main__':
    main()