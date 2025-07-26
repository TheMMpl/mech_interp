import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import umap
from scipy import sparse
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from utils import compute_feature_density

sns.set(context="paper", style="white")

# Custom colormap for log-likelihood ratio (blue to red)
llr_colors = ['#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#eff3ff', 
              '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26']
llr_cmap = LinearSegmentedColormap.from_list('llr', llr_colors, N=256)


def compute_log_likelihood_ratio_fast(activations, feature_idx, threshold=0.1, num_bins=100):
    """
    Compute log-likelihood ratio log(P(s|feature)/P(s)) efficiently using histograms.
    
    This follows the Anthropic paper formulation where:
    - P(s|feature) is the probability of sequence s given the feature is active
    - P(s) is the overall probability of sequence s
    
    Args:
        activations: np.ndarray, shape (num_samples, num_features)
        feature_idx: int, index of the feature to analyze
        threshold: float, threshold for considering a feature "active"
        num_bins: int, number of bins for histogram-based density estimation
    
    Returns:
        np.ndarray: log-likelihood ratios for each sample
    """
    if sparse.issparse(activations):
        feature_acts = activations[:, feature_idx].toarray().ravel()
    else:
        feature_acts = activations[:, feature_idx]
    
    # Define active samples (where feature fires above threshold)
    active_mask = feature_acts > threshold
    n_active = np.sum(active_mask)
    n_total = len(feature_acts)
    
    if n_active == 0 or n_active == n_total:
        return np.zeros_like(feature_acts)
    
    # Create bins for density estimation
    min_val, max_val = np.min(feature_acts), np.max(feature_acts)
    if min_val == max_val:
        return np.zeros_like(feature_acts)
    
    bins = np.linspace(min_val, max_val, num_bins + 1)
    bin_width = bins[1] - bins[0]
    
    # Compute P(s|feature) - probability distribution when feature is active
    active_acts = feature_acts[active_mask]
    hist_active, _ = np.histogram(active_acts, bins=bins, density=True)
    
    # Compute P(s) - overall probability distribution
    hist_all, _ = np.histogram(feature_acts, bins=bins, density=True)
    
    # Avoid division by zero
    hist_all = np.maximum(hist_all, 1e-10)
    hist_active = np.maximum(hist_active, 1e-10)
    
    # Compute log-likelihood ratio for each bin
    llr_per_bin = np.log(hist_active / hist_all)
    
    # Assign LLR to each sample based on which bin it falls into
    bin_indices = np.digitize(feature_acts, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(llr_per_bin) - 1)
    
    llr = llr_per_bin[bin_indices]
    
    return llr


def plot_feature_activation_distribution_anthropic(activations, feature_idx, title=None, 
                                                 threshold=0.1, bins=100, figsize=(12, 8)):
    """
    Plot feature activation distribution in Anthropic paper style with log-likelihood ratio coloring.
    
    Following the paper's formulation: log(P(s|feature)/P(s))
    
    Args:
        activations: np.ndarray or sparse matrix, shape (num_samples, num_features)
        feature_idx: int, feature index to visualize
        title: str, plot title
        threshold: float, activation threshold
        bins: int, number of histogram bins
        figsize: tuple, figure size
    """
    if sparse.issparse(activations):
        total_acts = activations[:, feature_idx].toarray().ravel()
    else:
        total_acts = activations[:, feature_idx]
    
    feature_acts=total_acts[total_acts>threshold/10]
    # Compute log-likelihood ratios using fast method
    llr = compute_log_likelihood_ratio_fast(activations, feature_idx, threshold, num_bins=bins)[total_acts>threshold/10]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Main histogram with LLR coloring
    hist, bin_edges = np.histogram(feature_acts, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Compute average LLR for each bin
    bin_llrs = np.zeros(len(bin_centers))
    for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (feature_acts >= left) & (feature_acts < right)
        if np.sum(mask) > 0:
            bin_llrs[i] = np.mean(llr[mask])
    
    # Normalize LLRs for colormap (center around 0)
    if np.ptp(bin_llrs) > 0:
        max_abs_llr = np.max(np.abs(bin_llrs))
        normalized_llrs = (bin_llrs + max_abs_llr) / (2 * max_abs_llr)
    else:
        normalized_llrs = np.full_like(bin_llrs, 0.5)
    
    # Plot colored histogram
    for i, (center, height, norm_llr) in enumerate(zip(bin_centers, hist, normalized_llrs)):
        if height > 0:  # Only plot non-empty bins
            color = llr_cmap(norm_llr)
            ax1.bar(center, height, width=bin_width, color=color, alpha=0.8, edgecolor='none')
    
    # Add threshold line
    ax1.axvline(threshold, color='black', linestyle='--', alpha=0.7, linewidth=2, 
                label=f'Active threshold ({threshold})')
    
    ax1.set_xlabel('Feature Activation Value')
    ax1.set_ylabel('Number of Sequences')
    ax1.set_title(title or f'Feature {feature_idx} Activation Distribution\n(Colored by log(P(sequence|feature)/P(sequence)))')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text box with statistics
    active_count = np.sum(feature_acts > threshold)
    total_count = len(total_acts)
    ax1.text(0.02, 0.98, f'Active sequences: {active_count}/{total_count} ({100*active_count/total_count:.1f}%)', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Scatter plot showing individual activations colored by LLR
    # Subsample for performance
    subsample_size = min(5000, len(feature_acts))
    if len(feature_acts) > subsample_size:
        indices = np.random.choice(len(feature_acts), subsample_size, replace=False)
        scatter_x = feature_acts[indices]
        scatter_llr = llr[indices]
    else:
        scatter_x = feature_acts
        scatter_llr = llr
    
    scatter_y = np.random.normal(0, 0.1, len(scatter_x))  # Add jitter
    
    im = ax2.scatter(scatter_x, scatter_y, c=scatter_llr, cmap=llr_cmap, alpha=0.6, s=1)
    ax2.set_xlabel('Feature Activation Value')
    ax2.set_ylabel('Random\nJitter')
    ax2.set_ylim(-0.5, 0.5)
    ax2.axvline(threshold, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar with proper label
    cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.15)
    cbar.set_label('log(P(sequence|feature) / P(sequence))')
    
    plt.tight_layout()
    plt.show()

def plot_feature_activation_expected_anthropic(activations, feature_idx, title=None, 
                                                 threshold=0.1, bins=100, figsize=(12, 8)):
    """
    Plot feature activation distribution in Anthropic paper style with log-likelihood ratio coloring.
    
    Following the paper's formulation: log(P(s|feature)/P(s))
    
    Args:
        activations: np.ndarray or sparse matrix, shape (num_samples, num_features)
        feature_idx: int, feature index to visualize
        title: str, plot title
        threshold: float, activation threshold
        bins: int, number of histogram bins
        figsize: tuple, figure size
    """
    if sparse.issparse(activations):
        total_acts = activations[:, feature_idx].toarray().ravel()
    else:
        total_acts = activations[:, feature_idx]
    
    feature_acts=total_acts[total_acts>threshold/10]
    # Compute log-likelihood ratios using fast method
    llr = compute_log_likelihood_ratio_fast(activations, feature_idx, threshold, num_bins=bins)[total_acts>threshold/10]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Main histogram with LLR coloring
    hist, bin_edges = np.histogram(feature_acts, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Compute average LLR for each bin
    bin_llrs = np.zeros(len(bin_centers))
    for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (feature_acts >= left) & (feature_acts < right)
        if np.sum(mask) > 0:
            bin_llrs[i] = np.mean(llr[mask])
    
    # Normalize LLRs for colormap (center around 0)
    if np.ptp(bin_llrs) > 0:
        max_abs_llr = np.max(np.abs(bin_llrs))
        normalized_llrs = (bin_llrs + max_abs_llr) / (2 * max_abs_llr)
    else:
        normalized_llrs = np.full_like(bin_llrs, 0.5)
    
    # Plot colored histogram
    for i, (center, height, norm_llr) in enumerate(zip(bin_centers, hist*bin_centers, normalized_llrs)):
        if height > 0:  # Only plot non-empty bins
            color = llr_cmap(norm_llr)
            ax1.bar(center, height, width=bin_width, color=color, alpha=0.8, edgecolor='none')
    
    # Add threshold line
    ax1.axvline(threshold, color='black', linestyle='--', alpha=0.7, linewidth=2, 
                label=f'Active threshold ({threshold})')
    
    ax1.set_xlabel('Feature Activation Value')
    ax1.set_ylabel('sequences*activation (expected)')
    ax1.set_title(title or f'Feature {feature_idx} Activation Distribution\n(Colored by log(P(sequence|feature)/P(sequence)))')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text box with statistics
    active_count = np.sum(feature_acts > threshold)
    total_count = len(total_acts)
    ax1.text(0.02, 0.98, f'Active sequences: {active_count}/{total_count} ({100*active_count/total_count:.1f}%)', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Scatter plot showing individual activations colored by LLR
    # Subsample for performance
    subsample_size = min(5000, len(feature_acts))
    if len(feature_acts) > subsample_size:
        indices = np.random.choice(len(feature_acts), subsample_size, replace=False)
        scatter_x = feature_acts[indices]
        scatter_llr = llr[indices]
    else:
        scatter_x = feature_acts
        scatter_llr = llr
    
    scatter_y = np.random.normal(0, 0.1, len(scatter_x))  # Add jitter
    
    im = ax2.scatter(scatter_x, scatter_y, c=scatter_llr, cmap=llr_cmap, alpha=0.6, s=1)
    ax2.set_xlabel('Feature Activation Value')
    ax2.set_ylabel('Random\nJitter')
    ax2.set_ylim(-0.5, 0.5)
    ax2.axvline(threshold, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar with proper label
    cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.15)
    cbar.set_label('log(P(sequence|feature) / P(sequence))')
    
    plt.tight_layout()
    plt.show()



def plot_activation_expected_value_distribution(activations, bins=50, figsize=(10, 6)):
    """
    Plot distribution of expected activation values across features.
    
    Args:
        activations: np.ndarray or sparse matrix, shape (num_samples, num_features)
        bins: int, number of histogram bins
        figsize: tuple, figure size
    """
    if sparse.issparse(activations):
        # For sparse matrices, compute mean of all elements (including zeros) per feature
        expected_values = np.array(activations.mean(axis=0)).ravel()
    else:
        expected_values = np.mean(activations, axis=0)
    
    plt.figure(figsize=figsize)
    
    # Main histogram
    n, bins_edges, patches = plt.hist(expected_values, bins=bins, alpha=0.7, 
                                     color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Color bars based on their position (gradient effect)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
    if np.ptp(bin_centers) > 0:
        normalized_positions = (bin_centers - np.min(bin_centers)) / np.ptp(bin_centers)
        for patch, norm_pos in zip(patches, normalized_positions):
            patch.set_facecolor(plt.cm.viridis(norm_pos))
    
    # Add statistics
    mean_val = np.mean(expected_values)
    median_val = np.median(expected_values)
    std_val = np.std(expected_values)
    
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    
    plt.xlabel('Expected Feature Activation')
    plt.ylabel('Number of Features')
    plt.title(f'Distribution of Feature Expected Activation Values\n(σ = {std_val:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return expected_values


def plot_logit_weights_distribution(weights, feature_indices=None, bins=50, figsize=(12, 8)):
    """
    Plot distribution of logit weights for selected features.
    
    Args:
        weights: np.ndarray, shape (num_features, vocab_size) - decoder weights
        feature_indices: list of int, features to analyze (if None, use all)
        bins: int, number of histogram bins
        figsize: tuple, figure size
    """
    if feature_indices is None:
        feature_indices = list(range(min(100, weights.shape[0])))  # Limit to first 100 features
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    # 1. Distribution of all weights
    all_weights = weights[feature_indices].flatten()
    axes[0].hist(all_weights, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Weight Value')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of All Logit Weights')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Distribution of weight magnitudes
    weight_magnitudes = np.abs(all_weights)
    axes[1].hist(weight_magnitudes, bins=bins, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('|Weight Value|')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Weight Magnitudes')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Max weight per feature
    max_weights = np.max(np.abs(weights[feature_indices]), axis=1)
    axes[2].hist(max_weights, bins=bins//2, alpha=0.7, color='mediumseagreen', edgecolor='black')
    axes[2].set_xlabel('Max |Weight| per Feature')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Distribution of Maximum Weights per Feature')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Weight sparsity (fraction of near-zero weights)
    sparsity_threshold = 1e-4
    sparsity_per_feature = np.mean(np.abs(weights[feature_indices]) < sparsity_threshold, axis=1)
    axes[3].hist(sparsity_per_feature, bins=bins//2, alpha=0.7, color='gold', edgecolor='black')
    axes[3].set_xlabel('Fraction of Near-Zero Weights')
    axes[3].set_ylabel('Count')
    axes[3].set_title(f'Weight Sparsity per Feature\n(threshold = {sparsity_threshold})')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_feature_comparison_anthropic(activations, feature_indices, figsize=(15, 10)):
    """
    Create a comparison plot of multiple features in Anthropic style.

    Args:
        activations: np.ndarray or sparse matrix
        feature_indices: list of int, features to compare
        figsize: tuple, figure size
    """
    n_features = len(feature_indices)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    elif n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, feat_idx in enumerate(feature_indices):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        if sparse.issparse(activations):
            feature_acts = activations[:, feat_idx].toarray().ravel()
        else:
            feature_acts = activations[:, feat_idx]

        # Filter out zero values for log scale
        nonzero_acts = feature_acts[feature_acts > 0]
        if len(nonzero_acts) > 0:
            n, bins, patches = ax.hist(nonzero_acts, bins=50, alpha=0.7,
                                       color='steelblue', edgecolor='black', linewidth=0.5)

            bin_centers = (bins[:-1] + bins[1:]) / 2
            if np.ptp(bin_centers) > 0:
                normalized_centers = (bin_centers - np.min(bin_centers)) / np.ptp(bin_centers)
                for patch, norm_center in zip(patches, normalized_centers):
                    patch.set_facecolor(plt.cm.plasma(norm_center))

            ax.set_yscale('log')

        ax.set_xlabel('Feature Activation')
        ax.set_ylabel('Number of Sequences (log scale)')
        ax.set_title(f'Feature {feat_idx}')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_act = np.mean(feature_acts)
        nonzero_frac = np.mean(feature_acts > 0)
        max_act = np.max(feature_acts)
        ax.text(0.05, 0.95, f'μ={mean_act:.3f}\nmax={max_act:.2f}\n%>0={nonzero_frac:.1%}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)

    # Hide unused subplots
    for i in range(n_features, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_density_histogram(densities, feature_label, bins, log_scale=True):
    """
    Plot a histogram of feature densities with enhanced styling.
    Args:
        densities: np.ndarray of feature densities
        feature_label: str, label for the plot
        bins: int, number of bins
        log_scale: bool, if True, apply log scale to x-axis
    """
    plt.figure(figsize=(10, 6))
    
    # Use log scale on densities if requested
    plot_data = np.log10(densities[densities > 0]) if log_scale else densities

    n, bins_edges, patches = plt.hist(plot_data, bins=bins, alpha=0.7, 
                                      edgecolor='black', linewidth=0.5)
    
    # Color gradient based on density values
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
    normalized_centers = (bin_centers - np.min(bin_centers)) / np.ptp(bin_centers)
    
    for patch, norm_center in zip(patches, normalized_centers):
        patch.set_facecolor(plt.cm.RdYlBu_r(norm_center))

    
    xlabel = 'Feature Density (log scale)' if log_scale else 'Feature Density'
    plt.xlabel(xlabel)
    plt.ylabel('Number of Features')
    plt.title(f'Distribution of Feature Densities\n{feature_label}')
    
    # Add statistics
    mean_density = np.mean(plot_data)
    median_density = np.median(plot_data)
    plt.axvline(mean_density, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_density:.4f}')
    plt.axvline(median_density, color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {median_density:.4f}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sparsity_histogram(activations, threshold, bins):
    """
    Plot a histogram of the number of active features per data point with enhanced styling.
    Args:
        activations: scipy.sparse.csr_matrix or np.ndarray, shape (num_data_points, num_features)
        threshold: float, activation threshold
        bins: int, number of bins
    """
    if sparse.issparse(activations):
        sparsity = np.array(activations.getnnz(axis=1))
    else:
        sparsity = (activations > threshold).sum(axis=1)

    plt.figure(figsize=(10, 6))
    
    n, bins_edges, patches = plt.hist(sparsity, bins=bins, alpha=0.7, 
                                     edgecolor='black', linewidth=0.5)
    
    # Color gradient
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
    normalized_centers = (bin_centers - np.min(bin_centers)) / np.ptp(bin_centers)
    
    for patch, norm_center in zip(patches, normalized_centers):
        patch.set_facecolor(plt.cm.viridis(norm_center))
    
    plt.xlabel('Number of Active Features per Data Point')
    plt.ylabel('Count')
    plt.title('Sparsity Distribution')
    
    # Add statistics
    mean_sparsity = np.mean(sparsity)
    median_sparsity = np.median(sparsity)
    plt.axvline(mean_sparsity, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_sparsity:.1f}')
    plt.axvline(median_sparsity, color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {median_sparsity:.1f}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_umap_projection(vectors, title, random_state=42, metric='cosine'):
    """
    Project high-dimensional vectors to 2D using UMAP and plot  with enhanced styling.
    Args:
        vectors: np.ndarray or scipy.sparse.csr_matrix, shape (n_vectors, dim)
        title: str, plot title
        random_state: int, random seed for UMAP
        metric: str, distance metric for UMAP
    """
    if sparse.issparse(vectors):
        vectors = vectors.toarray()

    reducer = umap.UMAP(random_state=random_state, metric=metric, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(vectors)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color points based on their norm
    norms = np.linalg.norm(vectors, axis=1)
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=norms, 
                         cmap="viridis", s=0.5, alpha=0.6)
    
    plt.colorbar(scatter, label='Vector Norm')
    plt.setp(ax, xticks=[], yticks=[])
    plt.title(title, fontsize=18)
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
        for line in lines:
            if line.startswith('tensor('):
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
        for i in range(min(topk, len(activations))):
            print(f"{activations[i]:.4f}\t{tokens[i]}")


def main():
    """
    Main function for feature and activation analysis and visualization.
    """
    parser = argparse.ArgumentParser(description="Feature/activation analysis with Anthropic-style visualizations.")
    parser.add_argument('--activations', type=str, required=True, help='Path to activations .npz file (sparse format) or .npy file (dense format)')
    parser.add_argument('--threshold', type=float, default=0.0, help='Activation threshold for density/sparsity (default: 0.0)')
    parser.add_argument('--features', type=str, default=None, help='Comma-separated list of feature indices to visualize (default: all)')
    parser.add_argument('--bins', type=int, default=50, help='Number of bins for histogram (default: 50)')
    parser.add_argument('--experiment_results', type=str, default=None, help='Path to experiment_results folder for acts_and_tokens files')
    parser.add_argument('--topk', type=int, default=10, help='Number of top activations/tokens to show per feature (default: 10)')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights .npy file (shape: [num_features, dim]) for UMAP projection')
    parser.add_argument('--decoder_weights', type=str, default=None, help='Path to decoder weights .npy file (shape: [num_features, vocab_size]) for logit analysis')
    parser.add_argument('--umap_gaussian', action='store_true', help='Also plot UMAP for Gaussian random vectors of same shape as weights')
    parser.add_argument('--anthropic_style', action='store_true', help='Use Anthropic paper visualization styles')
    parser.add_argument('--feature_comparison', type=str, default=None, help='Comma-separated feature indices for comparison plot')
    parser.add_argument('--case_feature', type=int, default=1, help='feature index for case study')
    args = parser.parse_args()

    # Load activations
    if args.activations.endswith('.npz'):
        activations = sparse.load_npz(args.activations)
    else:
        activations = np.load(args.activations)

    if not sparse.issparse(activations) and activations.ndim != 2:
        raise ValueError(f"Activations should be 2D, got shape {activations.shape}")

    # Feature selection
    if args.features is not None:
        feature_indices = [int(idx) for idx in args.features.split(',')]
        if sparse.issparse(activations):
            selected_activations = activations[:, feature_indices]
        else:
            selected_activations = activations[:, feature_indices]
        feature_label = f"Selected features ({len(feature_indices)})"
    else:
        selected_activations = activations
        if sparse.issparse(activations):
            feature_indices = list(range(activations.shape[1]))
        else:
            feature_indices = list(range(activations.shape[1]))
        feature_label = f"All features ({activations.shape[1]})"

    # Compute and plot densities
    if sparse.issparse(selected_activations):
        densities = np.array(selected_activations.getnnz(axis=0)) / selected_activations.shape[0]
    else:
        densities = compute_feature_density(selected_activations, threshold=args.threshold)

    plot_density_histogram(densities, feature_label, args.bins)

    # Plot sparsity histogram
    plot_sparsity_histogram(selected_activations, args.threshold, args.bins)

    # Anthropic-style visualizations
    if args.anthropic_style:
        print("Creating Anthropic-style visualizations...")
        
        # Plot expected activation value distribution
        expected_values = plot_activation_expected_value_distribution(selected_activations, bins=args.bins)
        
        # Plot individual feature with LLR coloring (first feature or specified)
        demo_feature = feature_indices[args.case_feature] if feature_indices else 0
        plot_feature_activation_distribution_anthropic(
            activations, demo_feature, 
            title=f"Feature {demo_feature} - Anthropic Style",
            threshold=args.threshold
        )
        plot_feature_activation_expected_anthropic(
            activations, demo_feature, 
            title=f"Feature {demo_feature} - Anthropic Style",
            threshold=args.threshold
        )
        
        # Feature comparison plot
        if args.feature_comparison:
            comparison_features = [int(idx) for idx in args.feature_comparison.split(',')]
            plot_feature_comparison_anthropic(activations, comparison_features)

    # Logit weights analysis
    if args.decoder_weights is not None:
        decoder_weights = np.load(args.decoder_weights)
        plot_logit_weights_distribution(decoder_weights, feature_indices, bins=args.bins)

    # Print top activations/tokens
    if args.experiment_results is not None:
        print_top_activations_and_tokens(args.experiment_results, feature_indices, args.topk)

    # UMAP projection of feature vectors (weights)
    if args.weights is not None:
        weights = np.load(args.weights)
        plot_umap_projection(weights, title="UMAP projection of feature vectors (model weights)")
        if args.umap_gaussian:
            gaussian_data = np.random.normal(loc=0.0, scale=1.0, size=weights.shape)
            plot_umap_projection(gaussian_data, title="UMAP projection of Gaussian random vectors")


if __name__ == '__main__':
    main()
