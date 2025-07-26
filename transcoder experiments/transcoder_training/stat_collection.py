import torch
import wandb
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformer_lens import HookedTransformer
from scipy import sparse
import os

from utils import prepare_sampling_data
from model import SparseTranscoder, AutoEncoder
from consts import *
import pickle

# perhaps we can pass tokenized data- we will just untokenize at the end seeing as it will make it easier to maintain a dictionary, or there is no point since w can highliht the token in question
def collect_activation_samples(conf, loader, tokenizer, model, num_features, top_k=500, deterministic=True):
    """
    Collect top activation samples for each feature.
    Args:
        conf: Configuration dictionary
        loader: DataLoader yielding batches
        tokenizer: Tokenizer for the data
        model: Model with .forward() method
        num_features: Number of features to sample
        top_k: Number of top activations to keep per feature
        deterministic: If True, use arange for features; else random
    Returns:
        features, top_values, top_key_rows, top_specific_keys
    """
    if deterministic:
        features = torch.arange(end=conf['d_emb'] * conf['dict_mult'])[:num_features]
    else:
        features = torch.randperm(conf['d_emb'] * conf['dict_mult'])[:num_features]
    print(f"Selected features: {features.shape}")

    top_values = {feat: torch.empty(0, device='cpu') for feat in range(num_features)}
    top_key_rows = {feat: torch.empty((0, 50), dtype=torch.long, device='cpu') for feat in range(num_features)}
    top_specific_keys = {feat: torch.empty(0, dtype=torch.long, device='cpu') for feat in range(num_features)}

    for batch in tqdm(loader, desc='Collecting activation samples'):
        texts = batch['text']
        input_tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=50)['input_ids']
        with torch.no_grad():
            logits, model_cache = model.llm.run_with_cache(input_tokens)
            if type(model.layer) == int:
                input_acts = model_cache[f'blocks.{model.layer}.ln2.hook_normalized']
            preds, acts = model.forward(input_acts)
            samples = acts[:, :, features].to('cpu')

        batch_indices = torch.arange(samples.shape[0], device='cpu').repeat_interleave(50)
        values_flat = samples.view(-1, num_features)
        keys_flat = input_tokens.view(-1)
        keys_full_rows = input_tokens[batch_indices, :]

        for feat in range(num_features):
            feat_values = values_flat[:, feat]  # [batch_size*50]

            # stack existing with new
            all_values = torch.cat([top_values[feat], feat_values], dim=0)
            all_key_rows = torch.cat([top_key_rows[feat], keys_full_rows], dim=0)
            all_specific_keys = torch.cat([top_specific_keys[feat], keys_flat], dim=0)

            # get topk
            if all_values.shape[0] > top_k:
                topk = torch.topk(all_values, k=top_k)
                idxs = topk.indices
                top_values[feat] = all_values[idxs]
                top_key_rows[feat] = all_key_rows[idxs]
                top_specific_keys[feat] = all_specific_keys[idxs]
            else:
                # if still under top_k, no need to cut
                top_values[feat] = all_values
                top_key_rows[feat] = all_key_rows
                top_specific_keys[feat] = all_specific_keys
    return features, top_values, top_key_rows, top_specific_keys

def collect_smallest_activations(conf, loader, tokenizer, model, num_features, top_k=500, deterministic=True):
    """
    Collect top activation samples for each feature.
    Args:
        conf: Configuration dictionary
        loader: DataLoader yielding batches
        tokenizer: Tokenizer for the data
        model: Model with .forward() method
        num_features: Number of features to sample
        top_k: Number of top activations to keep per feature
        deterministic: If True, use arange for features; else random
    Returns:
        features, top_values, top_key_rows, top_specific_keys
    """
    if deterministic:
        features = torch.arange(end=conf['d_emb'] * conf['dict_mult'])[:num_features]
    else:
        features = torch.randperm(conf['d_emb'] * conf['dict_mult'])[:num_features]
    print(f"Selected features: {features.shape}")

    top_values = {feat: torch.empty(0, device='cpu') for feat in range(num_features)}
    top_key_rows = {feat: torch.empty((0, 50), dtype=torch.long, device='cpu') for feat in range(num_features)}
    top_specific_keys = {feat: torch.empty(0, dtype=torch.long, device='cpu') for feat in range(num_features)}

    for batch in tqdm(loader, desc='Collecting activation samples'):
        texts = batch['text']
        input_tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=50)['input_ids']
        with torch.no_grad():
            logits, model_cache = model.llm.run_with_cache(input_tokens)
            if type(model.layer) == int:
                input_acts = model_cache[f'blocks.{model.layer}.ln2.hook_normalized']
            preds, acts = model.forward(input_acts)
            samples = acts[:, :, features].to('cpu')

        batch_indices = torch.arange(samples.shape[0], device='cpu').repeat_interleave(50)
        values_flat = samples.view(-1, num_features)
        keys_flat = input_tokens.view(-1)
        keys_full_rows = input_tokens[batch_indices, :]

        for feat in range(num_features):
            feat_values = values_flat[:, feat]  # [batch_size*50]

            # stack existing with new
            all_values = torch.cat([top_values[feat], feat_values], dim=0)
            all_key_rows = torch.cat([top_key_rows[feat], keys_full_rows], dim=0)
            all_specific_keys = torch.cat([top_specific_keys[feat], keys_flat], dim=0)

            # get topk
            if all_values.shape[0] > top_k:
                topk = torch.topk(all_values,largest=False, k=top_k)
                idxs = topk.indices
                top_values[feat] = all_values[idxs]
                top_key_rows[feat] = all_key_rows[idxs]
                top_specific_keys[feat] = all_specific_keys[idxs]
            else:
                # if still under top_k, no need to cut
                top_values[feat] = all_values
                top_key_rows[feat] = all_key_rows
                top_specific_keys[feat] = all_specific_keys
    return features, top_values, top_key_rows, top_specific_keys

def basic_data_logging(features, top_values, top_key_rows, top_specific_keys, filename, tokenizer):
    """
    Log top activations and tokens for each feature to text files.
    """
    for i, feat in enumerate(features):
        filename_rows = filename + f'{feat}_rows.txt'
        with open(filename_rows, 'w') as f:
            f.write(str(feat) + '\n-----------\n')
            for j in range(500):
                f.write(f'{j}\n')
                f.write(tokenizer.decode(top_key_rows[i][j, :]) + '\n-----------\n')
        filename_acts_and_tokens = filename + f'{feat}_acts_and_tokens.txt'
        with open(filename_acts_and_tokens, 'w') as f:
            f.write(str(feat) + '\n-----------\n')
            for j in range(500):
                f.write(f'{j} {top_values[i][j]} {tokenizer.decode(top_specific_keys[i][j])}\n\n')

def full_context_logging(features, top_values, top_key_rows, top_specific_keys, filename, tokenizer):
    """
    Log full context for each feature to text files.
    """
    for i, feat in enumerate(features):
        filename_acts_and_tokens = filename + f'{feat}_full_context.txt'
        with open(filename_acts_and_tokens, 'w') as f:
            f.write(str(feat) + '\n-----------\n')
            for j in range(500):
                f.write(f'{j} {top_values[i][j]} {tokenizer.decode(top_specific_keys[i][j])}\n')
                f.write(tokenizer.decode(top_key_rows[i][j, :]) + '\n___\n')

def save_all_activations_as_numpy(loader, tokenizer, model, output_path, feature_indices=None):
    """
    Collect all activations for all data points and save as a sparse matrix in .npz format.
    Args:
        loader: DataLoader yielding batches of text or tokens
        tokenizer: Tokenizer for the data
        model: Model with .forward() method returning (recon, acts)
        output_path: Path to save the .npz file
        feature_indices: Optional list or array of feature indices to select (default: None, meaning all features)
    """
    all_acts = []
    total_rows = 0

    # First pass to collect all activations and count total rows
    for batch in tqdm(loader, desc='Collecting activations for all data points'):
        texts = batch['text'] if 'text' in batch else batch
        input_tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=50)['input_ids']
        with torch.no_grad():
            logits, model_cache = model.llm.run_with_cache(input_tokens)
            if type(model.layer) == int:
                input_acts = model_cache[f'blocks.{model.layer}.ln2.hook_normalized']
            else:
                raise NotImplementedError()
            _, acts = model.forward(input_acts)
            # acts: [batch, seq_len, num_features]
            acts = acts.cpu().numpy()
            if feature_indices is not None:
                acts = acts[..., feature_indices]
            acts = acts.reshape(-1, acts.shape[-1])  # flatten batch and seq_len
            all_acts.append(acts)
            total_rows += acts.shape[0]

    # Convert to sparse matrix
    num_features = all_acts[0].shape[1]
    sparse_matrix = sparse.csr_matrix((total_rows, num_features))

    # Fill the sparse matrix
    current_row = 0
    for acts in all_acts:
        # Convert to sparse format and add to the main matrix
        acts_sparse = sparse.csr_matrix(acts)
        sparse_matrix[current_row:current_row + acts.shape[0]] = acts_sparse
        current_row += acts.shape[0]

    # Save the sparse matrix
    sparse.save_npz(output_path, sparse_matrix)
    print(f"Saved sparse activations to {output_path}")
    print(f"Matrix shape: {sparse_matrix.shape}")
    print(f"Number of non-zero elements: {sparse_matrix.nnz}")
    print(f"Sparsity: {1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.4%}")

def load_sparse_activations(file_path):
    """
    Load sparse activations from an .npz file.
    Args:
        file_path: Path to the .npz file containing sparse activations
    Returns:
        sparse_matrix: scipy.sparse.csr_matrix containing the activations
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No activation file found at {file_path}")
    return sparse.load_npz(file_path)

if __name__ == '__main__':
    run = wandb.init(project="transcoders")
    loader, tokenizer = prepare_sampling_data()
    artifact = run.use_artifact(MODEL_CHECKPOINT, type="model")
    artifact_dir = artifact.download()
    llm = HookedTransformer.from_pretrained_no_processing(
        MODEL_PATH,
        device=DEVICE,
        default_padding_side='left',
    )
    model = SparseTranscoder.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", config=CONFIG, hooked_transformer=llm, layer=0)

    num_features = CONFIG['d_emb'] * CONFIG['dict_mult']
    #features, top_values, top_key_rows, top_specific_keys = collect_activation_samples(CONFIG, loader, tokenizer, model, num_features, deterministic=True)
    #basic_data_logging(features, top_values, top_key_rows, top_specific_keys, 'experiments_layer3/feature', tokenizer)
    #full_context_logging(features, top_values, top_key_rows, top_specific_keys, 'experiments/feature', tokenizer)

    # Save activations in sparse format
    #output_path = 'activations_layer3.npz'
    #save_all_activations_as_numpy(loader, tokenizer, model, output_path, feature_indices=list(range(4000)))

    # # Example of loading the sparse activations
    # sparse_acts = load_sparse_activations(output_path)
    # print(f"Loaded sparse activations with shape: {sparse_acts.shape}")
    # print(f"Number of non-zero elements: {sparse_acts.nnz}")
    # print(f"Sparsity: {1 - sparse_acts.nnz / (sparse_acts.shape[0] * sparse_acts.shape[1]):.4%}")
    features, top_values, top_key_rows, top_specific_keys = collect_smallest_activations(CONFIG, loader, tokenizer, model, num_features, deterministic=True)
    basic_data_logging(features, top_values, top_key_rows, top_specific_keys, 'experiments_layer0_smallest/feature', tokenizer)
