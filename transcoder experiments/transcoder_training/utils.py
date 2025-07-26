from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from consts import BATCH_SIZE, TOKENIZERS_PARALLELISM, NUM_WORKERS
import numpy as np

def prepare_training_data():
    """
    Prepare training and validation data loaders and tokenizer for TinyStories dataset.
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        tokenizer: HuggingFace tokenizer
    """
    ds_train = load_dataset("roneneldan/TinyStories", split='train')
    ds_val = load_dataset("roneneldan/TinyStories", split='validation')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    def tokenize(data):
        tokenizer.pad_token = tokenizer.eos_token
        #note - padding to largest in batch would introduce many eos tokens to inputs
        # I don't see a way to apply an attention mask to a hookedtransformer
        # since we are after activations, i think it is sensible to shorten the inputs to avoid padding- they will have reasonable context anyways
        return tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True, max_length=50)

    ds_train.set_transform(tokenize)
    ds_val.set_transform(tokenize)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    return train_loader, val_loader, tokenizer

def prepare_sampling_data():
    """
    Prepare a data loader and tokenizer for sampling/validation from TinyStories.
    Returns:
        test_loader: DataLoader for validation set
        tokenizer: HuggingFace tokenizer
    """
    ds_test = load_dataset("roneneldan/TinyStories", split='validation')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE)
    tokenizer.pad_token = tokenizer.eos_token
    return test_loader, tokenizer

def compute_feature_density(activations: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Compute the density of each feature in a 2D activations array.
    Args:
        activations: np.ndarray of shape (num_data_points, num_features)
        threshold: float, activation threshold to consider a feature as 'active' (default: 0.0)
    Returns:
        densities: np.ndarray of shape (num_features,), fraction of data points where each feature is active
    """
    assert activations.ndim == 2, "activations must be a 2D array"
    num_data_points = activations.shape[0]
    # Boolean array: True where activation > threshold
    active = activations > threshold
    # Sum over data points (rows) for each feature (column)
    densities = active.sum(axis=0) / num_data_points
    return densities