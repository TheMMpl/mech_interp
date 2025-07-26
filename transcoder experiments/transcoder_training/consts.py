import torch
import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

def load_config(path=CONFIG_PATH):
    """
    Load configuration from a YAML file.
    Args:
        path: Path to the YAML config file.
    Returns:
        config: dict with configuration values
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

# Convenience constants
MODEL_PATH = CONFIG['MODEL_PATH']
DEVICE = torch.device(CONFIG['DEVICE'])
DTYPES = {k: getattr(torch, v) for k, v in CONFIG['DTYPES'].items()}
NUM_WORKERS = CONFIG['NUM_WORKERS']
TOKENIZERS_PARALLELISM = CONFIG['TOKENIZERS_PARALLELISM']
BATCH_SIZE = CONFIG['BATCH_SIZE']
MAX_EPOCHS = CONFIG['MAX_EPOCHS']
LOG_STEPS = CONFIG['LOG_STEPS']
MODEL_CHECKPOINT = CONFIG['MODEL_CHECKPOINT']
# ... add more as needed
