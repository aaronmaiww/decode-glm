"""Utility functions for InterPLM training adapted for Nucleotide Transformer."""

from pathlib import Path

import torch


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def _convert_paths_to_str(config_dict):
    """Convert Path objects to strings for serialization."""
    if isinstance(config_dict, dict):
        return {k: _convert_paths_to_str(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [_convert_paths_to_str(item) for item in config_dict]
    elif isinstance(config_dict, Path):
        return str(config_dict)
    else:
        return config_dict


# Data directory constant
DATA_DIR = "data"
