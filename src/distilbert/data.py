"""Module to load training data from file and to prepare
train / validation dataloaders
"""
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def load_training_set(filepath: Path) -> tuple[list[str], list[int]]:
    """Loads (X, Y) training data from pickle file.

    Returns: a tuple (x, y) with:
        x: list of strings with length n_samples
        y: list of int in (0, 1) with length n_samples
    """
    df = pd.read_pickle(filepath)
    x = df["Title"] + " " + df["Abstract"] + " " + df["Keywords"]
    x = x.astype(str).to_list()
    y = df.Screening.map({"included": 1, "excluded": 0}).to_numpy()

    logger.info(f"Loaded training data with {len(x)} (x,y) samples")
    logger.info(f"First sample:  \nx={x[0]}, \ny={y[0]},")

    return x, y


def get_dataloaders(
    tokens, labels, test_size: float, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """Prepare training and validation data loaders.

    Args:
        tokens: all tokens
        labels: all labels (0 or 1)
        test_size: proportion of samples to be used for validation
        batch_size: number of samples per batch

    Returns: tuple (train dataloader, validation dataloader)
    """
    (
        train_tokens,
        val_tokens,
        train_attention_masks,
        val_attention_masks,
        train_labels,
        val_labels,
    ) = train_test_split(
        tokens["input_ids"],
        tokens["attention_mask"],
        labels,
        test_size=test_size,
        random_state=42,
        shuffle=False,
    )

    train_dataset = TensorDataset(train_tokens, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_tokens, val_attention_masks, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
