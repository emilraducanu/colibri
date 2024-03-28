"""Module that loads the model and tokenizer"""
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from transformers import DistilBertModel, DistilBertTokenizer

MODEL_NAME = "distilbert-base-uncased"


def get_tokenizer(model_name=MODEL_NAME) -> DistilBertTokenizer:
    """Loads Distilbert tokenizer and add new tokens."""
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    new_tokens = [
        "SOC",
        "SOM",
        "meta-analysis",
        "metaanalysis",
        "meta-analyses",
        "metaanalyses",
    ]

    tokenizer.add_tokens(new_tokens)
    return tokenizer


def get_model_head(model: DistilBertModel, dropout: float = 0.3) -> nn.Sequential:
    """Returns non-trained last layer of the model ("classification head")

    Args:
        model: a Distilbert model
        dropout: dropout fraction for sequential layer
    """
    return nn.Sequential(
        nn.Linear(model.config.hidden_size, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 2),
    )


def load_pretrained_model(
    model_name: str = MODEL_NAME,
) -> tuple[DistilBertTokenizer, DistilBertModel, nn.Sequential]:
    """Loads tokenizer, pre-trained model and new model head."""
    logger.info(f"Loading pretrained {model_name} model...")
    tokenizer = get_tokenizer(model_name=model_name)
    model = DistilBertModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model_head = get_model_head(model=model)
    logger.info(f"Model parameters: {model.num_parameters()}")
    return tokenizer, model, model_head


def load_finetuned_model(
    model_dir: Path, device: str = "cpu", model_name: str = MODEL_NAME
):
    """Loads tokenizer and fine-tuned model from saved weights.

    Args:
        model_dir: directory containing model weights and model head weights
        device: either 'cpu' or 'gpu'
        model_name: HuggingFace model name
    """
    tokenizer, model, model_head = load_pretrained_model(model_name=model_name)
    model_head.load_state_dict(
        torch.load(model_dir / "fine_tuned_head.pt", map_location=device)
    )
    model.model_head = model_head
    model.eval()

    return tokenizer, model
