"""Module with the predict() function to run the trained model on a given text."""

import torch
from torch import nn
from transformers import DistilBertModel, DistilBertTokenizer


def predict(
    tokenizer: DistilBertTokenizer, model: DistilBertModel, text: str, max_length: int
):
    """Perform inference using the fine-tuned DistilBERT model on the given text.
    Classifies whereas a text describes a MA studying the impact of human practices
    on SOC (included) or not (excluded).

    Args:
        tokenizer: a tokenizer
        model: a fine-tuned DistilBERT model
        text: text to be classified (concatenation of the title,
            the abstract and the keywords of an article).
        max_length: input text will be cropped to max_length

    Returns: O (excluded) or 1 (included)
    """
    # Tokenize and encode the text
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    with torch.no_grad():
        outputs = model(**inputs)

    logits = model.model_head(outputs.last_hidden_state[:, 0, :])
    probas = nn.functional.softmax(logits, dim=1)
    proba, y_pred = torch.max(probas, 1)

    return y_pred.detach().numpy()[0]
