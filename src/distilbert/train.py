"""Functions to train DistilbertModel"""

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertModel


def _validate_one_batch(
    model: DistilBertModel,
    model_head: nn.Sequential,
    tokens,
    attention_masks,
    labels,
    device: str,
) -> tuple[list, list]:
    """Runs model validation on one batch of (X,Y) data."""
    # Send labels to the device
    labels = labels.to(device)

    with torch.no_grad():
        # Validation of the model
        outputs = model(
            input_ids=tokens.to(device),
            attention_mask=attention_masks.to(device),
        )

    logits = model_head(outputs.last_hidden_state[:, 0, :])
    _, y_pred = torch.max(logits, 1)
    labels_true = [label.item() for label in labels]
    labels_pred = [label.item() for label in y_pred]

    return labels_true, labels_pred


def _validate_one_epoch(
    model: DistilBertModel,
    model_head: nn.Sequential,
    val_loader: DataLoader,
    epoch: int,
    num_epochs: int,
    device,
) -> tuple[list, list, float, float]:
    """Runs model validation on one epoch."""
    labels_true, labels_pred = [], []
    model.eval()
    for tokens_, attention_masks_, labels_ in tqdm(
        val_loader,
        total=len(val_loader),
        desc=f"Epoch {epoch + 1}/{num_epochs} (Validation)",
        leave=False,
        unit=" batch",
    ):
        labels_true_, labels_pred_ = _validate_one_batch(
            model=model,
            model_head=model_head,
            tokens=tokens_,
            attention_masks=attention_masks_,
            labels=labels_,
            device=device,
        )
        labels_true.extend(labels_true_)
        labels_pred.extend(labels_pred_)

    f1_score_ = f1_score(labels_true, labels_pred, zero_division=1, average="weighted")
    accuracy_ = accuracy_score(labels_true, labels_pred)
    print(f"Accuracy: {accuracy_:.4f} F1-score: {f1_score_:.4f}")

    return labels_true, labels_pred, f1_score_, accuracy_


def _train_one_batch(
    model: DistilBertModel,
    model_head: nn.Sequential,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    tokens,
    attention_masks,
    labels,
) -> tuple[DistilBertModel, nn.Sequential, nn.CrossEntropyLoss, torch.optim.Optimizer]:
    """Runs model training on one batch of (X,Y) data."""
    optimizer.zero_grad()
    with torch.no_grad():
        output = model(input_ids=tokens, attention_mask=attention_masks)

    logits = model_head(output.last_hidden_state[:, 0, :])
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    return model, model_head, loss, optimizer


def _train(
    model,
    model_head: nn.Sequential,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    num_epochs: int,
    device,
):
    """Trains model on (X,Y) data."""
    # Loss function and optimizer
    params = model_head.parameters()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    # Training loop
    f1_scores, accuracies = [], []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Create progress bar for the training loop
        train_progress_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs} (Training)",
            leave=False,
            unit="batch",
        )
        for tokens, attention_masks, labels in train_progress_bar:
            model, model_head, loss, optimizer = _train_one_batch(
                model=model,
                model_head=model_head,
                loss_fn=loss_fn,
                optimizer=optimizer,
                tokens=tokens,
                attention_masks=attention_masks,
                labels=labels,
            )
            train_loss += loss.item()

        # Validation loop
        labels_true, labels_pred, f1_score_, accuracy_ = _validate_one_epoch(
            model=model,
            model_head=model_head,
            val_loader=val_loader,
            epoch=epoch,
            num_epochs=num_epochs,
            device=device,
        )
        f1_scores.append(f1_score_)
        accuracies.append(accuracy_)

    return labels_true, labels_pred, f1_scores, accuracies
