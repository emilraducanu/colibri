"""Script that fine-tunes the pretrained Distilbert model
and saves model weights to files.
"""


import json
import time
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

from src.distilbert.data import get_dataloaders, load_training_set
from src.distilbert.model import load_pretrained_model
from src.distilbert.train import _train
from src.distilbert.viz import plot_confusion_matrix, plot_f1_scores


def train(
    train_set_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    padding_length: int,
    test_size: float,
) -> None:
    """Main training function"""
    args_passed = locals().copy()
    start_time = time.time()

    # Load tokenizer, pre-trained model and model head
    tokenizer, model, model_head = load_pretrained_model()

    # Prepare training data
    x, y = load_training_set(Path(train_set_path))
    tokens = tokenizer(
        x,
        padding="max_length",
        truncation=True,
        max_length=padding_length,
        return_tensors="pt",
    )
    labels = torch.tensor(y)
    train_loader, val_loader = get_dataloaders(
        tokens=tokens, labels=labels, test_size=test_size, batch_size=batch_size
    )

    # Fine-tune model
    labels_true, labels_pred, f1_scores, accuracies = _train(
        model=model,
        model_head=model_head,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        num_epochs=epochs,
        device="cpu",
    )

    # Create output directory
    root_dir = Path(__file__).parent.parent.parent
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target_dir = root_dir / "data" / "distilbert_runs" / now
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save the fine-tuned model head to file
    torch.save(model_head.state_dict(), target_dir / "fine_tuned_head.pt")

    # Save results to figures
    plot_confusion_matrix(y_true=labels_true, y_pred=labels_pred, output_dir=target_dir)
    plot_f1_scores(f1_scores=f1_scores, output_dir=target_dir)

    # Save run info to file
    run_info = {
        **args_passed,
        "duration": time.time() - start_time,
        "last_f1_score": f1_scores[-1],
    }
    with open(target_dir / "run_info.json", "w") as output_file:
        json.dump(run_info, output_file)

    logger.info(f"Model and results saved to {target_dir}")


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent
    train_set_path = root_dir / "data" / "distilbert_trainset" / "trainset.pkl"
    train(
        train_set_path=str(train_set_path),
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        padding_length=100,
        test_size=0.2,
    )
