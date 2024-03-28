"""Script that runs the trained Distilbert model on
the whole training dataset and computes classification metrics.
"""

from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from src.distilbert.data import load_training_set
from src.distilbert.model import load_finetuned_model
from src.distilbert.predict import predict

if __name__ == "__main__":
    # Manage data paths
    root_dir = Path(__file__).parent.parent.parent
    train_set_path = root_dir / "data" / "distilbert_trainset" / "trainset.pkl"
    model_dir = root_dir / "data" / "distilbert_runs" / "2024-03-28_10-43-37"

    # Loads fine-tuned model
    tokenizer, model = load_finetuned_model(model_dir=model_dir, device="cpu")

    # Loads text data
    x, labels_true = load_training_set(train_set_path)

    # Run predictions
    labels_pred = [
        predict(tokenizer=tokenizer, model=model, text=text, max_length=100)
        for text in x
    ]

    # Compute classification performance
    f1_score_ = f1_score(labels_true, labels_pred, zero_division=1, average="weighted")
    accuracy_ = accuracy_score(labels_true, labels_pred)
    print(f"Accuracy: {accuracy_:.4f} F1-score: {f1_score_:.4f}")
