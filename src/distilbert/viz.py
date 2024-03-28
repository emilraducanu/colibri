"""Functions to plot figures from run results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: list[int], y_pred: list[int], output_dir: Path
) -> None:
    """Plots confusion matrix and save figure to .png file"""
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm / np.sum(cm), annot=True, fmt=".2%", cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion matrix")
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


def plot_f1_scores(f1_scores: list[float], output_dir: Path) -> None:
    """Plots F1-scores across training epochs and save figure to .png file"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(f1_scores) + 1), f1_scores, marker="o")
    plt.title("F1-score evolution")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.grid()
    plt.savefig(output_dir / "f1-score_evolution.png")
    plt.close()
