"""
plots.py

Plotting utilities for training curves, confusion matrices, and model comparisons.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plots save without a display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         title="Training Curves", out_path=None):
    """Plot loss and accuracy curves for a single model."""
    epochs = np.arange(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, label="Train", linewidth=1.5)
    ax1.plot(epochs, val_losses, label="Validation", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, label="Train", linewidth=1.5)
    ax2.plot(epochs, val_accs, label="Validation", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix",
                          out_path=None):
    """Plot a confusion matrix as a seaborn heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_comparison_curves(history_plain, history_resnet, out_path=None):
    """Overlay PlainCNN vs ResidualCNN on the same loss and accuracy axes."""
    epochs = np.arange(1, len(history_plain["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Loss comparison ---
    ax1.plot(epochs, history_plain["val_loss"],
             label="PlainCNN (val)", linestyle="--", linewidth=1.5)
    ax1.plot(epochs, history_resnet["val_loss"],
             label="ResidualCNN (val)", linewidth=2)
    ax1.plot(epochs, history_plain["train_loss"],
             label="PlainCNN (train)", linestyle="--", alpha=0.5)
    ax1.plot(epochs, history_resnet["train_loss"],
             label="ResidualCNN (train)", alpha=0.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("PlainCNN vs ResidualCNN — Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Accuracy comparison ---
    ax2.plot(epochs, history_plain["val_acc"],
             label="PlainCNN (val)", linestyle="--", linewidth=1.5)
    ax2.plot(epochs, history_resnet["val_acc"],
             label="ResidualCNN (val)", linewidth=2)
    ax2.plot(epochs, history_plain["train_acc"],
             label="PlainCNN (train)", linestyle="--", alpha=0.5)
    ax2.plot(epochs, history_resnet["train_acc"],
             label="ResidualCNN (train)", alpha=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("PlainCNN vs ResidualCNN — Accuracy")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)
