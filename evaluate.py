"""
evaluate.py

Evaluation helpers for computing accuracy and generating classification reports.
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# CIFAR-10 class names in label order
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_predictions(model, dataloader, device):
    """Run the model over the full dataloader; returns (all_preds, all_targets) as numpy arrays."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    return np.array(all_preds), np.array(all_targets)


def evaluate_model(model, dataloader, device):
    """Return a formatted classification report string."""
    preds, targets = get_predictions(model, dataloader, device)
    report = classification_report(
        targets, preds, target_names=CIFAR10_CLASSES, digits=4
    )
    return report


def compute_confusion_matrix(model, dataloader, device):
    """Return the confusion matrix as a 2-D numpy array."""
    preds, targets = get_predictions(model, dataloader, device)
    cm = confusion_matrix(targets, preds)
    return cm
