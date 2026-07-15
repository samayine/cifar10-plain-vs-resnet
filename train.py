"""
train.py

Training loop and checkpoint utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from utils import save_checkpoint


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch; returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Evaluate on a validation/test set; returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs=20,
    lr=0.1,
    checkpoint_path="best_model.pth",
):
    """Train for the given number of epochs; save the best checkpoint by val accuracy.

    Uses SGD + CosineAnnealingLR (the standard ResNet training recipe).
    Returns a history dict with train/val loss and accuracy per epoch.
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"lr {current_lr:.5f} | "
            f"train_loss {train_loss:.4f}  train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}"
        )

        # Keep the checkpoint with the highest validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )
            print(f"  ↳ saved best checkpoint (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    return history


if __name__ == "__main__":
    print("Run training from main.py, not this file directly.")
