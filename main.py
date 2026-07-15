#!/usr/bin/env python3
"""
main.py

Entry point for training and evaluating both PlainCNN and ResidualCNN on
CIFAR-10.  Run with:

    python main.py                  # defaults: 20 epochs, batch 128, lr 0.1
    python main.py --epochs 5       # quick smoke test
    python main.py --help           # see all options
"""

import argparse
import json
import os
import sys

import torch

from dataset import get_cifar10_dataloaders
from models import PlainCNN, ResidualCNN
from train import train_model
from evaluate import evaluate_model, compute_confusion_matrix, CIFAR10_CLASSES
from plots import plot_training_curves, plot_confusion_matrix, plot_comparison_curves
from utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PlainCNN and ResidualCNN on CIFAR-10"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for data loaders (default: 128)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="Initial learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory to download/cache CIFAR-10 (default: ./data)",
    )
    parser.add_argument(
        "--figures-dir", type=str, default="report/figures",
        help="Directory to save figures (default: report/figures)",
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default="checkpoints",
        help="Directory to save model checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use a small synthetic dataset instead of downloading CIFAR-10 (for fast testing)",
    )
    return parser.parse_args()


def count_parameters(model):
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")
    print("=" * 70)

    # Create output directories
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load data                                                        #
    # ------------------------------------------------------------------ #
    print("\n[1/6] Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size, data_dir=args.data_dir, use_synthetic=args.synthetic
    )
    print(f"  Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # ------------------------------------------------------------------ #
    # 2. Train PlainCNN                                                   #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("[2/6] Training PlainCNN...")
    print("=" * 70)
    plain_model = PlainCNN(num_classes=10)
    print(f"  Parameters: {count_parameters(plain_model):,}")

    history_plain = train_model(
        plain_model,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=os.path.join(args.checkpoints_dir, "plain_cnn_best.pth"),
    )

    # ------------------------------------------------------------------ #
    # 3. Train ResidualCNN                                                #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("[3/6] Training ResidualCNN...")
    print("=" * 70)
    resnet_model = ResidualCNN(num_classes=10)
    print(f"  Parameters: {count_parameters(resnet_model):,}")

    history_resnet = train_model(
        resnet_model,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=os.path.join(args.checkpoints_dir, "resnet_cnn_best.pth"),
    )

    # ------------------------------------------------------------------ #
    # 4. Evaluate both models                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("[4/6] Evaluating models on test set...")
    print("=" * 70)

    # Reload best checkpoints for evaluation
    plain_model.load_state_dict(
        torch.load(
            os.path.join(args.checkpoints_dir, "plain_cnn_best.pth"),
            map_location=device,
        )["model_state"]
    )
    resnet_model.load_state_dict(
        torch.load(
            os.path.join(args.checkpoints_dir, "resnet_cnn_best.pth"),
            map_location=device,
        )["model_state"]
    )
    plain_model.to(device)
    resnet_model.to(device)

    print("\n--- PlainCNN Classification Report ---")
    report_plain = evaluate_model(plain_model, test_loader, device)
    print(report_plain)

    print("\n--- ResidualCNN Classification Report ---")
    report_resnet = evaluate_model(resnet_model, test_loader, device)
    print(report_resnet)

    # ------------------------------------------------------------------ #
    # 5. Generate plots                                                   #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("[5/6] Generating plots...")
    print("=" * 70)

    # Individual training curves
    plot_training_curves(
        history_plain["train_loss"], history_plain["val_loss"],
        history_plain["train_acc"], history_plain["val_acc"],
        title="PlainCNN",
        out_path=os.path.join(args.figures_dir, "plain_cnn_curves.png"),
    )

    plot_training_curves(
        history_resnet["train_loss"], history_resnet["val_loss"],
        history_resnet["train_acc"], history_resnet["val_acc"],
        title="ResidualCNN",
        out_path=os.path.join(args.figures_dir, "resnet_cnn_curves.png"),
    )

    # Side-by-side comparison
    plot_comparison_curves(
        history_plain, history_resnet,
        out_path=os.path.join(args.figures_dir, "comparison_curves.png"),
    )

    # Confusion matrices
    cm_plain = compute_confusion_matrix(plain_model, test_loader, device)
    cm_resnet = compute_confusion_matrix(resnet_model, test_loader, device)

    plot_confusion_matrix(
        cm_plain, CIFAR10_CLASSES, title="PlainCNN Confusion Matrix",
        out_path=os.path.join(args.figures_dir, "plain_cnn_confusion.png"),
    )
    plot_confusion_matrix(
        cm_resnet, CIFAR10_CLASSES, title="ResidualCNN Confusion Matrix",
        out_path=os.path.join(args.figures_dir, "resnet_cnn_confusion.png"),
    )

    # ------------------------------------------------------------------ #
    # 6. Save histories as JSON                                           #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("[6/6] Saving training histories...")
    print("=" * 70)

    with open(os.path.join(args.checkpoints_dir, "history_plain.json"), "w") as f:
        json.dump(history_plain, f, indent=2)
    with open(os.path.join(args.checkpoints_dir, "history_resnet.json"), "w") as f:
        json.dump(history_resnet, f, indent=2)
    print("  Saved history_plain.json and history_resnet.json")

    # ------------------------------------------------------------------ #
    # Done                                                                #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("All done!")
    print(f"  Figures  → {args.figures_dir}/")
    print(f"  Models   → {args.checkpoints_dir}/")
    print(f"  PlainCNN best val acc:    {max(history_plain['val_acc']):.4f}")
    print(f"  ResidualCNN best val acc: {max(history_resnet['val_acc']):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
