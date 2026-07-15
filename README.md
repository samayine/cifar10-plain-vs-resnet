# CIFAR-10: PlainCNN vs ResidualCNN

A PyTorch implementation comparing a plain CNN baseline against a ResNet-style CNN on the CIFAR-10 image classification dataset. The goal is to empirically validate the benefit of residual (skip) connections as described in [He et al. (2016)](https://arxiv.org/abs/1512.03385).

## Results (20 epochs, GPU)

| Model       | Parameters | Best Val. Accuracy |
|-------------|------------|--------------------|
| PlainCNN    | 94,762     | 79.71%             |
| ResidualCNN | 696,618    | 90.81%             |

## Project Structure

```
├── main.py                 # Entry point — trains, evaluates, and saves results
├── models.py               # PlainCNN and ResidualCNN definitions
├── dataset.py              # CIFAR-10 dataloaders and augmentations
├── train.py                # Training and validation loops
├── evaluate.py             # Classification report and confusion matrix
├── plots.py                # Training curve and confusion matrix plots
├── utils.py                # Checkpoint and device helpers
├── fast_cifar_download.py  # Fast CIFAR-10 download via Hugging Face
├── checkpoints/            # Saved model weights and training history
└── report/figures/         # Generated plots
```

## Usage

**Quick smoke test (no download needed):**
```bash
python main.py --synthetic --epochs 1
```

**Full training (recommended on GPU):**
```bash
python main.py --epochs 20
```

**All options:**
```bash
python main.py --help
```

## Requirements

```bash
pip install -r requirements.txt
```
