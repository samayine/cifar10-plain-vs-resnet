#!/usr/bin/env python3
"""
fast_cifar_download.py

Downloads the CIFAR-10 dataset from Hugging Face Hub (which is extremely fast)
and formats it into the pickled binary batches directory structure expected by
torchvision.datasets.CIFAR10. This bypasses the throttled University of Toronto servers.
"""

import os
import sys
import pickle
import numpy as np

try:
    from datasets import load_dataset
except ImportError:
    print("Error: The 'datasets' library is required to run this helper script.")
    print("Please install it by running: pip install datasets")
    sys.exit(1)


def main():
    print("======================================================================")
    print("🚀 Downloading CIFAR-10 dataset from Hugging Face Hub...")
    print("======================================================================")
    
    # Load dataset from Hugging Face
    hf_dataset = load_dataset("uoft-cs/cifar10")
    
    out_dir = "data/cifar-10-batches-py"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Save batches.meta
    print("\n[1/3] Generating batches.meta...")
    meta = {
        "num_cases_per_batch": 10000,
        "label_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ],
        "num_vis": 3072
    }
    with open(os.path.join(out_dir, "batches.meta"), "wb") as f:
        pickle.dump(meta, f)
    
    # Helper to convert images to the flat numpy format (3072 bytes per image)
    def convert_to_cifar_format(hf_subset):
        data_list = []
        labels_list = []
        for item in hf_subset:
            # Convert PIL image to numpy array of shape (32, 32, 3)
            img_np = np.array(item["img"])
            # Transpose to (3, 32, 32) (channels first)
            img_transposed = img_np.transpose(2, 0, 1)
            # Flatten to 3072 bytes
            img_flat = img_transposed.flatten()
            data_list.append(img_flat)
            labels_list.append(item["label"])
        
        data_array = np.vstack(data_list)
        return {
            "data": data_array,
            "labels": labels_list
        }
    
    # 2. Save training batches (5 batches of 10,000 images each)
    print("\n[2/3] Processing training batches (50,000 images total)...")
    train_data = hf_dataset["train"]
    for i in range(5):
        batch_name = f"data_batch_{i+1}"
        print(f"  → Generating {batch_name}...")
        slice_data = train_data.select(range(i * 10000, (i + 1) * 10000))
        batch_dict = convert_to_cifar_format(slice_data)
        batch_dict["filenames"] = [f"image_{j}.png" for j in range(i * 10000, (i + 1) * 10000)]
        with open(os.path.join(out_dir, batch_name), "wb") as f:
            pickle.dump(batch_dict, f, protocol=2)
            
    # 3. Save test batch (10,000 images)
    print("\n[3/3] Processing test batch (10,000 images total)...")
    test_data = hf_dataset["test"]
    batch_dict = convert_to_cifar_format(test_data)
    batch_dict["filenames"] = [f"image_{j}.png" for j in range(10000)]
    with open(os.path.join(out_dir, "test_batch"), "wb") as f:
        pickle.dump(batch_dict, f, protocol=2)
        
    # Create a dummy cifar-10-python.tar.gz file so torchvision doesn't crash on check
    with open("data/cifar-10-python.tar.gz", "wb") as f:
        f.write(b"")
        
    print("\n======================================================================")
    print("✅ CIFAR-10 dataset successfully downloaded and structured for PyTorch!")
    print("======================================================================")


if __name__ == "__main__":
    main()
