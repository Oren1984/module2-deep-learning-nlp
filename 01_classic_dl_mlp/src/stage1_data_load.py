# 01_classic_dl_mlp/src/stage1_data_load.py
# This script loads the Fashion-MNIST dataset, applies necessary transformations,
# and prepares DataLoaders for training, validation, and testing.

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define class names for Fashion-MNIST
CLASSES = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

# Function to get device
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load data and create DataLoaders
def get_dataloaders(batch_size=64, val_ratio=0.1, seed=42):
    os.makedirs("data/raw", exist_ok=True)

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),                  # [0..1]
        transforms.Normalize((0.5,), (0.5,))    # [-1..1] roughly
    ])
    
    # Load datasets
    train_full = datasets.FashionMNIST(
        root="data/raw", train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root="data/raw", train=False, download=True, transform=transform
    )
    
    # Split training set into train and validation
    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    
    # Ensure reproducibility- set seed
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=g)
    
    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

# Main function to demonstrate data loading
def main():
    train_loader, val_loader, test_loader = get_dataloaders()

    x, y = next(iter(train_loader))
    print("✅ STAGE 1 — Data Load")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    print(f"One batch X shape: {x.shape} (B,1,28,28)")
    print(f"One batch y shape: {y.shape} (B,)")
    print("Labels example:", y[:10].tolist())

if __name__ == "__main__":
    main()
