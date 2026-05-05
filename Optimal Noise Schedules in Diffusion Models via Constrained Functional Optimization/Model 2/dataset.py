import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loader(batch_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # scale to [-1, 1]
    ])
    root = os.environ.get("CIFAR10_ROOT", "./data")
    if os.environ.get("CIFAR10_URL"):
        datasets.CIFAR10.url = os.environ["CIFAR10_URL"]
    download = os.environ.get("CIFAR10_DOWNLOAD", "1") != "0"
    dataset = datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    return loader
