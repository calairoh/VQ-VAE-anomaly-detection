"""
    Create dataloaders to feed data into the neural network
    Default MNIST dataset is used and standard train/test split is performed
    """

import torch
from torchvision import datasets, transforms


def create_train_dataset(batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)


def create_test_dataset():
    return torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
        batch_size=1)
