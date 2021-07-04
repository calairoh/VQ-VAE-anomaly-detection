import torchvision
from torch.utils.data import DataLoader


def get_training_set(batch_size, transform):
    # training set and train data loader
    trainset = torchvision.datasets.MNIST(
        root='../dataset', train=True, download=True, transform=transform
    )

    return DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )


def get_test_set(batch_size, transform):
    # validation set and validation data loader
    testset = torchvision.datasets.MNIST(
        root='../dataset', train=False, download=True, transform=transform
    )

    return DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )
