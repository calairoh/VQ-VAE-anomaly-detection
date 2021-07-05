import torchvision
from torch.utils.data import DataLoader


def get_training_set(transform):
    # training set
    return torchvision.datasets.MNIST(
        root='../dataset', train=True, download=True, transform=transform
    )


def get_training_dataloader(trainset, batch_size):
    # train data loader
    return DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )


def get_test_set(transform):
    # validation set
    return torchvision.datasets.MNIST(
        root='../dataset', train=False, download=True, transform=transform
    )


def get_test_dataloader(testset, batch_size):
    # validation data loader
    return DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )
