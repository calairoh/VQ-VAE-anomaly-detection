from torch.utils.data import DataLoader


def get_training_dataloader(trainset, batch_size):
    # train data loader
    return DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )


def get_validation_dataloader(validationset, batch_size):
    # validation data loader
    return DataLoader(
        validationset, batch_size=batch_size, shuffle=True
    )


def get_test_dataloader(testset, batch_size):
    # test data loader
    return DataLoader(
        testset, batch_size=batch_size, shuffle=True
    )
