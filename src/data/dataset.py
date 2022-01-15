from torch.utils.data import DataLoader


def get_training_dataloader(trainset, batch_size, num_workers):
    # train data loader
    return DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def get_validation_dataloader(validationset, batch_size, num_workers):
    # validation data loader
    return DataLoader(
        validationset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def get_test_dataloader(testset, batch_size, num_workers):
    # test data loader
    return DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
