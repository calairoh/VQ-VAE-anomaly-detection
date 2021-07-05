import matplotlib
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

import model
from engine import train, validate
from utils import save_reconstructed_images, image_to_vid, save_loss_plot


def start(trainloader, trainset, testloader, testset, epochs, lr, device):

    # initialize the model
    net = model.ConvVAE().to(device)
    # set the learning parameters

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='sum')
    # a list to save all the reconstructed images in PyTorch grid format
    grid_images = []

    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = train(
            net, trainloader, trainset, device, optimizer, criterion
        )
        valid_epoch_loss, recon_images = validate(
            net, testloader, testset, device, criterion
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        # save the reconstructed images from the validation loop
        save_reconstructed_images(recon_images, epoch + 1)
        # convert the reconstructed images to PyTorch image grid format
        image_grid = make_grid(recon_images.detach().cpu())
        grid_images.append(image_grid)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {valid_epoch_loss:.4f}")

    # save the reconstructions as a .gif file
    image_to_vid(grid_images)
    # save the loss plots to disk
    save_loss_plot(train_loss, valid_loss)
    print('TRAINING COMPLETE')

    return net
