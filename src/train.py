import sys

import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from torchsummary import summary

from src.models import BaseModel
from engine import train, validate
from utils import save_reconstructed_images, image_to_vid, save_loss_plot, save_original_images, save_model


def start(net,
          trainloader,
          trainset,
          testloader,
          testset,
          epochs,
          lr,
          device):

    best_val_loss = sys.maxsize
    best_epoch = 1

    # summary
    summary(net, (3, 256, 256))

    # set the learning parameters
    optimizer = optim.Adadelta(net.parameters())
    criterion = nn.MSELoss(reduction='sum')
    # a list to save all the reconstructed images in PyTorch grid format
    grid_images = []

    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = train(
            net, trainloader, trainset, device, optimizer, criterion
        )
        valid_epoch_loss, recon_images, original_images = validate(
            net, testloader, testset, device, criterion
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        # save the reconstructed images from the validation loop
        save_reconstructed_images(recon_images, epoch + 1)
        save_original_images(original_images, epoch + 1)
        # save model
        save_model(net, epoch + 1)
        # convert the reconstructed images to PyTorch image grid format
        image_grid = make_grid(recon_images.detach().cpu())
        grid_images.append(image_grid)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {valid_epoch_loss:.4f}")

        if valid_epoch_loss < best_val_loss:
            best_val_loss = valid_epoch_loss
            best_epoch = epoch + 1

    # save the reconstructions as a .gif file
    image_to_vid(grid_images)
    # save the loss plots to disk
    save_loss_plot(train_loss, valid_loss)
    print('TRAINING COMPLETE')

    return net, best_epoch
