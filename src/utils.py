import imageio
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

to_pil_image = transforms.ToPILImage()


def image_to_vid(images):
    images = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('../outputs/images/generated_images.gif', images)


def save_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"../outputs/images/output{epoch}.jpg")


def save_original_images(original_images, epoch):
    save_image(original_images.cpu(), f"../outputs/images/output{epoch}-or.jpg")


def save_loss_plot(train_loss, valid_loss):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/images/loss.jpg')
    plt.show()


def save_model(model, epoch):
    torch.save(model.state_dict(), f'../outputs/params/model-{epoch}')


def load_model(epoch):
    return torch.load(f'../outputs/params/model-{epoch}')
