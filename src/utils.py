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


def save_elab_image(image, count):
    image.save(f"../outputs/test/{count}-segmented-elaborated.jpg", "JPEG")


def save_test_images(original_image, recon_image, segmented_image, count):
    save_image(original_image.cpu(), f"../outputs/test/{count}-original.jpg")
    save_image(recon_image.cpu(), f"../outputs/test/{count}-recon.jpg")
    segmented_image.save(f"../outputs/test/{count}-segmented.jpg", "JPEG")


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


def build_segmentation_plot(original, reconstruction, diff, elaborated, count):
    plt.figure()

    # ORIGINAL
    ax = plt.subplot(1, 4, 1)
    plt.tight_layout()
    ax.set_title('Original')
    ax.axis('off')
    plt.imshow(np.transpose(np.squeeze(original), (1, 2, 0)))

    # RECONSTRUCTION
    ax = plt.subplot(1, 4, 2)
    plt.tight_layout()
    ax.set_title('Reconstructed')
    ax.axis('off')
    plt.imshow(np.transpose(np.squeeze(reconstruction.detach().numpy()), (1, 2, 0)))

    # RAW DIFFERENCE
    ax = plt.subplot(1, 4, 3)
    plt.tight_layout()
    ax.set_title('Raw Diff')
    ax.axis('off')
    plt.imshow(diff)

    # FINAL SEGMENTATION RESULT
    ax = plt.subplot(1, 4, 4)
    plt.tight_layout()
    ax.set_title('Result')
    ax.axis('off')
    plt.imshow(elaborated)

    plt.savefig(f'../outputs/test/image{count}.jpg')
