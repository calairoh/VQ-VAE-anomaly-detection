from utils import *

class EpochInfo:
  def __init__(self,
    model,
    epoch,
    recon_images,
    original_image,
    train_epoch_loss,
    valid_epoch_loss):
    self.model = model
    self.epoch = epoch
    self.recon_images = recon_images
    self.original_image = original_image
    self.train_epoch_loss = train_epoch_loss
    self.valid_epoch_loss = valid_epoch_loss
  
  def save(self):
    # save the reconstructed images from the validation loop
    save_reconstructed_images(self.recon_images, self.epoch + 1)
    save_original_images(self.original_image, self.epoch + 1)
    # save model
    save_model(self.model, self.epoch + 1)