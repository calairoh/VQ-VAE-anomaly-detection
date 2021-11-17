from torchvision.utils import *
from utils import *
import sys

class TrainingInfo:
  def __init__(self):
    self.train_loss = []
    self.valid_loss = []

    self.grid_images = []

    self.best_val_loss = sys.maxsize
    self.best_epoch = 1

  def add_grid_image(self, recon_images):
    image_grid = make_grid(recon_images.detach().cpu())
    self.grid_images.append(image_grid)
  
  def add_epoch_info(self, epoch_info):
    if epoch_info.valid_epoch_loss < self.best_val_loss:
      self.best_val_loss = epoch_info.valid_epoch_loss
      self.best_epoch = epoch_info.epoch + 1
    
  def save_validation_gif_results(self):
    # save the reconstructions as a .gif file
    image_to_vid(self.grid_images)
  
  def save_loss_plot(self):
    # save the loss plots to disk
    save_loss_plot(self.train_loss, self.valid_loss)
