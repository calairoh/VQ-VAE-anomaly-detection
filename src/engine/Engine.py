
import torch
import numpy as np
from tqdm import tqdm
from torchsummary import summary
import torch.nn.functional as F
from torchvision import transforms

from image.ImageDifference import ImageDifference
from image.ImageElaboration import ImageElaboration
from image.RGB import RGB

from helpers.EpochInfo import EpochInfo
from helpers.TrainingInfo import TrainingInfo
from helpers.PlotManager import PlotManager

from utils import plot_roc_curve, build_segmentation_plot, save_image
from sklearn.metrics import roc_curve, roc_auc_score

class Engine:
  def __init__(self,
    model,
    trainloader,
    trainset,
    testloader,
    testset,
    epochs,
    optimizer,
    criterion,
    device,
    input_shape,
    compute_loss):

    self.model = model
    self.trainloader = trainloader
    self.trainset = trainset
    self.testloader = testloader
    self.testset = testset

    self.epochs = epochs
    self.optimizer = optimizer
    self.criterion = criterion
    self.device = device
    self.input_shape = input_shape
    self.compute_loss = compute_loss

  def start(self):
    summary(self.model, self.input_shape)

    training_info = TrainingInfo()

    for epoch in range(self.epochs):
      print(f"Epoch {epoch + 1} of {self.epochs}")
      train_epoch_loss = self.train_step()
      valid_epoch_loss, recon_images, original_images = self.validate_step()

      epoch_info = EpochInfo(model=self.model,
        epoch=epoch,
        recon_images=recon_images,
        original_image=original_images,
        train_epoch_loss=train_epoch_loss,
        valid_epoch_loss=valid_epoch_loss)

      epoch_info.save()
      training_info.add_grid_image(recon_images=recon_images)
      training_info.add_epoch_info(epoch_info=epoch_info)

      print(f"Train Loss: {train_epoch_loss*1000:.4f}")
      print(f"Val Loss: {valid_epoch_loss*1000:.4f}")

    training_info.save_validation_gif_results()
    training_info.save_loss_plot()
    print('TRAINING COMPLETE')

    return self.model, training_info.best_epoch


  def train_step(self):
    self.model.train()
    running_loss = 0.0
    counter = 0

    for i, data in tqdm(enumerate(self.trainloader), total=int(len(self.trainset) / self.trainloader.batch_size)):
      counter += 1
      data = data['image']
      data = data.to(self.device)
      self.optimizer.zero_grad()
      reconstruction, loss, perplexity = self.model(data)
      #bce_loss = self.criterion(reconstruction, data)
      #loss = self.compute_loss(bce_loss, mu, logvar)
      recon_error = F.mse_loss(reconstruction, data)
      loss = recon_error + loss
      loss.backward()
      running_loss += loss.item()
      self.optimizer.step()
    train_loss = running_loss / (counter * self.trainloader.batch_size)
    return train_loss
  
  def validate_step(self):
    self.model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
      for i, data in tqdm(enumerate(self.testloader), total=int(len(self.testset) / self.testloader.batch_size)):
        counter += 1
        data = data['image']
        data = data.to(self.device)
        reconstruction, loss, perplexity = self.model(data)
        #bce_loss = self.criterion(reconstruction, data)
        #loss = self.compute_loss(bce_loss, mu, logvar)
        recon_error = F.mse_loss(reconstruction, data)
        loss = recon_error + loss
        running_loss += loss.item()

        # save the last batch input and output of every epoch
        if i == int(len(self.testloader) / self.testloader.batch_size) - 1:
          recon_images = reconstruction
          original_images = data
    val_loss = running_loss / counter
    return val_loss, recon_images, original_images
  
  def visualization(self):
    plot = PlotManager(self.model, self.testset, self.criterion)
    plot.visualization()
  
  def roc_curve_computation(self, testloader, testset):
    y_true = []
    y_pred = []
    for i, data in tqdm(enumerate(testloader), total=len(testset)):
        img = data['image']
        img = img.to(self.device)
        reconstruction, loss, perplexity = self.model(img)
        recon_error = F.mse_loss(reconstruction, img)
        loss = recon_error + loss

        y_pred.append(float(loss))
        y_true.append(int(data['label']))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    norm_pred = (y_pred - min(y_pred)) / (max(y_pred) - min(y_pred))

    fpr, tpr, threshold = roc_curve(y_true, norm_pred)

    plot_roc_curve(fpr, tpr)

    print(f'ROC AUC score: {roc_auc_score(y_true, norm_pred)}')

  def segmentation_performance_computation(self, model, testloader, testset):
    counter = 0
    for i, data in tqdm(enumerate(testloader), total=len(testset)):
      counter += 1
      img = data['image']
      label = data['label']
      img = img.to(self.device)
      reconstruction, _, _ = model(img)

      # if loss > threshold:
      diff = ImageDifference(img, reconstruction).difference()
      elaborated = ImageElaboration(diff)
      elaborated.keep_only(RGB.RED)
      elaborated.negative()
      build_segmentation_plot(img, reconstruction, diff, elaborated.get(), counter, reconstruction, label)

      save_image(reconstruction.cpu(), f"./outputs/test/reconstructed-{i}.jpg")
      save_image(img.cpu(), f"./outputs/test/original-{i}.jpg")

      convert_tensor = transforms.ToTensor()

      save_image(convert_tensor(elaborated.get()), f"./outputs/test/elaborated-{i}.jpg")

  @staticmethod
  def final_loss(bce_loss, mu, logvar):
      """
      This function will add the reconstruction loss (BCELoss) and the
      KL-Divergence.
      KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
      :param bce_loss: reconstruction loss
      :param mu: the mean from the latent vector
      :param logvar: log variance from the latent vector
      """
      bce = bce_loss
      kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
      return bce + kld