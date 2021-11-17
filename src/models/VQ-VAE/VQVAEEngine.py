import torch
import torch.nn.functional as F
import torchsummary as summary
import tqdm

class VQVAEEngine:
  def __init__(self,
    model,
    num_training_updates,
    training_loader,
    trainset,
    validation_loader,
    validationset,
    optimizer,
    device,
    data_variance,
    epochs):

    self.model = model
    self.num_training_updates = num_training_updates
    self.training_loader = training_loader
    self.trainset = trainset
    self.validation_loader = validation_loader
    self.validationset = validationset
    self.device = device
    self.optimizer = optimizer
    self.data_variance = data_variance
    self.epochs = epochs

  def start(self):
    # summary
    summary(self.model, (3, 256, 256))
    
    for epoch in range(self.epochs):
      print(f"Epoch {epoch + 1} of {self.epochs}")
      
      train_epoch_loss = self.train(
        model=self.net,
        dataloader=self.trainloader,
        dataset=self.trainset,
        device=self.device,
        optimizer=self.optimizer
      )
      valid_epoch_loss, recon_images, original_images = self.validate(
        model=self.net,
        dataloader=self.testloader,
        dataset=self.testset,
        device=self.device
      )
  
  def train(self, model, dataloader, dataset, device, optimizer):
    model.train()
    running_loss = 0.0
    counter = 0

    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
      counter += 1
      data = data['image']
      data = data.to(device)
      optimizer.zero_grad()

      vq_loss, data_recon, perplexity = self.model(data)
      recon_error = F.mse_loss(data_recon, data) / self.data_variance
      loss = recon_error + vq_loss
      loss.backward()

      running_loss += loss.item()

      optimizer.step()

    train_loss = running_loss / (counter * dataloader.batch_size)
    return train_loss

  def validate(self, model, dataloader, dataset, device):
    model.eval()
    running_loss = 0.0
    counter = 0

    with torch.no_grad():
      for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        counter += 1
        data = data['image']
        data = data.to(device)

        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / self.data_variance
        loss = recon_error + vq_loss
        
        running_loss += loss.item()

        # save the last batch input and output of every epoch
        if i == int(len(dataloader) / dataloader.batch_size) - 1:
          recon_images = vq_loss
          original_images = data
    val_loss = running_loss / counter
    return val_loss, recon_images, original_images
        