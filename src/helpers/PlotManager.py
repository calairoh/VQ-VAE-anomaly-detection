import torch
import numpy as np
import matplotlib.pyplot as plt

class PlotManager:
  def __init__(self, model, testset, criterion):
    self.model = model
    self.testset = testset
    self.criterion = criterion
  
  def visualization(self, slot_num=2):
    self.model.eval()

    plt.figure()

    with torch.no_grad():
      for i in range(len(self.testset)):
        sample = self.testset[i]['image']

        reconstruction, mu, logvar = self.model(sample[None, ...])

        loss = self.criterion(reconstruction, sample[None, ...])

        ax = plt.subplot(1, slot_num * 2, (i * 2) + 1)
        plt.tight_layout()
        ax.set_title('Original'.format(i + 1))
        ax.axis('off')
        plt.imshow(np.transpose(sample, (1, 2, 0)))

        ax = plt.subplot(1, slot_num * 2, (i * 2) + 2)
        plt.tight_layout()
        ax.set_title('Recon error: {}'.format(loss))
        ax.axis('off')
        plt.imshow(np.transpose(np.squeeze(reconstruction), (1, 2, 0)))

        if i == (slot_num - 1):
          plt.show()
          break