import matplotlib
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from src.data.MendeleyDataset import MendeleyDataset, MendeleyPlant
from src.train import start
from src.visualization import visualization
from src.dataset import *

"""MatPlotLib"""
matplotlib.style.use('ggplot')
matplotlib.use('TkAgg')
plt.ion()

"""CUDA"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
PARAMETERS
"""
validationSplit = 0.2

lr = 0.001
epochs = 10
batch_size = 4
img_width = 128
img_height = 128

transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
])

"""
DATASET GENERATION
"""
if not MendeleyDataset.csvExists():
    MendeleyDataset.create_csv('../data/mendeley')

mendeleyDatasetTrain = MendeleyDataset(csv_file='../data/mendeley/mendeley.csv',
                                       root_dir='../data/mendeley',
                                       healthy_only=True,
                                       plants=list([MendeleyPlant.ALSTONIA_SCHOLARIS]),
                                       validation=False,
                                       validationSplit=validationSplit,
                                       transform=transform)

mendeleyDatasetTest = MendeleyDataset(csv_file='../data/mendeley/mendeley.csv',
                                      root_dir='../data/mendeley',
                                      healthy_only=True,
                                      plants=list([MendeleyPlant.ALSTONIA_SCHOLARIS]),
                                      validation=True,
                                      validationSplit=validationSplit,
                                      transform=transform)

fig = plt.figure()

for i in range(len(mendeleyDatasetTrain)):
    sample = mendeleyDatasetTrain[i]

    sample = np.transpose(sample, (1, 2, 0))

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample)

    if i == 3:
        plt.show()
        break

trainloader = get_training_dataloader(mendeleyDatasetTrain, batch_size)
testloader = get_test_dataloader(mendeleyDatasetTest, batch_size)

"""
MODEL TRAINING
"""
net = start(trainloader, mendeleyDatasetTrain, testloader, mendeleyDatasetTest, epochs, lr, device)

"""
MODEL VISUALIZATION
"""
visualization(net, testloader, device)
