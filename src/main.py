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

"""CUDA"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
PARAMETERS
"""
validationSplit = 0.2

lr = 0.001
epochs = 10
batch_size = 64
img_width = 32
img_height = 32

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
                                       validationSplit=validationSplit)

mendeleyDatasetTest = MendeleyDataset(csv_file='../data/mendeley/mendeley.csv',
                                      root_dir='../data/mendeley',
                                      healthy_only=True,
                                      plants=list([MendeleyPlant.ALSTONIA_SCHOLARIS]),
                                      validation=True,
                                      validationSplit=validationSplit)

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
