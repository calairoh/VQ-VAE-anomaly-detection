import matplotlib
import torch
import torchvision.transforms as transforms

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
trainset = get_training_set(transform)
testset = get_test_set(transform)
trainloader = get_training_dataloader(trainset, batch_size)
testloader = get_test_dataloader(testset, batch_size)

"""
MODEL TRAINING
"""
net = start(trainloader, trainset, testloader, testset, epochs, lr, device)

"""
MODEL VISUALIZATION
"""
visualization(net, testloader, device)
