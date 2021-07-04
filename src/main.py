import torch
import torchvision.transforms as transforms

from src.train import start
from src.visualization import visualization
from src.dataset import *

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
trainloader = get_training_set(batch_size, transform)
testloader = get_test_set(batch_size, transform)

"""
MODEL TRAINING
"""
net = start(trainloader, testloader, epochs, lr, device)

"""
MODEL VISUALIZATION
"""
visualization(net, testloader, device)
