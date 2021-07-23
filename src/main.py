import os

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from performance.classification import classification_performance_computation
from src.data.PlantVillageDataset import PlantVillage
from src.dataset import *
from src.models import BaseModel, PoolBaseModel, BatchNormBaseModel, FaceGenModel
from src.train import start
from src.utils import load_model
from src.visualization import visualization

"""MatPlotLib"""
matplotlib.style.use('ggplot')
matplotlib.use('TkAgg')
plt.ion()

"""CUDA"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
SETTINGS
"""
LOAD_BEST_MODEL = True

"""
PARAMETERS
"""
# DATASET
validationSplit = 0.1
batch_size = 4
img_width = 256
img_height = 256

# FACE GEN MODEL
kernel_size_face_gen = 4
init_channels_face_gen = 16
stride_face_gen = 1
padding_face_gen = 0

# TRAINING
lr = 0.005
epochs = 20

transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

"""
DATASET GENERATION
"""
# Plant Village dataset
if not os.path.exists('../data/plantvillage/cherry/train/data.csv'):
    PlantVillage.create_csv("../data/plantvillage/cherry/train")

if not os.path.exists('../data/plantvillage/cherry/val/data.csv'):
    PlantVillage.create_csv("../data/plantvillage/cherry/val")

if not os.path.exists('../data/plantvillage/cherry/test/data.csv'):
    PlantVillage.create_csv("../data/plantvillage/cherry/test")

plantVillageTrain = PlantVillage(csv_file='../data/plantvillage/cherry/train/data.csv',
                                 root_dir='../data/plantvillage/cherry/train',
                                 transform=transform)

plantVillageVal = PlantVillage(csv_file='../data/plantvillage/cherry/val/data.csv',
                               root_dir='../data/plantvillage/cherry/val',
                               transform=transform)

plantVillageTest = PlantVillage(csv_file='../data/plantvillage/cherry/test/data.csv',
                                root_dir='../data/plantvillage/cherry/test',
                                transform=transform)

trainloader = get_training_dataloader(plantVillageTrain, batch_size)
validationloader = get_validation_dataloader(plantVillageVal, batch_size=1)
testloader = get_test_dataloader(plantVillageTest, batch_size=1)

"""
MODEL TRAINING
"""

# initialize the model
baseModel = BaseModel.ConvVAE().to(device)

net, best_epoch = start(net=baseModel,
                        trainloader=trainloader,
                        trainset=plantVillageTrain,
                        testloader=validationloader,
                        testset=plantVillageTest,
                        epochs=epochs,
                        lr=lr,
                        device=device)

if LOAD_BEST_MODEL:
    print('Loading epoch #{}'.format(best_epoch))
    load_model(best_epoch)
else:
    print('Using net as-is')

"""
MODEL VISUALIZATION
"""
visualization(net, plantVillageTest, slot_num=2)

"""
CLASSIFICATION TEST
"""
thresholds = []
for threshold in range(0, 6000, 100):
    thresholds.append(threshold)

criterion = nn.MSELoss(reduction='sum')
classification_performance_computation(net, testloader, plantVillageTest, device, criterion, thresholds)
