import os

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR

from models.CAE.CAEEngine import CAEEngine
from models.CAE.ConvAE import ConvAE
from models.CVAE.CVAEEngine import CVAEEngine
from src.data.PlantVillageDataset import PlantVillage
from data.dataset import *
from src.utils import load_model

"""MatPlotLib"""
matplotlib.style.use('ggplot')
matplotlib.use('TkAgg')
plt.ion()

"""Reproducibility"""
torch.manual_seed(0)

"""CUDA"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
SETTINGS
"""
TRAIN = True
LOAD_BEST_MODEL = True
PARAMS_TO_LOAD = 100

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
epochs = 200

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
model = ConvAE().to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = opt.Adam(model.parameters(), lr=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.95)

engine = CAEEngine(net=model,
                   trainloader=trainloader,
                   trainset=plantVillageTrain,
                   testloader=validationloader,
                   testset=plantVillageTest,
                   epochs=epochs,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   criterion=criterion,
                   device=device)

if TRAIN:
    model, best_epoch = engine.start()

    if LOAD_BEST_MODEL:
        print('Loading epoch #{}'.format(best_epoch))
        load_model(model, best_epoch)
    else:
        print('Using net as-is')
else:
    load_model(model, PARAMS_TO_LOAD)

"""
MODEL VISUALIZATION
"""
engine.visualization(model, plantVillageTest, slot_num=2)

"""
CLASSIFICATION TEST
"""
thresholds = []
for threshold in range(1000, 3000, 50):
    thresholds.append(threshold)

best_th = engine.classification_performance_computation(model, testloader, plantVillageTest, thresholds)

"""
SEGMENTATION TEST
"""
engine.segmentation_performance_computation(model, testloader, plantVillageTest, best_th)