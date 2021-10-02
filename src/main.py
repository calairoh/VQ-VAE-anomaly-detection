import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR

from data import DatasetGenerator
from models.CAE.CAEEngine import CAEEngine
from models.CAE.ConvAE import ConvAE
from data.dataset import *
from models.CVAE.CVAEEngine import CVAEEngine
from models.CVAE.ConvVAE import ConvVAE
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
epochs = 100

transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

"""
DATASET GENERATION
"""

plant = 'cherry'
plantVillageTrain, plantVillageVal, plantVillageTest = DatasetGenerator.generateDataset(plant, transform)

trainloader = get_training_dataloader(plantVillageTrain, batch_size)
validationloader = get_validation_dataloader(plantVillageVal, batch_size=1)
testloader = get_test_dataloader(plantVillageTest, batch_size=1)

"""
MODEL TRAINING
"""

# initialize the model
model = ConvVAE().to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = opt.Adadelta(model.parameters(), lr=1.5)
scheduler = ExponentialLR(optimizer, gamma=0.99)

engine = CVAEEngine(net=model,
                    trainloader=trainloader,
                    trainset=plantVillageTrain,
                    testloader=validationloader,
                    testset=plantVillageTest,
                    epochs=epochs,
                    optimizer=optimizer,
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
