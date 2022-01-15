import matplotlib
import time
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from data import DatasetGenerator
from ddp import get_distributed_model, setup_ddp
from engine.Engine import Engine
from models.CAE.CAEEngine import CAEEngine
from models.CAE.ConvAE import ConvAE
from models.VQVAE.VQVAEModel import VQVAEModel
from data.dataset import *
from models.CVAE.CVAEEngine import CVAEEngine
from models.CVAE.ConvVAE import ConvVAE
from utils import load_model, setup

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
PARAMS_TO_LOAD = 9

"""
PARAMETERS
"""
# DATASET
validationSplit = 0.1
batch_size = 16
img_width = 256
img_height = 256

# FACE GEN MODEL
kernel_size_face_gen = 4
init_channels_face_gen = 16
stride_face_gen = 1
padding_face_gen = 0

# TRAINING
epochs = 1

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

trainloader = get_training_dataloader(plantVillageTrain, batch_size, num_workers=0)
validationloader = get_validation_dataloader(plantVillageVal, batch_size=1, num_workers=0)
testloader = get_test_dataloader(plantVillageTest, batch_size=1, num_workers=0)

"""
MODEL TRAINING
"""

num_training_updates = 15000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

# initialize the model
model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens,
    num_embeddings, embedding_dim, commitment_cost, decay).to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = opt.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
scheduler = ExponentialLR(optimizer, gamma=0.99)
compute_loss = lambda a, b, c : a
input_shape = (3, img_width, img_height)

"""
DDP
"""
setup_ddp()
ddp_model = get_distributed_model(model)
print(torch.get_num_threads())
print(torch.distributed.get_backend(group=None))

engine = Engine(model=ddp_model,
                trainloader=trainloader,
                trainset=plantVillageTrain,
                testloader=validationloader,
                testset=plantVillageTest,
                epochs=epochs,
                optimizer=optimizer,
                criterion=criterion,
                input_shape=input_shape,
                compute_loss=compute_loss,
                device=device)

if TRAIN:
    start = time.time()
    model, best_epoch = engine.start()
    total_time = time.time() - start
    print('Total time: {}'.format(total_time))

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
engine.visualization()

"""
CLASSIFICATION TEST
"""
thresholds = []
for threshold in range(1000, 3000, 50):
    thresholds.append(threshold)

engine.roc_curve_computation(testloader, plantVillageTest)

"""
SEGMENTATION TEST
"""
engine.segmentation_performance_computation(model, testloader, plantVillageTest)
