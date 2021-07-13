import matplotlib
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from src.data.MendeleyDataset import MendeleyDataset, MendeleyPlant
from src.data.PlantVillageDataset import PlantVillage, PlantVillageStatus
from src.train import start
from src.dataset import *
from src.models import BaseModel, PoolBaseModel, BatchNormBaseModel, FaceGenModel
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
img_width = 64
img_height = 64

# FACE GEN MODEL
kernel_size_face_gen = 4
init_channels_face_gen = 16
stride_face_gen = 1
padding_face_gen = 0

# TRAINING
lr = 0.005
epochs = 50

transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

"""
DATASET GENERATION
"""
# Mendeley dataset
# if not MendeleyDataset.csvExists():
#     MendeleyDataset.create_csv('../data/mendeley')
#
# mendeleyDatasetTrain = MendeleyDataset(csv_file='../data/mendeley/mendeley.csv',
#                                        root_dir='../data/mendeley',
#                                        healthy_only=True,
#                                        plants=list([MendeleyPlant.ALSTONIA_SCHOLARIS]),
#                                        validation=False,
#                                        validationSplit=validationSplit,
#                                        transform=transform)
#
# mendeleyDatasetTest = MendeleyDataset(csv_file='../data/mendeley/mendeley.csv',
#                                       root_dir='../data/mendeley',
#                                       healthy_only=True,
#                                       plants=list([MendeleyPlant.ALSTONIA_SCHOLARIS]),
#                                       validation=True,
#                                       validationSplit=validationSplit,
#                                       transform=transform)

# Plant Village dataset
if not PlantVillage.csvExists():
    PlantVillage.create_csv("../data/plantvillage/cherry")

plantVillageTrain = PlantVillage(csv_file='../data/plantvillage/cherry/cherry.csv',
                                 root_dir='../data/plantvillage/cherry',
                                 status=list([PlantVillageStatus.HEALTHY]),
                                 validation=False,
                                 validation_split=validationSplit,
                                 transform=transform)

plantVillageVal = PlantVillage(csv_file='../data/plantvillage/cherry/cherry.csv',
                               root_dir='../data/plantvillage/cherry',
                               status=list([PlantVillageStatus.HEALTHY]),
                               validation=True,
                               validation_split=validationSplit,
                               transform=transform)

plantVillageTest = PlantVillage(csv_file='../data/plantvillage/cherry/cherry.csv',
                                root_dir='../data/plantvillage/cherry',
                                status=list(PlantVillageStatus),
                                validation=True,
                                validation_split=validationSplit,
                                transform=transform)

# Example images
# fig = plt.figure()
#
# for i in range(len(plantVillageTrain)):
#     sample = plantVillageTrain[i]
#
#     sample = np.transpose(sample, (1, 2, 0))
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample)
#
#     if i == 3:
#         plt.show()
#         break

trainloader = get_training_dataloader(plantVillageTrain, batch_size)
testloader = get_test_dataloader(plantVillageVal, batch_size)

"""
MODEL TRAINING
"""

# initialize the model
baseModel = BaseModel.ConvVAE().to(device)

poolBaseModel = PoolBaseModel.ConvVAE()

faceGenModel = FaceGenModel.ConvVAE(kernel_size_face_gen, init_channels_face_gen, stride_face_gen, padding_face_gen,
                                    image_channels=3)

batchNormBaseModel = BatchNormBaseModel.ConvVAE()

net, best_epoch = start(net=baseModel,
                        trainloader=trainloader,
                        trainset=plantVillageTrain,
                        testloader=testloader,
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
