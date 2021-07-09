import matplotlib
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from src.data.MendeleyDataset import MendeleyDataset, MendeleyPlant
from src.train import start
from src.dataset import *
from src.models import BaseModel, FaceGenModel, PoolBaseModel, BatchNormBaseModel

"""MatPlotLib"""
matplotlib.style.use('ggplot')
matplotlib.use('TkAgg')
plt.ion()

"""CUDA"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
PARAMETERS
"""
# DATASET
validationSplit = 0.1
batch_size = 1
img_width = 64
img_height = 64


# FACE GEN MODEL
kernel_size_face_gen = 4
init_channels_face_gen = 16
stride_face_gen = 1
padding_face_gen = 0

# TRAINING
lr = 0.005
epochs = 100

transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
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

# Example images
# fig = plt.figure()
#
# for i in range(len(mendeleyDatasetTrain)):
#     sample = mendeleyDatasetTrain[i]
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

trainloader = get_training_dataloader(mendeleyDatasetTrain, batch_size)
testloader = get_test_dataloader(mendeleyDatasetTest, batch_size)

"""
MODEL TRAINING
"""

# initialize the model
baseModel = BaseModel.ConvVAE().to(device)

poolBaseModel = PoolBaseModel.ConvVAE()

batchNormBaseModel = BatchNormBaseModel.ConvVAE()

net = start(net=baseModel,
            trainloader=trainloader,
            trainset=mendeleyDatasetTrain,
            testloader=testloader,
            testset=mendeleyDatasetTest,
            epochs=epochs,
            lr=lr,
            device=device)

"""
MODEL VISUALIZATION
"""
# visualization(net, testloader, device)
