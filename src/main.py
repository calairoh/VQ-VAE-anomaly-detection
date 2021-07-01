"""
Import necessary libraries to create a variational autoencoder
The code is mainly developed using the PyTorch library
"""
from src.model import VAE
from src.training import training
from src.visualization import visualization
from dataset import *

"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = create_train_dataset(128)
test_dataset = create_test_dataset()

net = training(VAE(), device, train_dataset)

visualization(net, test_dataset, device)
