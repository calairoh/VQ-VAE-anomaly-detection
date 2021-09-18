import torch
import torch.nn as nn
import torch.nn.functional as F


# define a Conv VAE
class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()

        init_channels = 8
        image_channels = 3
        latent_dim = 32
        dense_dim = 128

        # encoder
        self.enc1 = nn.Conv2d(image_channels, init_channels, kernel_size=4, stride=1, padding=0)
        self.enc2 = nn.Conv2d(init_channels, init_channels * 2, kernel_size=4, stride=1, padding=0)
        self.enc3 = nn.Conv2d(init_channels * 2, init_channels * 4, kernel_size=4, stride=1, padding=0)
        self.enc4 = nn.Conv2d(init_channels * 4, init_channels * 8, kernel_size=4, stride=1, padding=0)
        self.enc5 = nn.Conv2d(init_channels * 8, init_channels * 16, kernel_size=4, stride=1, padding=0)
        self.maxPool = nn.MaxPool2d(2)


        # fully connected layers for learning representations
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(init_channels * 8, dense_dim)
        self.fc1 = nn.Linear(12800, dense_dim)
        self.fc_z = nn.Linear(dense_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, dense_dim)

        # decoder
        self.dec1 = nn.ConvTranspose2d(dense_dim, init_channels * 8, kernel_size=4, stride=1, padding=0)
        self.dec2 = nn.ConvTranspose2d(init_channels * 8, init_channels * 4, kernel_size=4, stride=4, padding=0)
        self.dec3 = nn.ConvTranspose2d(init_channels * 4, init_channels * 2, kernel_size=4, stride=4, padding=0)
        self.dec4 = nn.ConvTranspose2d(init_channels * 2, init_channels, kernel_size=4, stride=2, padding=1)
        self.dec5 = nn.ConvTranspose2d(init_channels, image_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.maxPool(x)
        x = F.relu(self.enc2(x))
        x = self.maxPool(x)
        x = F.relu(self.enc3(x))
        x = self.maxPool(x)
        x = F.relu(self.enc4(x))
        x = self.maxPool(x)
        x = F.relu(self.enc5(x))

        batch, _, _, _ = x.shape
        # x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.flatten(x)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        x = self.fc_z(hidden)
        x = self.fc2(x)
        x = x.view(-1, 128, 1, 1)

        # decoding
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        reconstruction = torch.sigmoid(self.dec5(x))

        return reconstruction
