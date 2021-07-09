import torch
import torch.nn as nn


def sample_z(mean, logvar):
    stddev = torch.exp(0.5 * logvar)
    noise = torch.randn(stddev.size())
    return (noise * stddev) + mean


class ConvVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_latent = 32

        #ENCODER
        self.batchNorm0 = nn.BatchNorm2d(3, eps=1e-5)
        self.enc1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32, eps=1e-5)
        self.leakyRelu1 = nn.LeakyReLU()
        self.enc2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64, eps=1e-5)
        self.leakyRelu2 = nn.LeakyReLU()

        # REPARAMETRIZATION
        self.z_mean = nn.Linear(32, self.n_latent)
        self.z_var = nn.Linear(32, self.n_latent)
        self.z_develop = nn.Linear(self.n_latent, 64)

        # DECODER
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0)
        self.batchNormDec1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.dec2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1)
        self.sig = nn.Sigmoid()

    def encode(self, x):
        x = self.batchNorm0(x)
        x = self.enc1(x)
        x = self.batchNorm1(x)
        x = self.leakyRelu1(x)
        x = self.enc2(x)
        x = self.batchNorm2(x)
        x = self.leakyRelu2(x)

        #x = x.view(x.size(0), -1)

        mean = self.z_mean(x)
        var = self.z_var(x)

        return mean, var

    def decode(self, z):
        out = self.z_develop(z)
        #out = out.view(z.size(0), 64, self.z_dim, self.z_dim)

        out = self.dec1(out)
        out = self.batchNormDec1(out)
        out = self.relu1(out)
        out = self.dec2(out)
        out = self.sig(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar
