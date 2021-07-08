import torch
import torch.nn as nn
import torch.nn.functional as F


# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self, kernel_size, init_channels, image_channels, latent_dim):
        super(ConvVAE, self).__init__()

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels,
            out_channels=init_channels * 2,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels * 2,
            out_channels=init_channels * 4,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels * 4,
            out_channels=init_channels * 8,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.enc5 = nn.Conv2d(
            in_channels=init_channels * 8,
            out_channels=init_channels * 16,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(init_channels * 16, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_log_var = nn.Linear(32, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)

        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=init_channels * 8,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels * 8,
            out_channels=init_channels * 4,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels * 4,
            out_channels=init_channels * 2,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels * 2,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_channels,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.dec6 = nn.ConvTranspose2d(
            in_channels=init_channels,
            out_channels=image_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)

        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        #x = F.relu(self.dec5(x))
        reconstruction = torch.sigmoid(self.dec6(x))

        return reconstruction, mu, log_var
