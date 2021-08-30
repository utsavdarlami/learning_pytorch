"""Models need for training the AAE."""
import torch
from torch import nn


class Reshape(nn.Module):
    """Reshape class."""

    def __init__(self, *args):
        """Reshape  init."""
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        """Reshape as h,w,c."""
        x = x.view(self.shape)
        return x


class Trim(nn.Module):
    """Trim ."""

    def __init__(self, h, w):
        """Trim init."""
        super(Trim, self).__init__()
        self.h, self.w = h, w  # 128,128

    def forward(self, x):
        """Trim h and w."""
        return x[:, :, :self.h, :self.w]


class Encoder(nn.Module):
    """Encoder for 128*128 image."""

    def __init__(self, device, in_channels=3, latent_dim=100):
        """Initialize Encoder(device, 3, 100)."""
        super(Encoder, self,).__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(32, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Flatten(),
        )

        self.z_mean = nn.Linear(4096, latent_dim)
        self.z_log_var = nn.Linear(4096, latent_dim)
        self.device = device

    def reparameterize(self, x, z_mu, z_log_var):
        """Z = mu + x * var."""
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        z = z_mu + eps * torch.exp(z_log_var/2.0)
        return z

    def forward(self, x):
        """Finiding the encodings."""
        x = self.encode(x)

        z_mu = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        # get z using mean and var
        encoded = self.reparameterize(x, z_mu, z_log_var)

        return encoded


class Decoder(nn.Module):
    """Decoder."""

    def __init__(self, latent_dim=100):
        """Initialize the Decoder."""
        super(Decoder, self,).__init__()

        self.decode = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            Reshape(-1, 64, 8, 8),
            #
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(32, 3, stride=2, kernel_size=3, padding=1),
            #
            Trim(128, 128),  # 3x129x129 -> 3x128x128
            nn.Sigmoid()
        )

    def forward(self, encodings):
        """Decode the image from the encodings."""
        decoded = self.decode(encodings)
        return decoded


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, latent_dim=100):
        """Initialize discriminator."""
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, latent_var):
        """Fake or real target value."""
        pred = self.model(latent_var)
        return pred
