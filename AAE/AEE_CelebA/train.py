"""Training script."""
import torch
from torch import nn, optim
from tqdm import tqdm
from itertools import chain
# custom module
from loss import reconstruction_loss
from models import Encoder, Decoder, Discriminator
from utils import show_grid_tensor


def train_AAE(train_loader, latent_dim, device,
              epochs=1):
    """Training the AAE."""
    n_images = 3
    fixed_noise = torch.rand(n_images, latent_dim).to(device)

    encoder = Encoder(device, in_channels=3, latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)
    discriminator = Discriminator(latent_dim=latent_dim).to(device)

    # Defining optimizer
    lr = 0.0002
    generator_optimizer = optim.Adam(chain(encoder.parameters(),
                                           decoder.parameters()),
                                     lr=lr)

    discriminator_optimizer = optim.Adam(discriminator.parameters(),
                                         lr=lr)
    # loss function
    discriminator_loss = nn.BCELoss()  # sigmoid input, target

    # Training loop
    for epoch in range(epochs):

        loop = tqdm(train_loader,
                    total=len(train_loader),
                    leave=True)

        encoder.train()
        decoder.train()

        for x, _ in loop:

            real_labels = torch.ones([x.shape[0]]).to(device)
            fake_labels = torch.zeros([x.shape[0]]).to(device)
            fool_labels = real_labels

            """
            # Reconstrcution phase
            # For our generator
            """
            generator_optimizer.zero_grad()
            # x_1d = x.reshape([-1, 1*28*28]).to(device)
            latent_var = encoder(x.to(device))
            x_pred = decoder(latent_var)

            rc_loss = reconstruction_loss(x.to(device), x_pred, x.shape[0])

            # fool the discrminator
            dis_fool = discriminator(latent_var)
            # fool_labels=real_labels
            fool_loss = discriminator_loss(dis_fool.reshape(-1), fool_labels)

            # fool_loss.backward()
            # rc_loss.backward()
            generator_loss = fool_loss + rc_loss
            generator_loss.backward()
            generator_optimizer.step()

            """
            # regularization phase, for our discriminator
            """
            discriminator_optimizer.zero_grad()

            # sampling from true prior, real latent variables
            # consider it  comes from normal distribution then
            real_latent_var = torch.randn([x.shape[0], latent_dim]).to(device)
            dis_true = discriminator(real_latent_var)

            # fake latent variables
            dis_fake = discriminator(latent_var.detach())

            dis_loss = 0.5 * (
                discriminator_loss(dis_true.reshape(-1), real_labels)
                + discriminator_loss(dis_fake.reshape(-1), fake_labels))

            dis_loss.backward()
            discriminator_optimizer.step()

            # progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(reconstruct_loss=rc_loss.item(),
                             disc_loss=dis_loss.item(),
                             f_loss=fool_loss.item())

        """
        Checking on the fixed noise for each epoch
        """
        if epoch % 3 == 0 or epoch == 9:
            decoder.eval()

            with torch.no_grad():
                x_random = decoder(fixed_noise).detach()

            gen_x = x_random.reshape(n_images, 128, 128, 3)

            ax1 = show_grid_tensor(gen_x.cpu(), n_images)

            ax1.show()
            # break
