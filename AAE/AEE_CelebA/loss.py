"""Loss functions for the model."""
# import torch
import torch.nn.functional as F


def reconstruction_loss(features, decoded, batch_size):
    """
    Reconstruction loss, original image vs decoded image loss.

    1st calculte the pixel wise mse then sum for single image
    and finally mean for all the iamges in the batch.
    """
    pixel_wise = F.mse_loss(decoded, features, reduction='none')
    pixel_wise = pixel_wise.view(batch_size, -1).sum(axis=1)

    return pixel_wise.mean()

# discriminator_loss = nn.BCELoss() # sigmoid input, target
