"""MNIST dataset loader."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def train_test_loader(batch_size=64, device='cpu'):
    """
    Get the mnist train and test loader from pytorch dataset.

    Args:
        batch_size: `int`
        device:
    Returns:
        Tuple consisting of train and test loader.
        Tuple(Dataloader, Dataloader)
    """
    train_set = datasets.MNIST(root="pytorch_dataset/",
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())
    test_set = datasets.MNIST(root="pytorch_dataset/",
                              train=False,
                              download=True,
                              transform=transforms.ToTensor())
    # dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=True)
    return (train_loader, test_loader)
