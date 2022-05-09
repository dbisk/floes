"""
torch_mnist_common.py - common definitions used across the PyTorch MNIST
examples.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class MNISTModel(torch.nn.Module):
    """
    A torch.nn.Module subclass representing the simple model used for MNIST
    classification. This model is a 3 layer model, with a single convolutional
    layer and two fully connected layers. The output is a single vector of
    length 10, holding the probability the input image is of that digit.
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(32 * 26 * 26, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)


def evaluate_model(model: torch.nn.Module, device: str) -> Dict:
    """
    Evaluates the given model on the MNIST test dataset. It is assumed that the
    model outputs a vector of size 10 which represents the probabilities that
    each image is of that digit.

    Args:
        model: `torch.nn.Module`
            The model to evaluate.
        device: `str`
            The device to put the model and data on ('cuda' or 'cpu').
    Returns:
        Dictionary with some metrics of the evaluation, such as loss and 
        accuracy.
    """
    # load and prepare the MNIST test dataset
    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # perform the evaluation loop
    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size

    return {'accuracy': 100. * correct, 'loss': test_loss}
