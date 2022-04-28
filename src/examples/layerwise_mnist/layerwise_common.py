"""
layerwise_common.py - common module for the layer-wise experiment with MNIST.

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
    Very simple model for MNIST classification. 3 layers: 1 convolutional and 2
    fully connected layers. 
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


def evaluate_model(model: torch.nn.Module) -> Dict:
    """
    Evaluates the given model on the MNIST test dataset. Downloads the dataset
    to `data/` if not already downloaded.

    Args:
        model: `torch.nn.Module`
            The model to evaluate. Will be placed into `eval` mode after this
            function is finished.
    Returns:
        dict
            Dictionary of evaluation stats, such as accuracy and loss.
    """
    # load and prepare the MNIST test dataset
    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
