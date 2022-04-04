"""
torch_mnist_client.py - Implementation of a PyTorch client using FLoES. Trains
a simple network on the MNIST classification dataset.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

import floes.client


class MNISTClient(floes.client.PyTorchClient):
    
    def __init__(self):
        super().__init__(MNISTModel())

    def set_train_parameters(self, trainloader, criterion, device):
        self.trainloader = trainloader
        self.criterion = criterion
        self.device = device

    def train(self):
        """
        perform one epoch of training
        """
        trainloader = self.trainloader
        criterion = self.criterion
        device = self.device

        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

        self.model = self.model.to(device)

        self.model.train()
        with tqdm(trainloader, unit='batches') as tbatch:
            count = 0 # only needed for progress bar
            for X, y in tbatch:
                X, y = X.to(device), y.to(device)

                # compute prediction error
                pred = self.model(X)
                loss = criterion(pred, y)

                # backpropogation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update the progress bar
                count = count % 10 + 1
                if count == 1:
                    tbatch.set_postfix_str(f'Loss: {loss.item():.3f}')
        

class MNISTModel(torch.nn.Module):
    
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


def main():
    # load and prepare toy dataset (MNIST)
    train_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # only keep a certain subset of the training dataset for this client
    fraction = 0.1
    idxs = np.random.randint(0, len(train_data), size=int(len(train_data) * fraction))
    train_data = Subset(train_data, idxs)

    # define dataloaders for the datasets
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # create the floes client
    client = MNISTClient()

    # set the training info for the client
    client.set_train_parameters(train_dataloader, criterion, device)
    
    # set address information
    address = 'localhost:50051'

    # start the GRPC connection and client loop
    # this will continue until server indicates it is done
    print("Awaiting signal from server to begin")
    trained_model = floes.client.start_torch_client(client, address)

    # for metrics, we can evaluate the final model on the client side
    print("Server indicates training done. Evaluating new model...")
    metrics = evaluate_model(trained_model.model)
    print(metrics)
    

if __name__ == '__main__':
    main()
    