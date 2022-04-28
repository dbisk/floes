"""
layerwise_client.py - implementation of the client for the layer-wise training
experiment with MNIST. Uses PyTorch.

This script should be started with the index of the layer to train passed in as
a parameter. Currently, only training one layer is supported per client.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import argparse
from typing import Dict, Callable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import floes.client

from layerwise_common import MNISTModel, evaluate_model


class MNISTClient(floes.client.PyTorchClient):

    def __init__(self, train_layers: Dict[str, bool]):
        super().__init__(MNISTModel())
        self.train_layers = train_layers

    def train(self, train_dataloader: DataLoader, criterion: Callable, device: str):
        """
        Performs one epoch of training.

        Args:
            train_dataloader: `DataLoader`
                Torch dataloader for the training dataset.
            criterion: `Callable`
                Loss function to be optimized.
            device: `str`
                The device to train on. Either `cpu` or `cuda`.
        """
        # set the model layers to be trainable based on initialization params
        for k, v in self.train_layers.items():
            for name, param in self.model.named_parameters():
                if name == k:
                    param.requires_grad = v
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

        self.model = self.model.to(device)
        self.model.train()
        with tqdm(train_dataloader, unit='batches') as tbatch:
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

                # update progress bar
                count = count % 10 + 1
                if count == 1:
                    tbatch.set_postfix_str(f'Loss: {loss.item():.3f}')


def main(train_layer_index):
    train_layers = {
        'conv1.weight': (train_layer_index == 0),
        'conv1.bias': (train_layer_index == 0),
        'fc1.weight': (train_layer_index == 1),
        'fc1.bias': (train_layer_index == 1),
        'fc2.weight': (train_layer_index == 2),
        'fc2.bias': (train_layer_index == 2),
    }

    # load and prepare dataset
    train_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # define dataloader for the dataset
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # create the floes client
    client = MNISTClient(train_layers)

    # set address information
    address = 'localhost:50051'

    # start the GRPC connection and client loop
    # this will continue until server indicates it is done
    print("Awaiting signal from server to begin.")
    print(f"Configured to train layer(s): {train_layer_index}")
    trained_client = floes.client.start_layerwise_client(
        client, address, train_layers, 
        train_dataloader=train_dataloader,
        criterion=criterion,
        device=device
    )

    # for metrics, we can evaluate the final model on the client side
    print("Server indicates training done. Evaluating new model...")
    metrics = evaluate_model(trained_client.model)
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'train_layer_index',
        help="The index of the layer in the network that will be trained.",
        type=int
    )
    train_layer_index = parser.parse_args().train_layer_index
    main(train_layer_index)

