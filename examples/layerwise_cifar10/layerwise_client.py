"""
layerwise_client.py - implementation of the client for the layer-wise training
experiment with CIFAR10. Uses PyTorch.

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

from layerwise_common import CIFAR10Model, evaluate_model

GACCS = []
LACCS = []

class CIFAR10Client(floes.client.PyTorchClient):

    def __init__(self, train_layers: Dict[str, bool]):
        super().__init__(CIFAR10Model())
        self.train_layers = train_layers

    def train(
        self, train_dataloader: DataLoader, criterion: Callable, device: str,
        evaluate: bool = False
    ):
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
        if evaluate:
            metrics = evaluate_model(self.model)
            GACCS.append(metrics['accuracy'])
            print(metrics)

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
        
        # optional, but useful to see effect of aggregation
        # evaluate the model at the end of this training round
        if evaluate:
            metrics = evaluate_model(self.model)
            LACCS.append(metrics['accuracy'])
            print(metrics)


def main(train_layer_index: int, addr: str, evaluate: bool):
    train_layers = {
        'conv1.weight': (train_layer_index == 0),
        'conv1.bias': (train_layer_index == 0),
        'fc1.weight': (train_layer_index == 1),
        'fc1.bias': (train_layer_index == 1),
        'fc2.weight': (train_layer_index == 2),
        'fc2.bias': (train_layer_index == 2),
    }

    # load and prepare dataset
    train_data = datasets.CIFAR10(
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
    client = CIFAR10Client(train_layers)

    # start the GRPC connection and client loop
    # this will continue until server indicates it is done
    print("Awaiting signal from server to begin.")
    print(f"Configured to train layer(s): {train_layer_index}")
    trained_client = floes.client.start_layerwise_client(
        client, addr, train_layers, 
        train_dataloader=train_dataloader,
        criterion=criterion,
        device=device,
        evaluate=evaluate
    )

    if evaluate:
        # for metrics, we can evaluate the final model on the client side
        print("Server indicates training done. Evaluating new model...")
        metrics = evaluate_model(trained_client.model)
        GACCS.append(metrics['accuracy'])
        print(metrics)

        # save the metrics
        import pickle
        with open(f'{train_layer_index}_gaccs.pkl', 'wb') as f:
            pickle.dump(GACCS, f)
        with open(f'{train_layer_index}_laccs.pkl', 'wb') as f:
            pickle.dump(LACCS, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'train_layer_index',
        help="The index of the layer in the network that will be trained.",
        type=int
    )
    parser.add_argument(
        '--evaluate',
        action='store_true'
    )
    parser.add_argument(
        '--addr',
        type=str,
        default='localhost:50051'
    )
    args = parser.parse_args()
    main(args.train_layer_index, args.addr, args.evaluate)
