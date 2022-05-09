"""
disjoint_client.py - Implementation of a client that trains on a subset of the
MNIST dataset that only includes two of the MNIST classes. Note that this
implementation actually downloads the full dataset to each client, so may need
to be tweaked for very storage limited devices.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import argparse
from typing import Callable, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import floes.client

from torch_mnist_common import MNISTModel, evaluate_model


class MNISTClient(floes.client.PyTorchClient):

    def __init__(self):
        super().__init__(MNISTModel())
    
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
            print(evaluate_model(self.model))

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

                # update the progress bar
                count = count % 10 + 1
                if count == 1:
                    tbatch.set_postfix_str(f'Loss: {loss.item():.3f}')
        
        # optional, but useful to see effect of aggregation
        # evaluate the model at the end of this training round
        if evaluate:
            print(evaluate_model(self.model))


def load_and_prepare_dataset(
    keep_idxs: List, fpath: str = 'data', bs: int = 10
):
    train_data = datasets.MNIST(
        root=fpath,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    # only keep the specified indices
    idxs = np.array([np.array(train_data.targets) == i for i in keep_idxs])
    idxs = np.logical_or.reduce(idxs, axis=0)
    train_data.targets = train_data.targets[idxs]
    train_data.data = train_data.data[idxs]
    assert train_data.data.shape[0] == len(train_data.targets) # sanity check

    # define the dataloader for the dataset and return it
    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True)
    return train_dataloader


def main(keep_idxs: List, addr: str, evaluate: bool, bs: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataloader = load_and_prepare_dataset(keep_idxs, bs=bs)

    # define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # create the floes client
    client = MNISTClient()

    # set the server's address
    address = addr

    # start the GRPC connection and client loop
    # this will continue until server indicates it is done
    print("Awaiting signal from server to begin")
    trained_client = floes.client.start_client(
        client,
        address,
        train_dataloader=train_dataloader,
        criterion=criterion,
        device=device,
        evaluate=evaluate
    )

    if evaluate:
        # for metrics, we can evaluate the final model on the client side
        print("Server indicates training done. Evaluating new model...")
        metrics = evaluate_model(trained_client.model, device)
        print(metrics)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'keep_idxs',
        type=int,
        nargs='+',
        help='The indicies of the training labels to keep for this client.'
    )
    parser.add_argument(
        '--addr',
        type=str,
        default='localhost:50051'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true'
    )
    parser.add_argument(
        '--bs',
        type=int,
        default=10,
        help='The batch size to use with this client. Default 10.'
    )
    args = parser.parse_args()
    main(args.keep_idxs, args.addr, args.evaluate, args.bs)
