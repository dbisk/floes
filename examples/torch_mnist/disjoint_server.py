"""
disjoint_server.py - An implementation of the federated server for training
using clients that each have a disjoint subset of the MNIST training dataset.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import argparse

from floes.core import FloesParameters
import floes.server
import floes.strategy

from torch_mnist_common import *


def main(addr: str, rounds: int):
    # create the model
    model = MNISTModel()

    # convert model to FloesParameters (or OrderedDict works too)
    model = FloesParameters(
        {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    )

    # start the server
    # note: this never returns
    floes.server.start_server(
        model,
        addr,
        rounds,
        floes.strategy.UnweightedFedAvg()
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        type=str,
        default='[::]:50051',
        help='The address the server will be broadcasting on.'
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="The number of federated learning rounds to run."
    )
    
    main(parser.parse_args().address)
