"""
layerwise_server.py - implementation of the server for the layer-wise training
experiment with MNIST. Uses PyTorch.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import argparse
from collections import OrderedDict

import torch

from floes.core import FloesParameters
import floes.server
import floes.strategy

from layerwise_common import MNISTModel


def main(addr: str, rounds: int, await_termination: bool = False):
    # create the model
    model = MNISTModel()

    # convert model to FloesParameters
    params = FloesParameters(
        {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    )

    # start the server
    params = floes.server.start_server(
        params,
        addr,
        rounds,
        floes.strategy.LayerwiseFedAvg(),
        await_termination=await_termination
    )

    state_dict = {k: torch.Tensor(v) for k, v in params.items()}
    model.load_state_dict(OrderedDict(state_dict), strict=True)
    return model


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
    args = parser.parse_args()

    main(args.address, args.rounds)
