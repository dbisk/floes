"""
layerwise_server.py - implementation of the server for the layer-wise training
experiment with MNIST. Uses PyTorch.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from floes.core import FloesParameters
import floes.server
import floes.strategy

from layerwise_common import MNISTModel


def main():
    address = '[::]:50051'

    # create the model
    model = MNISTModel()

    # convert model to FloesParameters
    model = FloesParameters(
        {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    )

    # start the server
    floes.server.start_server(
        model,
        address,
        3,
        floes.strategy.LayerwiseFedAvg()
    )


if __name__ == '__main__':
    main()
