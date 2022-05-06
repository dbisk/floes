"""
haiku_mnist_server.py - Implementation of a Haiku server using FLoES. Trains a
simple network on the MNIST classification dataset.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import haiku as hk
import jax
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets

from floes.core import FloesParameters
import floes.server
import floes.strategy

from haiku_mnist_common import net_fn, numpy_collate, TorchToJAXTransform, Batch

def main():
    address = '[::]:50051'

    # load the test dataset
    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=TorchToJAXTransform()
    )

    test_dataloader = DataLoader(test_data, batch_size=10, collate_fn=numpy_collate, shuffle=False)
    
    # initialize the network; note we draw an input to get shapes
    net = hk.without_apply_rng(hk.transform(net_fn))
    batch = next(iter(test_dataloader))
    batch = {"image": batch[0], "label": batch[1]}
    params = net.init(jax.random.PRNGKey(42), batch)
    
    # convert the network to something FLoES understands (FloesParameters or
    # OrderedDict)
    floes_params = {}
    for top_key, lower_dict in params.items():
        for lower_key, weights in lower_dict.items():
            floes_params[f'{top_key}:{lower_key}'] = np.array(weights)
    floes_params = FloesParameters(floes_params)
    
    # start the server
    # note: this never returns
    floes.server.start_server(
        floes_params,
        address,
        3,
        floes.strategy.UnweightedFedAvg()
    )


if __name__ == '__main__':
    main()
