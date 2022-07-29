"""
sudo_client.py - Client example for a SuDORM-RF model [1]. See 
https://arxiv.org/abs/2007.06833.

[1]: Efthymios Tzinis, Zhepei Wang, and Paris Smaragdis. "Sudo rm -rf:
Efficient Networks for Universal Audio Source Separation". MLSP 2020.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import argparse

import torch

import floes.client
from groupcomm_sudormrf_v2 import GroupCommSudoRmRf
from sudo_common import evaluate_model


class SuDOClient(floes.client.PyTorchClient):
    
    def __init__(self, model: GroupCommSudoRmRf):
        super().__init__(model)
    
    def set_device(self, device):
        self.device = device
    
    def train(self):
        """
        Fake training using random data.
        """
        # load random data as the dataset
        dummy_inputs = torch.rand(1, 1, 8000)
        dummy_targets = torch.rand(1, 2, 8000)

        # define a totally fake loss function & optimizer
        criterion = lambda x, y: torch.mean(torch.abs(x - y))
        optimizer = torch.optim.Adam(self.model.parameters())

        device = self.device
        dummy_inputs = dummy_inputs.to(device)
        dummy_targets = dummy_targets.to(device)
        self.model = self.model.to(device)

        self.model.train()

        # backwards pass
        optimizer.zero_grad()
        estimated_sources = self.model(dummy_inputs)
        loss = criterion(estimated_sources, dummy_targets)
        loss.backward()
        optimizer.step()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create the client
    client = SuDOClient(GroupCommSudoRmRf())
    client.set_device(device)

    # set address information
    address = args.address

    # start the GRPC connection and client loop
    # this will continue until server indicates it is done
    print("Awaiting signal from server to begin")
    trained_model = floes.client.start_client(client, address)

    # for metrics, just print them
    print("Server indicates training done. Evaluating new model...")
    metrics = evaluate_model(trained_model.model.to('cpu'))
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        type=str,
        default="localhost:50051",
        help="The address of the server to connect to."
    )
    parser.add_argument(
        "--evaluate",
        action='store_true',
        help="Whether to evaluate the model after federated training is over."
    )
    
    args = parser.parse_args()
    main(args)
