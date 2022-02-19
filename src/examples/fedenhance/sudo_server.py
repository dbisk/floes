"""
sudo_server.py - Server example for a SuDORM-RF model [1]. See 
https://arxiv.org/abs/2007.06833.

[1]: Efthymios Tzinis, Zhepei Wang, and Paris Smaragdis. "Sudo rm -rf:
Efficient Networks for Universal Audio Source Separation". MLSP 2020.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import torch

import floe.server
import floe.strategy


class SUDOModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


def main():
    address = '[::]:50051'

    # create the model
    model = SUDOModel()
    model = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # start the server
    # note: this never returns
    floe.server.start_server(
        model,
        address,
        3,
        floe.strategy.UnweightedFedAvg()
    )


if __name__ == '__main__':
    main()
