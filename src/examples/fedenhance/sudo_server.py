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

import floes.server
import floes.strategy

from sudormrf import SuDORMRF


def main():
    address = '[::]:50051'

    # create the model
    model = SuDORMRF()
    model = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # start the server
    # note: this never returns
    floes.server.start_server(
        model,
        address,
        3,
        floes.strategy.UnweightedFedAvg()
    )


if __name__ == '__main__':
    main()
