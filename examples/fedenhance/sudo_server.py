"""
sudo_server.py - Server example for a SuDORM-RF model [1]. See 
https://arxiv.org/abs/2007.06833.

[1]: Efthymios Tzinis, Zhepei Wang, and Paris Smaragdis. "Sudo rm -rf:
Efficient Networks for Universal Audio Source Separation". MLSP 2020.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import argparse
import os

import torch
from floes.core.params import FloesParameters

import floes.server
import floes.strategy

from groupcomm_sudormrf_v2 import GroupCommSudoRmRf


def main(args):
    # create the model
    model = GroupCommSudoRmRf()
    params = FloesParameters(
        {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    )

    # start the server
    params = floes.server.start_server(
        model=params,
        address=args.address,
        rounds=args.rounds,
        strategy=floes.strategy.UnweightedFedAvg(),
        await_termination=False,
        save_dir=args.save_model_dir,
        min_clients=args.min_clients,
        max_clients=args.max_clients
    )

    # save the final model
    state_dict = {k: torch.Tensor(v) for k, v in params.items()}
    torch.save(state_dict, os.path.join(args.save_model_dir, 'final_model_sd.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", "-a",
        type=str,
        default='[::]:50051',
        help='The address the server will be broadcasting on.'
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=3,
        help="The number of federated learning rounds to run."
    )
    parser.add_argument(
        '--save_model_dir',
        type=str,
        default='.',
        help="The directory to save the models from each FL round to."
    )
    parser.add_argument(
        '--min_clients',
        type=int,
        default=2,
        help="The minimum number of clients before a federated round begins."
    )
    parser.add_argument(
        '--max_clients',
        type=int,
        default=100,
        help="The maximum number of clients the federated server will accept."
    )

    args = parser.parse_args()
    main(args)
