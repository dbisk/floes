"""
run_eval_all_chkpts.py - Runs evaluation on the test dataset for LibriFSD50K
for all checkpoints in the given directory.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import argparse
import os
import pickle

import torch

from groupcomm_sudormrf_v2 import GroupCommSudoRmRf
from sudo_common import evaluate_model


def main(args):
    hparams = vars(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results_list = []
    for chkpt_path in os.listdir(hparams['checkpoint_dir']):
        if chkpt_path[-4:] == '.pkl':
            # get the learning round number
            chkpt_number = int(chkpt_path[10:-4])

            # retrieve model
            model = GroupCommSudoRmRf()
            with open(os.path.join(hparams['checkpoint_dir'], chkpt_path), 'rb') as f:
                params = pickle.load(f)
            state_dict = {k: torch.Tensor(v) for k, v in params.items()}
            model.load_state_dict(state_dict)

            # perform evaluation
            results_list.append(
                (chkpt_number, evaluate_model(model, hparams['data_dir'], hparams, device))
            )
    
    with open(hparams['save_path'], 'wb') as f:
        pickle.dump(results_list, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help="The directory with the checkpoints."
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="The path to the root directory of LibriFSD50K."
    )
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help="The path to save the resulting stats to. Uses pickle."
    )
    parser.add_argument(
        '--batch_size',
        default=6,
        type=int
    )
    parser.add_argument(
        '--fs',
        type=float,
        default=16000
    )
    parser.add_argument(
        "--audio_timelength",
        type=float,
        default=4.0
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1
    )
    args = parser.parse_args()
    main(args)
