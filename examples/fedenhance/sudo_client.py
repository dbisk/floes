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
from tqdm import tqdm

import mixture_consistency
import dataset_loaders
import floes.client
from groupcomm_sudormrf_v2 import GroupCommSudoRmRf
from sudo_common import * # PEP8: bad style


class SuDOClient(floes.client.PyTorchClient):
    
    def __init__(self, model: GroupCommSudoRmRf):
        super().__init__(model)
    
    def set_device(self, device):
        self.device = device
    
    def fake_train_test(self):
        self.model = fake_train_one_sample(self.model, self.device)
    
    def train(self, **kwargs):
        """
        Training loop for one epoch of training.
        """
        # parse keyword arguments
        if 'lr' not in kwargs:
            kwargs['lr'] = 0.005
        if 'dataloader' not in kwargs:
            raise NameError(f"'dataloader' not in kwargs for train().")
        
        # create local optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs['lr'])
        self.model = self.model.to(self.device)
        self.model.train()

        # local stats
        local_loss_sum = 0

        # loop through the data
        with tqdm(kwargs['dataloader']) as pbar:
            for cnt, data in enumerate(pbar):
                optimizer.zero_grad()

                # data format:
                # <speech> (batch, time_samples),
                # <noise> (batch, time_samples)
                input_active_speakers, noise_wavs, extra_noise_waves = data
                input_active_speakers = input_active_speakers.unsqueeze(1)
                input_noises = torch.stack([noise_wavs, extra_noise_waves], 1)

                # create a mask for zeroing out the second noise
                zero_out_mask = (
                    torch.rand([kwargs['bs'], 1]) > kwargs['p_single_mix']
                ).to(torch.float32)

                # zero out mixture
                # equal probability to zero out a noise mixture or the mixture
                # also containing the speaker
                if (torch.rand([1]) < 0.5).item():
                    input_noises[:, 1] = input_noises[:, 1] * zero_out_mask
                else:
                    input_noises[:, 0] = input_noises[:, 0] * zero_out_mask
                    input_active_speakers[:, 0] = \
                        input_active_speakers[:, 0] * zero_out_mask
                
                # send to gpu
                input_active_speakers = input_active_speakers.to(self.device)
                input_noises = input_noises.to(self.device)

                # create mixture of mixtures
                input_mom = input_active_speakers.sum(1, keepdim=True) + \
                            input_noises.sum(1, keepdim=True)
                input_mom = input_mom.to(self.device)
                input_mix_std = input_mom.std(-1, keepdim=True)
                input_mix_mean = input_mom.mean(-1, keepdim=True)
                input_mom = (input_mom - input_mix_mean) / (input_mix_std + 1e-9)

                # forward
                rec_sources_wavs = self.model(input_mom)
                rec_sources_wavs = mixture_consistency.apply(
                    rec_sources_wavs, input_mom
                )

                # if the node is supervised then use the appropriate loss.
                if kwargs['is_supervised']:
                    loss_fn = sup_sisdr(
                        rec_sources_wavs,
                        input_active_speakers,
                        input_noises,
                        input_mom,
                        use_activity_masks=False
                    )
                else:
                    loss_fn = unsup_mixit(
                        rec_sources_wavs,
                        input_active_speakers,
                        input_noises,
                        input_mom
                    )
                
                # backward
                loss_fn.backward()
                if 'clip_grad_norm' in kwargs:
                    if kwargs['clip_grad_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            kwargs['clip_grad_norm']
                        )
                
                # optimize
                optimizer.step()

                # statistics
                local_loss_sum += loss_fn.detach().item()

                if cnt % 10 == 0:
                    # update pbar
                    pbar.set_description_str(
                        f'Avg loss: {local_loss_sum / (cnt + 1):.4f}.'
                    )
        return


def main(args):
    hparams = vars(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create the client
    client = SuDOClient(GroupCommSudoRmRf())
    client.set_device(device)

    # set address information
    address = args.address

    # get the dataloader
    dataloader = dataset_loaders.client_loader(args.data_dir, False, hparams)

    # start the GRPC connection and client loop
    # this will continue until server indicates it is done
    print("Awaiting signal from server to begin")
    trained_model = floes.client.start_client(
        client, address,
        dataloader=dataloader,
        lr=0.001,
        clip_grad_norm=5.0,
        is_supervised=False,
        bs=args.batch_size,
        p_single_mix=0.0
    )

    # for metrics, just print them
    print("Server indicates training done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        type=str,
        default="localhost:50051",
        help="The address of the server to connect to."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The root directory path of the dataset."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1
    )
    parser.add_argument(
        "--audio_timelength",
        type=float,
        default=4.0
    )
    parser.add_argument(
        "--available_speech_percentage",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=16000
    )
    
    args = parser.parse_args()
    main(args)
