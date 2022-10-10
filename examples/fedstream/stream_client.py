"""
stream_client.py - the main client script for the FedStream example.

@author Dean Biskup
@email dbiskup2@illinois.edu
@org University of Illinois, Urbana-Champaign Audio Group
"""

import argparse

import torch
from tqdm import tqdm

import asteroid_sdr as asteroid_sdr_lib
import floes.client
from groupcomm_sudormrf_v2 import GroupCommSudoRmRf
import mixture_consistency
from rec_dataloader import MicrophoneDataset


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


def unsup_mixit(rec_sources_wavs, input_active_speakers, input_noises,
                input_mom):
    ref_mix_1 = normalize_tensor_wav(
        input_active_speakers + input_noises[:, 0:1])
    ref_mix_2 = normalize_tensor_wav(input_noises[:, 1:2])

    ref_mix1_powers = torch.sum(
        (input_active_speakers + input_noises[:, 0:1]) ** 2,
        dim=-1, keepdim=True)
    input_mom_powers = torch.sum(input_mom ** 2, dim=-1, keepdim=True)
    mixtures_input_snr = 10. * torch.log10(
        ref_mix1_powers / (input_mom_powers + 1e-9))
    ref_mix1_activity_mask = mixtures_input_snr.ge(0.001)

    ref_mix2_powers = torch.sum(input_noises[:, 1:2]**2, dim=-1, keepdim=True)
    mixtures_input_snr = 10. * torch.log10(
        ref_mix2_powers / (input_mom_powers + 1e-9))
    ref_mix2_activity_mask = mixtures_input_snr.ge(0.001)

    er_00 = ref_mix1_activity_mask * torch.clamp(
        asteroid_sdr_lib.pairwise_neg_sisdr(
            rec_sources_wavs[:, 0:1] + rec_sources_wavs[:, 1:2], ref_mix_1),
        min=-50., max=50.)
    er_01 = ref_mix2_activity_mask * torch.clamp(
        asteroid_sdr_lib.pairwise_neg_sisdr(
            rec_sources_wavs[:, 2:3], ref_mix_2), min=-50., max=50.)

    er_10 = ref_mix1_activity_mask * torch.clamp(
        asteroid_sdr_lib.pairwise_neg_sisdr(
            rec_sources_wavs[:, 0:1] + rec_sources_wavs[:, 2:3], ref_mix_1),
        min=-50., max=50.)
    er_11 = ref_mix2_activity_mask * torch.clamp(
        asteroid_sdr_lib.pairwise_neg_sisdr(
            rec_sources_wavs[:, 1:2], ref_mix_2), min=-50., max=50.)

    errors = torch.cat([er_00 + er_01,
                        er_10 + er_11], 1)
    return torch.mean(torch.min(errors, 1)[0])


class SuDOClient(floes.client.PyTorchClient):
    
    def __init__(self, model: GroupCommSudoRmRf):
        super().__init__(model)
    
    def set_device(self, device):
        self.device = device
    
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
                input_active_speakers, input_noises = data
                input_active_speakers = input_active_speakers.unsqueeze(1)
                
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create the client
    client = SuDOClient(GroupCommSudoRmRf())
    client.set_device(device)

    # create the dataset and dataloader
    num_samples = args.audio_seconds_per_round / args.audio_timelength
    dset = MicrophoneDataset.get_default(
        num_samples, args.audio_timelength, args.fs
    )
    dataloader = dset.get_generator()

    # start the GRPC connection and client loop
    # this will continue until server indicates it is done
    print("Connecting to server...")
    trained_model = floes.client.start_client(
        client, args.address,
        dataloader=dataloader,
        lr=0.001,
        clip_grad_norm=5.0,
    )

    # finished!
    print("Server indicates training done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        type=str,
        required=True,
        help="The address of the server to connect to."
    )
    parser.add_argument(
        "--audio_timelength",
        type=float,
        default=4.0
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=16000
    )
    parser.add_argument(
        "--audio_seconds_per_round",
        type=int,
        default=100,
        help="The number of seconds of audio to constitute one federated learning round."
    )
    
    args = parser.parse_args()
    main(args)
