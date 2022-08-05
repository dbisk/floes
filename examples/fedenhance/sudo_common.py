"""
sudo_common.py - Common functions between the SuDO-RM-RF clients and server.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import copy
from typing import Dict

import torch

import asteroid_sdr as asteroid_sdr_lib


def evaluate_model(model: torch.nn.Module) -> Dict:
    """
    This function currently does not evaluate the model in any performance
    terms. It only determines whether the model outputs an appropriately shaped
    output.

    Args:
        model: `torch.nn.Module`
            The trained model to evaluate.
    Returns:
        Dict
            Dictionary of relevant evaluation metrics
    """
    model.eval()
    random_input = torch.rand(3, 1, 8000)
    estimated_sources = model(random_input)
    out_shape = estimated_sources.shape
    return {'output_shape': out_shape}


def fake_train_one_sample(model: torch.nn.Module, device = 'cpu'):
    """
    Sanity check test function that makes sure that the model can perform a
    backward pass with an arbitrary loss function and random (properly sized)
    data.

    Args:
        model: `torch.nn.Module`
            The model that will be "trained".
        device: `torch.device`
            The device to train on.
    Returns:
        torch.nn.Module
            The model "trained" on one data sample.
    """
    # load random data as the dataset
    dummy_inputs = torch.rand(1, 1, 8000)
    dummy_targets = torch.rand(1, 2, 8000)

    # define a totally fake loss function & optimizer
    criterion = lambda x, y: torch.mean(torch.abs(x - y))
    optimizer = torch.optim.Adam(model.parameters())

    dummy_inputs = dummy_inputs.to(device)
    dummy_targets = dummy_targets.to(device)
    model = model.to(device)
    model.train()

    # backwards pass
    optimizer.zero_grad()
    estimated_sources = model(dummy_inputs)
    loss = criterion(estimated_sources, dummy_targets)
    loss.backward()
    optimizer.step()

    return model


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


def snr_loss(est, ref, eps=1e-9):
    error_element = est - ref
    error = torch.sum(error_element**2, dim=-1)
    power_ref = torch.sum(ref**2, dim=-1)
    return 10. * torch.log10(error + eps + 0.001 * power_ref)


def sup_sisdr(rec_sources_wavs, input_active_speakers, input_noises,
              input_mom, use_activity_masks):
    supervised_ast_loss = asteroid_sdr_lib.PITLossWrapper(
        asteroid_sdr_lib.pairwise_neg_sisdr, pit_from='pw_mtx'
    )
    ref_speech = normalize_tensor_wav(input_active_speakers)
    ref_noises = normalize_tensor_wav(input_noises)

    if use_activity_masks:
        ref_speech_powers = torch.sum(input_active_speakers ** 2, dim=-1,
                                      keepdim=True)
        input_mom_powers = torch.sum(input_mom ** 2, dim=-1, keepdim=True)
        mixtures_input_snr = 10. * torch.log10(
            ref_speech_powers / (input_mom_powers + 1e-9))
        ref_speech_activity_mask = mixtures_input_snr.ge(0.001)

        ref_noise_powers = torch.sum(input_noises ** 2, dim=-1,
                                     keepdim=True)
        mixtures_input_snr = 10. * torch.log10(
            ref_noise_powers / (input_mom_powers + 1e-9))
        ref_noise_activity_mask = mixtures_input_snr.ge(0.001)

        speech_error = ref_speech_activity_mask * torch.clamp(
            asteroid_sdr_lib.pairwise_neg_sisdr(
                rec_sources_wavs[:, 0:1], ref_speech), min=-50., max=50.)

        noise_error = ref_noise_activity_mask * torch.clamp(
            supervised_ast_loss(
                rec_sources_wavs[:, 1:], ref_noises), min=-50., max=50.)
    else:
        speech_error = torch.clamp(
            asteroid_sdr_lib.pairwise_neg_sisdr(
                rec_sources_wavs[:, 0:1], ref_speech), min=-50.,
            max=50.)

        noise_error = torch.clamp(
            supervised_ast_loss(
                rec_sources_wavs[:, 1:], ref_noises), min=-50., max=50.)

    return speech_error.mean() + noise_error


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


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
