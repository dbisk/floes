"""!
@brief Pytorch dataloader for Libri+FSD50k federated dataset.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import os
import numpy as np
import glob2

import abstract_dataset
from scipy.io import wavfile
import warnings
from time import time

EPS = 1e-8


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for WHAM source separation and speech enhancement tasks.

    Example of kwargs:
        root_dirpath='/mnt/data/wham', task='enh_single',
        split='tr', sample_rate=8000, timelength=4.0,
        normalize_audio=False, n_samples=0, zero_pad=False
    """
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        warnings.filterwarnings("ignore")
        self.kwargs = kwargs

        self.available_speech_p = self.get_arg_and_check_validness(
            'available_speech_percentage', known_type=float,
            extra_lambda_checks=[lambda y: 0 < y <= 1])

        self.zero_pad = self.get_arg_and_check_validness(
            'zero_pad', known_type=bool)

        self.augment = self.get_arg_and_check_validness(
            'augment', known_type=bool)

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['train', 'test', 'val'])

        self.sample_rate = self.get_arg_and_check_validness(
            'sample_rate', known_type=int, choices=[16000])

        self.root_path = self.get_arg_and_check_validness(
            'root_dirpath', known_type=str,
            extra_lambda_checks=[lambda y: os.path.lexists(y)])
        self.dataset_dirpath = self.get_path()

        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)

        self.time_samples = int(self.sample_rate * self.timelength)

        # Create the indexing for the dataset.
        available_speaker_ids = os.listdir(self.dataset_dirpath)
        available_speaker_ids_ints = sorted(map(int, available_speaker_ids))

        self.sources_paths = []
        self.extra_noise_paths = []
        for speaker_id in available_speaker_ids_ints:
            this_dirpath = os.path.join(self.dataset_dirpath, str(speaker_id))
            noise_paths = glob2.glob(this_dirpath+'/noise/*.wav')
            speech_paths = glob2.glob(this_dirpath+'/speech/*.wav')
            assert len(noise_paths) == len(speech_paths)

            number_of_available_sp_uttncs = int(
                self.available_speech_p * len(speech_paths))
            self.extra_noise_paths += noise_paths[number_of_available_sp_uttncs:]
            noise_paths = noise_paths[:number_of_available_sp_uttncs]
            speech_paths = speech_paths[:number_of_available_sp_uttncs]

            this_sources_info = [{
                'noise_path': noise_path,
                'speech_path': speech_path
            } for (noise_path, speech_path) in zip(noise_paths, speech_paths)]
            self.sources_paths += this_sources_info

    def get_path(self):
        path = os.path.join(self.root_path, self.split)
        if os.path.lexists(path):
            return path
        else:
            raise IOError('Dataset path: {} not found!'.format(path))

    def safe_pad(self, tensor_wav):
        if self.zero_pad and tensor_wav.shape[0] < self.time_samples:
            appropriate_shape = tensor_wav.shape
            padded_wav = torch.zeros(
                list(appropriate_shape[:-1]) + [self.time_samples],
                dtype=torch.float32)
            padded_wav[:tensor_wav.shape[0]] = tensor_wav
            return padded_wav[:self.time_samples]
        else:
            return tensor_wav[:self.time_samples]

    def __len__(self):
        return len(self.sources_paths)

    def get_padded_tensor(self, numpy_waveform):
        max_len = len(numpy_waveform)
        rand_start = 0
        if self.augment and max_len > self.time_samples:
            rand_start = np.random.randint(0, max_len - self.time_samples)
        noise_waveform = numpy_waveform[
                         rand_start:rand_start + self.time_samples]
        np_noise_wav = np.array(noise_waveform)
        noise_wav = torch.tensor(np_noise_wav, dtype=torch.float32)
        return self.safe_pad(noise_wav)

    def __getitem__(self, idx):
        if self.augment:
            the_time = int(np.modf(time())[0] * 100000000)
            np.random.seed(the_time)

        example_sources_paths = self.sources_paths[idx]

        _, noise_waveform = wavfile.read(example_sources_paths['noise_path'])
        _, speech_waveform = wavfile.read(example_sources_paths['speech_path'])
        noise_wav = self.get_padded_tensor(noise_waveform)
        speech_wav = self.get_padded_tensor(speech_waveform)

        # Also draw a random noise waveform if available.
        if self.extra_noise_paths:
            file_idx = np.random.randint(0, len(self.extra_noise_paths))
            _, extra_noise_waveform = wavfile.read(
                self.extra_noise_paths[file_idx])
            extra_noise_wav = self.get_padded_tensor(extra_noise_waveform)
        else:
            extra_noise_wav = torch.zeros_like(noise_wav)

        return speech_wav, noise_wav, extra_noise_wav

    def get_generator(self, batch_size=4, shuffle=True, num_workers=4):
        generator_params = {'batch_size': batch_size,
                            'shuffle': shuffle,
                            'num_workers': num_workers,
                            'drop_last': True}
        return torch.utils.data.DataLoader(self, **generator_params,
                                           pin_memory=True)


def test_generator():
    dataset_root_p = '/mnt/data/ChunkFedEnhance/'
    batch_size = 3
    sample_rate = 16000
    timelength = 4.0
    speaker_ids = [x for x in range(2)]
    time_samples = int(sample_rate * timelength)
    max_abs_snr = 5.
    data_loader = Dataset(
        root_dirpath=dataset_root_p,
        speaker_ids=speaker_ids,
        available_speech_percentage=0.5,
        split='test', sample_rate=sample_rate, timelength=timelength,
        zero_pad=True, augment=True)
    generator = data_loader.get_generator(batch_size=batch_size, num_workers=1)

    for speech_wavs, noise_wavs, extra_noise_wavs in generator:
        assert speech_wavs.shape == (batch_size, time_samples)
        assert noise_wavs.shape == (batch_size, time_samples)
        assert extra_noise_wavs.shape == (batch_size, time_samples)
        break

if __name__ == "__main__":
    test_generator()
