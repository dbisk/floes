"""
rec_dataloader.py - definiton of a PyTorch Dataloader that records a short
audio snippet as the next data sample.

@author Dean Biskup
@email dbiskup2@illinois.edu
@org University of Illinois, Urbana-Champaign Audio Group
"""

import os
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
import torch.utils.data

import hw_interface


class MicrophoneDataset(torch.utils.data.IterableDataset):
    """
    Dataset class for recording a small snippet from the microphone for speech
    enhancement tasks.

    Allows for the definition of a custom 'recording' function that is called
    when a new sample is requested with __getitem__(). 

    Init Parameters:
        rec_fn: `Callable`
            Recording function that will be called with `meta_args` when a new
            sample is requested.
        meta_args: `dict`
            Dictionary of keyword arguments to be passed into `rec_fn`.
    """
    def __init__(
        self, total_samples: int, rec_fn: Callable, meta_args: Dict,
        voice_only: bool = False
    ):
        super().__init__()
        self.rec_fn = rec_fn
        self.meta_args = meta_args
        self.total_samples = int(total_samples)
        self.noise_1 = np.load(
            os.path.join(Path(__file__).parent, 'noise_4s_1.npy')
        )
        self.noise_2 = np.load(
            os.path.join(Path(__file__).parent, 'noise_4s_2.npy')
        )
        self.voice_only = voice_only

    def __iter__(self):
        for i in range(self.total_samples):
            if self.voice_only:
                # kind of bad code practice here - using assumed 'fs' dict key
                hw_interface.block_until_voice_detected(self.meta_args['fs'])
            rtn = (
                torch.tensor(self.rec_fn(**self.meta_args), dtype=torch.float32),
                torch.tensor(self.noise_1),
                torch.tensor(self.noise_2)
            )
            yield rtn

    def get_generator(self) -> torch.utils.data.DataLoader:
        """
        Returns the DataLoader object that generates samples using this
        dataset.

        Returns:
            `DataLoader`
                The torch dataloader generating samples using this dataset.
        """
        return torch.utils.data.DataLoader(self)
    
    def __len__(self):
        return self.total_samples

    @classmethod
    def get_default(cls, total_samples: int, length: float, fs: int, channels: int = 1) -> 'MicrophoneDataset':
        """
        Creates a default instance of this class, used on Raspberry Pis using
        the `sounddevice` python library. 

        Args:
            length: `float`
                The length of the audio sample to record.
            fs: `int`
                The sample rate of the audio recording.
            channels: `int`, default 1
                The number of channels to record on.
        Returns:
            `MicrophoneDataset`
                An instance of the `MicrophoneDataset` class using a default
                recording function.
        """
        meta_args = {
            'sample_rate': fs,
            'length': length,
            'channels': channels,
        }
        return cls(total_samples, hw_interface.record_audio, meta_args, voice_only=True)
