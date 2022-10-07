"""
hw_interface.py - hardware interfacing code for using the ReSpeaker Microphone
arrays on the Raspberry Pi. This is used to get live speech data.

@author Dean Biskup
@email dbiskup2@illinois.edu
@org University of Illinois, Urbana-Champaign Audio Group
"""

import numpy as np
import sounddevice as sd


def record_audio(
    sample_rate: int,
    length: float,
    channels: int = 1
) -> np.ndarray:
    """
    Records audio from a microphone of length `length`, in seconds.
    
    Args:
        sample_rate: `int`
            The sample rate to record the audio at.
        length: `float`
            The length of time, in seconds, to record.
    Returns:
        `np.ndarray`
            Numpy array of the recorded data, as `float`.
    """
    arr = sd.rec(
        int(length * sample_rate),
        samplerate=sample_rate,
        channels=channels
    )
    # rec runs in background, but we need to wait until recording is finished
    sd.wait()
    return arr.flatten().astype(np.float32)
