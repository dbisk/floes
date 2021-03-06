"""
client.py
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from floes.core import FloesParameters


class Client(ABC):
    """
    Class encapsulation for clients. A `Client` object itself should never be
    instantiated, as it is an abstract class.
    """

    def __init__(self, model_timestamp: str = None):
        self.model_timestamp = model_timestamp

    def set_model_timestamp(self, model_timestamp: str):
        self.model_timestamp = model_timestamp

    @abstractmethod
    def get_parameters(self) -> FloesParameters:
        raise NotImplementedError
    
    @abstractmethod
    def set_parameters(self, parameters: FloesParameters):
        raise NotImplementedError
    
    @abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError
