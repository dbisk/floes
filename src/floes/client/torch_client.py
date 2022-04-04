"""
torch_client.py - module containing a client using PyTorch. 
"""

from collections import OrderedDict
from typing import List

import numpy as np
import torch

from floes.core import FloesParameters

from .client import Client


class PyTorchClient(Client):
    """
    A class representing a federated client that uses PyTorch as the machine
    learning library. This class **DOES NOT** implement the `train()` function
    required of all clients, so users will need to extend this class with
    another subclass that implements that function.

    Initialization args:
        model: torch.nn.Module
            The model that this PyTorch client will be operating with.
        model_timestamp: str, optional, default: None
            The starting timestamp of the model. Usually this will be `None`.
    """

    def __init__(self, model: torch.nn.Module, model_timestamp: str = None):
        super().__init__(model_timestamp)
        self.model = model
    
    def get_parameters(self) -> FloesParameters:
        numpy_dict = {
            k: v.cpu().numpy() for k, v in self.model.state_dict().items()
        }
        return FloesParameters(numpy_dict)
    
    def set_parameters(self, params: FloesParameters):
        state_dict = {k: torch.Tensor(v) for k, v in params.items()}
        self.model.load_state_dict(OrderedDict(state_dict), strict=True)
