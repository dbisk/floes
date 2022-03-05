"""
torch_client.py - module containing a client using PyTorch. 
"""

from collections import OrderedDict
from typing import List

import numpy as np
import torch

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
    
    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
