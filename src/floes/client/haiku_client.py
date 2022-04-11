"""
haiku_client.py - module containing a client implementation using Haiku.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from typing import Dict, Tuple

import haiku as hk
import jax.numpy as jnp
import numpy as np

from floes.core import FloesParameters

from .client import Client


class HaikuClient(Client):
    """
    A class representing a federated client that uses Haiku (and JAX) as the ML
    library. This class **DOES NOT** implement the `train()` function required
    of all clients, so users will need to extend this class with another
    subclass that implements that function based on their specific use case.

    Args:
        params: haiku.Params
            The Haiku parameters associated with this client. 
        net: haiku.Transformed
            The transformed Haiku function that the `hk_params` are associated
            with. (e.g. the Haiku module after being converted to a pure
            function using `hk.transform(net_fn)`.)
        
    """
    def __init__(
        self,
        params: hk.Params,
        net: hk.Transformed,
        model_timestamp: str = None
    ):
        super().__init__(model_timestamp=model_timestamp)
        self.params = params
        self.net = net
        self._keymap: Dict[str, Tuple[str, str]] = self._create_keymap(params)
    
    def _create_keymap(self, params: hk.Params):
        """
        Creates the keymap for mapping Haiku's nested dict structure to FLoES's
        single dict structure.
        """
        keymap = {}
        for top_key, lower_dict in params.items():
            for lower_key in lower_dict.keys():
                keymap[f'{top_key}:{lower_key}'] = (top_key, lower_key)
        return keymap

    def get_parameters(self) -> FloesParameters:
        params_dict = {}
        for top_key, lower_dict in self.params.items():
            for lower_key, weights in lower_dict.items():
                params_dict[f'{top_key}:{lower_key}'] = np.array(weights)
        return FloesParameters(params_dict)
    
    def set_parameters(self, params: FloesParameters):
        for layer, weights in params.items():
            top_key, lower_key = self._keymap[layer]
            self.params[top_key][lower_key] = jnp.array(weights)
