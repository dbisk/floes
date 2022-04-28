"""
fedavg.py - Federated Averaging implementation for FLoES.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import copy
from typing import List, Dict

import numpy as np

from floes.core import FloesParameters

from .strategy import Strategy


class UnweightedFedAvg(Strategy):

    def aggregate(
        self, params_list: List[FloesParameters], **kwargs
    ) -> FloesParameters:
        """
        Performs a naive, unweighted federated averaging on the passed in list
        of `FloesParameters`. Essentially just computes the average for each
        parameter.

        Args:
            parameters_list: `List[FloesParameters]`
                The list of parameters (which are themselves OrderedDicts) that
                will be averaged to create the global model.
        Returns:
            `FloesParameters`
                The aggregated, averaged parameters.
        """
        # start with the 0th node as initialization
        final_params = copy.deepcopy(params_list[0])

        # find the average parameters for each layer
        for k in final_params.keys():
            final_params[k] = np.mean([p[k] for p in params_list], axis=0)
        
        return final_params

class LayerwiseFedAvg(Strategy):
    
    def aggregate(
        self,
        params_list: List[FloesParameters],
        layers_list: List[Dict[str, bool]]
    ) -> FloesParameters:
        # start with the 0th node as initialization
        final_params = copy.deepcopy(params_list[0])

        # average parameters for each layer among nodes that trained that layer
        for k in final_params.keys():
            client_indices = []
            for i, d in enumerate(layers_list):
                if d[k] == True:
                    client_indices.append(i)
            this_layer_param_list = [params_list[idx] for idx in client_indices]
            final_params[k] = np.mean(
                [p[k] for p in this_layer_param_list], axis=0
            )

        return final_params
