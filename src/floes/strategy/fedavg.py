"""
fedavg.py - Federated Averaging implementation for FLoES.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import copy
from typing import List

import numpy as np

from floes.core import FloesParameters

from .strategy import Strategy


class UnweightedFedAvg(Strategy):

    def aggregate(self, params_list: List[FloesParameters]) -> FloesParameters:
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
