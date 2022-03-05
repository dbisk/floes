"""
fedavg.py - Federated Averaging implementation for FLoES.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import copy
from typing import List

import numpy as np

from .strategy import Strategy


class UnweightedFedAvg(Strategy):

    def aggregate(self, parameters_list: List[List[np.ndarray]]):
        """
        Performs a naive, unweighted federated averaging on the passed in list
        of `np.ndarray`s. Essentially just computes the average for each
        parameter.

        Args:
            parameters_list: `List[List[np.ndarray]]`
                The list of parameters (which are themselves a list of arrays)
                that will be averaged to create the global model.
        Returns:
            `List[np.ndarray]`
                The aggregated, averaged parameters.
        """
        
        num_layers = len(parameters_list[0])
        num_nodes = len(parameters_list)

        # start with the 0th node as initialization
        final_params = copy.deepcopy(parameters_list[0])

        for i in range(num_layers):
            # sum across all the nodes
            for n in range(1, num_nodes):
                final_params[i] = parameters_list[n][i] + final_params[i]
            
            # divide by number of nodes
            final_params[i] = final_params[i] / float(num_nodes)
        
        return final_params
