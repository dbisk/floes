"""
sudo_common.py - Common functions between the SuDO-RM-RF clients and server.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from typing import Dict

import torch


def evaluate_model(model: torch.nn.Module) -> Dict:
    """
    This function currently does not evaluate the model in any performance
    terms. It only determines whether the model outputs an appropriately shaped
    output.

    Args:
        model: `torch.nn.Module`
            The trained model to evaluate.
    Returns:
        Dict
            Dictionary of relevant evaluation metrics
    """
    model.eval()
    random_input = torch.rand(3, 1, 8000)
    estimated_sources = model(random_input)
    out_shape = estimated_sources.shape
    return {'output_shape': out_shape}
