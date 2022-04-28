"""
condecon.py - Module with utilities for construction/deconstruction of Tensor
lists for FloeMessages.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from typing import Dict

import numpy as np

from floes.core import FloesParameters
from floes.proto.floes_pb2 import Tensor, Parameters, BoolDict


def parameters_to_proto(parameters: FloesParameters) -> Parameters:
    """
    Constructs a protobuf `Parameters` object from a FloesParameters object.

    Args:
        parameters: `FloesParameters`
            The `FloesParameters` (OrderedDict) object to convert to protobuf.
    Returns:
        `Parameters`
            Protobuf `Parameters` object that can be used in gRPC communication.
    """
    keys_list = list(parameters.keys())
    weights = list(parameters.values()) # should be a Numpy array
    weights = [Tensor(shape=a.shape, data=a.tobytes()) for a in weights]
    return Parameters(keys=keys_list, weights=weights)


def proto_to_parameters(parameters: Parameters) -> FloesParameters:
    """
    Deconstructs a protobuf `Parameters` object into a FloesParameters object.

    Args:
        parameters: `Parameters`
            The protobuf parameters that will be converted to FloesParameters.
    Returns:
        `FloesParameters`
            The parameters of the model as a `FloesParameters` object
            (OrderedDict).
    """
    keys = parameters.keys
    tlist = parameters.weights

    # NOTE: assumes np.float32. Tensor protobuf message should probably be
    # changed to allow for a type definition to override this.
    received_data = [np.frombuffer(t.data, dtype=np.float32) for t in tlist]
    shapes = [tuple(t.shape) for t in tlist]
    for i in range(len(shapes)):
        received_data[i] = np.reshape(received_data[i], shapes[i])
    
    return FloesParameters(zip(keys, received_data))


def booldict_to_proto(booldict: Dict[str, bool]) -> BoolDict:
    """
    Constructs a protobuf `BoolDict` object from a `str, bool` dictionary.

    Args:
        booldict: `Dict[str, bool]`
            The dictionary of `str, bool` pairs to be converted to protobuf.
    Returns:
        `BoolDict`
            The protobuf `BoolDict` object created from the given dictionary.
    """
    keys_list = list(booldict.keys())
    vals_list = list(booldict.values())

    return BoolDict(keys=keys_list, values=vals_list)


def proto_to_booldict(booldict: BoolDict) -> Dict[str, bool]:
    """
    Deconstructs a protobuf `BoolDict` object into a dictionary of `str, bool`
    pairs.

    Args:
        booldict: `BoolDict`
            The protobuf BoolDict that will be converted a Python dictionary.
    Returns:
        `Dict[str, bool]`
            The `str, bool` dictionary created.
    """
    keys = booldict.keys
    values = booldict.values
    
    return {k: values[i] for i, k in enumerate(keys)}
