"""
condecon.py - Module with utilities for construction/deconstruction of Tensor
lists for FloeMessages.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from typing import List

import numpy as np

from floe.proto.floe_pb2 import Tensor


def construct_from_alist(alist: List[np.ndarray]) -> List[Tensor]:
    """
    Constructs a list of protobuf `Tensor` objects from a list of `np.ndarray`.

    Args:
        alist: `List[np.ndarray]`
            The list of `np.ndarray`s that will become a list of protobuf
            `Tensor`s.
    Returns:
        `List[Tensor]`
            List of protobuf `Tensor`s that can be directly used as the
            `weights` parameter of a `FloeMessage` construction.
    """
    weights = [Tensor(shape=a.shape, data=a.tobytes()) for a in alist]
    return weights


def deconstruct_from_tlist(tlist: List[Tensor]) -> List[np.ndarray]:
    """
    Deconstructs a list of protobuf `Tensor`s into a list of `np.ndarray`s.

    Args:
        tlist: `List[Tensor]`
            The list of protobuf `Tensor`s that will become a list of
            `np.ndarray`s.
    Returns:
        `List[np.ndarray]`
            List of `np.ndarray`s that represent the model.
    """
    # NOTE: assumes np.float32. Tensor protobuf message should probably be
    # changed to allow for a type definition to override this.
    received_data = [np.frombuffer(t.data, dtype=np.float32) for t in tlist]
    shapes = [tuple(t.shape) for t in tlist]
    for i in range(len(shapes)):
        received_data[i] = np.reshape(received_data[i], shapes[i])
    return received_data
