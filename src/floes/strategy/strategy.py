"""
strategy.py - module containing the definition for the abstract base class
representing an aggregation strategy.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from abc import ABC, abstractmethod
from typing import List

from floes.core import FloesParameters


class Strategy(ABC):
    """
    An abstract class representing a federated aggregation strategy. The
    abstract class requires just one method: `aggregate`, which takes in a list
    of parameters and performs the aggregation algorithm on them, returning a
    single set of parameters.
    """

    @abstractmethod
    def aggregate(
        self, parameters: List[FloesParameters], **kwargs
    ) -> FloesParameters:
        raise NotImplementedError

