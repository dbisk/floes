"""
server.py - module representing the server in the federated learning scenario.
All servers should extend the `Server` class.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import copy
from datetime import datetime
import logging
import queue
from typing import Dict, List

import numpy as np

from .client_stub import ClientStub
from floes.core import floes_logger, FloesParameters
from floes.strategy import Strategy, UnweightedFedAvg
from floes.proto.floes_pb2 import FloesMessage


class Server:
    """
    Generic class that represents a federated learning server.
    """
    def __init__(self, max_clients: int = 100, min_clients: int = 2):
        self.max_clients = max_clients
        self.min_clients = min_clients
        self._clients: Dict[str, ClientStub] = {}
        self._contributors: List[str] = []
        self.model: FloesParameters = None
        self.model_timestamp: str = None
        self.model_queue: queue.Queue = queue.Queue()
        self.broadcast_message: FloesMessage = None
        self._strategy = UnweightedFedAvg()
    
    @property
    def num_clients(self) -> int:
        return len(self._clients)
    
    @property
    def num_contributors(self) -> int:
        return len(self._contributors)
    
    def get_model(self) -> FloesParameters:
        return self.model
    
    def get_model_timestamp(self) -> str:
        return self.model_timestamp
    
    def set_model(self, model: FloesParameters):
        """
        Sets the global model for the server. Creates a deep copy of the given
        list of ndarrays, so the original model passed into this function can
        be modified without affecting the stored global model.

        Args:
            model: `FloesParameters`
                The FloesParameters object representing the model.
        """
        self.model = copy.deepcopy(model)
        self.model_timestamp = str(datetime.now())
    
    def set_strategy(self, strategy: Strategy):
        """
        Sets the strategy this Server uses to aggregate the weights.

        Args:
            strategy: `Strategy`
                The strategy this Server will use when aggregating the weights.
        """
        self._strategy = strategy
    
    def source_model_from_clients(self, timeout: float = 300.0):
        """
        Sources a new model from the registered contributors the Server is
        aware of. The server will wait for a maximum of `timeout` seconds
        between successive contributor updates.

        Args:
            timeout: `float`, default 300
                The amount of time to wait for the clients before returning, if
                not all of the registered contributors are ready by that time.
        """

        model_list = []
        counter = 0
        
        while True:
            if (counter >= self.num_contributors):
                # we've gotten a model from every contributor
                break

            try:
                model, ts = self.model_queue.get(block=True, timeout=timeout)
            except queue.Empty:
                # we exceeded our timeout
                floes_logger.logger.write(
                    'Timeout exceeded waiting for contributors. Continuing '
                    'without remaining contributors.',
                    logging.WARNING
                )
                break

            # double check whether this model has the right timestamp
            if ts == self.model_timestamp:
                # we got a model back, so lets keep it in our model list
                model_list.append(model)
                counter += 1
        
        # raise an error if we didn't get anything back
        if counter == 0:
            raise TimeoutError(
                'No client contribution received before timeout.'
            )

        # send the model list to our strategy
        new_model = self._strategy.aggregate(model_list)

        # set the server model to the new model
        self.set_model(new_model)
    
    def broadcast(self, msg: FloesMessage):
        """
        Broadcasts a message to all registered clients. Also sets the Server's
        current `broadcast_message` property.

        Args:
            msg: FloesMessage
                The message to send to all registered clients.
        """
        for id in self._clients:
            self._clients[id].write(msg)
        
        self.broadcast_message = msg

    def add_client(self, client: ClientStub) -> bool:
        """
        Adds the provided Client to the Server. Returns True if successful,
        otherwise raises an error.

        Args:
            client: ClientStub
                The ClientStub representing the client to be added.
        """
        if client.id in self._clients:
            raise NameError(f'Requested client already registered.')
        
        if len(self._clients) >= self.max_clients:
            raise ValueError(f'Maximum clients already reached.')
        
        self._clients[client.id] = client

        # check if the client is a contributor, in which case add to list
        if client.is_contributor:
            self._contributors.append(client.id)
        
        # broadcast the current broadcast message to this client
        if self.broadcast_message is not None:
            client.write(self.broadcast_message)

        floes_logger.logger.write(f'Added client {client.id}.')
        return True

    def remove_client(self, client_id: str) -> bool:
        """
        Removes a client (labelled by ID) from the Server.

        Args:
            client_id: str
                The string client id for the client to be removed.
        """
        if client_id in self._clients:
            self._clients[client_id].close()
            if self._clients[client_id].is_contributor:
                self._contributors.remove(client_id)
            del self._clients[client_id]
            floes_logger.logger.write(f'Removed client {client_id}.')
            return True
        else:
            floes_logger.logger.write(
                f'Failed to remove client {client_id} as it does not exist.',
                logging.WARNING
            )
            return False
