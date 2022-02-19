"""
grpc_client.py
"""

from typing import List, Tuple, Iterator
import logging

import numpy as np

import floe.core.condecon as condecon
import floe.core.floe_logger as floe_logger
from floe.proto.floe_pb2 import FloeMessage, Tensor
from floe.proto.floe_pb2_grpc import FloeServiceStub


class GRPCCLient(object):

    def __init__(self):
        self._is_contributor = False
        self._is_subscriber = False

    def get_model_from_server(
        self, 
        stub: FloeServiceStub
    ) -> Tuple[List[np.ndarray], str]:
        """
        Gets the most recent model from the server, as well as the timestamp
        of that model for versioning purposed.

        Args:
            stub: `FloeServiceStub`
                The `FloeServiceStub` connecting the client to the server.
        Returns:
            `tuple` of the received model as `List[np.ndarray]`, and the new
            model timestamp as `str`.
        """
        
        # request the newest model from the server
        req = FloeMessage(msg='GetModel:NEWEST')
        response = stub.GetModel(req)

        # get the new model timestamp
        new_model_timestamp = response.timestamp

        # deconstruct the response
        received_model = condecon.deconstruct_from_tlist(response.weights)
        
        return received_model, new_model_timestamp
    
    def register_as_contributor(
        self,
        stub: FloeServiceStub
    ) -> Iterator[FloeMessage]:
        """
        Registers this client as a collaborator in the federated learning
        scenario. This means that this client will be offering models after
        each training round to the server.

        Args:
            stub: `FloeServiceStub`
                The `FloeServiceStub` connecting the client to the server
        """

        if self._is_contributor:
            # we are already a collaborator so we don't do anything new
            floe_logger.logger.write(
                f'This client already registered as a contributor.',
                logging.WARNING
            )
            return
        
        # set flags for this client
        self._is_contributor = True

        # subscribe to server messages as a contributor
        return self.register_as_subscriber(stub)
    
    def register_as_subscriber(
        self,
        stub: FloeServiceStub
    ) -> Iterator[FloeMessage]:
        """
        Registers this client as a subscriber in the federated learning
        scenario. This means that the client will listen for server broadcast
        messages, but will not contribute model updates to the server.

        Args:
            stub: `FloeServiceStub`
                The `FloeServiceStub` connecting the client to the server.
        """

        if self._is_subscriber:
            # we are already a subscriber so we don't do anything new
            floe_logger.logger.write(
                f'This client already registered as a subscriber.',
                logging.WARNING
            )
            return

        # set flags for this client
        self._is_subscriber = True
        
        # construct the FloeMessage to be sent to the server indicating our
        # intention to join
        msg = 'Subscribe:CONTRIBUTOR' if self._is_contributor else 'Subscribe:SUBSCRIBER'
        request = FloeMessage(msg=msg)
        
        # tell the server we are joining
        server_message_iterator = stub.Subscribe(request)
        return server_message_iterator
    
    def contribute_model_to_server(
        self,
        stub: FloeServiceStub,
        params: List[np.ndarray],
        timestamp: str
    ):
        """
        Offers the given model parameters to the server. Can only be called if
        this client is registered as a collaborator.

        Args:
            stub: `FloeServiceStub`
                The `FloeServiceStub` connecting the client to the server.
            params: `List[np.ndarray]`
                The model parameters being offered to the server.
            timestamp: `str`
                The model timestamp being offered to the server.
        """

        if not self._is_contributor:
            floe_logger.logger.write(
                f'This client is not a registered contributor',
                logging.ERROR
            )
            return

        # construct the FloeMessage
        weights = condecon.construct_from_alist(params)
        msg = 'OK'
        request = FloeMessage(
            msg=msg, 
            weights=weights, 
            timestamp=timestamp
        )

        # contribute the model to the server
        response = stub.ContributeModel(request)
    