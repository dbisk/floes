"""
grpc_client.py
"""

from typing import Dict, Tuple, Iterator
import logging

from floes.core import FloesParameters
import floes.core.condecon as condecon
import floes.core.floes_logger as floes_logger
from floes.proto.floes_pb2 import FloesMessage
from floes.proto.floes_pb2_grpc import FloesServiceStub


class GRPCCLient(object):

    def __init__(self):
        self._is_contributor = False
        self._is_subscriber = False

    def get_model_from_server(
        self, 
        stub: FloesServiceStub
    ) -> Tuple[FloesParameters, str]:
        """
        Gets the most recent model from the server, as well as the timestamp
        of that model for versioning purposed.

        Args:
            stub: `FloesServiceStub`
                The `FloesServiceStub` connecting the client to the server.
        Returns:
            `tuple` of the received model as `FloesParameters`, and the new
            model timestamp as `str`.
        """
        
        # request the newest model from the server
        req = FloesMessage(msg='GetModel:NEWEST')
        response = stub.GetModel(req)

        # get the new model timestamp
        new_model_timestamp = response.timestamp

        # deconstruct the response
        received_model = condecon.proto_to_parameters(response.params)
        
        return received_model, new_model_timestamp
    
    def register_as_contributor(
        self,
        stub: FloesServiceStub
    ) -> Iterator[FloesMessage]:
        """
        Registers this client as a collaborator in the federated learning
        scenario. This means that this client will be offering models after
        each training round to the server.

        Args:
            stub: `FloesServiceStub`
                The `FloesServiceStub` connecting the client to the server
        """

        if self._is_contributor:
            # we are already a collaborator so we don't do anything new
            floes_logger.logger.write(
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
        stub: FloesServiceStub
    ) -> Iterator[FloesMessage]:
        """
        Registers this client as a subscriber in the federated learning
        scenario. This means that the client will listen for server broadcast
        messages, but will not contribute model updates to the server.

        Args:
            stub: `FloesServiceStub`
                The `FloesServiceStub` connecting the client to the server.
        """

        if self._is_subscriber:
            # we are already a subscriber so we don't do anything new
            floes_logger.logger.write(
                f'This client already registered as a subscriber.',
                logging.WARNING
            )
            return

        # set flags for this client
        self._is_subscriber = True
        
        # construct the FloesMessage to be sent to the server indicating our
        # intention to join
        msg = 'Subscribe:CONTRIBUTOR' if self._is_contributor else 'Subscribe:SUBSCRIBER'
        request = FloesMessage(msg=msg)
        
        # tell the server we are joining
        server_message_iterator = stub.Subscribe(request)
        return server_message_iterator
    
    def contribute_model_to_server(
        self,
        stub: FloesServiceStub,
        params: FloesParameters,
        timestamp: str,
        layers: Dict[str, bool] = None,
    ):
        """
        Offers the given model parameters to the server. Can only be called if
        this client is registered as a collaborator.

        Args:
            stub: `FloesServiceStub`
                The `FloesServiceStub` connecting the client to the server.
            params: `FloesParameters`
                The model parameters being offered to the server.
            timestamp: `str`
                The model timestamp being offered to the server.
        """

        if not self._is_contributor:
            floes_logger.logger.write(
                f'This client is not a registered contributor',
                logging.ERROR
            )
            return

        # construct the FloesMessage
        weights = condecon.parameters_to_proto(params)
        msg = 'OK'
        if layers:
            request = FloesMessage(
                msg=msg, 
                params=weights, 
                timestamp=timestamp,
                trainlayers=condecon.booldict_to_proto(layers)
            )
        else:
            request = FloesMessage(
                msg=msg, 
                params=weights, 
                timestamp=timestamp
            )

        # contribute the model to the server
        response = stub.ContributeModel(request)
    