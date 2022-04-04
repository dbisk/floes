"""
grpc_server.py - interface between the `Server` class and the actual gRPC
connection between the server and the clients.

See `https://grpc.io/docs/languages/python/basics/` for docs and details
regarding implementation of this module.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import logging
from typing import Iterator

import grpc

from .client_stub import ClientStub
import floes.core.condecon as condecon
from floes.core.connection import ServerClientConnection
import floes.core.floes_logger as floes_logger
from floes.server.server import Server
import floes.proto.floes_pb2_grpc as floes_pb2_grpc
from floes.proto.floes_pb2 import FloesMessage, Parameters


def _register_client(
    server: Server,
    client: ClientStub,
    context: grpc.ServicerContext
) -> bool:
    """Registers a client with the given server."""
    if server.add_client(client):
        # register a callback function that closes the client in case the RPC
        # connection is terminated for some reason
        def rpc_termination_callback():
            floes_logger.logger.write(
                f'RPC Termination Callback called on {client.id}.'
            )
            server.remove_client(client.id)
        
        context.add_callback(rpc_termination_callback)
        return True
    else:
        return False


class FloesServiceServicer(floes_pb2_grpc.FloesServiceServicer):
    
    def __init__(self):
        self.server = Server()
    
    def GetModel(
        self,
        request: FloesMessage,
        context: grpc.ServicerContext
    ) -> FloesMessage:
        """
        Servicer function for GetModel. See `floes.proto` for function
        declarations.
        """

        peer = context.peer()
        floes_logger.logger.write(
            f'GetModel request received from {peer}.',
            level=logging.INFO
        )

        if request.msg == 'GetModel:NEWEST':
            model = self.server.get_model()
            model = condecon.parameters_to_proto(model)
            model_timestamp = self.server.get_model_timestamp()
            response = FloesMessage(
                msg='OK',
                params=model,
                timestamp=model_timestamp
            )
        else:
            response = FloesMessage(msg='GetModel:REJECTED')
        return response
    
    def ContributeModel(
        self,
        request: FloesMessage,
        context: grpc.ServicerContext
    ) -> FloesMessage:
        """
        Servicer function for ContributeModel. See `floes.proto` for function
        declarations.
        """

        peer = context.peer()
        floes_logger.logger.write(
            f'Client {peer} is offering a model.',
            level=logging.INFO
        )

        # check the timestamp to make sure the client has the newest version
        ts = request.timestamp
        if ts == self.server.get_model_timestamp():
            # correct timestamp, let's deconstruct the message
            model = condecon.proto_to_parameters(request.params)

            # add the model to the server's queue
            self.server.model_queue.put((model, ts))

            # logging and response
            floes_logger.logger.write(
                f'Model from Client {peer} accepted with valid timestamp.',
                level=logging.INFO
            )
            response = FloesMessage(msg="ContributeModel:ACCEPTED")
        else:
            # incorrect timestamp, reject
            floes_logger.logger.write(
                f'Model from Client {peer} REJECTED with invalid timestamp.',
                level=logging.WARNING
            )
            response = FloesMessage(msg="ContributeModel:REJECTED")
        
        return response

    def Subscribe(
        self,
        request: FloesMessage,
        context: grpc.ServicerContext
    ) -> Iterator[FloesMessage]:
        """
        Servicer function for Subscribe. See `floes.proto` for function
        declarations.
        """

        # get the role of the incoming client
        job = request.msg
        job = 'CONTRIBUTOR' if job == 'Subscribe:CONTRIBUTOR' else 'SUBSCRIBER'

        peer = context.peer()
        floes_logger.logger.write(f'Client {peer} joined as {job}.')

        # register the client with the server
        conn = ServerClientConnection()
        if job == 'CONTRIBUTOR':
            cstub = ClientStub(peer, conn, is_contributor=True)
        else:
            cstub = ClientStub(peer, conn, is_contributor=False)

        if _register_client(self.server, cstub, context):
            # yield the messages from the server whenever they are ready
            server_message_generator = conn.message_generator()
            while True:
                msg = next(server_message_generator)
                yield msg
