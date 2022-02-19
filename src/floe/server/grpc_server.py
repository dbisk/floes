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
import floe.core.condecon as condecon
from floe.core.connection import ServerClientConnection
import floe.core.floe_logger as floe_logger
from floe.server.server import Server
import floe.proto.floe_pb2_grpc as floe_pb2_grpc
from floe.proto.floe_pb2 import FloeMessage, Tensor


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
            floe_logger.logger.write(
                f'RPC Termination Callback called on {client.id}.'
            )
            server.remove_client(client.id)
        
        context.add_callback(rpc_termination_callback)
        return True
    else:
        return False


class FloeServiceServicer(floe_pb2_grpc.FloeServiceServicer):
    
    def __init__(self):
        self.server = Server()
    
    def GetModel(
        self,
        request: FloeMessage,
        context: grpc.ServicerContext
    ) -> FloeMessage:
        """
        Servicer function for GetModel. See `floe.proto` for function
        declarations.
        """

        peer = context.peer()
        floe_logger.logger.write(
            f'GetModel request received from {peer}.',
            level=logging.INFO
        )

        if request.msg == 'GetModel:NEWEST':
            model = self.server.get_model()
            model = condecon.construct_from_alist(model)
            model_timestamp = self.server.get_model_timestamp()
            response = FloeMessage(
                msg='OK',
                weights=model,
                timestamp=model_timestamp
            )
        else:
            response = FloeMessage(msg='GetModel:REJECTED')
        return response
    
    def ContributeModel(
        self,
        request: FloeMessage,
        context: grpc.ServicerContext
    ) -> FloeMessage:
        """
        Servicer function for ContributeModel. See `floe.proto` for function
        declarations.
        """

        peer = context.peer()
        floe_logger.logger.write(
            f'Client {peer} is offering a model.',
            level=logging.INFO
        )

        # check the timestamp to make sure the client has the newest version
        ts = request.timestamp
        if ts == self.server.get_model_timestamp():
            # correct timestamp, let's deconstruct the message
            model = condecon.deconstruct_from_tlist(request.weights)

            # add the model to the server's queue
            self.server.model_queue.put((model, ts))

            # logging and response
            floe_logger.logger.write(
                f'Model from Client {peer} accepted with valid timestamp.',
                level=logging.INFO
            )
            response = FloeMessage(msg="ContributeModel:ACCEPTED")
        else:
            # incorrect timestamp, reject
            floe_logger.logger.write(
                f'Model from Client {peer} REJECTED with invalid timestamp.',
                level=logging.WARNING
            )
            response = FloeMessage(msg="ContributeModel:REJECTED")
        
        return response

    def Subscribe(
        self,
        request: FloeMessage,
        context: grpc.ServicerContext
    ) -> Iterator[FloeMessage]:
        """
        Servicer function for Subscribe. See `floe.proto` for function
        declarations.
        """

        # get the role of the incoming client
        job = request.msg
        job = 'CONTRIBUTOR' if job == 'Subscribe:CONTRIBUTOR' else 'SUBSCRIBER'

        peer = context.peer()
        floe_logger.logger.write(f'Client {peer} joined as {job}.')

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
