"""
generic.py - an example of a generic client that participates in every update
round. 

NOTE: this file should probably be either renamed or moved to a different
location.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""
from typing import Dict, Tuple

import grpc

from .grpc_client import GRPCCLient
from .client import Client
from floes.proto.floes_pb2_grpc import FloesServiceStub

MAX_MESSAGE_LENGTH = 536_870_912 # == 512 * 1024 * 1024


def start_grpc_client(addr: str) -> Tuple[GRPCCLient, FloesServiceStub]:
    """
    Start a GRPC Client object to interact with the server at the given
    address. The returned `FloesServiceStub` should be passed into all of the
    GRPCClient's methods. The GRPCClient's methods will be called to interact
    with the server.

    Args:
        addr: `str`
            The address of the server. Usually an IP address + port. 
    Returns:
        Tuple of a started GRPC client object and the FloesServiceStub
        associated with the connection.
    """
    # initialize the grpc client connection
    grpc_client = GRPCCLient()

    # start the connection to the server
    channel = grpc.insecure_channel(
        addr,
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
    )
    stub = FloesServiceStub(channel)
    return grpc_client, stub


def start_client(client: Client, addr: str, **kwargs) -> Client:
    """
    Starts a generic client that participates in every update round. 

    Args:
        client: `Client`
            The client that will be participating in the federated setup.
        address: `str`
            The IP address of the GRPC channel (generally <server IP>:<port>)
        **kwargs:
            Any remaining keyword arguments to be passed into the client's
            `train` function.
    Returns:
        The trained client after the federated learning rounds are over.
    """
    return start_layerwise_client(client, addr, None, **kwargs)


def start_layerwise_client(
    client: Client, addr: str, layers: Dict[str, bool], **kwargs
) -> Client:
    """
    Starts a client that only trains some specified layers in each update
    round. 

    Args:
        client: `Client`
            The client that will be participating in the federated setup.
        address: `str`
            The IP address of the GRPC channel (generally <serverIP>:<port>)
        layers: `Dict[str, bool]`
            The trainable layers dictionary where the keys are the layer name
            and the values are `True` if this client trains it, and `False`
            otherwise.
        **kwargs:
            Any remaining keyword arguments to be passed into the client's
            `train` function.
    Returns:
        `Client`
            The trained client after the federated learning rounds are over.
    """

    # initialize the grpc client connection
    grpc_client, stub = start_grpc_client(addr)

    # register this client as a contributor
    server_message_iterator = grpc_client.register_as_contributor(stub)

    # wait for server to indicate that a starting model is ready
    msg = next(server_message_iterator)
    while (True):
        # grab the newest model from the server
        new_model, new_timestamp = grpc_client.get_model_from_server(stub)

        # don't need to train if the most recent model is still the last model
        if (
            (client.model_timestamp is not None) and
            (new_timestamp == client.model_timestamp)
        ):
            continue

        # set the client's model to the one received from the server
        client.set_parameters(new_model)
        client.set_model_timestamp(new_timestamp)

        # if we're done, we can exit
        if msg is not None and msg.msg == 'Subscribe:DONE':
            break

        # perform a training round with the new model
        client.train(**kwargs)

        # offer the model to the server
        params = client.get_parameters()
        grpc_client.contribute_model_to_server(
            stub,
            params,
            client.model_timestamp,
            layers=layers
        )

        # wait for the server to indicate that a new message is available
        msg = next(server_message_iterator)
    
    # return the model
    return client
