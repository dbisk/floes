"""
generic.py - an example of a generic client that participates in every update
round. 

NOTE: this file should probably be either renamed or moved to a different
location.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import grpc

from .grpc_client import GRPCCLient
from .torch_client import PyTorchClient
from floes.proto.floes_pb2_grpc import FloesServiceStub


MAX_MESSAGE_LENGTH = 536_870_912 # == 512 * 1024 * 1024


def start_torch_client(torch_client: PyTorchClient, addr: str) -> PyTorchClient:
    """
    Starts a generic torch client that participates in every update round. 

    Args:
        model: torch.nn.Module
            The torch model that will be trained.
        address: str
            The IP address of the GRPC channel (generally server IP + port)
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

    # register this client as a contributor
    server_message_iterator = grpc_client.register_as_contributor(stub)

    # wait for server to indicate that a starting model is ready
    msg = next(server_message_iterator)
    while (True):
        # grab the newest model from the server
        new_model, new_timestamp = grpc_client.get_model_from_server(stub)

        # don't need to train if the most recent model is still the last model
        if (
            (torch_client.model_timestamp is not None) and
            (new_timestamp == torch_client.model_timestamp)
        ):
            continue

        # set the client's model to the one received from the server
        torch_client.set_parameters(new_model)
        torch_client.set_model_timestamp(new_timestamp)

        # if we're done, we can exit
        if msg is not None and msg.msg == 'Subscribe:DONE':
            break

        # perform a training round with the new model
        torch_client.train()

        # offer the model to the server
        params = torch_client.get_parameters()
        grpc_client.contribute_model_to_server(
            stub,
            params,
            torch_client.model_timestamp
        )

        # wait for the server to indicate that a new message is available
        msg = next(server_message_iterator)
    
    # return the model
    return torch_client
