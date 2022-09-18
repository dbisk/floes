"""
generic.py - an example of a generic server that coordinates federated learning
over several rounds.

NOTE: like the `generic.py` in the `floe.client`, this should probably be
either renamed or moved to a different location.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from concurrent import futures
import logging
import os
import pickle
import time
from typing import List

import grpc

from floes.core import FloesParameters
import floes.core.floes_logger as floes_logger
from floes.server.grpc_server import FloesServiceServicer
from floes.strategy import Strategy
from floes.proto.floes_pb2 import FloesMessage
import floes.proto.floes_pb2_grpc as floes_pb2_grpc


MAX_MESSAGE_LENGTH = 536_870_912 # == 512 * 1024 * 1024


def start_grpc_server(
    model: FloesParameters, 
    address: str,
    strategy: Strategy,
    options: List = None,
    min_clients: int = 2,
    max_clients: int = 100
):
    if options is None:
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ("grpc.http2.max_pings_without_data", 0)
        ]
    
    servicer = FloesServiceServicer(min_clients=min_clients, max_clients=max_clients)
    servicer.server.set_model(model)
    servicer.server.set_strategy(strategy)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_clients),
        options=options
    )
    floes_pb2_grpc.add_FloesServiceServicer_to_server(
        servicer, server
    )
    server.add_insecure_port(address)
    server.start()

    return server, servicer


def start_server(
    model: FloesParameters,
    address: str,
    rounds: int,
    strategy: Strategy,
    await_termination: bool = True,
    client_timeout: int = None,
    save_dir: str = None,
    min_clients: int = 2,
    max_clients: int = 100
) -> FloesParameters:
    # start the grpc server
    server, servicer = start_grpc_server(
        model, address, strategy,
        min_clients=min_clients,
        max_clients=max_clients
    )

    floes_logger.logger.write(
        f'Server started. Awaiting minimum number of clients ({min_clients}) to start rounds.',
        logging.INFO
    )

    # wait until the minimum amount of clients have joined
    while (servicer.server.num_contributors < servicer.server.min_clients):
        time.sleep(1)
    
    # start the federated learning rounds
    for i in range(rounds):
        # logging
        floes_logger.logger.write(
            f'Beginning federated round {i}.',
            level=logging.INFO
        )
        
        # broadcast that a model is available to the clients
        servicer.server.broadcast(FloesMessage(msg='Subscribe:NEW_AVAILABLE'))

        # wait for the clients to respond with their updated models
        servicer.server.source_model_from_clients(timeout=client_timeout)

        # save the model if option is set
        if save_dir:
            with open(os.path.join(save_dir, f'checkpoint{i}.pkl'), 'wb') as f:
                pickle.dump(servicer.server.get_model(), f)

    # broadcast a new model available to the clients one more time and notify
    # that the server is done.
    servicer.server.broadcast(FloesMessage(msg='Subscribe:DONE'))

    # await termination by Ctrl+C
    if await_termination:
        server.wait_for_termination()
    else:
        time.sleep(3)
        return servicer.server.get_model()
