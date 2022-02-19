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
import time
from typing import List

import grpc
import numpy as np

import floe.core.floe_logger as floe_logger
from floe.server.grpc_server import FloeServiceServicer
from floe.strategy import Strategy
from floe.proto.floe_pb2 import FloeMessage
import floe.proto.floe_pb2_grpc as floe_pb2_grpc


MAX_MESSAGE_LENGTH = 536_870_912 # == 512 * 1024 * 1024


def start_grpc_server(
    model: List[np.ndarray], 
    address: str,
    strategy: Strategy,
    options: List = None
):
    if options is None:
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ("grpc.http2.max_pings_without_data", 0)
        ]
    
    servicer = FloeServiceServicer()
    servicer.server.set_model(model)
    servicer.server.set_strategy(strategy)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=options
    )
    floe_pb2_grpc.add_FloeServiceServicer_to_server(
        servicer, server
    )
    server.add_insecure_port(address)
    server.start()

    return server, servicer


def start_server(
    model: List[np.ndarray],
    address: str,
    rounds: int,
    strategy: Strategy
):
    # start the grpc server
    server, servicer = start_grpc_server(model, address, strategy)

    floe_logger.logger.write(
        'Server started. Awaiting minimum number of clients to start rounds.',
        logging.INFO
    )

    # wait until the minimum amount of clients have joined
    while (servicer.server.num_contributors < servicer.server.min_clients):
        time.sleep(1)
    
    # start the federated learning rounds
    for i in range(rounds):
        # logging
        floe_logger.logger.write(
            f'Beginning federated round {i}.',
            level=logging.INFO
        )
        
        # broadcast that a model is available to the clients
        servicer.server.broadcast(FloeMessage(msg='Subscribe:NEW_AVAILABLE'))

        # wait for the clients to respond with their updated models
        servicer.server.source_model_from_clients(timeout=600)

    # broadcast a new model available to the clients one more time and notify
    # that the server is done.
    servicer.server.broadcast(FloeMessage(msg='Subscribe:DONE'))

    # await termination by Ctrl+C
    server.wait_for_termination()
