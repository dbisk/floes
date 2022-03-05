"""
connection.py - contains class definition representing the connection between
a server and client. This `ServerClientConnection` class holds a message
generator that will notify the recipient when a message becomes available. 

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from enum import Enum
import threading
from typing import Iterator

from floes.proto.floes_pb2 import FloesMessage

class Status(Enum):

    STANDBY = 0
    MESSAGE_READY = 1
    CLOSED = 2

class ServerClientConnection(object):
    """
    A simple class that ensures that the server and client don't try to write
    to the connection stream at the same time by utilizing a Condition
    variable.
    """

    def __init__(self):
        self._cv = threading.Condition()
        self._status = Status.STANDBY
        self._message = None

    def _transition(self, state: Status):
        # NOTE: does not do any checking to verify valid transition
        self._status = state
        self._cv.notify_all()
    
    def set_message(self, msg: FloesMessage):
        with self._cv:
            self._message = msg
            self._transition(Status.MESSAGE_READY)

    def close(self):
        with self._cv:
            self._transition(Status.CLOSED)

    def message_generator(self) -> Iterator[FloesMessage]:
        while not (self._status == Status.CLOSED):
            with self._cv:
                self._cv.wait_for(
                    lambda: self._status in [
                        Status.CLOSED,
                        Status.MESSAGE_READY
                    ]
                )

                if self._status == Status.CLOSED:
                    self._raise_error()

                # send the server message and return to standby
                msg = self._message
                self._message = None
                self._transition(Status.STANDBY)
            
            # make sure to release condition variable by exiting the `with`
            # context before yielding (and thus pausing execution)
            yield msg

    def _raise_error(self, msg: str = None):
        if msg is None:
            raise Exception('Connection error.')
        else:
            raise Exception(msg)
