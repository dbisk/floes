"""
client_stub.py - module containing the definition of a client stub for use on
the server.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from floes.core.connection import ServerClientConnection
from floes.proto.floes_pb2 import FloesMessage


class ClientStub(object):
    """
    Encapsulation for the "idea" of a client on the server side. The server
    will use these `ClientStub` objects to interact with the true clients with
    implementation on the individual devices.

    Parameters:
        id: str
            The identifier for the client this stub is representing.
        connection: ServerClientConnection
            The `ServerClientConnection` object associated with this client.
        is_contributor: bool, optional
            Whether this client is a contributor. Default false.
    """

    def __init__(self,
        id: str,
        connection: ServerClientConnection,
        is_contributor: bool = False
    ):
        self.id = id
        self._connection = connection
        self.is_contributor = is_contributor

    def write(self, msg: FloesMessage):
        """
        Sends a message to the client.

        Args:
            msg: `FloesMessage`
                The message to send.
        """
        self._connection.set_message(msg)
    
    def close(self):
        """
        Closes the `ServerClientConnection` with this client.
        """
        self._connection.close()
