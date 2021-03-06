# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import floes.proto.floes_pb2 as floes__pb2


class FloesServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetModel = channel.unary_unary(
                '/floes.FloesService/GetModel',
                request_serializer=floes__pb2.FloesMessage.SerializeToString,
                response_deserializer=floes__pb2.FloesMessage.FromString,
                )
        self.ContributeModel = channel.unary_unary(
                '/floes.FloesService/ContributeModel',
                request_serializer=floes__pb2.FloesMessage.SerializeToString,
                response_deserializer=floes__pb2.FloesMessage.FromString,
                )
        self.Subscribe = channel.unary_stream(
                '/floes.FloesService/Subscribe',
                request_serializer=floes__pb2.FloesMessage.SerializeToString,
                response_deserializer=floes__pb2.FloesMessage.FromString,
                )


class FloesServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ContributeModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Subscribe(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FloesServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetModel': grpc.unary_unary_rpc_method_handler(
                    servicer.GetModel,
                    request_deserializer=floes__pb2.FloesMessage.FromString,
                    response_serializer=floes__pb2.FloesMessage.SerializeToString,
            ),
            'ContributeModel': grpc.unary_unary_rpc_method_handler(
                    servicer.ContributeModel,
                    request_deserializer=floes__pb2.FloesMessage.FromString,
                    response_serializer=floes__pb2.FloesMessage.SerializeToString,
            ),
            'Subscribe': grpc.unary_stream_rpc_method_handler(
                    servicer.Subscribe,
                    request_deserializer=floes__pb2.FloesMessage.FromString,
                    response_serializer=floes__pb2.FloesMessage.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'floes.FloesService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FloesService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/floes.FloesService/GetModel',
            floes__pb2.FloesMessage.SerializeToString,
            floes__pb2.FloesMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ContributeModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/floes.FloesService/ContributeModel',
            floes__pb2.FloesMessage.SerializeToString,
            floes__pb2.FloesMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Subscribe(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/floes.FloesService/Subscribe',
            floes__pb2.FloesMessage.SerializeToString,
            floes__pb2.FloesMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
