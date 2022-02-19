# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: floe.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='floe.proto',
  package='floe',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\nfloe.proto\x12\x04\x66loe\"_\n\x0b\x46loeMessage\x12\x0b\n\x03msg\x18\x01 \x01(\t\x12\x1d\n\x07weights\x18\x02 \x03(\x0b\x32\x0c.floe.Tensor\x12\x16\n\ttimestamp\x18\x03 \x01(\tH\x00\x88\x01\x01\x42\x0c\n\n_timestamp\"%\n\x06Tensor\x12\r\n\x05shape\x18\x01 \x03(\x05\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c\x32\xb3\x01\n\x0b\x46loeService\x12\x32\n\x08GetModel\x12\x11.floe.FloeMessage\x1a\x11.floe.FloeMessage\"\x00\x12\x39\n\x0f\x43ontributeModel\x12\x11.floe.FloeMessage\x1a\x11.floe.FloeMessage\"\x00\x12\x35\n\tSubscribe\x12\x11.floe.FloeMessage\x1a\x11.floe.FloeMessage\"\x00\x30\x01\x62\x06proto3'
)




_FLOEMESSAGE = _descriptor.Descriptor(
  name='FloeMessage',
  full_name='floe.FloeMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg', full_name='floe.FloeMessage.msg', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='weights', full_name='floe.FloeMessage.weights', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='floe.FloeMessage.timestamp', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='_timestamp', full_name='floe.FloeMessage._timestamp',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=20,
  serialized_end=115,
)


_TENSOR = _descriptor.Descriptor(
  name='Tensor',
  full_name='floe.Tensor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='shape', full_name='floe.Tensor.shape', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data', full_name='floe.Tensor.data', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=117,
  serialized_end=154,
)

_FLOEMESSAGE.fields_by_name['weights'].message_type = _TENSOR
_FLOEMESSAGE.oneofs_by_name['_timestamp'].fields.append(
  _FLOEMESSAGE.fields_by_name['timestamp'])
_FLOEMESSAGE.fields_by_name['timestamp'].containing_oneof = _FLOEMESSAGE.oneofs_by_name['_timestamp']
DESCRIPTOR.message_types_by_name['FloeMessage'] = _FLOEMESSAGE
DESCRIPTOR.message_types_by_name['Tensor'] = _TENSOR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FloeMessage = _reflection.GeneratedProtocolMessageType('FloeMessage', (_message.Message,), {
  'DESCRIPTOR' : _FLOEMESSAGE,
  '__module__' : 'floe_pb2'
  # @@protoc_insertion_point(class_scope:floe.FloeMessage)
  })
_sym_db.RegisterMessage(FloeMessage)

Tensor = _reflection.GeneratedProtocolMessageType('Tensor', (_message.Message,), {
  'DESCRIPTOR' : _TENSOR,
  '__module__' : 'floe_pb2'
  # @@protoc_insertion_point(class_scope:floe.Tensor)
  })
_sym_db.RegisterMessage(Tensor)



_FLOESERVICE = _descriptor.ServiceDescriptor(
  name='FloeService',
  full_name='floe.FloeService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=157,
  serialized_end=336,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetModel',
    full_name='floe.FloeService.GetModel',
    index=0,
    containing_service=None,
    input_type=_FLOEMESSAGE,
    output_type=_FLOEMESSAGE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ContributeModel',
    full_name='floe.FloeService.ContributeModel',
    index=1,
    containing_service=None,
    input_type=_FLOEMESSAGE,
    output_type=_FLOEMESSAGE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Subscribe',
    full_name='floe.FloeService.Subscribe',
    index=2,
    containing_service=None,
    input_type=_FLOEMESSAGE,
    output_type=_FLOEMESSAGE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_FLOESERVICE)

DESCRIPTOR.services_by_name['FloeService'] = _FLOESERVICE

# @@protoc_insertion_point(module_scope)
