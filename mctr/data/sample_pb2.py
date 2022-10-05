# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sample.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import feaconf_pb2 as feaconf__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='sample.proto',
  package='xdl.io',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x0csample.proto\x12\x06xdl.io\x1a\rfeaconf.proto\"\xb4\x01\n\x0bSampleGroup\x12\x12\n\nsample_ids\x18\x01 \x03(\t\x12\x1d\n\x06labels\x18\x02 \x03(\x0b\x32\r.xdl.io.Label\x12,\n\x0e\x66\x65\x61ture_tables\x18\x03 \x03(\x0b\x32\x14.xdl.io.FeatureTable\x12\x1c\n\x05props\x18\x04 \x03(\x0b\x32\r.xdl.io.Label\x12&\n\nextensions\x18\x05 \x03(\x0b\x32\x12.xdl.io.Extensions\"t\n\nExtensions\x12\x34\n\textension\x18\x01 \x03(\x0b\x32!.xdl.io.Extensions.ExtensionEntry\x1a\x30\n\x0e\x45xtensionEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x17\n\x05Label\x12\x0e\n\x06values\x18\x01 \x03(\x02\":\n\x0c\x46\x65\x61tureTable\x12*\n\rfeature_lines\x18\x01 \x03(\x0b\x32\x13.xdl.io.FeatureLine\"?\n\x0b\x46\x65\x61tureLine\x12!\n\x08\x66\x65\x61tures\x18\x01 \x03(\x0b\x32\x0f.xdl.io.Feature\x12\r\n\x05refer\x18\x02 \x01(\x05\"`\n\x07\x46\x65\x61ture\x12!\n\x04type\x18\x01 \x02(\x0e\x32\x13.xdl.io.FeatureType\x12\x0c\n\x04name\x18\x02 \x01(\t\x12$\n\x06values\x18\x03 \x03(\x0b\x32\x14.xdl.io.FeatureValue\"H\n\x0c\x46\x65\x61tureValue\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\x02\x12\x0e\n\x06vector\x18\x03 \x03(\x02\x12\x0c\n\x04hkey\x18\x04 \x01(\x03')
  ,
  dependencies=[feaconf__pb2.DESCRIPTOR,])




_SAMPLEGROUP = _descriptor.Descriptor(
  name='SampleGroup',
  full_name='xdl.io.SampleGroup',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sample_ids', full_name='xdl.io.SampleGroup.sample_ids', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='labels', full_name='xdl.io.SampleGroup.labels', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='feature_tables', full_name='xdl.io.SampleGroup.feature_tables', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='props', full_name='xdl.io.SampleGroup.props', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='extensions', full_name='xdl.io.SampleGroup.extensions', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=40,
  serialized_end=220,
)


_EXTENSIONS_EXTENSIONENTRY = _descriptor.Descriptor(
  name='ExtensionEntry',
  full_name='xdl.io.Extensions.ExtensionEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='xdl.io.Extensions.ExtensionEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='xdl.io.Extensions.ExtensionEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=290,
  serialized_end=338,
)

_EXTENSIONS = _descriptor.Descriptor(
  name='Extensions',
  full_name='xdl.io.Extensions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='extension', full_name='xdl.io.Extensions.extension', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_EXTENSIONS_EXTENSIONENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=222,
  serialized_end=338,
)


_LABEL = _descriptor.Descriptor(
  name='Label',
  full_name='xdl.io.Label',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='values', full_name='xdl.io.Label.values', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=340,
  serialized_end=363,
)


_FEATURETABLE = _descriptor.Descriptor(
  name='FeatureTable',
  full_name='xdl.io.FeatureTable',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature_lines', full_name='xdl.io.FeatureTable.feature_lines', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=365,
  serialized_end=423,
)


_FEATURELINE = _descriptor.Descriptor(
  name='FeatureLine',
  full_name='xdl.io.FeatureLine',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='features', full_name='xdl.io.FeatureLine.features', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='refer', full_name='xdl.io.FeatureLine.refer', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=425,
  serialized_end=488,
)


_FEATURE = _descriptor.Descriptor(
  name='Feature',
  full_name='xdl.io.Feature',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='xdl.io.Feature.type', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='xdl.io.Feature.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='values', full_name='xdl.io.Feature.values', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=490,
  serialized_end=586,
)


_FEATUREVALUE = _descriptor.Descriptor(
  name='FeatureValue',
  full_name='xdl.io.FeatureValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='xdl.io.FeatureValue.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='xdl.io.FeatureValue.value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vector', full_name='xdl.io.FeatureValue.vector', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hkey', full_name='xdl.io.FeatureValue.hkey', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=588,
  serialized_end=660,
)

_SAMPLEGROUP.fields_by_name['labels'].message_type = _LABEL
_SAMPLEGROUP.fields_by_name['feature_tables'].message_type = _FEATURETABLE
_SAMPLEGROUP.fields_by_name['props'].message_type = _LABEL
_SAMPLEGROUP.fields_by_name['extensions'].message_type = _EXTENSIONS
_EXTENSIONS_EXTENSIONENTRY.containing_type = _EXTENSIONS
_EXTENSIONS.fields_by_name['extension'].message_type = _EXTENSIONS_EXTENSIONENTRY
_FEATURETABLE.fields_by_name['feature_lines'].message_type = _FEATURELINE
_FEATURELINE.fields_by_name['features'].message_type = _FEATURE
_FEATURE.fields_by_name['type'].enum_type = feaconf__pb2._FEATURETYPE
_FEATURE.fields_by_name['values'].message_type = _FEATUREVALUE
DESCRIPTOR.message_types_by_name['SampleGroup'] = _SAMPLEGROUP
DESCRIPTOR.message_types_by_name['Extensions'] = _EXTENSIONS
DESCRIPTOR.message_types_by_name['Label'] = _LABEL
DESCRIPTOR.message_types_by_name['FeatureTable'] = _FEATURETABLE
DESCRIPTOR.message_types_by_name['FeatureLine'] = _FEATURELINE
DESCRIPTOR.message_types_by_name['Feature'] = _FEATURE
DESCRIPTOR.message_types_by_name['FeatureValue'] = _FEATUREVALUE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SampleGroup = _reflection.GeneratedProtocolMessageType('SampleGroup', (_message.Message,), dict(
  DESCRIPTOR = _SAMPLEGROUP,
  __module__ = 'sample_pb2'
  # @@protoc_insertion_point(class_scope:xdl.io.SampleGroup)
  ))
_sym_db.RegisterMessage(SampleGroup)

Extensions = _reflection.GeneratedProtocolMessageType('Extensions', (_message.Message,), dict(

  ExtensionEntry = _reflection.GeneratedProtocolMessageType('ExtensionEntry', (_message.Message,), dict(
    DESCRIPTOR = _EXTENSIONS_EXTENSIONENTRY,
    __module__ = 'sample_pb2'
    # @@protoc_insertion_point(class_scope:xdl.io.Extensions.ExtensionEntry)
    ))
  ,
  DESCRIPTOR = _EXTENSIONS,
  __module__ = 'sample_pb2'
  # @@protoc_insertion_point(class_scope:xdl.io.Extensions)
  ))
_sym_db.RegisterMessage(Extensions)
_sym_db.RegisterMessage(Extensions.ExtensionEntry)

Label = _reflection.GeneratedProtocolMessageType('Label', (_message.Message,), dict(
  DESCRIPTOR = _LABEL,
  __module__ = 'sample_pb2'
  # @@protoc_insertion_point(class_scope:xdl.io.Label)
  ))
_sym_db.RegisterMessage(Label)

FeatureTable = _reflection.GeneratedProtocolMessageType('FeatureTable', (_message.Message,), dict(
  DESCRIPTOR = _FEATURETABLE,
  __module__ = 'sample_pb2'
  # @@protoc_insertion_point(class_scope:xdl.io.FeatureTable)
  ))
_sym_db.RegisterMessage(FeatureTable)

FeatureLine = _reflection.GeneratedProtocolMessageType('FeatureLine', (_message.Message,), dict(
  DESCRIPTOR = _FEATURELINE,
  __module__ = 'sample_pb2'
  # @@protoc_insertion_point(class_scope:xdl.io.FeatureLine)
  ))
_sym_db.RegisterMessage(FeatureLine)

Feature = _reflection.GeneratedProtocolMessageType('Feature', (_message.Message,), dict(
  DESCRIPTOR = _FEATURE,
  __module__ = 'sample_pb2'
  # @@protoc_insertion_point(class_scope:xdl.io.Feature)
  ))
_sym_db.RegisterMessage(Feature)

FeatureValue = _reflection.GeneratedProtocolMessageType('FeatureValue', (_message.Message,), dict(
  DESCRIPTOR = _FEATUREVALUE,
  __module__ = 'sample_pb2'
  # @@protoc_insertion_point(class_scope:xdl.io.FeatureValue)
  ))
_sym_db.RegisterMessage(FeatureValue)


_EXTENSIONS_EXTENSIONENTRY._options = None
# @@protoc_insertion_point(module_scope)