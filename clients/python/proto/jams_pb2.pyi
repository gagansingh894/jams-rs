from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PredictRequest(_message.Message):
    __slots__ = ("model_name", "input")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    input: str
    def __init__(self, model_name: _Optional[str] = ..., input: _Optional[str] = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ("output",)
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    output: str
    def __init__(self, output: _Optional[str] = ...) -> None: ...

class GetModelsResponse(_message.Message):
    __slots__ = ("total", "models")
    class Model(_message.Message):
        __slots__ = ("name", "framework", "path", "last_updated")
        NAME_FIELD_NUMBER: _ClassVar[int]
        FRAMEWORK_FIELD_NUMBER: _ClassVar[int]
        PATH_FIELD_NUMBER: _ClassVar[int]
        LAST_UPDATED_FIELD_NUMBER: _ClassVar[int]
        name: str
        framework: str
        path: str
        last_updated: str
        def __init__(self, name: _Optional[str] = ..., framework: _Optional[str] = ..., path: _Optional[str] = ..., last_updated: _Optional[str] = ...) -> None: ...
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    total: int
    models: _containers.RepeatedCompositeFieldContainer[GetModelsResponse.Model]
    def __init__(self, total: _Optional[int] = ..., models: _Optional[_Iterable[_Union[GetModelsResponse.Model, _Mapping]]] = ...) -> None: ...

class AddModelRequest(_message.Message):
    __slots__ = ("model_name",)
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    def __init__(self, model_name: _Optional[str] = ...) -> None: ...

class UpdateModelRequest(_message.Message):
    __slots__ = ("model_name",)
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    def __init__(self, model_name: _Optional[str] = ...) -> None: ...

class DeleteModelRequest(_message.Message):
    __slots__ = ("model_name",)
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    def __init__(self, model_name: _Optional[str] = ...) -> None: ...
