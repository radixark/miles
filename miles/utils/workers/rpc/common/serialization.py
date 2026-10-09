from __future__ import annotations

import dataclasses
import math
from typing import Any

import typing_extensions
from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model

NON_FINITE_FLOAT_TAG = "__miles_non_finite_float__"

_WIRE_CONFIG = ConfigDict(ser_json_inf_nan="constants", ser_json_bytes="base64", val_json_bytes="base64")

_TOKEN_BY_FLOAT = {math.inf: "inf", -math.inf: "-inf"}
_FLOAT_BY_TOKEN = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


@dataclasses.dataclass(frozen=True)
class RpcSerializer:
    query_model: type[BaseModel]
    result_adapter: TypeAdapter[Any]

    @classmethod
    def create(cls, *, query_model_name: str, query_fields: dict[str, Any], result_annotation: Any) -> RpcSerializer:
        query_model = create_model(
            query_model_name, __config__=ConfigDict(extra="forbid", **_WIRE_CONFIG), **query_fields
        )
        result_config = None if _carries_own_config(result_annotation) else _WIRE_CONFIG
        return cls(query_model=query_model, result_adapter=TypeAdapter(result_annotation, config=result_config))

    def encode_query(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return _NonFiniteFloatCodec.encode(self.query_model(**kwargs).model_dump(mode="json"))

    def decode_query(self, query: dict[str, Any]) -> dict[str, Any]:
        return dict(self.query_model(**_NonFiniteFloatCodec.decode(query)))

    def encode_result(self, result: Any) -> Any:
        return _NonFiniteFloatCodec.encode(self.result_adapter.dump_python(result, mode="json", warnings="error"))

    def decode_result(self, payload: Any) -> Any:
        return self.result_adapter.validate_python(_NonFiniteFloatCodec.decode(payload))


class _NonFiniteFloatCodec:
    @classmethod
    def encode(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return {NON_FINITE_FLOAT_TAG: _TOKEN_BY_FLOAT.get(value, "nan")}
        if isinstance(value, dict):
            if NON_FINITE_FLOAT_TAG in value:
                raise ValueError(f"rpc payloads must not contain the reserved key {NON_FINITE_FLOAT_TAG!r}")
            return {key: cls.encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls.encode(item) for item in value]
        return value

    @classmethod
    def decode(cls, value: Any) -> Any:
        if isinstance(value, dict):
            token = value.get(NON_FINITE_FLOAT_TAG)
            if len(value) == 1 and token in _FLOAT_BY_TOKEN:
                return _FLOAT_BY_TOKEN[token]
            return {key: cls.decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls.decode(item) for item in value]
        return value


def _carries_own_config(annotation: Any) -> bool:
    if not isinstance(annotation, type):
        return False
    return (
        issubclass(annotation, BaseModel)
        or dataclasses.is_dataclass(annotation)
        or typing_extensions.is_typeddict(annotation)
    )
