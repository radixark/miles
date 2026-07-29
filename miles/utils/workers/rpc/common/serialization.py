from __future__ import annotations

import dataclasses
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model


@dataclasses.dataclass(frozen=True)
class RpcSerializer:
    query_model: type[BaseModel]
    result_adapter: TypeAdapter[Any]

    @classmethod
    def create(cls, *, query_model_name: str, query_fields: dict[str, Any], result_annotation: Any) -> RpcSerializer:
        query_model = create_model(query_model_name, __config__=ConfigDict(extra="forbid"), **query_fields)
        return cls(query_model=query_model, result_adapter=TypeAdapter(result_annotation))

    def encode_query(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return self.query_model(**kwargs).model_dump(mode="json")

    def decode_query(self, query: dict[str, Any]) -> dict[str, Any]:
        return dict(self.query_model(**query))

    def encode_result(self, result: Any) -> Any:
        return self.result_adapter.dump_python(result, mode="json")

    def decode_result(self, payload: Any) -> Any:
        return self.result_adapter.validate_python(payload)
