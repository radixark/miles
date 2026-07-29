from typing import Any

import pytest
from pydantic import ValidationError

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.common.metadata import DEFAULT_CONCURRENCY_GROUP, collect_rpc_method_specs, rpc


class _Payload(StrictBaseModel):
    text: str
    count: int = 1


class _GoodWorker:
    demo_class_attribute = 3

    def demo_default_arg(self, a: int, b: int = 10) -> int:
        return a + b

    async def demo_async_model(self, payload: _Payload) -> _Payload:
        return payload

    @rpc(concurrency_group="heavy")
    def demo_grouped(self, step: int) -> None:
        pass

    @classmethod
    def demo_classmethod(cls, x: int) -> int:
        return x

    @staticmethod
    def demo_staticmethod(x: int) -> int:
        return x

    @property
    def demo_property(self) -> int:
        return 1

    def _demo_private(self, x):
        pass


class TestCollectSpecs:
    def test_collects_public_methods_only(self):
        """Public methods are collected; underscore-prefixed ones are skipped."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert set(specs) == {"demo_default_arg", "demo_async_model", "demo_grouped"}

    def test_non_instance_method_members_are_skipped(self):
        """Classmethods, staticmethods and properties are skipped like plain attributes."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert {"demo_classmethod", "demo_staticmethod", "demo_property", "demo_class_attribute"}.isdisjoint(specs)

    def test_default_concurrency_group(self):
        """Undecorated methods fall into the default concurrency group."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_default_arg"].concurrency_group == DEFAULT_CONCURRENCY_GROUP

    def test_decorated_concurrency_group(self):
        """@rpc(concurrency_group=...) is picked up by introspection."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_grouped"].concurrency_group == "heavy"

    def test_is_async_flag(self):
        """Coroutine methods are flagged async, plain ones are not."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_async_model"].is_async and not specs["demo_default_arg"].is_async

    def test_async_method_with_non_default_concurrency_group_rejected(self):
        """A concurrency group on an async method would be silently ignored, so it is refused."""

        class Worker:
            @rpc(concurrency_group="heavy")
            async def demo_async_grouped(self) -> None:
                pass

        with pytest.raises(TypeError, match="concurrency groups only serialize sync methods"):
            collect_rpc_method_specs(Worker)


class TestQueryModel:
    def test_decode_query_applies_defaults(self):
        """An omitted defaulted parameter is filled in on the server side."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_default_arg"].serializer.decode_query({"a": 1}) == {"a": 1, "b": 10}

    def test_decode_query_parses_nested_model(self):
        """A nested model argument is revived as a model instance."""
        specs = collect_rpc_method_specs(_GoodWorker)
        decoded = specs["demo_async_model"].serializer.decode_query({"payload": {"text": "hi"}})
        assert decoded == {"payload": _Payload(text="hi")}

    def test_missing_required_param_rejected(self):
        """A missing required argument fails validation instead of defaulting."""
        specs = collect_rpc_method_specs(_GoodWorker)
        with pytest.raises(ValidationError):
            specs["demo_default_arg"].serializer.decode_query({})

    def test_unknown_param_rejected(self):
        """An argument the method does not declare fails validation."""
        specs = collect_rpc_method_specs(_GoodWorker)
        with pytest.raises(ValidationError):
            specs["demo_default_arg"].serializer.decode_query({"a": 1, "nope": 2})

    def test_wrong_type_rejected(self):
        """An argument that cannot be coerced fails validation."""
        specs = collect_rpc_method_specs(_GoodWorker)
        with pytest.raises(ValidationError):
            specs["demo_default_arg"].serializer.decode_query({"a": "not-int"})


class TestResultAdapter:
    def test_result_roundtrip(self):
        """A model result encodes to json-safe data and decodes back to a model."""
        serializer = collect_rpc_method_specs(_GoodWorker)["demo_async_model"].serializer
        encoded: Any = serializer.encode_result(_Payload(text="hi", count=2))
        assert encoded == {"text": "hi", "count": 2}
        assert serializer.decode_result(encoded) == _Payload(text="hi", count=2)
