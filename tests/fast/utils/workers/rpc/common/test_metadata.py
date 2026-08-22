import functools
import inspect
from collections.abc import Callable
from typing import Annotated, Any

import pytest
from pydantic import Field, ValidationError

from tests.fast.utils.workers.rpc.common.postponed_annotation_worker import LatePayload, PostponedWorker

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.common.metadata import DEFAULT_CONCURRENCY_GROUP, collect_rpc_method_specs, rpc


class _Payload(StrictBaseModel):
    text: str
    count: int = 1


def _passthrough(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return wrapper


def _inject_ray_trace_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(fn)
    fn.__signature__ = signature.replace(
        parameters=[
            *signature.parameters.values(),
            inspect.Parameter("_ray_trace_ctx", inspect.Parameter.KEYWORD_ONLY, default=None),
        ]
    )
    return fn


def _opaque_passthrough(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    wrapper.__wrapped__ = fn
    return wrapper


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

    def test_default_serialized_outcome_limit(self) -> None:
        """An undecorated RPC method declares no outcome limit, so nothing is reserved for it."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_default_arg"].max_serialized_outcome_bytes is None

    def test_decorated_serialized_outcome_limit(self) -> None:
        """A method can declare a smaller proven serialized outcome bound."""

        class Worker:
            @rpc(max_serialized_outcome_bytes=2048)
            def demo(self) -> str:
                return "ok"

        assert collect_rpc_method_specs(Worker)["demo"].max_serialized_outcome_bytes == 2048

    def test_control_plane_marker_is_explicit_in_the_method_spec(self) -> None:
        """Heartbeat-class methods carry an explicit bounded control-admission marker."""

        class Worker:
            @rpc(control_plane=True)
            def heartbeat(self) -> str:
                return "alive"

        assert collect_rpc_method_specs(Worker)["heartbeat"].control_plane
        assert not collect_rpc_method_specs(_GoodWorker)["demo_default_arg"].control_plane

    def test_too_small_serialized_outcome_limit_is_rejected(self) -> None:
        """A result bound must leave enough room for a terminal protocol envelope."""
        with pytest.raises(ValueError, match="at least"):
            rpc(max_serialized_outcome_bytes=1)

    def test_is_async_flag(self):
        """Coroutine methods are flagged async, plain ones are not."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_async_model"].is_async and not specs["demo_default_arg"].is_async


class TestDecoratorChainConcurrencyGroup:
    def test_marker_above_wrapper_is_found(self):
        """@rpc applied outside a functools.wraps wrapper still declares its concurrency group."""

        class Worker:
            @rpc(concurrency_group="heavy")
            @_passthrough
            def demo_marker_outside(self, x: int) -> int:
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_marker_outside"].concurrency_group == "heavy"

    def test_marker_below_wrapper_is_found(self):
        """@rpc applied inside a functools.wraps wrapper still declares its concurrency group."""

        class Worker:
            @_passthrough
            @rpc(concurrency_group="heavy")
            def demo_marker_inside(self, x: int) -> int:
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_marker_inside"].concurrency_group == "heavy"

    def test_marker_between_two_wrapper_layers_is_found(self):
        """@rpc sandwiched between two wrapper layers is still found by walking the decorator chain."""

        class Worker:
            @_passthrough
            @rpc(concurrency_group="heavy")
            @_passthrough
            def demo_marker_nested(self, x: int) -> int:
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_marker_nested"].concurrency_group == "heavy"

    def test_marker_hidden_by_a_wrapper_that_copies_nothing_is_found(self):
        """A wrapper that does not copy the wrapped function's attributes cannot hide the marker."""

        class Worker:
            @_opaque_passthrough
            @rpc(concurrency_group="heavy")
            def demo_marker_hidden(self, x: int) -> int:
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_marker_hidden"].concurrency_group == "heavy"

    def test_outermost_marker_wins_over_an_inner_one(self):
        """When two decorator layers each declare a group, the outermost declaration decides."""

        class Worker:
            @rpc(concurrency_group="outer")
            @_opaque_passthrough
            @rpc(concurrency_group="inner")
            def demo_marker_conflict(self, x: int) -> int:
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_marker_conflict"].concurrency_group == "outer"


class TestQueryModel:
    def test_decode_query_applies_defaults(self):
        """Omitted parameters with defaults resolve to their default values."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_default_arg"].serializer.decode_query({"a": 5}) == {"a": 5, "b": 10}

    def test_decode_query_parses_nested_model(self):
        """Nested pydantic payloads are revived into real model instances."""
        specs = collect_rpc_method_specs(_GoodWorker)
        kwargs = specs["demo_async_model"].serializer.decode_query({"payload": {"text": "hi"}})
        assert kwargs["payload"] == _Payload(text="hi")

    def test_missing_required_param_rejected(self):
        """Missing required parameters raise a validation error."""
        specs = collect_rpc_method_specs(_GoodWorker)
        with pytest.raises(ValidationError):
            specs["demo_default_arg"].serializer.decode_query({})

    def test_unknown_param_rejected(self):
        """Extra unknown parameters raise a validation error."""
        specs = collect_rpc_method_specs(_GoodWorker)
        with pytest.raises(ValidationError):
            specs["demo_default_arg"].serializer.decode_query({"a": 1, "unknown": 2})

    def test_wrong_type_rejected(self):
        """Type-mismatched parameters raise a validation error."""
        specs = collect_rpc_method_specs(_GoodWorker)
        with pytest.raises(ValidationError):
            specs["demo_default_arg"].serializer.decode_query({"a": "not-an-int"})


class TestParameterKinds:
    def test_keyword_only_parameters_are_supported(self):
        """Keyword-only parameters are accepted, land in the query model and decode like normal ones."""

        class Worker:
            def demo_keyword_only(self, *, required: int, optional: str = "fallback") -> int:
                return required

        specs = collect_rpc_method_specs(Worker)
        serializer = specs["demo_keyword_only"].serializer
        assert serializer.decode_query({"required": 5}) == {"required": 5, "optional": "fallback"}
        with pytest.raises(ValidationError):
            serializer.decode_query({"optional": "only"})

    def test_positional_only_receiver_is_supported(self):
        """A positional-only self is accepted because the receiver never reaches the wire."""

        class Worker:
            def demo_positional_receiver(self, /, value: int) -> int:
                return value

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_positional_receiver"].serializer.decode_query({"value": 7}) == {"value": 7}


class TestAnnotatedParameters:
    def test_annotated_parameter_constraints_are_preserved(self):
        """Constraints carried by Annotated metadata survive hint resolution and are enforced."""

        class Worker:
            def demo_constrained(self, value: Annotated[int, Field(ge=1)]) -> int:
                return value

        serializer = collect_rpc_method_specs(Worker)["demo_constrained"].serializer
        assert serializer.decode_query({"value": 3}) == {"value": 3}
        with pytest.raises(ValidationError):
            serializer.decode_query({"value": 0})


class TestPostponedAnnotations:
    def test_string_annotations_resolved_in_worker_module(self):
        """A worker module using postponed annotations still builds real typed models."""
        specs = collect_rpc_method_specs(PostponedWorker)
        kwargs = specs["demo_transform"].serializer.decode_query({"payload": {"text": "hi"}})
        assert kwargs["payload"] == LatePayload(text="hi")

    def test_string_return_annotation_resolved(self):
        """A postponed return annotation resolves into a working result adapter."""
        specs = collect_rpc_method_specs(PostponedWorker)
        assert specs["demo_transform"].serializer.decode_result({"text": "hi"}) == LatePayload(text="hi")


class TestInheritance:
    def test_inherited_methods_collected(self):
        """Methods inherited from a base worker class are exposed too."""

        class Child(_GoodWorker):
            def demo_child_only(self, x: int) -> int:
                return x

        specs = collect_rpc_method_specs(Child)
        assert {"demo_default_arg", "demo_async_model", "demo_grouped", "demo_child_only"} <= set(specs)


class TestResultAdapter:
    def test_result_roundtrip(self):
        """Return values are encoded as plain json data and decode back into the model."""
        specs = collect_rpc_method_specs(_GoodWorker)
        serializer = specs["demo_async_model"].serializer
        dumped = serializer.encode_result(_Payload(text="hi"))
        assert not isinstance(dumped, _Payload)
        assert dumped == {"text": "hi", "count": 1}
        assert serializer.decode_result(dumped) == _Payload(text="hi")

    def test_none_return_annotation(self):
        """Methods annotated -> None get a NoneType result adapter."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_grouped"].serializer.decode_result(None) is None


class TestFailLoud:
    def test_async_method_with_non_default_concurrency_group_rejected(self):
        """An async method with a non-default concurrency group fails at collection time."""

        class Worker:
            @rpc(concurrency_group="train")
            async def demo_async_grouped(self) -> int:
                return 0

        with pytest.raises(TypeError, match="concurrency_group"):
            collect_rpc_method_specs(Worker)

    def test_missing_param_annotation_rejected(self):
        """A parameter without a type annotation fails at collection time."""

        class Worker:
            def demo_unannotated_arg(self, x) -> int:
                return 0

        with pytest.raises(TypeError, match="must be type-annotated"):
            collect_rpc_method_specs(Worker)

    def test_missing_return_annotation_rejected(self):
        """A method without a return annotation fails at collection time."""

        class Worker:
            def demo_unannotated_return(self, x: int):
                return x

        with pytest.raises(TypeError, match="return type annotation"):
            collect_rpc_method_specs(Worker)

    def test_var_positional_rejected(self):
        """*args signatures fail at collection time."""

        class Worker:
            def demo_var_positional(self, *x: int) -> int:
                return 0

        with pytest.raises(TypeError, match="args"):
            collect_rpc_method_specs(Worker)

    def test_var_keyword_rejected(self):
        """**kwargs signatures fail at collection time."""

        class Worker:
            def demo_var_keyword(self, **x: int) -> int:
                return 0

        with pytest.raises(TypeError, match="kwargs"):
            collect_rpc_method_specs(Worker)

    def test_positional_only_rejected(self):
        """Positional-only parameters fail at collection time since calls pass kwargs."""

        class Worker:
            def demo_positional_only(self, x: int, /) -> int:
                return x

        with pytest.raises(TypeError, match="positional-only"):
            collect_rpc_method_specs(Worker)

    def test_non_self_receiver_rejected(self):
        """An unconventionally named receiver is refused rather than silently dropped."""

        class Worker:
            def demo_odd_receiver(this, x: int) -> int:
                return x

        with pytest.raises(TypeError, match="receiver parameter 'self'"):
            collect_rpc_method_specs(Worker)

    def test_forgotten_self_is_rejected_instead_of_eating_the_first_argument(self):
        """A method that forgets self would otherwise lose its first parameter off the wire."""

        class Worker:
            def demo_forgot_self(a: int, b: int) -> int:
                return a + b

        with pytest.raises(TypeError, match="receiver parameter 'self'"):
            collect_rpc_method_specs(Worker)

    def test_keyword_only_receiver_rejected(self):
        """A keyword-only self is refused at collection time instead of blowing up on call."""

        class Worker:
            def demo_keyword_only_receiver(*, self) -> int:
                return 0

        with pytest.raises(TypeError, match="receiver parameter positionally"):
            collect_rpc_method_specs(Worker)

    def test_method_without_any_parameter_rejected(self):
        """A method taking no parameters at all is refused for lacking a receiver."""

        class Worker:
            def demo_no_parameters() -> int:
                return 0

        with pytest.raises(TypeError, match="must take a receiver parameter"):
            collect_rpc_method_specs(Worker)

    def test_public_nested_model_class_rejected_as_non_method(self):
        """A public nested model class is refused as not being a method at all."""

        class Worker:
            class Config(StrictBaseModel):
                text: str

            def demo_ok(self, x: int) -> int:
                return x

        with pytest.raises(TypeError) as excinfo:
            collect_rpc_method_specs(Worker)
        assert "not a method" in str(excinfo.value)
        assert "receiver" not in str(excinfo.value)

    def test_public_class_alias_attribute_rejected_as_non_method(self):
        """A public class alias attribute is refused as not being a method at all."""

        class Worker:
            demo_alias = _Payload

            def demo_ok(self, x: int) -> int:
                return x

        with pytest.raises(TypeError) as excinfo:
            collect_rpc_method_specs(Worker)
        assert "not a method" in str(excinfo.value)
        assert "receiver" not in str(excinfo.value)

    def test_wrapped_async_method_stays_async(self):
        """A functools.wraps-decorated async method is still detected as async."""

        def passthrough(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                return await fn(*args, **kwargs)

            return wrapper

        class Worker:
            @passthrough
            async def demo_wrapped_async(self, x: int) -> int:
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_wrapped_async"].is_async
        assert specs["demo_wrapped_async"].serializer.decode_query({"x": 1}) == {"x": 1}

    def test_no_public_methods_collects_an_empty_surface(self):
        """A worker whose whole value is a lifecycle side effect answers no call, and still has to be servable."""

        class Worker:
            def _demo_hidden(self, x: int) -> int:
                return x

        assert collect_rpc_method_specs(Worker) == {}

    def test_any_annotation_allowed(self):
        """Any-annotated parameters are accepted and passed through."""

        class Worker:
            def demo_any(self, x: Any) -> Any:
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_any"].serializer.decode_query({"x": [1, "a"]}) == {"x": [1, "a"]}

    def test_only_positional_or_keyword_parameters_may_be_filled_positionally(self):
        """A caller's positional arguments are named in declaration order, and keyword-only names are not in it."""

        class Worker:
            def demo_mixed(self, a: int, b: int, *, c: int) -> int:
                return a + b + c

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_mixed"].positional_parameter_names == ("a", "b")


class TestRayInjectedTraceContext:
    def test_a_method_carrying_rays_injected_trace_context_is_still_collected(self):
        """Ray rewrites actor method signatures with a keyword-only _ray_trace_ctx, which must not reach the wire."""

        class Worker:
            @_inject_ray_trace_context
            def demo_traced(self, a: int, b: int) -> int:
                return a + b

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_traced"].positional_parameter_names == ("a", "b")
        assert specs["demo_traced"].serializer.decode_query({"a": 1, "b": 2}) == {"a": 1, "b": 2}
        with pytest.raises(ValidationError):
            specs["demo_traced"].serializer.decode_query({"a": 1, "b": 2, "_ray_trace_ctx": {}})

    def test_an_unannotated_parameter_beside_the_injected_one_is_still_rejected(self):
        """Skipping ray's parameter must not excuse a genuinely unannotated parameter on the same method."""

        class Worker:
            @_inject_ray_trace_context
            def demo_traced(self, x) -> int:
                return x

        with pytest.raises(TypeError, match="parameter 'x' must be type-annotated"):
            collect_rpc_method_specs(Worker)

    def test_an_ordinary_parameter_borrowing_rays_name_is_still_rejected(self):
        """Only ray's keyword-only injection is skipped, not any parameter that happens to share its name."""

        class Worker:
            def demo_shadowed(self, _ray_trace_ctx) -> int:
                return 0

        with pytest.raises(TypeError, match="parameter '_ray_trace_ctx' must be type-annotated"):
            collect_rpc_method_specs(Worker)
