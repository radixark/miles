import inspect
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
from tests.fast.utils.fake_ray_ids import fake_ray_node_id

from miles.utils.tracking_utils import prometheus_utils


class TestCollectorConstruction:
    def test_the_collector_actor_is_constructed_with_keyword_arguments(self, monkeypatch):
        """A positional handoff silently binds to the wrong parameter once the actor grows another one."""
        recorded: dict[str, object] = {}
        args = SimpleNamespace(prometheus_run_name="run")

        class _FakeActorClass:
            def options(self, **_options):
                return self

            def remote(self, *call_args, **call_kwargs):
                recorded["call_args"] = call_args
                recorded["call_kwargs"] = call_kwargs
                return SimpleNamespace(ping=SimpleNamespace(remote=lambda: None))

        monkeypatch.setattr(
            prometheus_utils,
            "ray",
            SimpleNamespace(
                remote=lambda _cls: _FakeActorClass(),
                get=lambda _ref: None,
                get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: fake_ray_node_id(0)),
            ),
        )
        monkeypatch.setattr(prometheus_utils, "_collector_handle", None)

        prometheus_utils.init_prometheus(args, start_server=True)

        assert (recorded["call_args"], recorded["call_kwargs"]) == ((), {"args": args})


class _CollectorInit(NamedTuple):
    actor_class: type
    call_args: tuple
    call_kwargs: dict


def _run_init_prometheus(monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace) -> _CollectorInit:
    captured: dict[str, Any] = {}

    class _FakeActorClass:
        def options(self, **_options: Any) -> "_FakeActorClass":
            return self

        def remote(self, *call_args: Any, **call_kwargs: Any) -> SimpleNamespace:
            captured["call_args"] = call_args
            captured["call_kwargs"] = call_kwargs
            return SimpleNamespace(ping=SimpleNamespace(remote=lambda: None))

    def _fake_remote(cls: type) -> _FakeActorClass:
        captured["actor_class"] = cls
        return _FakeActorClass()

    monkeypatch.setattr(
        prometheus_utils,
        "ray",
        SimpleNamespace(
            remote=_fake_remote,
            get=lambda _ref: None,
            get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: fake_ray_node_id(0)),
        ),
    )
    monkeypatch.setattr(prometheus_utils, "_collector_handle", None)

    prometheus_utils.init_prometheus(args, start_server=True)

    return _CollectorInit(
        actor_class=captured["actor_class"],
        call_args=captured["call_args"],
        call_kwargs=captured["call_kwargs"],
    )


class TestCollectorKeywordOnlyConstruction:
    def test_the_collector_refuses_a_positional_args_object(self):
        """Constructing the collector positionally must fail so a later parameter cannot silently steal the slot."""
        with pytest.raises(TypeError):
            prometheus_utils._PrometheusCollector(SimpleNamespace(prometheus_port=0))

    def test_every_collector_constructor_parameter_is_keyword_only(self):
        """No collector constructor parameter may be positionally bindable."""
        parameters = list(inspect.signature(prometheus_utils._PrometheusCollector.__init__).parameters.values())[1:]

        assert [parameter.kind for parameter in parameters] == [inspect.Parameter.KEYWORD_ONLY] * len(parameters)
        assert parameters

    def test_the_recorded_collector_keywords_bind_to_the_collector_constructor(self, monkeypatch):
        """The keywords the call site sends must name real collector constructor parameters."""
        init = _run_init_prometheus(monkeypatch, SimpleNamespace(prometheus_run_name="run"))

        assert init.actor_class is prometheus_utils._PrometheusCollector
        assert init.call_args == ()
        inspect.signature(init.actor_class.__init__).bind(object(), **init.call_kwargs)
