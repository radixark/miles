from __future__ import annotations

import inspect

import pytest
import ray

from miles.utils.http_utils import MILES_HOST_IP_ENV
from miles.utils.misc import NodeProbeMixin, get_free_port
from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine
from miles.utils.workers.command_actor import CommandActor


def _public_methods(cls) -> set[str]:
    """Public methods (and known semi-private helpers) of ``cls``."""
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__  # unwrap @ray.remote
    keep_underscored = {name for name in vars(NodeProbeMixin) if not name.startswith("__")}
    return {
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("__") and (not name.startswith("_") or name in keep_underscored)
    }


def _comparable_params(func) -> list[tuple[str, str, bool]]:
    """Parameter name, kind and default-ness of ``func``, dropping a leading ``self`` and
    the ``_ray_trace_ctx`` argument that ``ray.remote`` injects."""
    params = list(inspect.signature(func).parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    return [
        (p.name, p.kind.name, p.default is not inspect.Parameter.empty) for p in params if p.name != "_ray_trace_ctx"
    ]


# Helpers the mock exposes for tests to drive it; the real actor has no reason to carry them.
_MOCK_ONLY_METHODS: set[str] = {
    "get_calls",
    "get_http_paths",
    "get_http_payloads_of",
    "get_server_args",
    "set_fault",
}


# ----------------------------- contract tests -----------------------------


class TestApiContractMatchesRealEngine:
    def test_the_mock_answers_every_call_the_real_actor_answers(self) -> None:
        """A method only the real actor has would AttributeError in production while
        every mock-driven test stays green."""
        missing = _public_methods(CommandActor) - _public_methods(MockSGLangEngine)
        assert missing == set(), (
            f"MockSGLangEngine cannot stand in for CommandActor: {sorted(missing)}. "
            f"Add stub implementations to mock_sglang_engine.py."
        )

    def test_the_mock_invents_no_method_the_real_actor_lacks(self) -> None:
        """A stub with no counterpart lets a mock-driven test go green on a call that
        AttributeErrors in production."""
        invented = _public_methods(MockSGLangEngine) - _public_methods(CommandActor) - _MOCK_ONLY_METHODS
        assert invented == set(), (
            f"MockSGLangEngine exposes methods CommandActor does not: {sorted(invented)}. "
            f"Either add them to CommandActor or list them in _MOCK_ONLY_METHODS."
        )

    def test_signature_compat_for_run(self) -> None:
        """``run`` is the most important signature to keep aligned because
        the rollout code hands it the launch command and env map."""
        real_sig = inspect.signature(CommandActor.run)
        mock_sig = inspect.signature(MockSGLangEngine.__ray_actor_class__.run)
        # ray injects _ray_trace_ctx into the methods of a class it decorates.
        ignored = {"self", "_ray_trace_ctx"}
        assert set(real_sig.parameters) - ignored == set(mock_sig.parameters) - ignored

    def test_signature_compat_for_every_shared_method(self) -> None:
        """Every method shared by mock and real engine declares the same parameter names, kinds and default-ness."""
        mock_cls = MockSGLangEngine.__ray_actor_class__
        shared: set[str] = _public_methods(CommandActor) & _public_methods(MockSGLangEngine)

        mismatches: dict[str, dict[str, list[tuple[str, str, bool]]]] = {}
        for name in sorted(shared):
            mock_params = _comparable_params(getattr(mock_cls, name))
            if any(kind == "VAR_KEYWORD" for _name, kind, _has_default in mock_params):
                continue
            real_params = _comparable_params(getattr(CommandActor, name))
            if mock_params != real_params:
                mismatches[name] = {"mock": mock_params, "real": real_params}

        assert not mismatches, (
            f"MockSGLangEngine signatures drifted from CommandActor: {mismatches}. "
            f"Update mock_sglang_engine.py so the mock accepts exactly what the real engine accepts."
        )


class TestNodeProbeHelpersAreCovered:
    def test_every_node_probe_helper_takes_part_in_the_contract_comparison(self) -> None:
        """A probe helper left out of the comparison set lets the mock silently lose a stub the real actor has."""
        probe_helpers = {name for name in vars(NodeProbeMixin) if not name.startswith("__")}

        assert probe_helpers
        assert probe_helpers <= _public_methods(CommandActor)
        assert probe_helpers <= _public_methods(MockSGLangEngine)


# ----------------------------- real Ray smoke tests -----------------------------


class TestRealRayActorLifecycle:
    def test_actor_construction_and_method_round_trip(self, ray_local_mode):
        """End-to-end: every method rollout code touches round-trips through
        Ray with the right args, and the call log preserves ordering."""
        actor = MockSGLangEngine.options(num_cpus=0.1, num_gpus=0).remote()
        try:
            port = get_free_port(start_port=20000)
            cmd = f"python -m sglang.launch_server --model-path /fake/model --host 127.0.0.1 --port {port}"
            ray.get(actor.run.remote(cmd=cmd, envs={}))
            ray.get(actor._get_free_port_block.remote(start_port=20100, count=1))
            ray.get(actor.kill_subprocess.remote())

            calls = ray.get(actor.get_calls.remote())
            method_names = [name for name, _, _ in calls]
            assert method_names == [
                "run",
                "_get_free_port_block",
                "kill_subprocess",
            ]
            server_args = ray.get(actor.get_server_args.remote())
            assert server_args["host"] == "127.0.0.1" and server_args["port"] == port
        finally:
            try:
                ray.get(actor.shutdown.remote())
            finally:
                ray.kill(actor)

    def test_fault_injection_round_trips_through_ray(self, ray_local_mode):
        """``set_fault`` schedules an exception; it must surface back via
        ``ray.get`` and be one-shot (cleared after firing)."""
        actor = MockSGLangEngine.options(num_cpus=0.1, num_gpus=0).remote()
        try:
            ray.get(actor.set_fault.remote("shutdown", RuntimeError("boom")))
            with pytest.raises(ray.exceptions.RayTaskError, match="boom"):
                ray.get(actor.shutdown.remote())
            # Fault is one-shot — second call must succeed.
            assert ray.get(actor.shutdown.remote()) is True
        finally:
            ray.kill(actor)

    def test_inject_fault_records_acknowledgement_lifetime_request(self, ray_local_mode) -> None:
        """Fault injection preserves the request to keep the actor alive through acknowledgement."""
        actor = MockSGLangEngine.options(num_cpus=0.1, num_gpus=0).remote()
        try:
            ray.get(actor.inject_fault.remote("sigkill", keep_actor_alive_until_ack=True))

            assert ray.get(actor.get_calls.remote()) == [
                (
                    "inject_fault",
                    (),
                    {"mode": "sigkill", "keep_actor_alive_until_ack": True},
                )
            ]
        finally:
            ray.kill(actor)


class TestPortProbe:
    def test_the_mock_reports_every_port_as_free_and_records_the_probe(self) -> None:
        """A mock-driven launch must never be refused a pinned port by the real machine's port state."""
        engine = MockSGLangEngine.__ray_actor_class__()

        assert engine._is_port_available(port=30001) is True
        assert engine.get_calls() == [("_is_port_available", (), {"port": 30001})]


class TestNodeAddress:
    def test_the_mock_reports_the_node_ip_instead_of_loopback(self) -> None:
        """A mock engine placed on another node must publish an address its peers can reach."""
        engine = MockSGLangEngine.__ray_actor_class__()

        assert engine._get_node_ip() == NodeProbeMixin._get_node_ip()
        assert engine._get_node_ip() != "127.0.0.1"


class TestNodeIpReporting:
    def test_get_node_ip_reports_the_reachable_node_ip_and_records_the_call(self) -> None:
        """Driven in process, the mock hands back the node's routable ip rather than loopback."""
        engine = MockSGLangEngine.__ray_actor_class__()

        node_ip = engine._get_node_ip()

        assert node_ip == NodeProbeMixin._get_node_ip()
        assert node_ip != "127.0.0.1"
        assert engine.get_calls() == [("_get_node_ip", (), {})]

    def test_the_mock_honours_the_node_ip_override(self, monkeypatch) -> None:
        """A real worker publishes MILES_HOST_IP when it is set, so a mock that ignores it tests the wrong address."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "10.9.9.9")

        assert MockSGLangEngine.__ray_actor_class__()._get_node_ip() == "10.9.9.9"

    def test_the_mock_and_the_real_actor_report_the_same_node_ip(self, monkeypatch) -> None:
        """Tests drive the mock in place of the real actor, so the two must derive an address the same way."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "10.9.9.9")

        assert MockSGLangEngine.__ray_actor_class__()._get_node_ip() == CommandActor()._get_node_ip()
