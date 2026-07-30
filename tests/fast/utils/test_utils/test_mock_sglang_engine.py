from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.sglang_engine import SGLangEngine
from miles.utils.misc import get_current_node_ip, get_free_port
from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine

# tests/fast/utils/test_utils/test_mock_sglang_engine.py → 4 levels up → repo root
ROLLOUT_DIR = Path(__file__).resolve().parents[4] / "miles" / "ray" / "rollout"


def _grep_engine_method_calls(directory: Path) -> set[str]:
    """Find every ``<engine|actor_handle>.<method>.remote(...)`` call in the
    rollout dir. The set returned is method names that the rollout code
    expects to exist on every SGLangEngine actor."""
    pattern = re.compile(r"(?:engine|actor_handle|rollout_engine)\.([a-zA-Z_][a-zA-Z0-9_]*)\.remote\(")
    methods: set[str] = set()
    for py in directory.rglob("*.py"):
        for m in pattern.finditer(py.read_text()):
            methods.add(m.group(1))
    return methods


def _public_methods(cls) -> set[str]:
    """Public methods (and known semi-private helpers) of ``cls``."""
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__  # unwrap @ray.remote
    keep_underscored = {"_get_free_port_block", "_get_node_ip", "_get_gpu_uuids"}
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


@pytest.fixture(scope="module")
def used_methods() -> set[str]:
    used = _grep_engine_method_calls(ROLLOUT_DIR)
    assert used, f"Expected to find engine.<method>.remote(...) calls under {ROLLOUT_DIR}"
    return used


# ----------------------------- contract tests -----------------------------


class TestApiContractMatchesRealEngine:
    def test_mock_implements_every_method_used_in_rollout_dir(self, used_methods: set[str]) -> None:
        real_methods = _public_methods(SGLangEngine)
        mock_methods = _public_methods(MockSGLangEngine)

        must_have = used_methods & real_methods
        missing_on_mock = must_have - mock_methods
        assert not missing_on_mock, (
            f"MockSGLangEngine is missing real-API methods that are called in "
            f"miles/ray/rollout/: {sorted(missing_on_mock)}. "
            f"Add stub implementations to mock_sglang_engine.py before adding the dependent test."
        )

    def test_mock_does_not_invent_methods_outside_real_api(self, used_methods: set[str]) -> None:
        """Mock must not declare methods that rollout code calls but the real
        engine does not implement — that would produce false positives where
        the mock test passes but real code AttributeErrors."""
        real_methods = _public_methods(SGLangEngine)
        mock_methods = _public_methods(MockSGLangEngine)

        invented = (mock_methods & used_methods) - real_methods
        assert not invented, (
            f"MockSGLangEngine declares methods that are called by rollout code but "
            f"do not exist on the real SGLangEngine: {sorted(invented)}."
        )

    def test_signature_compat_for_init(self) -> None:
        """``init`` is the most important signature to keep aligned because
        the rollout code passes addr/port kwargs from addr_allocator."""
        real_sig = inspect.signature(SGLangEngine.init)
        mock_sig = inspect.signature(MockSGLangEngine.__ray_actor_class__.init)
        real_params = set(real_sig.parameters) - {"self"}
        mock_params = set(mock_sig.parameters) - {"self"}

        # Mock accepts **kwargs catch-all; real signature lists explicit params.
        if "kwargs" not in mock_params:
            missing = real_params - mock_params
            assert not missing, f"MockSGLangEngine.init drops real params: {sorted(missing)}"

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


# ----------------------------- real Ray smoke tests -----------------------------


class TestRealRayActorLifecycle:
    def test_actor_construction_and_method_round_trip(self, ray_local_mode):
        """End-to-end: every method rollout code touches round-trips through
        Ray with the right args, and the call log preserves ordering."""
        args = make_args(rollout_num_gpus_per_engine=1)
        actor = MockSGLangEngine.options(num_cpus=0.1, num_gpus=0).remote(
            args,
            rank=0,
            worker_type="regular",
            base_gpu_id=0,
            sglang_overrides={},
            num_gpus_per_engine=1,
        )
        try:
            ray.get(actor.init.remote(host="127.0.0.1", port=get_free_port(start_port=20000)))
            ray.get(actor._get_free_port_block.remote(start_port=20100, count=1))
            ray.get(actor.simulate_crash.remote())

            calls = ray.get(actor.get_calls.remote())
            method_names = [name for name, _, _ in calls]
            assert method_names == [
                "init",
                "_get_free_port_block",
                "simulate_crash",
                "shutdown",
            ]
        finally:
            try:
                ray.get(actor.shutdown.remote())
            finally:
                ray.kill(actor)

    def test_fault_injection_round_trips_through_ray(self, ray_local_mode):
        """``set_fault`` schedules an exception; it must surface back via
        ``ray.get`` and be one-shot (cleared after firing)."""
        args = make_args(rollout_num_gpus_per_engine=1)
        actor = MockSGLangEngine.options(num_cpus=0.1, num_gpus=0).remote(
            args,
            rank=0,
            worker_type="regular",
            base_gpu_id=0,
            sglang_overrides={},
            num_gpus_per_engine=1,
        )
        try:
            ray.get(actor.set_fault.remote("shutdown", RuntimeError("boom")))
            with pytest.raises(ray.exceptions.RayTaskError, match="boom"):
                ray.get(actor.shutdown.remote())
            # Fault is one-shot — second call must succeed.
            assert ray.get(actor.shutdown.remote()) is True
        finally:
            ray.kill(actor)


class TestNodeAddress:
    def test_the_mock_reports_the_node_ip_instead_of_loopback(self) -> None:
        """A mock engine placed on another node must publish an address its peers can reach."""
        engine = MockSGLangEngine.__ray_actor_class__()

        assert engine._get_node_ip() == get_current_node_ip()
        assert engine._get_node_ip() != "127.0.0.1"


class TestNodeIpReporting:
    def test_get_node_ip_reports_the_reachable_node_ip_and_records_the_call(self) -> None:
        """Driven in process, the mock hands back the node's routable ip rather than loopback."""
        engine = MockSGLangEngine.__ray_actor_class__()

        node_ip = engine._get_node_ip()

        assert node_ip == get_current_node_ip()
        assert node_ip != "127.0.0.1"
        assert engine.get_calls() == [("_get_node_ip", (), {})]
