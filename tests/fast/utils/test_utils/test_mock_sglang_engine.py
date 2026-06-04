from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.sglang_engine import SGLangEngine
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
    keep_underscored = {"_get_current_node_ip_and_free_port"}
    return {
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("__") and (not name.startswith("_") or name in keep_underscored)
    }


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
            ray.get(actor.init.remote(host="127.0.0.1", port=20000))
            ray.get(actor.health_generate.remote(timeout=1.0))
            ray.get(actor.release_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
            ray.get(actor.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
            ray.get(actor.update_weights_from_disk.remote(model_path="/fake"))
            ray.get(actor.check_weights.remote(action="pre_update"))

            calls = ray.get(actor.get_calls.remote())
            method_names = [name for name, _, _ in calls]
            assert method_names == [
                "init",
                "health_generate",
                "release_memory_occupation",
                "resume_memory_occupation",
                "update_weights_from_disk",
                "check_weights",
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
            ray.get(actor.set_fault.remote("health_generate", RuntimeError("boom")))
            with pytest.raises(ray.exceptions.RayTaskError, match="boom"):
                ray.get(actor.health_generate.remote(timeout=1.0))
            # Fault is one-shot — second call must succeed.
            assert ray.get(actor.health_generate.remote(timeout=1.0)) is True
        finally:
            ray.kill(actor)
