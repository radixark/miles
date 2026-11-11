from types import SimpleNamespace

import pytest
from tests.fast.fixtures.args_fixtures import make_trainer_config
from tests.fast.ray.test_train_actor import _ActorWithoutReloadSupport

from miles.ray import train_actor
from miles.utils.init_once import InitOnce


class TestTrainProcessGroupBackendSelection:
    @pytest.mark.parametrize(
        "backend,offload,expected",
        [
            ("megatron", False, "nccl"),
            ("fsdp", False, "nccl"),
            ("fsdp", True, "cpu:gloo,cuda:nccl"),
            ("torchtitan", True, "cpu:gloo,cuda:nccl"),
        ],
    )
    def test_only_fsdp2_cpu_offload_selects_a_hybrid_process_group(
        self, monkeypatch: pytest.MonkeyPatch, backend: str, offload: bool, expected: str
    ) -> None:
        """Megatron initialization never reads FSDP2 settings while FSDP / torchtitan offload enables both transports."""
        overrides = (
            {"optimizer_cpu_offload": offload, "fsdp_cpu_backend": "gloo"} if backend in ("fsdp", "torchtitan") else {}
        )
        args = make_trainer_config(
            train_backend=backend,
            distributed_backend="nccl",
            distributed_timeout_minutes=1,
            debug_deterministic_collective=False,
            **overrides,
        )
        actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)
        actor._init_once = InitOnce("TrainRayActor")
        actor._heartbeat = SimpleNamespace(bump=lambda: None)
        transports: list[str] = []
        monkeypatch.setattr(
            train_actor,
            "current_platform",
            SimpleNamespace(
                bind_device_before_process_group=lambda _rank: None,
                bind_device_after_process_group=lambda: None,
                nvml_numa_affinity=False,
                device_type="cpu",
            ),
        )
        monkeypatch.setattr(
            train_actor.dist, "init_process_group", lambda **kwargs: transports.append(kwargs["backend"])
        )
        monkeypatch.setattr(train_actor.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(train_actor.dist, "get_world_size", lambda: 1)
        monkeypatch.setattr(train_actor, "init_gloo_group", lambda: None)
        monkeypatch.setattr(train_actor, "rebind_env_reporting", lambda _args: None)

        actor._init_common(args=args, role="actor")

        assert transports == [expected]
        assert args.backend.rank == 0
        assert args.backend.world_size == 1
