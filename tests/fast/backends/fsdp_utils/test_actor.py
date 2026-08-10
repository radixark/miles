import sys
from argparse import Namespace
from types import ModuleType

import pytest

from miles.backends.fsdp_utils import actor as actor_module
from miles.utils import distributed_utils
from miles.utils.ft_utils.heartbeat_utils import SimpleHeartbeat
from miles.utils.ft_utils.indep_dp import IndepDPInfo


class TestFSDPInit:
    def test_fsdp_init_rejects_an_independent_dp_quorum_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FSDP initialization rejects a non-null independent-DP quorum store address."""

        def fail_if_initialization_continues() -> None:
            raise RuntimeError("FSDP initialization continued past quorum validation")

        def ignore_device(_device: object) -> None:
            return None

        def ignore_process_group(**_kwargs: object) -> None:
            return None

        def zero() -> int:
            return 0

        def one() -> int:
            return 1

        def false() -> bool:
            return False

        class FakeDumper:
            apply_source_patches = staticmethod(fail_if_initialization_continues)

        dumper_module = ModuleType("sglang.srt.debug_utils.dumper")
        dumper_module.dumper = FakeDumper()
        monkeypatch.setitem(sys.modules, "sglang.srt.debug_utils.dumper", dumper_module)
        monkeypatch.setattr(actor_module.torch.cuda, "set_device", ignore_device)
        monkeypatch.setattr(actor_module.dist, "init_process_group", ignore_process_group)
        monkeypatch.setattr(actor_module.dist, "get_rank", zero)
        monkeypatch.setattr(actor_module.dist, "get_world_size", one)
        monkeypatch.setattr(actor_module.dist, "is_initialized", false)
        monkeypatch.setattr(actor_module.dist, "new_group", lambda **_kwargs: object())
        monkeypatch.setattr(distributed_utils, "GLOO_GROUP", None)

        actor = object.__new__(actor_module.FSDPTrainRayActor)
        actor._rank = 0
        actor._heartbeat = SimpleHeartbeat()
        args = Namespace(
            debug_deterministic_collective=False,
            distributed_backend="nccl",
            distributed_timeout_minutes=1,
            dumper_enable=True,
            env_report=None,
            num_gpus_per_node=1,
        )

        with pytest.raises(AssertionError):
            actor.init(
                args,
                "actor",
                indep_dp_info=IndepDPInfo.create_trivial(),
                indep_dp_store_addr="10.0.0.9:1234",
            )
