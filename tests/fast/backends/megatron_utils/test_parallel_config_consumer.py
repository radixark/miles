from types import SimpleNamespace
from typing import Any

import pytest

from miles.backends.megatron_utils import parallel
from miles.utils.ft_utils.process_group_utils import GroupInfo


class TestParallelConfigConsumer:
    @pytest.mark.parametrize("comm_type,expected", [(None, False), (["p2p"], False), (["a2a"], True)])
    def test_context_communication_config_selects_the_real_ulysses_consumer(
        self, monkeypatch: pytest.MonkeyPatch, comm_type: list[str] | None, expected: bool
    ) -> None:
        """SDK parallel state construction preserves communication choice for Ulysses dispatch."""
        group = None
        sdk: dict[str, Any] = {}

        def rank(*args: Any, **kwargs: Any) -> int:
            return 0

        def size(*args: Any, **kwargs: Any) -> int:
            return 4

        def process_group(*args: Any, **kwargs: Any) -> None:
            return group

        for prefix in (
            "data_parallel",
            "context_parallel",
            "tensor_model_parallel",
            "pipeline_model_parallel",
            "expert_model_parallel",
            "expert_tensor_parallel",
            "expert_data_parallel",
        ):
            sdk[f"get_{prefix}_rank"] = rank
            sdk[f"get_{prefix}_world_size"] = size
            sdk[f"get_{prefix}_group"] = process_group
        sdk["get_data_parallel_group_gloo"] = process_group
        sdk["get_tensor_and_data_parallel_group"] = process_group
        sdk["get_virtual_pipeline_model_parallel_world_size"] = lambda: None
        sdk["is_pipeline_last_stage"] = lambda: True
        monkeypatch.setattr(parallel, "mpu", SimpleNamespace(**sdk))
        monkeypatch.setattr(parallel, "get_args", lambda: SimpleNamespace(cp_comm_type=comm_type))
        monkeypatch.setattr(parallel.torch.distributed, "get_rank", rank)
        monkeypatch.setattr(parallel.torch.distributed, "get_world_size", size)
        independent = GroupInfo(rank=0, size=1, group=group)

        state = parallel.create_megatron_parallel_state(indep_dp=independent)

        assert state.is_ulysses_cp is expected
        assert state.cp.group is group
        assert state.indep_dp is independent
