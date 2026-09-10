from argparse import Namespace

import pytest

from miles.backends.training_utils import data as data_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.utils.dp_schedule import TrainParallelConfig


class TestGetRolloutDataParallelWiring:
    def test_loader_uses_the_current_effective_rank_with_the_received_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The loader gives the splitter the live independent-DP rank and typed config."""
        trivial = GroupInfo(rank=0, size=1, group=None)
        state = ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=GroupInfo(rank=1, size=2, group=None),
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=GroupInfo(rank=2, size=3, group=None),
        )
        config = state.train_parallel_config(supports_precomputed_schedule=True)
        seen: list[tuple[int, TrainParallelConfig, object]] = []
        store_result = object()

        def process(
            args: Namespace,
            rollout_data_ref: object,
            *,
            dp_rank: int,
            train_parallel_config: TrainParallelConfig,
            witness_info: object,
        ) -> tuple[dict[str, list[list[int]]], object]:
            seen.append((dp_rank, train_parallel_config, witness_info))
            return {"tokens": [[1]], "loss_masks": [[1]]}, store_result

        monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
        monkeypatch.setattr(data_utils, "process_rollout_data", process)
        monkeypatch.setattr(data_utils.torch.cuda, "current_device", lambda: "cpu")
        args = Namespace(enable_witness=False, qkv_format="thd")

        rollout, result = data_utils.get_rollout_data(
            args=args, rollout_data_ref=object(), train_parallel_config=config
        )

        assert seen == [(2, config, None)]
        assert rollout["tokens"][0].device.type == "cpu"
        assert result is store_result
