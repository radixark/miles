from argparse import Namespace

import pytest

from miles.backends.training_utils import data as data_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.ray.rollout.train_data_conversion import split_train_data_by_dp, split_train_data_by_dp_scheduled_raw
from miles.utils.dp_schedule import TrainParallelConfig


def _parallel_state(*, dp_size: int) -> ParallelState:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    dp_group = GroupInfo(rank=0, size=dp_size, group=None)
    return ParallelState(
        intra_dp=dp_group,
        intra_dp_cp=dp_group,
        cp=trivial_group,
        tp=trivial_group,
        pp=trivial_group,
        ep=trivial_group,
        etp=trivial_group,
        indep_dp=trivial_group,
    )


def _static_args() -> Namespace:
    return Namespace(
        qkv_format="thd",
        global_batch_size=256,
        use_dynamic_global_batch_size=False,
        use_dynamic_batch_size=False,
        micro_batch_size=8,
    )


def _scheduled_args() -> Namespace:
    return Namespace(
        qkv_format="thd",
        global_batch_size=256,
        use_dynamic_global_batch_size=False,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=64,
        balance_data=False,
        balance_by_flops=False,
        allow_partial_train_step=False,
        multi_lora=False,
        enable_sample_ownership_checker=False,
    )


class TestGetDataIteratorTrainingSideSchedule:
    def test_rejects_a_dp_size_that_does_not_divide_the_global_batch(self, monkeypatch):
        """Batches the rollout side cannot schedule still need the live cell count to divide the global batch."""
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda: _parallel_state(dp_size=3))

        with pytest.raises(AssertionError, match="must be divisible by dp_size"):
            data_utils.get_data_iterator(_static_args(), model=None, rollout_data={"total_lengths": [4] * 86})

    def test_a_dividing_dp_size_keeps_the_fixed_size_micro_batches(self, monkeypatch):
        """The divisibility check must not disturb the legacy static path it guards."""
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda: _parallel_state(dp_size=4))

        data_iterators, num_microbatches = data_utils.get_data_iterator(
            _static_args(), model=None, rollout_data={"total_lengths": [4] * 64}
        )

        assert num_microbatches == [8]
        assert len(data_iterators) == 1


class TestGetDataIteratorPrecomputedSchedule:
    @pytest.mark.parametrize("vpp_size", [1, 2])
    def test_three_live_cells_consume_every_scheduled_shard_row(
        self, monkeypatch: pytest.MonkeyPatch, vpp_size: int
    ) -> None:
        """Precomputed 256-over-3 shards reach every VPP iterator without losing or repeating rows."""
        state = _parallel_state(dp_size=3)
        state.vpp_size = vpp_size
        state.microbatch_group_size_per_vp_stage = 2 if vpp_size > 1 else None
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
        args = _scheduled_args()
        raw = {
            "tokens": [[i] * ((i % 11) + 1) for i in range(256)],
            "sample_indices": list(range(256)),
            "rollout_ids": list(range(256)),
        }
        shards = split_train_data_by_dp_scheduled_raw(
            args, raw, train_parallel_config=state.train_parallel_config(supports_precomputed_schedule=True)
        )

        for shard in shards:
            iterators, counts = data_utils.get_data_iterator(args, model=None, rollout_data=shard)

            assert counts == shard["num_microbatches"]
            assert len(iterators) == vpp_size
            expected = [shard["sample_indices"][i] for batch in shard["micro_batch_indices"] for i in batch]
            assert len(expected) == len(shard["sample_indices"])
            for iterator in iterators:
                actual = [
                    i for _ in range(sum(counts)) for i in iterator.get_next(["sample_indices"])["sample_indices"]
                ]
                assert sorted(actual) == sorted(shard["sample_indices"])
                assert actual == expected
                assert iterator.reset().get_next(["sample_indices"])["sample_indices"] == [
                    shard["sample_indices"][i] for i in shard["micro_batch_indices"][0]
                ]

        assert sorted(i for shard in shards for i in shard["sample_indices"]) == list(range(256))


class TestGetDataIteratorUnsupportedSchedule:
    def test_a_backend_without_schedule_support_falls_back_and_rejects_three_cells(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without schedule support 256-over-3 shards carry no precomputed layout and the training side rejects them."""
        state = _parallel_state(dp_size=3)
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
        args = _scheduled_args()
        raw = {
            "tokens": [[i] * ((i % 11) + 1) for i in range(256)],
            "sample_indices": list(range(256)),
            "rollout_ids": list(range(256)),
        }

        shards = split_train_data_by_dp(
            args, raw, train_parallel_config=state.train_parallel_config(supports_precomputed_schedule=False)
        )

        assert len(shards) == 3
        assert all("micro_batch_indices" not in shard for shard in shards)
        with pytest.raises(AssertionError, match="must be divisible by dp_size"):
            data_utils.get_data_iterator(args, model=None, rollout_data=shards[0])


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
