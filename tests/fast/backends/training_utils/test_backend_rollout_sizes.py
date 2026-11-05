from argparse import Namespace

import pytest
from tests.fast.backends.training_utils.backend_owner_fixtures import cpu_rollout_store, native_backend_owner_state

from miles.backends.training_utils.data.rollout import get_data_iterator, get_num_rollouts, get_rollout_data
from miles.backends.training_utils.parallel import GroupInfo, get_parallel_state

__all__ = ["cpu_rollout_store", "native_backend_owner_state"]


class TestBackendRolloutSizes:
    @pytest.mark.parametrize("cp_size,ratios,expected", [(1, [4], 12), (2, [4], 16), (2, [], 11)])
    def test_rollout_padding_uses_backend_compression_ratios_and_context_topology(
        self,
        native_backend_owner_state: None,
        cpu_rollout_store: None,
        cp_size: int,
        ratios: list[int],
        expected: int,
    ) -> None:
        """Compression-aware padding grows for context-parallel mirrored regions without flat backend fields."""
        state = get_parallel_state()
        state.cp = GroupInfo(rank=0, size=cp_size, group=None)
        state.intra_dp_cp = GroupInfo(rank=0, size=cp_size, group=None)
        args = Namespace(
            backend=Namespace(compress_ratios=ratios),
            qkv_format="bshd",
            bshd_padding_mode="rollout",
            data_pad_size_multiplier=1,
            delay_split_train_data_by_dp=False,
            enable_witness=False,
        )

        batch, _ = get_rollout_data(
            args=args,
            rollout_data_ref=[object()],
            train_parallel_config=state.train_parallel_config(supports_precomputed_schedule=True),
        )

        assert batch["max_seq_lens"] == [expected]
        assert batch["total_lengths"] == [11]
        assert batch["tokens"][0].tolist() == list(range(11))

    @pytest.mark.parametrize("dynamic,steps,rollouts", [(None, [2, 2], [4, 4]), (8, [4], [8])])
    def test_iterator_and_rollout_counts_share_backend_static_and_dynamic_batch_sizes(
        self, native_backend_owner_state: None, dynamic: int | None, steps: list[int], rollouts: list[int]
    ) -> None:
        """The iterator schedule and rollout normalization agree on backend and dynamic batch sizes."""
        args = Namespace(
            backend=Namespace(global_batch_size=4, micro_batch_size=2),
            qkv_format="thd",
            use_dynamic_global_batch_size=dynamic is not None,
            use_dynamic_batch_size=False,
        )
        data = {"total_lengths": list(range(8))}
        if dynamic is not None:
            data["dynamic_global_batch_size"] = dynamic

        [iterator], counts = get_data_iterator(args=args, model=None, rollout_data=data)

        assert counts == steps
        assert get_num_rollouts(args=args, rollout_data=data, num_steps=len(counts)) == rollouts
        assert [iterator.get_next(["total_lengths"])["total_lengths"] for _ in range(sum(counts))] == [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ]

    def test_multi_lora_rejects_backend_micro_batches_that_cannot_partition_the_local_batch(
        self, native_backend_owner_state: None
    ) -> None:
        """Multi-LoRA cannot silently drop rows from an indivisible fixed micro-batch schedule."""
        args = Namespace(
            backend=Namespace(global_batch_size=4, micro_batch_size=3),
            qkv_format="thd",
            use_dynamic_global_batch_size=False,
            use_dynamic_batch_size=False,
        )

        with pytest.raises(ValueError, match="local_batch_size=4, micro_batch_size=3"):
            get_data_iterator(args=args, model=None, rollout_data={"total_lengths": [4] * 4, "adapter_slots": [0] * 4})
