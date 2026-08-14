from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed import HashStore

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
    UpdateWeightFromDistributed,
)
from miles.utils.distributed_lock import StoreTicketLock

_BROADCAST_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"
_PREFIX = "miles/weight_update"
_NEXT_KEY = f"{_PREFIX}/next"
_SERVING_KEY = f"{_PREFIX}/serving"


def _build_updater(
    *, pp_size: int, pp_rank: int = 0, tp_rank: int = 0, intra_dp_cp_rank: int = 0
) -> tuple[UpdateWeightFromDistributed, MagicMock]:
    parallel_state = SimpleNamespace(
        pp=SimpleNamespace(size=pp_size, rank=pp_rank),
        tp=SimpleNamespace(rank=tp_rank),
        intra_dp_cp=SimpleNamespace(rank=intra_dp_cp_rank),
    )
    with (
        patch(f"{_BROADCAST_MODULE}.get_parallel_state", return_value=parallel_state),
        patch(f"{_BROADCAST_MODULE}.create_world_ticket_lock") as create_lock,
        patch.object(UpdateWeightFromDistributed, "_init_lora"),
    ):
        updater = UpdateWeightFromDistributed(
            Namespace(),
            [],
            lambda: {},
            model_name="test-model",
            quantization_config=None,
        )
    return updater, create_lock


class TestEngineLockConstruction:
    def test_a_single_stage_world_builds_no_lock_at_all(self) -> None:
        """One PP stage means one source, so there is nothing to exclude and no store to host."""
        updater, create_lock = _build_updater(pp_size=1)

        create_lock.assert_not_called()
        assert isinstance(updater._engine_lock, nullcontext)

    @pytest.mark.parametrize(("pp_size", "pp_rank"), [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
    def test_every_pipelined_source_contends_on_the_shared_lock(self, pp_size: int, pp_rank: int) -> None:
        """Every PP stage broadcasts its own slice, so every stage must queue, not just the first."""
        updater, create_lock = _build_updater(pp_size=pp_size, pp_rank=pp_rank)

        create_lock.assert_called_once_with(prefix=_PREFIX, participates=True)
        assert updater._engine_lock is create_lock.return_value

    @pytest.mark.parametrize(("tp_rank", "intra_dp_cp_rank"), [(1, 0), (0, 1)])
    def test_a_rank_that_never_broadcasts_joins_the_collective_without_contending(
        self, tp_rank: int, intra_dp_cp_rank: int
    ) -> None:
        """Non-source ranks must still call in, since building the lock is collective."""
        _, create_lock = _build_updater(pp_size=3, tp_rank=tp_rank, intra_dp_cp_rank=intra_dp_cp_rank)

        create_lock.assert_called_once_with(prefix=_PREFIX, participates=False)

    def test_reconnecting_to_the_engines_does_not_rebuild_the_lock(self) -> None:
        """Reconnects are not collective, so entering one there would hang the ranks that skip it."""
        updater, create_lock = _build_updater(pp_size=3)
        lock_after_init = updater._engine_lock

        parallel_state = SimpleNamespace(pp=SimpleNamespace(rank=0), tp=SimpleNamespace(rank=0))
        with (
            patch(f"{_BROADCAST_MODULE}.get_parallel_state", return_value=parallel_state),
            patch.object(UpdateWeightFromDistributed, "_is_source", True),
            patch(f"{_BROADCAST_MODULE}.disconnect_rollout_engines_from_distributed"),
            patch(f"{_BROADCAST_MODULE}.connect_rollout_engines_from_distributed"),
        ):
            updater.connect_rollout_engines([MagicMock()])

        create_lock.assert_called_once()
        assert updater._engine_lock is lock_after_init


class TestUpdateWeightUnderTheEngineLock:
    @staticmethod
    def _make_self(store: HashStore) -> SimpleNamespace:
        return SimpleNamespace(
            _engine_lock=StoreTicketLock(store=store, prefix=_PREFIX, poll_interval=0.001),
            _group_name="miles-pp_0",
            _model_update_groups=MagicMock(name="nccl_group"),
            weight_version=7,
            rollout_engines=[MagicMock(name="engine")],
            _weight_update_selector="all",
        )

    @staticmethod
    def _run(
        fake_self: SimpleNamespace,
        named_tensors: list[tuple[str, torch.Tensor]],
        *,
        pbar: MagicMock | None = None,
        broadcast_side_effect=None,
        engine_failure: Exception | None = None,
    ) -> tuple[MagicMock, MagicMock]:
        with (
            patch(f"{_BROADCAST_MODULE}.update_weights_from_distributed") as broadcast,
            patch(f"{_BROADCAST_MODULE}.async_utils.wait_futures") as wait_futures,
        ):
            broadcast.side_effect = broadcast_side_effect
            broadcast.return_value = []
            wait_futures.side_effect = engine_failure
            wait_futures.return_value = []
            UpdateWeightFromDistributed._update_weight_implementation(fake_self, named_tensors, pbar=pbar)
        return broadcast, wait_futures

    def test_a_finished_update_hands_the_lock_to_the_next_source(self) -> None:
        """The common path: broadcast, drop the bucket, then call the next ticket."""
        store = HashStore()
        fake_self = self._make_self(store)
        named_tensors = [("w", torch.zeros(2))]
        pbar = MagicMock()

        broadcast, wait_futures = self._run(fake_self, named_tensors, pbar=pbar)

        broadcast.assert_called_once_with(
            "miles-pp_0",
            fake_self._model_update_groups,
            7,
            fake_self.rollout_engines,
            named_tensors,
            selector="all",
        )
        wait_futures.assert_called_once_with(broadcast.return_value)
        assert named_tensors == []
        assert store.add(_SERVING_KEY, 0) == 1
        pbar.update.assert_called_once_with(1)

    def test_the_broadcast_is_issued_while_the_lock_is_held(self) -> None:
        """A broadcast outside the critical section would be exactly the interleaving this prevents."""
        store = HashStore()
        store.add(_NEXT_KEY, 1)
        store.add(_SERVING_KEY, 1)
        fake_self = self._make_self(store)
        observed: list[tuple[int, int]] = []

        def record(*args, **kwargs) -> None:
            observed.append((store.add(_NEXT_KEY, 0), store.add(_SERVING_KEY, 0)))

        self._run(fake_self, [("w", torch.zeros(2))], broadcast_side_effect=record)

        assert observed == [(2, 1)]
        assert store.add(_SERVING_KEY, 0) == 2

    def test_an_engine_failure_keeps_the_lock_and_the_bucket(self) -> None:
        """Failing closed where it actually fails: awaiting the futures reports the dead engine."""
        store = HashStore()
        fake_self = self._make_self(store)
        named_tensors = [("w", torch.zeros(2))]
        pbar = MagicMock()

        with pytest.raises(RuntimeError, match="engine died"):
            self._run(fake_self, named_tensors, pbar=pbar, engine_failure=RuntimeError("engine died"))

        assert store.add(_SERVING_KEY, 0) == 0
        assert len(named_tensors) == 1
        pbar.update.assert_not_called()

    def test_a_broadcast_failure_keeps_the_lock_and_the_bucket(self) -> None:
        """A source that fails before returning futures must retain its ticket and bucket."""
        store = HashStore()
        fake_self = self._make_self(store)
        named_tensors = [("w", torch.zeros(2))]
        pbar = MagicMock()

        with pytest.raises(RuntimeError, match="broadcast failed"):
            self._run(
                fake_self,
                named_tensors,
                pbar=pbar,
                broadcast_side_effect=RuntimeError("broadcast failed"),
            )

        assert store.add(_SERVING_KEY, 0) == 0
        assert len(named_tensors) == 1
        pbar.update.assert_not_called()
        with pytest.raises(AssertionError):
            self._run(fake_self, named_tensors)

    def test_a_source_that_failed_once_refuses_to_broadcast_again(self) -> None:
        """The retained ticket must block this rank too, not just the ranks behind it."""
        store = HashStore()
        fake_self = self._make_self(store)

        with pytest.raises(RuntimeError):
            self._run(fake_self, [("w", torch.zeros(2))], engine_failure=RuntimeError("engine died"))

        with pytest.raises(AssertionError):
            self._run(fake_self, [("w", torch.zeros(2))])
