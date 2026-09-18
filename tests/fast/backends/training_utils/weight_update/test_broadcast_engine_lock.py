from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed import HashStore

from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocols.broadcast import UpdateWeightFromDistributed
from miles.utils.distributed_lock import StoreTicketLock

_BROADCAST_MODULE = "miles.backends.training_utils.weight_update.protocols.broadcast"
_PREFIX = "miles/weight_update"
_NEXT_KEY = f"{_PREFIX}/next"
_SERVING_KEY = f"{_PREFIX}/serving"


def _parallel_state(*, pp_rank: int, ep_rank: int, tp_rank: int = 0, etp_rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        pp=SimpleNamespace(rank=pp_rank),
        ep=SimpleNamespace(rank=ep_rank),
        tp=SimpleNamespace(rank=tp_rank),
        etp=SimpleNamespace(rank=etp_rank),
    )


def _configure_protocol(
    *,
    global_rank: int,
    coordinates: list[tuple[int, int]],
    pp_rank: int,
    ep_rank: int,
    placement: WeightUpdatePlacement | None = None,
    tp_rank: int = 0,
    etp_rank: int = 0,
) -> tuple[UpdateWeightFromDistributed, MagicMock]:
    protocol = UpdateWeightFromDistributed(Namespace())
    parallel_state = _parallel_state(pp_rank=pp_rank, ep_rank=ep_rank, tp_rank=tp_rank, etp_rank=etp_rank)
    placement = placement or WeightUpdatePlacement(gather_pp=False, gather_ep=False)

    def all_gather_coordinates(output, value, *, group) -> None:
        assert value[-2:] == (
            0 if placement.gather_pp else pp_rank,
            0 if placement.gather_ep else ep_rank,
        )
        output[:] = (
            [(rank, *coordinate) for rank, coordinate in enumerate(coordinates)] if len(value) == 3 else coordinates
        )

    with (
        patch(f"{_BROADCAST_MODULE}.get_gloo_group", return_value=MagicMock(name="gloo_group")),
        patch(f"{_BROADCAST_MODULE}.dist.get_rank", return_value=global_rank),
        patch(f"{_BROADCAST_MODULE}.dist.get_world_size", return_value=len(coordinates)),
        patch(f"{_BROADCAST_MODULE}.dist.all_gather_object", side_effect=all_gather_coordinates),
        patch(f"{_BROADCAST_MODULE}.create_world_ticket_lock") as create_lock,
    ):
        protocol.configure(parallel_state, placement)
    return protocol, create_lock


class TestBroadcastTopology:
    def test_required_placement_retains_pp_and_ep_shards(self) -> None:
        assert UpdateWeightFromDistributed.required_placement == WeightUpdatePlacement(
            gather_pp=False, gather_ep=False
        )

    def test_a_single_sender_builds_no_lock(self) -> None:
        protocol, create_lock = _configure_protocol(
            global_rank=0,
            coordinates=[(0, 0), (0, 0)],
            pp_rank=0,
            ep_rank=0,
        )

        create_lock.assert_not_called()
        assert isinstance(protocol._engine_lock, nullcontext)
        assert protocol.is_sender
        assert protocol._sends_dense
        assert protocol.group_name == "miles-pp_0-ep_0"

    @pytest.mark.parametrize(
        ("global_rank", "pp_rank", "ep_rank", "is_sender", "sends_dense"),
        [
            (0, 0, 0, True, True),
            (1, 0, 0, False, False),
            (2, 0, 1, True, False),
            (3, 0, 1, False, False),
        ],
    )
    def test_ep_senders_share_one_world_lock(
        self,
        global_rank: int,
        pp_rank: int,
        ep_rank: int,
        is_sender: bool,
        sends_dense: bool,
    ) -> None:
        protocol, create_lock = _configure_protocol(
            global_rank=global_rank,
            coordinates=[(0, 0), (0, 0), (0, 1), (0, 1)],
            pp_rank=pp_rank,
            ep_rank=ep_rank,
        )

        create_lock.assert_called_once_with(prefix=_PREFIX, participates=is_sender)
        assert protocol._engine_lock is create_lock.return_value
        assert protocol.is_sender is is_sender
        assert protocol._sends_dense is sends_dense

    def test_sender_selection_does_not_assume_regular_tp_matches_expert_tp(self) -> None:
        protocol, create_lock = _configure_protocol(
            global_rank=2,
            coordinates=[(0, 0), (0, 0), (0, 1), (0, 1)],
            pp_rank=0,
            ep_rank=1,
            tp_rank=1,
            etp_rank=0,
        )

        assert protocol.is_sender
        assert not protocol._sends_dense
        assert protocol.group_name == "miles-pp_0-ep_1"
        create_lock.assert_called_once_with(prefix=_PREFIX, participates=True)

    @pytest.mark.parametrize(
        ("global_rank", "pp_rank", "ep_rank", "expected_group"),
        [
            (0, 0, 0, "miles-pp_0-ep_0"),
            (1, 0, 1, "miles-pp_0-ep_1"),
            (2, 1, 0, "miles-pp_1-ep_0"),
            (3, 1, 1, "miles-pp_1-ep_1"),
        ],
    )
    def test_each_retained_pp_ep_shard_gets_a_unique_group(
        self, global_rank: int, pp_rank: int, ep_rank: int, expected_group: str
    ) -> None:
        protocol, _ = _configure_protocol(
            global_rank=global_rank,
            coordinates=[(0, 0), (0, 1), (1, 0), (1, 1)],
            pp_rank=pp_rank,
            ep_rank=ep_rank,
        )

        assert protocol.is_sender
        assert protocol.group_name == expected_group

    def test_gathered_ep_keeps_the_existing_pp_group_name(self) -> None:
        protocol, _ = _configure_protocol(
            global_rank=1,
            coordinates=[(0, 0), (1, 0)],
            pp_rank=1,
            ep_rank=3,
            placement=WeightUpdatePlacement(gather_pp=False, gather_ep=True),
        )

        assert protocol.group_name == "miles-pp_1"


class TestWeightUnitOwnership:
    @staticmethod
    def _unit(*names: str) -> list[tuple[str, torch.Tensor]]:
        return [(name, torch.empty(1)) for name in names]

    def test_dense_sender_accepts_every_unit(self) -> None:
        protocol, _ = _configure_protocol(
            global_rank=0,
            coordinates=[(0, 0), (0, 1)],
            pp_rank=0,
            ep_rank=0,
        )

        assert protocol.should_send_weight_unit(self._unit("model.embed_tokens.weight"))
        assert protocol.should_send_weight_unit(self._unit("model.layers.0.mlp.experts.0.gate_proj.weight"))

    @pytest.mark.parametrize(
        "name",
        [
            "model.embed_tokens.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.layers.0.mlp.shared_experts.gate_proj.weight",
            "model.layers.0.mlp.experts.gate_up_proj",
        ],
    )
    def test_expert_only_sender_rejects_units_without_a_numeric_expert_id(self, name: str) -> None:
        protocol, _ = _configure_protocol(
            global_rank=1,
            coordinates=[(0, 0), (0, 1)],
            pp_rank=0,
            ep_rank=1,
        )

        assert not protocol.should_send_weight_unit(self._unit(name))

    def test_expert_only_sender_accepts_an_atomic_unit_of_routed_expert_weights(self) -> None:
        protocol, _ = _configure_protocol(
            global_rank=1,
            coordinates=[(0, 0), (0, 1)],
            pp_rank=0,
            ep_rank=1,
        )

        assert protocol.should_send_weight_unit(
            self._unit(
                "model.layers.0.mlp.experts.37.gate_proj.weight",
                "model.layers.0.mlp.experts.37.up_proj.weight",
            )
        )

    def test_mixed_dense_and_expert_atomic_unit_is_rejected(self) -> None:
        protocol, _ = _configure_protocol(
            global_rank=1,
            coordinates=[(0, 0), (0, 1)],
            pp_rank=0,
            ep_rank=1,
        )

        with pytest.raises(AssertionError, match="mix"):
            protocol.should_send_weight_unit(
                self._unit(
                    "model.layers.0.mlp.experts.37.gate_proj.weight",
                    "model.layers.0.input_layernorm.weight",
                )
            )


class TestEngineReconnect:
    class _RecordingLock:
        def __init__(self, events: list[str]) -> None:
            self.events = events

        def __enter__(self):
            self.events.append("enter")

        def __exit__(self, *exc_info):
            self.events.append("exit")

    def test_reconnecting_to_the_engines_does_not_rebuild_the_lock(self) -> None:
        protocol, create_lock = _configure_protocol(
            global_rank=0,
            coordinates=[(0, 0), (0, 1)],
            pp_rank=0,
            ep_rank=0,
        )
        lock_after_init = protocol._engine_lock
        engines = [MagicMock()]
        parallel_state = _parallel_state(pp_rank=0, ep_rank=0)

        with (
            patch(f"{_BROADCAST_MODULE}.disconnect_rollout_engines_from_distributed"),
            patch(f"{_BROADCAST_MODULE}.connect_rollout_engines_from_distributed"),
        ):
            for _ in range(2):
                protocol.connect(
                    engines,
                    engine_gpu_counts=[2],
                    engine_gpu_offsets=None,
                    parallel_state=parallel_state,
                    placement=WeightUpdatePlacement(gather_pp=False, gather_ep=False),
                    selector="all",
                )

        create_lock.assert_called_once()
        assert protocol._engine_lock is lock_after_init

    def test_disconnect_and_reconnect_are_serialized_and_forward_engine_sizes(self) -> None:
        events: list[str] = []
        protocol, _ = _configure_protocol(
            global_rank=0,
            coordinates=[(0, 0), (0, 1)],
            pp_rank=0,
            ep_rank=0,
        )
        protocol._engine_lock = self._RecordingLock(events)
        engines = [MagicMock(name="engine")]
        old_group = MagicMock(name="old_group")
        new_group = MagicMock(name="new_group")
        protocol._model_update_groups = old_group

        with (
            patch(
                f"{_BROADCAST_MODULE}.disconnect_rollout_engines_from_distributed",
                side_effect=lambda *args: events.append("disconnect"),
            ) as disconnect,
            patch(
                f"{_BROADCAST_MODULE}.connect_rollout_engines_from_distributed",
                side_effect=lambda *args, **kwargs: (events.append("connect"), new_group)[1],
            ) as connect,
        ):
            protocol.connect(
                engines,
                engine_gpu_counts=[2],
                engine_gpu_offsets=None,
                parallel_state=_parallel_state(pp_rank=0, ep_rank=0),
                placement=WeightUpdatePlacement(gather_pp=False, gather_ep=False),
                selector="all",
            )

        assert events == ["enter", "disconnect", "connect", "exit"]
        disconnect.assert_called_once_with(protocol.args, "miles-pp_0-ep_0", old_group, engines)
        connect.assert_called_once_with(protocol.args, "miles-pp_0-ep_0", engines, engine_gpu_counts=[2])
        assert protocol._model_update_groups is new_group

    def test_non_sender_does_not_connect_to_the_engines(self) -> None:
        protocol, _ = _configure_protocol(
            global_rank=1,
            coordinates=[(0, 0), (0, 0), (0, 1)],
            pp_rank=0,
            ep_rank=0,
        )

        with (
            patch(f"{_BROADCAST_MODULE}.disconnect_rollout_engines_from_distributed") as disconnect,
            patch(f"{_BROADCAST_MODULE}.connect_rollout_engines_from_distributed") as connect,
        ):
            protocol.connect(
                [MagicMock()],
                engine_gpu_counts=[1],
                engine_gpu_offsets=None,
                parallel_state=_parallel_state(pp_rank=0, ep_rank=0),
                placement=WeightUpdatePlacement(gather_pp=False, gather_ep=False),
                selector="all",
            )

        disconnect.assert_not_called()
        connect.assert_not_called()


class TestSendBucketUnderTheEngineLock:
    @staticmethod
    def _make_self(store: HashStore) -> SimpleNamespace:
        return SimpleNamespace(
            _engine_lock=StoreTicketLock(store=store, prefix=_PREFIX, poll_interval=0.001),
            group_name="miles-pp_0",
            _model_update_groups=MagicMock(name="nccl_group"),
            rollout_engines=[MagicMock(name="engine")],
            _selector="all",
        )

    @staticmethod
    def _run(
        fake_self: SimpleNamespace,
        bucket: list[tuple[str, torch.Tensor]],
        *,
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
            UpdateWeightFromDistributed.send_bucket(fake_self, bucket)
        return broadcast, wait_futures

    def test_a_finished_update_hands_the_lock_to_the_next_source(self) -> None:
        """The common path: broadcast, drop the bucket, then call the next ticket."""
        store = HashStore()
        fake_self = self._make_self(store)
        bucket = [("w", torch.zeros(2))]

        broadcast, wait_futures = self._run(fake_self, bucket)

        broadcast.assert_called_once_with(
            "miles-pp_0",
            fake_self._model_update_groups,
            fake_self.rollout_engines,
            bucket,
            selector="all",
        )
        wait_futures.assert_called_once_with(broadcast.return_value)
        assert bucket == []
        assert store.add(_SERVING_KEY, 0) == 1

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
        bucket = [("w", torch.zeros(2))]

        with pytest.raises(RuntimeError, match="engine died"):
            self._run(fake_self, bucket, engine_failure=RuntimeError("engine died"))

        assert store.add(_SERVING_KEY, 0) == 0
        assert len(bucket) == 1

    def test_a_broadcast_failure_keeps_the_lock_and_the_bucket(self) -> None:
        """A source that fails before returning futures must retain its ticket and bucket."""
        store = HashStore()
        fake_self = self._make_self(store)
        bucket = [("w", torch.zeros(2))]

        with pytest.raises(RuntimeError, match="broadcast failed"):
            self._run(fake_self, bucket, broadcast_side_effect=RuntimeError("broadcast failed"))

        assert store.add(_SERVING_KEY, 0) == 0
        assert len(bucket) == 1
        with pytest.raises(AssertionError):
            self._run(fake_self, bucket)

    def test_a_source_that_failed_once_refuses_to_broadcast_again(self) -> None:
        """The retained ticket must block this rank too, not just the ranks behind it."""
        store = HashStore()
        fake_self = self._make_self(store)

        with pytest.raises(RuntimeError):
            self._run(fake_self, [("w", torch.zeros(2))], engine_failure=RuntimeError("engine died"))

        with pytest.raises(AssertionError):
            self._run(fake_self, [("w", torch.zeros(2))])
