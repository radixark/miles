import threading
from argparse import Namespace
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class _FakeReplica:
    def __init__(self, params: dict[str, torch.Tensor]):
        self._params = params

    def named_parameters(self):
        return list(self._params.items())

    def load_weights(self, named_tensors) -> None:
        pass


class _ReplicaFactory:
    def __init__(self):
        self.calls: list[SimpleNamespace] = []

    def __call__(self, parallelism_config, model_path, server_args, *, shared_params_dict, first_engine_rank):
        self.calls.append(SimpleNamespace(shared_params_dict=shared_params_dict, first_engine_rank=first_engine_rank))
        params = {"w": torch.zeros(2)} if first_engine_rank else dict(shared_params_dict)
        return _FakeReplica(params)


def _make_protocol(p2p, *, gathered_dp_rank: int = 0):
    with (
        patch.object(p2p, "RemoteTransferPlan"),
        patch.object(p2p, "dist") as dist_mock,
        patch.object(p2p, "get_gloo_group"),
    ):
        dist_mock.get_rank.return_value = 0
        protocol = p2p.UpdateWeightP2P(
            Namespace(hf_checkpoint="/ckpt", update_weight_engine_request_timeout=30.0, p2p_transfer_timeout=30.0)
        )
    protocol.transfer_plan._gathered_dp_rank = gathered_dp_rank
    return protocol


@contextmanager
def _patched_p2p(p2p, replica_factory: _ReplicaFactory):
    with (
        patch.object(p2p, "query_remote_weight_infos") as query,
        patch.object(p2p, "create_transfer_engine", side_effect=lambda: object()) as create_transfer_engine,
        patch.object(p2p, "_create_cpu_replica", side_effect=replica_factory),
        patch.object(p2p, "RankParallelismConfig"),
        patch.object(p2p, "ParameterMapper"),
    ):
        yield SimpleNamespace(query=query, create_transfer_engine=create_transfer_engine)


def _connect(
    protocol,
    patches,
    *,
    pairs: list[tuple[int, int]],
    session_prefix: str = "a",
    layout_of=lambda pair: {"tp_rank": pair[1]},
    quant_profile: str | None = None,
    engine_count: int | None = None,
    query_failures: dict[int, Exception] | None = None,
) -> None:
    query_failures = query_failures if query_failures is not None else {}
    protocol.transfer_plan.plan_p2p.return_value = [
        SimpleNamespace(engine_ind=engine_ind, engine_rank=engine_rank) for engine_ind, engine_rank in pairs
    ]
    answered = [pair for pair in pairs if pair[0] not in query_failures]
    targets_to_session_id = {pair: f"{session_prefix}-{pair[0]}-{pair[1]}" for pair in answered}
    patches.query.return_value = SimpleNamespace(
        remote_weight_infos_by_session_id={targets_to_session_id[pair]: ({}, layout_of(pair)) for pair in answered},
        targets_to_session_id=targets_to_session_id,
        session_id_to_server_args={
            targets_to_session_id[pair]: SimpleNamespace(rl_quant_profile=quant_profile) for pair in answered
        },
        failures_by_engine_ind=query_failures,
    )
    if engine_count is None:
        engine_count = 1 + max((engine_ind for engine_ind, _rank in pairs), default=-1)
    protocol.connect(
        [object()] * engine_count,
        [
            1 + max((rank for index, rank in pairs if index == engine_ind), default=0)
            for engine_ind in range(engine_count)
        ],
        None,
        [f"cell-{index}" for index in range(engine_count)],
        None,
        None,
        "all",
    )


def _blocked_executor(p2p, cell_id: str, release: threading.Event):
    executor = p2p._CellWriteExecutor(cell_id)
    started = threading.Event()

    def block() -> None:
        started.set()
        release.wait(timeout=30.0)

    executor.submit(block)
    assert started.wait(timeout=30.0)
    return executor


def _target_session_ids(protocol) -> list[str]:
    return [
        target.session_id
        for meta in protocol._transfer_engine_meta_list
        for cell_updater in meta.cell_updaters
        for target in cell_updater._target_by_engine_rank.values()
    ]


def _cell_updaters(protocol) -> list[object]:
    return [cell_updater for meta in protocol._transfer_engine_meta_list for cell_updater in meta.cell_updaters]


class TestReconnection:
    """The source side of a P2P connection outlives the targets it is currently writing to."""

    def test_a_repeated_connection_reuses_the_transfer_engine_and_the_cpu_replicas(self, p2p_protocol) -> None:
        """Rebuilding the pinned CPU model on every reconnect would cost a full model load and re-registration."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0), (1, 0)])
            engine_after_first = protocol._transfer_engine
            params_after_first = protocol._shared_params_dict
            _connect(protocol, patches, pairs=[(0, 0), (1, 0)], session_prefix="b")

            assert patches.create_transfer_engine.call_count == 1
        assert len(factory.calls) == 1
        assert protocol._transfer_engine is engine_after_first
        assert protocol._shared_params_dict is params_after_first

    def test_a_reconnection_replaces_every_target_session(self, p2p_protocol) -> None:
        """A stale session id points at an engine that was already killed, so writes would go nowhere."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0), (1, 0)])
            first_cell_updaters = _cell_updaters(protocol)
            _connect(protocol, patches, pairs=[(0, 0), (1, 0)], session_prefix="b")

        assert sorted(_target_session_ids(protocol)) == ["b-0-0", "b-1-0"]
        assert not set(map(id, _cell_updaters(protocol))) & set(map(id, first_cell_updaters))

    def test_a_reconnection_drains_the_pending_writes_before_dropping_the_targets(self, p2p_protocol) -> None:
        """A write still in flight reads the shared buffers, so the targets must not be replaced under it."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()
        observed_targets_at_drain: list[list[str]] = []
        real_drain = protocol._drain_pending_writes

        def recording_drain() -> None:
            observed_targets_at_drain.append(sorted(_target_session_ids(protocol)))
            real_drain()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            protocol._drain_pending_writes = recording_drain
            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")

        assert observed_targets_at_drain == [["a-0-0"]]

    def test_a_write_that_never_finished_is_kept_and_does_not_block_the_reconnection(self, p2p_protocol) -> None:
        """The trainer has to reconnect to its healthy cells, but must not forget a write still reading its buffers."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()
        protocol._transfer_timeout = 0.05
        release = threading.Event()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            engine_after_first = protocol._transfer_engine
            params_after_first = protocol._shared_params_dict
            cell_updater = protocol._cell_updaters_by_cell_id["cell-0"]
            cell_updater._transfer_timeout = 0.05
            stuck = cell_updater._executor.submit(lambda: release.wait(timeout=30.0))
            cell_updater._pending_writes.append(stuck)

            try:
                _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")
            finally:
                release.set()

        assert protocol._unfinished_writes == [stuck]
        assert len(protocol._stalled_executors) == 1
        assert protocol._transfer_engine is engine_after_first
        assert protocol._shared_params_dict is params_after_first
        assert sorted(_target_session_ids(protocol)) == ["b-0-0"]

    def test_a_rank_without_targets_drops_its_targets_and_keeps_its_buffers(self, p2p_protocol) -> None:
        """Losing every target must not free registered source memory the transfer engine still knows about."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            engine_after_first = protocol._transfer_engine
            params_after_first = protocol._shared_params_dict
            protocol._weight_memory_registry = {"w": (0x1000, 2, 4)}
            _connect(protocol, patches, pairs=[])

        assert protocol.is_sender is False
        assert protocol._transfer_engine_meta_list == []
        assert protocol._transfer_engine is engine_after_first
        assert protocol._shared_params_dict is params_after_first
        assert protocol._weight_memory_registry == {"w": (0x1000, 2, 4)}

    def test_a_newly_assigned_engine_rank_aliases_the_existing_shared_buffers(self, p2p_protocol) -> None:
        """A second sharding layout must reuse the registered buffers instead of allocating another pinned model."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            params_after_first = protocol._shared_params_dict
            _connect(protocol, patches, pairs=[(0, 0), (0, 1)], session_prefix="b")

        assert [call.first_engine_rank for call in factory.calls] == [True, False]
        assert factory.calls[1].shared_params_dict is params_after_first
        assert len(protocol._replicas_by_representation) == 2

    def test_the_source_memory_registration_survives_a_reconnection(self, p2p_protocol) -> None:
        """Re-registering the same buffers on every reconnect would leak registrations inside the transfer engine."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            with patch.object(p2p_protocol, "register_cpu_memory", return_value={"w": (0x1000, 2, 4)}) as register:
                protocol.begin_sync(1, lambda **_kwargs: iter([]))
                _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")
                protocol.begin_sync(2, lambda **_kwargs: iter([]))

        assert register.call_count == 1
        assert protocol._weight_memory_registry == {"w": (0x1000, 2, 4)}

    def test_a_reconnection_drops_the_tensors_staged_by_an_abandoned_sync(self, p2p_protocol) -> None:
        """Shards left over from an interrupted update would be flushed into the next version's first bucket."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()
        mapper = SimpleNamespace(
            map=lambda name: SimpleNamespace(sglang_name="w", num_shards=2, num_local_experts=None)
        )

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            protocol._model_param_stager.get_transfer_ready_params(
                [("hf.q", torch.zeros(1))], param_mapper=mapper, params_dict={"w": torch.zeros(1)}
            )
            with pytest.raises(AssertionError):
                protocol._model_param_stager.assert_all_done()

            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")

        protocol._model_param_stager.assert_all_done()


class TestStandaloneDisconnect:
    """A disconnect on its own must leave nothing behind that could still address the old engines."""

    def test_a_disconnect_drops_every_client_and_target_reference(self, p2p_protocol) -> None:
        """A retained engine client or session map lets a later call publish weights to engines that are gone."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0), (1, 0)])
            engine_after_connect = protocol._transfer_engine
            params_after_connect = protocol._shared_params_dict
            protocol._weight_memory_registry = {"w": (0x1000, 2, 4)}

            protocol.disconnect()

        assert protocol.is_sender is False
        assert list(protocol.rollout_engines) == []
        assert protocol._transfer_engine_meta_list == []
        assert protocol.remote_weight_infos_by_session_id == {}
        assert protocol.session_id_to_server_args == {}
        assert protocol._transfer_engine is engine_after_connect
        assert protocol._shared_params_dict is params_after_connect
        assert protocol._weight_memory_registry == {"w": (0x1000, 2, 4)}
        assert len(protocol._replicas_by_representation) == 1

    def test_a_disconnected_protocol_neither_registers_nor_writes(self, p2p_protocol) -> None:
        """The updater keeps driving every rank, so a disconnected one must publish nothing to the old targets."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            protocol.disconnect()
            protocol._model_registered = False

            with patch.object(p2p_protocol, "register_cpu_memory") as register:
                assert protocol.begin_sync(3, lambda **_kwargs: iter([])) is True

            bucket = [("hf.w", torch.zeros(1))]
            protocol.send_bucket(bucket)
            protocol.after_base_weights()

        register.assert_not_called()
        assert protocol._cell_updaters_by_cell_id == {}
        assert len(bucket) == 1

    def test_a_connection_after_a_standalone_disconnect_serves_the_new_targets(self, p2p_protocol) -> None:
        """Disconnecting must not poison the protocol: the next reconnection has to work from the retained source."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            engine_after_connect = protocol._transfer_engine
            params_after_connect = protocol._shared_params_dict
            protocol.disconnect()
            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")

        assert protocol.is_sender is True
        assert sorted(_target_session_ids(protocol)) == ["b-0-0"]
        assert protocol._transfer_engine is engine_after_connect
        assert protocol._shared_params_dict is params_after_connect
        assert len(factory.calls) == 1


class TestCpuReplicaReuse:
    """A cached CPU replica may only be reused for a target whose weight representation it actually matches."""

    def test_a_changed_quantization_profile_builds_its_own_replica(self, p2p_protocol) -> None:
        """Reusing a replica built for another quantization profile would convert every tensor into the wrong bytes."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)], quant_profile="fp8")
            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b", quant_profile="bf16")

        assert len(factory.calls) == 2
        assert len(protocol._replicas_by_representation) == 2

    def test_an_unchanged_quantization_profile_reuses_the_replica(self, p2p_protocol) -> None:
        """Reloading the pinned CPU model on every reconnect costs a full model load for nothing."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)], quant_profile="fp8")
            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b", quant_profile="fp8")

        assert len(factory.calls) == 1
        assert len(protocol._replicas_by_representation) == 1

    def test_a_replacement_cell_on_other_gpus_reuses_the_replica(self, p2p_protocol) -> None:
        """A healed cell holds the same shard from other GPUs, so its placement must not force a second pinned model."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(
                protocol,
                patches,
                pairs=[(0, 0)],
                layout_of=lambda _pair: {"tp_rank": 0, "tp_size": 2, "global_rank": 4, "local_rank": 0},
            )
            _connect(
                protocol,
                patches,
                pairs=[(0, 0)],
                session_prefix="b",
                layout_of=lambda _pair: {"tp_rank": 0, "tp_size": 2, "global_rank": 9, "local_rank": 1},
            )

        assert len(factory.calls) == 1
        assert len(protocol._replicas_by_representation) == 1


class TestOneReplicaPerEngineRank:
    """All the cells of one engine rank are served from a single CPU replica, so they must agree on its layout."""

    def test_targets_of_one_engine_rank_with_different_sharding_are_rejected(self, p2p_protocol) -> None:
        """A TP1 and a TP2 cell hold different shards of the same rank, and one conversion cannot serve both."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with (
            _patched_p2p(p2p_protocol, factory) as patches,
            pytest.raises(AssertionError, match="different weight representations"),
        ):
            _connect(
                protocol,
                patches,
                pairs=[(0, 0), (1, 0)],
                layout_of=lambda pair: {"tp_rank": 0, "tp_size": 1 + pair[0]},
            )

    def test_targets_of_one_engine_rank_that_differ_only_in_placement_are_served_together(self, p2p_protocol) -> None:
        """Two cells holding the same shard on different GPUs are exactly the case this backend must support."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(
                protocol,
                patches,
                pairs=[(0, 0), (1, 0)],
                layout_of=lambda pair: {"tp_rank": 0, "tp_size": 2, "global_rank": 4 * pair[0], "local_rank": 0},
            )

        assert len(factory.calls) == 1
        assert len(_cell_updaters(protocol)) == 2
        assert sorted(_target_session_ids(protocol)) == ["a-0-0", "a-1-0"]


class TestInferenceCellState:
    """Every cell handed to this rank gets failure state, whether or not the rank writes to it."""

    def test_a_rank_without_any_target_still_tracks_every_supplied_cell(self, p2p_protocol) -> None:
        """Cross-rank failure aggregation names cells, so a rank that sends nothing must still know them."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[], engine_count=3)

        assert protocol.is_sender is False
        assert protocol.inference_cell_health.cell_ids == ("cell-0", "cell-1", "cell-2")
        assert sorted(protocol._cell_updaters_by_cell_id) == ["cell-0", "cell-1", "cell-2"]
        assert protocol.inference_cell_health.healthy_cell_ids == ["cell-0", "cell-1", "cell-2"]

    def test_a_sender_tracks_the_cells_it_holds_no_target_for(self, p2p_protocol) -> None:
        """The driver rank drives the session of every assigned cell, not only of the shards it sends."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)], engine_count=2)

        assert protocol.is_sender is True
        assert sorted(protocol._cell_updaters_by_cell_id) == ["cell-0", "cell-1"]
        assert protocol._cell_updaters_by_cell_id["cell-1"]._target_by_engine_rank == {}
        assert protocol.inference_cell_health.healthy_cell_ids == ["cell-0", "cell-1"]

    def test_a_reconnection_starts_a_new_verdict_for_the_new_engine_set(self, p2p_protocol) -> None:
        """The cells of a new connection are freshly assigned, so an old verdict must not disable them."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            old_health = protocol.inference_cell_health
            old_health.mark_errored("cell-0", RuntimeError("boom"))
            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")

        assert protocol.inference_cell_health is not old_health
        assert protocol.inference_cell_health.errored_cell_ids == []
        assert protocol._cell_updaters_by_cell_id["cell-0"].is_errored is False
        assert old_health.errored_cell_ids == ["cell-0"]

    def test_the_cell_updater_of_the_previous_connection_stays_errored(self, p2p_protocol) -> None:
        """A worker still running for an old incarnation must never see itself become healthy again."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            old_cell_updater = protocol._cell_updaters_by_cell_id["cell-0"]
            old_cell_updater.mark_errored(RuntimeError("boom"))
            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")

        assert old_cell_updater.is_errored is True
        assert old_cell_updater.is_disposed is True
        assert protocol._cell_updaters_by_cell_id["cell-0"].is_errored is False

    def test_a_late_failure_of_an_old_cell_does_not_reach_the_new_one(self, p2p_protocol) -> None:
        """The old worker reports under the same cell id, and its verdict would condemn a healthy new engine."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            old_cell_updater = protocol._cell_updaters_by_cell_id["cell-0"]
            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")
            new_cell_updater = protocol._cell_updaters_by_cell_id["cell-0"]

            old_cell_updater.mark_errored(RuntimeError("the write of the previous incarnation failed"))

        assert protocol.inference_cell_health.errored_cell_ids == []
        assert new_cell_updater.is_errored is False
        assert old_cell_updater.is_errored is True

    def test_a_disposed_cell_updater_submits_no_further_write(self, p2p_protocol) -> None:
        """An old target endpoint addresses an engine that was replaced, so writing to it corrupts nothing but wastes."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            old_cell_updater = protocol._cell_updaters_by_cell_id["cell-0"]
            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")

        assert old_cell_updater.accepts_writes is False
        assert old_cell_updater.submit_write(engine_rank=0, names=["w"], weight_memory_registry={}) is None

    def test_a_disconnect_leaves_no_cell_to_report_on(self, p2p_protocol) -> None:
        """A disconnected rank has no assignment left, so it must not keep answering for the old cells."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            old_health = protocol.inference_cell_health
            old_health.mark_errored("cell-0", RuntimeError("boom"))
            protocol.disconnect()

        assert protocol.inference_cell_health is not old_health
        assert protocol.inference_cell_health.cell_ids == ()
        assert protocol.inference_cell_health.errored_cell_ids == []
        assert protocol._cell_updaters_by_cell_id == {}
        assert old_health.errored_cell_ids == ["cell-0"]


class TestConnectWithADeadTarget:
    """An engine that cannot be queried at connect time only costs its own cell."""

    def test_a_dead_engine_marks_its_cell_and_leaves_the_others_connected(self, p2p_protocol) -> None:
        """Failing the whole connect would kill a healthy trainer over one unreachable inference cell."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(
                protocol,
                patches,
                pairs=[(0, 0), (1, 0)],
                query_failures={0: ConnectionError("engine 0 is unreachable")},
            )

        assert protocol.is_sender is True
        assert protocol.inference_cell_health.errored_cell_ids == ["cell-0"]
        assert sorted(_target_session_ids(protocol)) == ["a-1-0"]
        assert [meta.engine_rank for meta in protocol._transfer_engine_meta_list] == [0]

    def test_a_dead_engine_is_not_written_to_by_the_next_bucket(self, p2p_protocol) -> None:
        """The cell updater of a cell that never handed over its targets must submit nothing."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(
                protocol,
                patches,
                pairs=[(0, 0), (1, 0)],
                query_failures={0: ConnectionError("engine 0 is unreachable")},
            )

        dead = protocol._cell_updaters_by_cell_id["cell-0"]
        assert dead.is_errored is True
        assert dead._target_by_engine_rank == {}
        assert dead.submit_write(engine_rank=0, names=["w"], weight_memory_registry={}) is None

    def test_every_engine_of_a_rank_failing_leaves_that_rank_without_a_replica(self, p2p_protocol) -> None:
        """Building a CPU replica for a rank whose every target is gone would load a model for nobody."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(
                protocol,
                patches,
                pairs=[(0, 0)],
                query_failures={0: ConnectionError("engine 0 is unreachable")},
            )

        assert protocol.is_sender is False
        assert protocol._transfer_engine_meta_list == []
        assert factory.calls == []
        assert protocol.inference_cell_health.errored_cell_ids == ["cell-0"]


class TestConnectWhoseEveryTargetIsDead:
    """A rank left without a single usable target must stay out of the way, not break the trainer."""

    def test_no_usable_target_neither_stages_nor_registers(self, p2p_protocol) -> None:
        """With no CPU replica there is no parameter mapper, so staging a bucket would raise on the source side."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(
                protocol,
                patches,
                pairs=[(0, 0)],
                query_failures={0: ConnectionError("engine 0 is unreachable")},
            )
            with patch.object(p2p_protocol, "register_cpu_memory") as register:
                assert protocol.begin_sync(1, lambda **_kwargs: iter([])) is True

            bucket = [("hf.w", torch.zeros(1))]
            protocol.send_bucket(bucket)
            protocol.after_base_weights()

        register.assert_not_called()
        assert protocol._model_registered is False
        assert protocol._weight_memory_registry == {}
        assert len(bucket) == 1

    def test_a_later_connection_to_a_healthy_target_registers_real_buffers(self, p2p_protocol) -> None:
        """A registration marked done against an empty dict would leave the real pinned buffers unknown to RDMA."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()
        registered: list[dict] = []

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(
                protocol,
                patches,
                pairs=[(0, 0)],
                query_failures={0: ConnectionError("engine 0 is unreachable")},
            )
            protocol.begin_sync(1, lambda **_kwargs: iter([]))

            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")
            with patch.object(
                p2p_protocol,
                "register_cpu_memory",
                side_effect=lambda params, engine: registered.append(dict(params)) or {"w": (0x1000, 2, 4)},
            ):
                protocol.begin_sync(2, lambda **_kwargs: iter([]))

        assert protocol.is_sender is True
        assert [sorted(params) for params in registered] == [["w"]]
        assert protocol._weight_memory_registry == {"w": (0x1000, 2, 4)}
        assert protocol._model_registered is True

    def test_a_healthy_target_beside_a_dead_one_still_sends(self, p2p_protocol) -> None:
        """Isolating the dead target is worthless if the surviving one stops being written to."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(
                protocol,
                patches,
                pairs=[(0, 0), (1, 0)],
                query_failures={0: ConnectionError("engine 0 is unreachable")},
            )

        assert protocol.is_sender is True
        assert sorted(_target_session_ids(protocol)) == ["a-1-0"]
        assert protocol._cell_updaters_by_cell_id["cell-1"].accepts_writes is True


class TestPerCellWriteThreads:
    """Each inference cell writes from its own thread, and a stuck one is bounded rather than repeated forever."""

    def test_every_cell_gets_its_own_executor(self, p2p_protocol) -> None:
        """One shared pool lets a few stuck cells exhaust the workers of every healthy cell."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0), (1, 0)])

        executors = [updater._executor for updater in protocol._cell_updaters_by_cell_id.values()]
        assert len(executors) == 2
        assert len({id(executor) for executor in executors}) == 2
        assert [executor.cell_id for executor in executors] == ["cell-0", "cell-1"]

    def test_an_idle_cell_releases_its_executor_on_reconnection(self, p2p_protocol) -> None:
        """A thread per cell per reconnection would grow without bound over a long healing run."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            first_executor = protocol._cell_updaters_by_cell_id["cell-0"]._executor
            first_executor.submit(lambda: None).result(timeout=30.0)
            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")

        assert first_executor.is_running is False
        assert protocol._stalled_executors == []

    def test_a_write_queued_before_a_reconnection_never_reaches_the_new_cell(self, p2p_protocol) -> None:
        """The queued write still carries the target of the replaced engine, whose memory now belongs to somebody else."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()
        protocol._transfer_timeout = 0.05
        release = threading.Event()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            old_cell_updater = protocol._cell_updaters_by_cell_id["cell-0"]
            old_cell_updater._transfer_timeout = 0.05
            old_cell_updater._pending_writes.append(
                old_cell_updater._executor.submit(lambda: release.wait(timeout=30.0))
            )
            queued = old_cell_updater.submit_write(
                engine_rank=0, names=["w"], weight_memory_registry={"w": (0x1000, 2, 4)}
            )

            try:
                _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")
            finally:
                release.set()

        assert queued.cancelled() is True
        assert old_cell_updater.is_disposed is True
        assert old_cell_updater.accepts_writes is False
        assert protocol._cell_updaters_by_cell_id["cell-0"].is_errored is False
        assert protocol.inference_cell_health.errored_cell_ids == []

    def test_a_long_run_of_healthy_reconnections_strands_nothing(self, p2p_protocol) -> None:
        """Counting reconnections instead of live threads would retire a perfectly healthy trainer rank."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            for attempt in range(65):
                _connect(protocol, patches, pairs=[(0, 0)], session_prefix=f"gen{attempt}")
                cell_updater = protocol._cell_updaters_by_cell_id["cell-0"]
                cell_updater._executor.submit(lambda: None).result(timeout=30.0)
            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="last")

        assert protocol._stalled_executors == []
        assert protocol._unfinished_writes == []

    def test_a_write_thread_that_ends_late_is_not_counted_as_stalled(self, p2p_protocol) -> None:
        """A thread that had not yet reached the sentinel is not stuck, and would retire the rank for a scheduling gap."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()
        release = threading.Event()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            stalled = protocol._cell_updaters_by_cell_id["cell-0"]._executor
            stalled.submit(lambda: release.wait(timeout=30.0))
            protocol._cell_updaters_by_cell_id["cell-0"].dispose()
            protocol._stalled_executors.append(stalled)
            release.set()
            assert stalled.close(timeout=30.0) is True

            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")

        assert protocol._stalled_executors == []

    def test_enough_stuck_write_threads_fail_the_trainer_rank(self, p2p_protocol) -> None:
        """Reconnecting forever around a transfer engine that never completes a write would leak threads instead."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()
        release = threading.Event()
        already_stuck = [
            _blocked_executor(p2p_protocol, f"retired-{index}", release)
            for index in range(p2p_protocol._MAX_STALLED_WRITE_THREADS)
        ]
        protocol._stalled_executors = list(already_stuck)
        protocol._transfer_timeout = 0.05

        try:
            with _patched_p2p(p2p_protocol, factory) as patches:
                _connect(protocol, patches, pairs=[(0, 0)])
                cell_updater = protocol._cell_updaters_by_cell_id["cell-0"]
                cell_updater._transfer_timeout = 0.05
                cell_updater._pending_writes.append(cell_updater._executor.submit(lambda: release.wait(timeout=30.0)))

                with pytest.raises(AssertionError, match="stuck inside a native transfer"):
                    _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")

                assert len(protocol._stalled_executors) == p2p_protocol._MAX_STALLED_WRITE_THREADS + 1
                assert all(executor.is_running for executor in already_stuck)
        finally:
            release.set()
            for executor in protocol._stalled_executors:
                executor.close(timeout=30.0)
