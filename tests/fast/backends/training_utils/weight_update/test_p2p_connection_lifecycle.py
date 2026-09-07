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


def _make_protocol(p2p, *, gathered_dp_rank: int = 0, rollout_num_gpus: int = 4):
    with (
        patch.object(p2p, "RemoteTransferPlan"),
        patch.object(p2p, "dist") as dist_mock,
        patch.object(p2p, "get_gloo_group"),
    ):
        dist_mock.get_rank.return_value = 0
        protocol = p2p.UpdateWeightP2P(Namespace(hf_checkpoint="/ckpt"))
    protocol.transfer_plan._gathered_dp_rank = gathered_dp_rank
    protocol.transfer_plan._rollout_num_gpus = rollout_num_gpus
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
) -> None:
    protocol.transfer_plan.plan_p2p.return_value = [
        SimpleNamespace(engine_ind=engine_ind, engine_rank=engine_rank) for engine_ind, engine_rank in pairs
    ]
    targets_to_session_id = {pair: f"{session_prefix}-{pair[0]}-{pair[1]}" for pair in pairs}
    patches.query.return_value = (
        {targets_to_session_id[pair]: ({}, layout_of(pair)) for pair in pairs},
        targets_to_session_id,
        {targets_to_session_id[pair]: SimpleNamespace(rl_quant_profile=quant_profile) for pair in pairs},
    )
    engine_count = 1 + max((engine_ind for engine_ind, _rank in pairs), default=-1)
    protocol.connect(
        [object()] * engine_count,
        None,
        None,
        [f"cell-{index}" for index in range(engine_count)],
        None,
        None,
        "all",
    )


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
        real_wait = protocol.transfer_manager.wait_transfers

        def recording_wait() -> None:
            observed_targets_at_drain.append(sorted(_target_session_ids(protocol)))
            real_wait()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            protocol.transfer_manager.wait_transfers = recording_wait
            _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")

        assert observed_targets_at_drain == [["a-0-0"]]

    def test_a_failed_drain_keeps_the_source_state_and_propagates(self, p2p_protocol) -> None:
        """The buffers a failed write may still be reading must stay alive, and the failure must not be hidden."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            engine_after_first = protocol._transfer_engine
            params_after_first = protocol._shared_params_dict

            def failing_wait() -> None:
                raise RuntimeError("[P2P] 1 of 1 transfers failed")

            protocol.transfer_manager.wait_transfers = failing_wait

            with pytest.raises(RuntimeError, match="transfers failed"):
                _connect(protocol, patches, pairs=[(0, 0)], session_prefix="b")

        assert protocol._transfer_engine is engine_after_first
        assert protocol._shared_params_dict is params_after_first
        assert sorted(_target_session_ids(protocol)) == ["a-0-0"]
        assert protocol.is_sender is True
        assert sorted(protocol.remote_weight_infos_by_session_id) == ["a-0-0"]

    def test_a_rank_without_targets_drops_its_targets_and_keeps_its_buffers(self, p2p_protocol) -> None:
        """Losing every target must not free registered source memory the transfer engine still knows about."""
        protocol = _make_protocol(p2p_protocol)
        factory = _ReplicaFactory()

        with _patched_p2p(p2p_protocol, factory) as patches:
            _connect(protocol, patches, pairs=[(0, 0)])
            engine_after_first = protocol._transfer_engine
            params_after_first = protocol._shared_params_dict
            protocol._weight_memory_registry = {"w": (0x1000, 2, 4)}
            protocol.transfer_plan._gathered_dp_rank = 99
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
        assert protocol.transfer_manager.transfer_futures == []
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
