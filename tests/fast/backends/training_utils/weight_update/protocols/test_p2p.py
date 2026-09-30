from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


def _server_args(rl_quant_profile: str | None = None) -> Any:
    return SimpleNamespace(rl_quant_profile=rl_quant_profile)


class _FakeReplica:
    def __init__(self, name: str) -> None:
        self.name = name
        self.loaded: list[list[tuple[str, Any]]] = []

    def named_parameters(self) -> list[tuple[str, Any]]:
        return [(f"{self.name}.weight", object())]

    def load_weights(self, tensors: list[tuple[str, Any]]) -> None:
        self.loaded.append(list(tensors))


def _manager(p2p_protocol: ModuleType, replicas: list[_FakeReplica], monkeypatch) -> Any:
    created: list[bool] = []

    def _create(parallelism_config, model_path, server_args, shared_params_dict, first_rollout_engine_rank=False):
        created.append(first_rollout_engine_rank)
        return replicas[len(created) - 1]

    monkeypatch.setattr(p2p_protocol, "_create_cpu_replica", _create)
    monkeypatch.setattr(
        p2p_protocol, "ParameterMapper", type("_Mapper", (), {"from_model": staticmethod(lambda m: m)})
    )
    manager = p2p_protocol._CPUReplicasManager(model_path="/model")
    return manager, created


class TestCPUReplicasManager:
    """Bookkeeping of the CPU replicas the p2p sender stages its weights through."""

    def test_a_fresh_manager_holds_no_replica_and_no_shared_state(self, p2p_protocol: ModuleType) -> None:
        """The shared buffers only exist once a replica has published them, so they must start empty."""
        manager = p2p_protocol._CPUReplicasManager(model_path="/model")

        assert manager.replicas == []
        assert manager.shared_params_dict == {}
        assert manager.shared_param_mapper is None

    def test_the_first_replica_publishes_the_shared_buffers_and_the_mapper(
        self, p2p_protocol: ModuleType, monkeypatch
    ) -> None:
        """Every later rank writes into these buffers, so the first replica has to hand them over."""
        replica = _FakeReplica("first")
        manager, created = _manager(p2p_protocol, [replica], monkeypatch)

        manager.create_replica(parallelism_config=object(), server_args=object())

        assert created == [True]
        assert list(manager.shared_params_dict) == ["first.weight"]
        assert manager.shared_param_mapper is replica

    def test_only_the_first_replica_allocates_its_own_buffers(self, p2p_protocol: ModuleType, monkeypatch) -> None:
        """A second allocation would give one rank buffers nobody else writes into."""
        first, second = _FakeReplica("first"), _FakeReplica("second")
        manager, created = _manager(p2p_protocol, [first, second], monkeypatch)

        manager.create_replica(parallelism_config=object(), server_args=object())
        manager.create_replica(parallelism_config=object(), server_args=object())

        assert created == [True, False]
        assert list(manager.shared_params_dict) == ["first.weight"]
        assert manager.shared_param_mapper is first

    def test_every_created_replica_is_kept(self, p2p_protocol: ModuleType, monkeypatch) -> None:
        """A replica the manager forgets is a shard layout nothing can stage into any more."""
        first, second = _FakeReplica("first"), _FakeReplica("second")
        manager, _created = _manager(p2p_protocol, [first, second], monkeypatch)

        manager.create_replica(parallelism_config=object(), server_args=object())
        manager.create_replica(parallelism_config=object(), server_args=object())

        assert manager.replicas == [first, second]


class TestShardLayoutKey:
    """What makes two rollout engine ranks able to share one CPU replica."""

    def test_ranks_differing_only_in_placement_share_one_shard_layout(self, p2p_protocol: ModuleType) -> None:
        """The process a rank runs in says nothing about which slice of the weights it holds."""
        first = p2p_protocol._shard_layout_key({"tp_rank": 0, "global_rank": 3, "local_rank": 3}, _server_args())
        second = p2p_protocol._shard_layout_key({"tp_rank": 0, "global_rank": 9, "local_rank": 1}, _server_args())

        assert first == second

    def test_a_different_shard_index_is_a_different_shard_layout(self, p2p_protocol: ModuleType) -> None:
        """Two tp ranks hold different slices, so one replica cannot serve both."""
        first = p2p_protocol._shard_layout_key({"tp_rank": 0, "global_rank": 0}, _server_args())
        second = p2p_protocol._shard_layout_key({"tp_rank": 1, "global_rank": 0}, _server_args())

        assert first != second

    def test_a_different_quantization_profile_is_a_different_shard_layout(self, p2p_protocol: ModuleType) -> None:
        """The quantization profile decides the dtype of every buffer the replica allocates."""
        first = p2p_protocol._shard_layout_key({"tp_rank": 0}, _server_args(rl_quant_profile=None))
        second = p2p_protocol._shard_layout_key({"tp_rank": 0}, _server_args(rl_quant_profile="fp8"))

        assert first != second

    def test_the_key_does_not_depend_on_the_order_of_the_parallelism_fields(self, p2p_protocol: ModuleType) -> None:
        """The remote answers a mapping, whose iteration order must not split one layout into two."""
        first = p2p_protocol._shard_layout_key({"tp_rank": 1, "ep_rank": 2}, _server_args())
        second = p2p_protocol._shard_layout_key({"ep_rank": 2, "tp_rank": 1}, _server_args())

        assert first == second


class TestGetOrCreateReplica:
    """One replica per shard layout, reused across reconnects."""

    def test_one_shard_layout_is_built_only_once(self, p2p_protocol: ModuleType, monkeypatch) -> None:
        """Rebuilding a replica the sender already holds wastes host memory and re-registers its buffers."""
        first, second = _FakeReplica("first"), _FakeReplica("second")
        manager, created = _manager(p2p_protocol, [first, second], monkeypatch)
        monkeypatch.setattr(
            p2p_protocol, "RankParallelismConfig", type("_Cfg", (), {"from_dict": staticmethod(lambda d: d)})
        )

        replica = manager.get_or_create_replica(parallelism_info={"tp_rank": 0}, server_args=_server_args())
        again = manager.get_or_create_replica(parallelism_info={"tp_rank": 0}, server_args=_server_args())

        assert (replica, again) == (first, first)
        assert created == [True]

    def test_another_shard_layout_gets_its_own_replica(self, p2p_protocol: ModuleType, monkeypatch) -> None:
        """Two ranks sharded differently cannot read the same replica without sending each other's shard."""
        first, second = _FakeReplica("first"), _FakeReplica("second")
        manager, created = _manager(p2p_protocol, [first, second], monkeypatch)
        monkeypatch.setattr(
            p2p_protocol, "RankParallelismConfig", type("_Cfg", (), {"from_dict": staticmethod(lambda d: d)})
        )

        manager.get_or_create_replica(parallelism_info={"tp_rank": 0}, server_args=_server_args())
        manager.get_or_create_replica(parallelism_info={"tp_rank": 1}, server_args=_server_args())

        assert manager.replicas == [first, second]
        assert created == [True, False]


class TestAssertOneShardLayout:
    """The guard protecting the CPU replica shared by the targets of one rollout engine rank."""

    def test_targets_holding_the_same_shard_agree(self, p2p_protocol: ModuleType) -> None:
        """Two engines whose rank 0 holds the same slice can be written from one replica."""
        p2p_protocol._assert_one_shard_layout(
            rollout_engine_rank=0,
            session_ids=["session-a", "session-b"],
            remote_weight_infos_by_session_id={
                "session-a": ({}, {"tp_rank": 0, "global_rank": 0}),
                "session-b": ({}, {"tp_rank": 0, "global_rank": 8}),
            },
            session_id_to_server_args={"session-a": _server_args(), "session-b": _server_args()},
        )

    def test_targets_holding_different_shards_are_rejected(self, p2p_protocol: ModuleType) -> None:
        """One replica can only hold one slice, so a mismatch would send the wrong weights."""
        with pytest.raises(AssertionError, match="rollout engine rank 1 hold different shard layouts"):
            p2p_protocol._assert_one_shard_layout(
                rollout_engine_rank=1,
                session_ids=["session-a", "session-b"],
                remote_weight_infos_by_session_id={
                    "session-a": ({}, {"tp_rank": 1}),
                    "session-b": ({}, {"tp_rank": 2}),
                },
                session_id_to_server_args={"session-a": _server_args(), "session-b": _server_args()},
            )

    def test_targets_quantized_differently_are_rejected(self, p2p_protocol: ModuleType) -> None:
        """The same slice in another quantization profile still needs its own replica."""
        with pytest.raises(AssertionError, match="cannot share one CPU replica"):
            p2p_protocol._assert_one_shard_layout(
                rollout_engine_rank=0,
                session_ids=["session-a", "session-b"],
                remote_weight_infos_by_session_id={
                    "session-a": ({}, {"tp_rank": 0}),
                    "session-b": ({}, {"tp_rank": 0}),
                },
                session_id_to_server_args={
                    "session-a": _server_args(),
                    "session-b": _server_args(rl_quant_profile="fp8"),
                },
            )


class TestConnectReusesOneShotResources:
    def test_a_reconnect_with_the_same_layout_reuses_the_transfer_engine_replicas_and_registration(
        self, p2p_sender: Any, make_rollout_api: Any
    ) -> None:
        """Rebuilding the replicas or the engine on reconnect would reallocate and re-register the pinned buffers."""
        protocol = p2p_sender.make_protocol()

        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=2)])
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=2, generation=2)])
        protocol.begin_sync(weight_version=2, iter_buckets=None)

        assert p2p_sender.transfer_engines_created == 1
        assert p2p_sender.replicas_created == [(0, True), (1, False)]
        assert len(p2p_sender.transfer_engine.registered) == 2

    def test_a_reconnect_writes_only_to_the_sessions_of_the_new_peers(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A restarted engine gets new sessions, so a leftover mapping would write into memory it released."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(
            protocol, [make_rollout_api("cell-a", gpu_count=2), make_rollout_api("cell-b", gpu_count=1)]
        )
        restarted = make_rollout_api("cell-a", gpu_count=2, generation=2)

        p2p_sender.connect(protocol, [restarted])
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert sorted(p2p_sender.transfer_engine.written_sessions()) == [
            restarted.session_id(0),
            restarted.session_id(1),
        ]
        assert list(protocol._cell_updaters_of_cell_id) == ["cell-a"]

    def test_a_round_without_reachable_peers_cuts_off_the_old_peers_and_keeps_the_resources(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A sender left without peers must stop writing to the last round's sessions yet keep its buffers for later."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=2)])
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()
        written_while_connected = p2p_sender.transfer_engine.written_sessions()

        p2p_sender.connect(protocol, [])
        protocol.begin_sync(weight_version=2, iter_buckets=None)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()
        written_while_cut_off = p2p_sender.transfer_engine.written_sessions()[len(written_while_connected) :]
        sender_while_cut_off = protocol.is_sender

        returned = make_rollout_api("cell-a", gpu_count=2, generation=3)
        p2p_sender.connect(protocol, [returned])
        protocol.begin_sync(weight_version=3, iter_buckets=None)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert sender_while_cut_off is False
        assert written_while_cut_off == []
        assert sorted(p2p_sender.transfer_engine.written_sessions()[len(written_while_connected) :]) == [
            returned.session_id(0),
            returned.session_id(1),
        ]
        assert p2p_sender.transfer_engines_created == 1
        assert p2p_sender.replicas_created == [(0, True), (1, False)]
        assert len(p2p_sender.transfer_engine.registered) == 2


class TestDisconnect:
    def test_a_disconnect_returns_only_after_the_write_in_flight_finished(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Forgetting a peer mid-write would let the next round overwrite the buffer that write still reads."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1)
        p2p_sender.connect(protocol, [api])
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        hold = p2p_sender.transfer_engine.hold(api.session_id(0))
        protocol.send_bucket(make_bucket("hf.w"))
        assert hold.entered.wait(timeout=10)

        disconnect = p2p_sender.call_in_thread(protocol.disconnect)
        state = disconnect.wait_until_draining_or_returned()
        hold.release.set()
        disconnect.join()

        assert state == "draining"
        assert ("write", api.session_id(0)) in disconnect.log_at_return

    def test_a_shard_staged_before_a_reconnect_does_not_leak_into_the_next_round(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A half-staged fused parameter from a lost round must not fail the next round's completeness check."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        protocol.send_bucket(make_bucket("hf.q"))

        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1, generation=2)])
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()


class TestSendBucket:
    def test_a_rank_is_written_before_the_next_rank_overwrites_the_shared_buffer(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """The next rank reuses the same pinned memory, so loading it early would ship its shard to this rank."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=2)
        p2p_sender.connect(protocol, [api])
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        hold = p2p_sender.transfer_engine.hold(api.session_id(0))

        send = p2p_sender.call_in_thread(lambda: protocol.send_bucket(make_bucket("hf.w")))
        assert hold.entered.wait(timeout=10)
        state = send.wait_until_draining_or_returned(extra=p2p_sender.loaded_event(1).is_set)
        hold.release.set()
        send.join()
        protocol.after_base_weights()

        assert state == "draining"
        assert p2p_sender.log == [
            ("load", 0, ("hf.w",)),
            ("write", api.session_id(0)),
            ("load", 1, ("hf.w",)),
            ("write", api.session_id(1)),
        ]
        assert p2p_sender.transfer_engine.payload_of(api.session_id(0)) == {
            api.target_address(0, "w"): [1.0, 2.0, 3.0, 4.0]
        }
        assert p2p_sender.transfer_engine.payload_of(api.session_id(1)) == {
            api.target_address(1, "w"): [101.0, 102.0, 103.0, 104.0]
        }

    def test_the_last_rank_is_left_in_flight_until_the_base_weights_are_done(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Nothing overwrites the buffer after the last rank, so it overlaps the stream and is drained at the end."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1)
        p2p_sender.connect(protocol, [api])
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        hold = p2p_sender.transfer_engine.hold(api.session_id(0))

        send = p2p_sender.call_in_thread(lambda: protocol.send_bucket(make_bucket("hf.w")))
        assert hold.entered.wait(timeout=10)
        send_state = send.wait_until_draining_or_returned()
        drain = p2p_sender.call_in_thread(protocol.after_base_weights)
        drain_state = drain.wait_until_draining_or_returned()
        hold.release.set()
        send.join()
        drain.join()

        assert send_state == "returned"
        assert drain_state == "draining"
        assert ("write", api.session_id(0)) in drain.log_at_return

    def test_an_empty_bucket_loads_and_sends_nothing_and_the_next_bucket_still_goes_out(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """An empty bucket must not reload the shared buffer or issue empty writes, nor wedge the stream."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=2)
        p2p_sender.connect(protocol, [api])
        protocol.begin_sync(weight_version=1, iter_buckets=None)

        protocol.send_bucket([])
        log_after_empty_bucket = list(p2p_sender.log)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert log_after_empty_bucket == []
        assert p2p_sender.log == [
            ("load", 0, ("hf.w",)),
            ("write", api.session_id(0)),
            ("load", 1, ("hf.w",)),
            ("write", api.session_id(1)),
        ]

    def test_a_fused_parameter_is_loaded_and_written_only_once_every_shard_arrived(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Loading one shard of a fused parameter would ship a half-overwritten buffer to the engine."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1)
        p2p_sender.connect(protocol, [api])
        protocol.begin_sync(weight_version=1, iter_buckets=None)

        protocol.send_bucket(make_bucket("hf.q"))
        log_after_first_shard = list(p2p_sender.log)
        protocol.send_bucket(make_bucket("hf.k"))
        protocol.after_base_weights()

        assert log_after_first_shard == []
        assert p2p_sender.log == [("load", 0, ("hf.q", "hf.k")), ("write", api.session_id(0))]
        assert p2p_sender.transfer_engine.payload_of(api.session_id(0)) == {
            api.target_address(0, "qk"): [5.0, 6.0, 7.0, 8.0]
        }

    def test_a_parameter_still_missing_a_shard_fails_the_end_of_the_base_weights(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A shard that never arrives would otherwise leave the engine serving a stale parameter silently."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])
        protocol.begin_sync(weight_version=1, iter_buckets=None)

        protocol.send_bucket(make_bucket("hf.q"))

        with pytest.raises(AssertionError, match="not transferred"):
            protocol.after_base_weights()


class TestCreateCPUReplica:
    def test_the_loader_post_load_hook_is_a_noop_only_while_the_model_loads(
        self, p2p_protocol: ModuleType, model_loader_sdk: Any, shared_buffers: dict[str, torch.Tensor]
    ) -> None:
        """get_model runs the hook internally and it may launch CUDA kernels, but it must come back afterwards."""
        p2p_protocol._create_cpu_replica({"tp_rank": 1}, "/model", _server_args(), shared_params_dict=shared_buffers)

        assert model_loader_sdk.hook_result_during_load is None
        assert model_loader_sdk.loader.post_load_weights is model_loader_sdk.original_post_load_weights

    def test_a_failing_model_load_still_restores_the_loader_post_load_hook(
        self, p2p_protocol: ModuleType, model_loader_sdk: Any, shared_buffers: dict[str, torch.Tensor]
    ) -> None:
        """A leaked no-op hook would silently skip post-processing for every later sglang model load."""
        model_loader_sdk.load_error = RuntimeError("checkpoint unreadable")

        with pytest.raises(RuntimeError, match="checkpoint unreadable"):
            p2p_protocol._create_cpu_replica(
                {"tp_rank": 1}, "/model", _server_args(), shared_params_dict=shared_buffers
            )

        assert model_loader_sdk.loader.post_load_weights is model_loader_sdk.original_post_load_weights

    def test_the_replicas_own_post_load_hook_is_a_noop(
        self, p2p_protocol: ModuleType, model_loader_sdk: Any, shared_buffers: dict[str, torch.Tensor]
    ) -> None:
        """Later load_weights calls invoke the model's hook, which would run CUDA-only code on the CPU replica."""
        replica = p2p_protocol._create_cpu_replica(
            {"tp_rank": 1}, "/model", _server_args(), shared_params_dict=shared_buffers
        )

        assert replica.post_load_weights() is None

    def test_a_later_replica_aliases_every_parameter_to_the_shared_buffers(
        self, p2p_protocol: ModuleType, model_loader_sdk: Any, shared_buffers: dict[str, torch.Tensor]
    ) -> None:
        """A copy instead of an alias would load into memory the transfer engine never reads."""
        replica = p2p_protocol._create_cpu_replica(
            {"tp_rank": 1}, "/model", _server_args(), shared_params_dict=shared_buffers
        )

        assert {name: param.data_ptr() for name, param in replica.named_parameters()} == {
            name: tensor.data_ptr() for name, tensor in shared_buffers.items()
        }

    def test_a_parameter_missing_from_the_shared_buffers_is_rejected(
        self, p2p_protocol: ModuleType, model_loader_sdk: Any, shared_buffers: dict[str, torch.Tensor]
    ) -> None:
        """A parameter with no shared buffer has no registered memory to be written from."""
        del shared_buffers["b"]

        with pytest.raises(AssertionError, match="Parameter b not found in shared buffers"):
            p2p_protocol._create_cpu_replica(
                {"tp_rank": 1}, "/model", _server_args(), shared_params_dict=shared_buffers
            )

    @pytest.mark.parametrize(
        "mismatched", [torch.zeros(4), torch.zeros(3, dtype=torch.bfloat16)], ids=["shape", "dtype"]
    )
    def test_a_shared_buffer_of_another_shape_or_dtype_cannot_be_aliased(
        self,
        p2p_protocol: ModuleType,
        model_loader_sdk: Any,
        shared_buffers: dict[str, torch.Tensor],
        mismatched: torch.Tensor,
    ) -> None:
        """Aliasing a differently laid out buffer would write this shard's bytes over the wrong elements."""
        shared_buffers["b"] = mismatched

        with pytest.raises(AssertionError, match="Parameter b cannot alias the shared buffer"):
            p2p_protocol._create_cpu_replica(
                {"tp_rank": 1}, "/model", _server_args(), shared_params_dict=shared_buffers
            )
