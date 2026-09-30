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
