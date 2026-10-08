from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import pytest

_WAIT_BOUND = 10.0


class TestSendBucket:
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

    def test_each_replica_loads_inside_its_parallelism_context(
        self,
        p2p_sender: Any,
        p2p_protocol: ModuleType,
        make_rollout_api: Any,
        make_bucket: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """sglang's sharded weight loaders read the rank at call time, which exists only inside its context."""

        @contextmanager
        def logged_parallelism_context(parallelism_config: Any) -> Iterator[None]:
            p2p_sender.log.append(("enter", parallelism_config.tp_rank))
            yield
            p2p_sender.log.append(("exit", parallelism_config.tp_rank))

        monkeypatch.setattr(p2p_protocol, "ParallelismContext", logged_parallelism_context)
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=2)
        p2p_sender.connect(protocol, [api])
        protocol.begin_sync(weight_version=1, iter_buckets=None)

        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert p2p_sender.log == [
            ("enter", 0),
            ("load", 0, ("hf.w",)),
            ("exit", 0),
            ("write", api.session_id(0)),
            ("enter", 1),
            ("load", 1, ("hf.w",)),
            ("exit", 1),
            ("write", api.session_id(1)),
        ]

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


class TestWriteThreads:
    def test_a_stuck_rollout_engine_does_not_hold_up_writes_to_another(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Each rollout engine has its own write thread, so a stuck engine delays only its own writes."""
        protocol = p2p_sender.make_protocol()
        stuck_api = make_rollout_api("cell-a", gpu_count=1)
        healthy_api = make_rollout_api("cell-b", gpu_count=1)
        p2p_sender.connect(protocol, [stuck_api, healthy_api])
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        stuck_write = p2p_sender.transfer_engine.hold(stuck_api.session_id(0))
        healthy_write = p2p_sender.transfer_engine.hold(healthy_api.session_id(0))

        protocol.send_bucket(make_bucket("hf.w"))

        assert stuck_write.entered.wait(timeout=_WAIT_BOUND)
        assert healthy_write.entered.wait(timeout=_WAIT_BOUND), "the healthy engine's write waited for the stuck one"
        p2p_sender.transfer_engine.release_all()
        protocol.after_base_weights()
        assert sorted(p2p_sender.transfer_engine.written_sessions()) == sorted(
            [stuck_api.session_id(0), healthy_api.session_id(0)]
        )


class TestWriteCompletion:
    def test_a_failed_write_to_the_last_rollout_engine_rank_fails_the_update(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A failed write leaves its rollout engine rank on the old weights, so the update must not succeed."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=2)
        p2p_sender.connect(protocol, [api])
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        p2p_sender.transfer_engine.failing_sessions = {api.session_id(1)}

        protocol.send_bucket(make_bucket("hf.w"))

        with pytest.raises(RuntimeError, match=api.session_id(1)):
            protocol.after_base_weights()

    def test_every_failed_write_is_named(self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any) -> None:
        """The error names each failed rollout engine rank, not only the first one to fail."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=2)
        p2p_sender.connect(protocol, [api])
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        p2p_sender.transfer_engine.failing_sessions = {api.session_id(0), api.session_id(1)}

        protocol.send_bucket(make_bucket("hf.w"))

        with pytest.raises(RuntimeError, match="2 of 2 p2p writes failed") as failure:
            protocol.after_base_weights()
        assert api.session_id(0) in str(failure.value)
        assert api.session_id(1) in str(failure.value)

    def test_a_write_still_running_at_the_timeout_fails_the_update(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A write that has not finished within --p2p-transfer-timeout fails the update instead of being forgotten."""
        protocol = p2p_sender.make_protocol(p2p_transfer_timeout=0.1)
        api = make_rollout_api("cell-a", gpu_count=1)
        p2p_sender.connect(protocol, [api])
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        p2p_sender.transfer_engine.hold(api.session_id(0))

        protocol.send_bucket(make_bucket("hf.w"))

        with pytest.raises(RuntimeError, match="still running after 0.1s"):
            protocol.after_base_weights()


class TestConnect:
    def test_ranks_are_assigned_over_the_rollout_engines_and_placement_handed_over(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Assigning over the engines in args queried engines that were not there and never wrote others; a fixed
        placement made every PP stage write the whole model under Megatron Bridge."""
        protocol = p2p_sender.make_protocol()
        protocol.args.rollout_num_gpus_per_engine = 1
        protocol.args.rollout_num_gpus = 1
        resolved_placement = object()
        api = make_rollout_api("cell-a", gpu_count=2)

        p2p_sender.connect(protocol, [api], placement=resolved_placement)
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert p2p_sender.assignment_inputs == [(resolved_placement, [2])]
        assert p2p_sender.transfer_engine.written_sessions() == [api.session_id(0), api.session_id(1)]
