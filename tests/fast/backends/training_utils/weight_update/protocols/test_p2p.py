from typing import Any

import pytest

_WAIT_BOUND = 10.0
_STILL_BLOCKED_SECONDS = 0.2


class TestSendBucket:
    def test_an_empty_bucket_loads_and_sends_nothing_and_the_next_bucket_still_goes_out(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """An empty bucket must not load a transfer buffer or issue empty writes, nor wedge the stream."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=2)
        p2p_sender.connect(protocol, [api])
        protocol.begin_sync(weight_version=1, iter_buckets=None)

        protocol.send_bucket([])
        log_after_empty_bucket = list(p2p_sender.log)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert log_after_empty_bucket == []
        assert [entry for entry in p2p_sender.log if entry[0] == "load"] == [
            ("load", 0, ("hf.w",)),
            ("load", 1, ("hf.w",)),
        ]
        assert p2p_sender.transfer_engine.written_sessions() == [api.session_id(0), api.session_id(1)]

    def test_a_fused_parameter_is_loaded_and_written_only_once_every_shard_arrived(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Loading one shard of a fused parameter would write a half-filled parameter to the engine."""
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


class TestTransferBuffers:
    def test_a_transfer_buffer_is_not_loaded_again_while_a_write_still_reads_it(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """With two buffers the third rank loads into the first rank's buffer; doing it while the first rank's write
        still reads that buffer would send the third rank's bytes to the first."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=3)
        p2p_sender.connect(protocol, [api])
        first_write = p2p_sender.transfer_engine.hold(api.session_id(0))

        call = p2p_sender.call_in_thread(lambda: protocol.send_bucket(make_bucket("hf.w")))

        assert first_write.entered.wait(timeout=_WAIT_BOUND)
        assert p2p_sender.loaded_event(1).wait(timeout=_WAIT_BOUND)
        assert not p2p_sender.loaded_event(2).wait(timeout=_STILL_BLOCKED_SECONDS)
        first_write.release.set()
        call.join()
        protocol.after_base_weights()
        assert p2p_sender.transfer_engine.payload_of(api.session_id(0)) == {
            api.target_address(0, "w"): [1.0, 2.0, 3.0, 4.0]
        }


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

    def test_rollout_engines_holding_one_rank_in_different_layouts_are_rejected(
        self, p2p_sender: Any, make_rollout_api: Any
    ) -> None:
        """One model replica serves all engines of a rank, so their layouts must match."""
        protocol = p2p_sender.make_protocol()
        trtllm_api = make_rollout_api("cell-a", gpu_count=1, moe_runner_backend="flashinfer_trtllm")
        triton_api = make_rollout_api("cell-b", gpu_count=1, moe_runner_backend="triton")

        with pytest.raises(AssertionError, match="different layouts"):
            p2p_sender.connect(protocol, [trtllm_api, triton_api])

    @pytest.mark.parametrize(
        "expert_placement",
        [{"ep_num_redundant_experts": 32}, {"init_expert_location": "/placement.json"}, {"enable_eplb": True}],
        ids=["redundant_experts", "init_expert_location", "eplb"],
    )
    def test_a_rollout_engine_placing_experts_a_replica_cannot_reproduce_is_rejected(
        self, p2p_sender: Any, make_rollout_api: Any, expert_placement: dict
    ) -> None:
        """The engine places these experts by metadata the replica does not have, so p2p would write them into the
        wrong slots."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1, expert_placement=expert_placement)

        with pytest.raises(AssertionError, match=f"rollout engine 0 places experts by {next(iter(expert_placement))}"):
            p2p_sender.connect(protocol, [api])

    def test_a_replica_that_does_not_match_the_published_weights_is_rejected(
        self, p2p_sender: Any, make_rollout_api: Any
    ) -> None:
        """Bytes loaded in a layout the rollout engine does not hold would land in the wrong places."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=1, published_weight_numel=2)

        with pytest.raises(AssertionError, match="does not match the weights rollout engine 0 rank 0 publishes"):
            p2p_sender.connect(protocol, [api])

    def test_a_reconnect_keeps_the_replica_and_the_registered_transfer_buffers(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Rebuilding the replica or its memory at a reconnect left the writes reading memory Mooncake never
        registered."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])
        protocol.begin_sync(weight_version=1, iter_buckets=None)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        replaced_api = make_rollout_api("cell-a", gpu_count=1, generation=2)
        p2p_sender.connect(protocol, [replaced_api])
        protocol.begin_sync(weight_version=2, iter_buckets=None)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert p2p_sender.transfer_engine.written_sessions()[-1] == replaced_api.session_id(0)
        assert len(p2p_sender.replicas_created) == 1
        assert p2p_sender.transfer_engines_created == 1
        assert len(p2p_sender.transfer_engine.registered) == 2
