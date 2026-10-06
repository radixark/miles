from typing import Any

import pytest

_WAIT_BOUND = 10.0


class TestSendBucket:
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
