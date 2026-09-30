from typing import Any

import pytest


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
