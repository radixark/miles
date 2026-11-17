from typing import Any

import pytest


class TestCellWrites:
    def test_a_worker_failure_marks_only_its_submitting_cell(
        self, make_cell_updater: Any, make_remote_shard: Any
    ) -> None:
        """A failed worker must blame its submitting cell without poisoning a healthy peer."""
        broken, healthy = make_cell_updater("broken"), make_cell_updater("healthy")
        calls = []
        error = RuntimeError("write failed")

        def fail() -> None:
            raise error

        with pytest.raises(RuntimeError, match="write failed"):
            broken.run_write(fail, remote_shard=make_remote_shard("broken"), sent_checksums=None)
        healthy.run_write(
            lambda: calls.append("healthy"), remote_shard=make_remote_shard("healthy"), sent_checksums=None
        )

        assert broken._error is error
        assert not healthy.is_errored
        assert calls == ["healthy"]

    def test_an_errored_cell_skips_the_native_call(self, make_cell_updater: Any, make_remote_shard: Any) -> None:
        """The executor must recheck cell state before any queued write enters native code."""
        updater = make_cell_updater()
        updater.mark_errored(RuntimeError("lost"))
        calls = []

        updater.run_write(lambda: calls.append("written"), remote_shard=make_remote_shard(), sent_checksums=None)

        assert calls == []

    def test_a_failed_rank_prevents_native_writes_from_queued_tasks(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """A failed rank must prevent later queued ranks from entering the native writer."""
        protocol = p2p_sender.make_protocol()
        api = make_rollout_api("cell-a", gpu_count=2)
        p2p_sender.connect(protocol, [api])
        p2p_sender.begin_sync(protocol, weight_version=1)
        hold = p2p_sender.transfer_engine.hold(api.session_id(0))
        p2p_sender.transfer_engine.failing_sessions.add(api.session_id(0))
        protocol.send_bucket(make_bucket("hf.w"))
        assert hold.entered.wait(timeout=10)
        updater = protocol.cell_updaters_of_cell_id["cell-a"]
        queued = [
            write.future
            for writes in protocol._transfer_buffers._pending_writes_by_buffer_index
            for write in writes
            if not write.future.running()
        ]
        assert len(queued) == 1

        hold.release.set()
        protocol.after_base_weights()

        assert updater.is_errored
        if not queued[0].cancelled():
            queued[0].result(timeout=10)
        assert p2p_sender.transfer_engine.written_sessions() == [api.session_id(0)]


# ============================ transfer checksums ============================


class TestWriteChecksumFailures:
    @pytest.mark.parametrize("failure", ["error", "timeout", "native"])
    def test_failed_native_or_checksum_requests_remove_only_their_cell(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any, failure: str
    ) -> None:
        """Unverified receivers must be excluded while healthy peers complete the same update."""
        protocol = p2p_sender.make_protocol(check_weight_transfer_checksum=True)
        protocol.args.update_weight_engine_request_timeout = 0.02
        bad, good = make_rollout_api("bad", gpu_count=1), make_rollout_api("good", gpu_count=1)
        p2p_sender.connect(protocol, [bad, good])
        p2p_sender.begin_sync(protocol, weight_version=1)
        if failure == "error":
            bad.request_failures["check_weights target"] = ConnectionError("checksum unavailable")
        elif failure == "timeout":
            bad.hanging_requests.add("check_weights target")
        else:
            p2p_sender.transfer_engine.failing_sessions.add(bad.session_id(0))

        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert protocol.cell_updaters_of_cell_id["bad"].is_errored
        assert not protocol.cell_updaters_of_cell_id["good"].is_errored
        assert good.check_weights_calls == [dict(action="raw_checksum", names=["w"], selector="target")]
        if failure == "native":
            assert bad.check_weights_calls == []
        elif failure == "error":
            assert isinstance(protocol.cell_updaters_of_cell_id["bad"]._error, ConnectionError)
        else:
            assert isinstance(protocol.cell_updaters_of_cell_id["bad"]._error, TimeoutError)
