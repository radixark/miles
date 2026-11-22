from collections.abc import Callable
from typing import Any

import torch


class TestReconnect:
    def test_queued_old_writes_are_cancelled_and_replacement_writes_do_not_wait(
        self, p2p_sender: Any, transport: Any, make_remote_shard: Callable[..., Any]
    ) -> None:
        """Reconnection cancels queued stale sends while allowing the replacement to progress."""
        tensor = torch.ones(4)
        transport.register_memory(tensor)
        running_shard = make_remote_shard(cell_id="old-running")
        queued_shard = make_remote_shard(cell_id="old-queued")
        replacement_shard = make_remote_shard(cell_id="replacement")
        hold = p2p_sender.transfer_engine.hold(running_shard.transport_peer)
        running, queued = transport.write(
            [running_shard, queued_shard], {"w": tensor}, run_write=lambda shard, write: write()
        )

        try:
            assert hold.entered.wait(timeout=5)
            assert not queued.done()
            assert transport.connect({}, [], [], []) == {}
            assert queued.cancelled()
            assert not running.done()

            (replacement,) = transport.write(
                [replacement_shard], {"w": tensor}, run_write=lambda shard, write: write()
            )
            replacement.result(timeout=5)
            assert p2p_sender.transfer_engine.written_sessions() == [replacement_shard.transport_peer]
        finally:
            hold.release.set()
            running.result(timeout=5)

        assert queued.cancelled()
        assert p2p_sender.transfer_engine.written_sessions() == [
            replacement_shard.transport_peer,
            running_shard.transport_peer,
        ]
        transport.unregister_memory(tensor)
        assert p2p_sender.transfer_engine.registered == []
