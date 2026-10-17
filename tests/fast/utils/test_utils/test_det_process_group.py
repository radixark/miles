from datetime import timedelta

import pytest
import torch

from tests.fast.utils.test_utils.det_process_group_fakes import DeferredProcessGroup


class TestCoalescedP2P:
    def test_aggregate_wait_completes_every_transfer_and_can_be_waited_again(self) -> None:
        """The aggregate preserves every deferred transfer and waits again for a new consumer."""
        group = DeferredProcessGroup()
        received = torch.full((2,), -1)
        group._start_coalescing(torch.device("cpu"))
        sent = group.send([torch.tensor([7])], 1, 0)
        recv = group.recv([received], 1, 0)
        aggregate = group._end_coalescing(torch.device("cpu"))

        assert group._inner.sent == []
        assert received.tolist() == [-1, -1]
        assert aggregate.wait()
        assert group._inner.sent[0].tolist() == [7]
        assert received.tolist() == [1, 1]
        assert aggregate.wait()
        assert sent.wait_count == recv.wait_count == 2

    def test_batches_have_independent_completion_and_plain_receives_keep_their_work(self) -> None:
        """Waiting a later batch cannot discard or complete an earlier batch's receive."""
        group = DeferredProcessGroup()
        first, second, plain = (torch.full((1,), -1) for _ in range(3))
        group._start_coalescing(torch.device("cpu"))
        group.recv([first], 1, 0)
        first_work = group._end_coalescing(torch.device("cpu"))
        group._start_coalescing(torch.device("cpu"))
        group.recv([second], 2, 0)
        second_work = group._end_coalescing(torch.device("cpu"))
        plain_work = group.recv([plain], 3, 0)

        assert second_work.wait()
        assert [first.item(), second.item(), plain.item()] == [-1, 2, -1]
        assert first_work.wait()
        assert plain_work.wait()
        assert [first.item(), second.item(), plain.item()] == [1, 2, 3]

    @pytest.mark.parametrize("raises", [False, True])
    def test_failed_transfer_is_not_reported_as_completed(self, raises: bool) -> None:
        """An unsuccessful or exceptional underlying wait remains visible to the caller."""
        group = DeferredProcessGroup()
        group._start_coalescing(torch.device("cpu"))
        work = group.recv([torch.zeros(1)], 1, 0)
        work.success = False
        if raises:
            work.error = RuntimeError("transfer failed")
        aggregate = group._end_coalescing(torch.device("cpu"))

        if raises:
            with pytest.raises(RuntimeError, match="transfer failed"):
                aggregate.wait()
        else:
            assert not aggregate.wait()

    def test_empty_batch_completes_and_nested_batches_are_rejected(self) -> None:
        """Empty batches complete while unsupported nesting raises immediately."""
        group = DeferredProcessGroup()
        group._start_coalescing(torch.device("cpu"))
        with pytest.raises(AssertionError, match="Nested"):
            group._start_coalescing(torch.device("cpu"))
        assert group._end_coalescing(torch.device("cpu")).wait()
        with pytest.raises(AssertionError, match="not started"):
            group._end_coalescing(torch.device("cpu"))

    def test_explicit_timeout_reaches_each_underlying_transfer(self) -> None:
        """Each native work retains its own timeout handling instead of silently ignoring it."""
        group = DeferredProcessGroup()
        group._start_coalescing(torch.device("cpu"))
        first = group.send([torch.ones(1)], 1, 0)
        second = group.recv([torch.zeros(1)], 1, 0)
        aggregate = group._end_coalescing(torch.device("cpu"))
        timeout = timedelta(seconds=2)

        assert aggregate.wait(timeout)
        assert first.timeouts == second.timeouts == [timeout]
