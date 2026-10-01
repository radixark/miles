"""Rank-local failures reach every rank instead of leaving peers on a collective."""

import pytest
from torch.distributed.checkpoint.api import CheckpointException

from miles.utils import distributed_utils
from miles.utils.distributed_utils import RankFailureError, raise_if_any_rank_failed, run_on_all_ranks, run_on_rank0


def _two_ranks(monkeypatch, peer_message):
    gathered = []

    def all_gather_object(output, local_message, group):
        gathered.append(local_message)
        output[:] = [local_message, peer_message]

    monkeypatch.setattr(distributed_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(distributed_utils.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", object())
    return gathered


def test_without_process_group_the_local_error_is_raised_unchanged():
    error = OSError("disk full")

    def fail():
        raise error

    with pytest.raises(OSError) as caught:
        run_on_all_ranks("saving", fail)
    assert caught.value is error
    run_on_all_ranks("saving", lambda: None)


def test_a_peer_failure_is_raised_on_a_healthy_rank(monkeypatch):
    gathered = _two_ranks(monkeypatch, "OSError: disk full")

    with pytest.raises(RankFailureError, match=r"saving failed \(rank 1: OSError: disk full\)"):
        run_on_all_ranks("saving", lambda: None)
    assert gathered == [None]


def test_a_local_failure_is_shared_and_chained(monkeypatch):
    gathered = _two_ranks(monkeypatch, None)
    error = ValueError("bad shard")

    with pytest.raises(RankFailureError, match="rank 0: ValueError: bad shard") as caught:
        raise_if_any_rank_failed("saving", error)
    assert caught.value.__cause__ is error
    assert gathered == ["ValueError: bad shard"]


def test_no_failures_is_a_no_op(monkeypatch):
    _two_ranks(monkeypatch, None)

    raise_if_any_rank_failed("saving", None)


def test_run_on_all_ranks_returns_the_result(monkeypatch):
    _two_ranks(monkeypatch, None)

    assert run_on_all_ranks("loading", max, 2, 5) == 5


def test_run_on_rank0_skips_the_work_elsewhere_but_shares_its_failure(monkeypatch):
    _two_ranks(monkeypatch, None)
    gathered = []

    def all_gather_object(output, local_message, group):
        gathered.append(local_message)
        output[:] = ["OSError: read-only", local_message]

    monkeypatch.setattr(distributed_utils.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(distributed_utils.dist, "get_rank", lambda: 1)
    calls = []

    with pytest.raises(RankFailureError, match="rank 0: OSError: read-only"):
        run_on_rank0("writing the tracker", calls.append, "tracker")
    assert calls == []
    assert gathered == [None]


def test_dcp_checkpoint_errors_are_shared_but_interrupts_are_not(monkeypatch):
    _two_ranks(monkeypatch, None)

    def fail_checkpoint():
        raise CheckpointException("shard write failed", {})

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(RankFailureError, match=r"\(rank 0: CheckpointException: [^\n]*\)$"):
        run_on_all_ranks("saving", fail_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        run_on_all_ranks("saving", interrupt)
