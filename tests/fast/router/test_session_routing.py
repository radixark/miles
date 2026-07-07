"""Unit tests for process-stable session routing (miles/rollout/session/sharding.py)."""

import os
import subprocess
import sys
import uuid

import pytest

from miles.rollout.session.sharding import worker_index_for_session


def test_worker_index_in_range_and_deterministic():
    sid = uuid.uuid4().hex
    for n in (1, 2, 4, 16):
        idx = worker_index_for_session(sid, n)
        assert 0 <= idx < n
        # deterministic: repeated calls agree
        assert worker_index_for_session(sid, n) == idx


def test_worker_index_distributes_across_workers():
    n = 8
    seen = {worker_index_for_session(uuid.uuid4().hex, n) for _ in range(2000)}
    # with 2000 ids over 8 workers, every worker should be hit
    assert seen == set(range(n))


def test_worker_index_rejects_bad_n():
    with pytest.raises(ValueError):
        worker_index_for_session(uuid.uuid4().hex, 0)


def test_mapping_is_process_stable_across_pythonhashseed():
    """The mapping must not depend on PYTHONHASHSEED (i.e. must not use builtin hash())."""
    sid = "0123456789abcdef0123456789abcdef"
    n = 7
    expected = worker_index_for_session(sid, n)

    def index_in_subproc(seed: str) -> int:
        code = (
            "from miles.rollout.session.sharding import worker_index_for_session;"
            f"print(worker_index_for_session({sid!r}, {n}))"
        )
        out = subprocess.check_output([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed})
        return int(out.strip())

    assert index_in_subproc("0") == expected
    assert index_in_subproc("12345") == expected
