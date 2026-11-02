"""Checkpoint expiry lives in one gateway-private file that survives restarts and never deletes on bad input."""

import json

import pytest

from miles.tinker.core.checkpoint_expiry import EXPIRY_FILENAME, CheckpointExpiry
from miles.tinker.core.types import UserInputError

PATH = "tinker://model/weights/ck"


def test_an_expiry_survives_a_restart_and_none_removes_it(tmp_path):
    CheckpointExpiry(str(tmp_path)).set(PATH, 3600, now=100.0)
    restarted = CheckpointExpiry(str(tmp_path))
    assert restarted.expires_at(PATH) == 3700.0
    assert restarted.expired(3699.0) == [] and restarted.expired(3700.0) == [PATH]
    restarted.set(PATH, None, now=0.0)
    assert CheckpointExpiry(str(tmp_path)).expires_at(PATH) is None


@pytest.mark.parametrize("ttl_seconds", [3599, 10 * 365 * 24 * 3600 + 1, 3600.5])
def test_ttl_outside_the_sdk_range_is_refused(tmp_path, ttl_seconds):
    with pytest.raises(UserInputError, match="ttl_seconds"):
        CheckpointExpiry(str(tmp_path)).set(PATH, ttl_seconds, now=0.0)


def test_an_entry_that_does_not_parse_is_an_error_not_a_deletion(tmp_path):
    (tmp_path / EXPIRY_FILENAME).write_text(json.dumps({"not-a-tinker-path": 1.0}))
    with pytest.raises(UserInputError, match="not a tinker path"):
        CheckpointExpiry(str(tmp_path))
