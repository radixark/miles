from unittest.mock import MagicMock

import pytest

import miles.backends.megatron_utils.multi_lora_utils as multi_lora_utils


def test_checkpoint_stage_failure_is_reported_to_peer_ranks(monkeypatch):
    monkeypatch.setattr(multi_lora_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(multi_lora_utils.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(multi_lora_utils, "get_gloo_group", MagicMock(return_value=object()))

    def gather(messages, _local_message, group):
        messages[:] = ["OSError: disk full", None]

    monkeypatch.setattr(multi_lora_utils.dist, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="PEFT checkpoint write.*disk full"):
        multi_lora_utils._raise_if_any_rank_failed(None, "PEFT checkpoint write")
