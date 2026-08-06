from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

_MODULE = "miles.backends.sglang_utils.sglang_engine"


def test_init_forwards_the_prebracketed_addressing_verbatim():
    """The addressing builder hands out display-ready v6 values; init must not re-format them."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.args = Namespace(env_report=None)
    engine.rank = 0
    engine.worker_type = "regular"
    engine.base_gpu_id = 0
    engine.sglang_overrides = {}
    engine.num_gpus_per_engine = 1

    forwarded: dict = {}

    def fake_compute_server_args(*args, **kwargs):
        forwarded["dist_init_addr"] = args[2]
        forwarded["host"] = args[4]
        return {"node_rank": 0, "host": "[fd00::2]", "port": 31007, "disaggregation_bootstrap_port": None}

    with (
        patch(f"{_MODULE}._compute_server_args", side_effect=fake_compute_server_args),
        patch(f"{_MODULE}.ServerArgs"),
        patch(f"{_MODULE}.launch_server_process", return_value=MagicMock(pid=4242)),
    ):
        engine.init(dist_init_addr="[fd00::1]:15003", port=31007, nccl_port=6000, host="[fd00::2]")

    assert forwarded == {"dist_init_addr": "[fd00::1]:15003", "host": "[fd00::2]"}
    assert engine.server_url == "http://[fd00::2]:31007"
