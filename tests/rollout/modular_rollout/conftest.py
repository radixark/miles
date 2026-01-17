from unittest.mock import patch

import pytest

from miles.utils.arguments import parse_args


def _build_mock_args(extra_argv: list[str] | None = None):
    argv = [
        "pytest",
        "--train-backend",
        "fsdp",
        "--rollout-batch-size",
        "2",
        "--n-samples-per-prompt",
        "1",
        "--num-rollout",
        "1",
        "--rollout-num-gpus",
        "4",
        "--rollout-num-gpus-per-engine",
        "2",
        "--hf-checkpoint",
        "Qwen/Qwen3-0.6B",
        "--prompt-data",
        "/dev/null",
        "--input-key",
        "input",
        "--label-key",
        "label",
        "--rm-type",
        "math",
        "--use-miles-router",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        "30000",
    ] + (extra_argv or [])
    with patch("sys.argv", argv):
        return parse_args()


@pytest.fixture
def mock_args():
    return _build_mock_args()
