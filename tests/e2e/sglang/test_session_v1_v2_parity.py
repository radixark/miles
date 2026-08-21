import os

import pytest
import torch
from huggingface_hub import snapshot_download
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.utils.sglang_server import start_sglang_server
from tests.session_parity_utils import V1, V2, assert_agentic_retry_trajectory_parity, run_agentic_retry_trajectories

from miles.utils.test_utils.session_verify_agent import build_initial_messages
from miles.utils.types import Sample

register_cuda_ci(est_time=190, suite="stage-c-2-gpu-h200", labels=["sglang"], hardware=["hopper"])

_MODEL_ID = "Qwen/Qwen3-8B"
_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
_DEFAULT_MODEL_PATH = "/root/models/Qwen3-8B"
_MODEL_PATH_OVERRIDE = os.environ.get("SGLANG_SESSION_PARITY_MODEL_PATH")
_MODEL_PATH = _MODEL_PATH_OVERRIDE or _DEFAULT_MODEL_PATH
_BATCH_SIZE = 16


@pytest.fixture(scope="module")
def sglang_server():
    assert torch.cuda.is_available()
    assert "H200" in torch.cuda.get_device_name(0)
    if _MODEL_PATH_OVERRIDE is None:
        snapshot_download(_MODEL_ID, revision=_MODEL_REVISION, local_dir=_DEFAULT_MODEL_PATH)

    server = start_sglang_server(
        model_path=_MODEL_PATH,
        enable_deterministic_inference=True,
        extra_args=[
            "--attention-backend",
            "fa3",
            "--reasoning-parser",
            "qwen3",
            "--tool-call-parser",
            "qwen25",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.5",
        ],
    )
    try:
        yield server
    finally:
        server.stop()


def test_qwen3_8b_h200_fa3_agentic_v2_drop_retries_matches_v1_training_payload_bitwise(sglang_server):
    v1_runs = run_agentic_retry_trajectories(
        backend_url=sglang_server.base_url,
        hf_checkpoint=_MODEL_PATH,
        version=V1,
        input_samples=_build_input_samples(),
    )
    v2_runs = run_agentic_retry_trajectories(
        backend_url=sglang_server.base_url,
        hf_checkpoint=_MODEL_PATH,
        version=V2,
        input_samples=_build_input_samples(),
    )

    assert len(v1_runs) == len(v2_runs) == _BATCH_SIZE
    for index, (v1, v2) in enumerate(zip(v1_runs, v2_runs, strict=True)):
        assert v1.samples[0].index == v2.samples[0].index == index
        assert_agentic_retry_trajectory_parity(v1, v2)


def _build_input_samples() -> list[Sample]:
    return [
        Sample(
            index=index,
            prompt=build_initial_messages(),
            reward=0.25,
            metadata={"source": "parity"},
        )
        for index in range(_BATCH_SIZE)
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
