import base64
import os
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch
from huggingface_hub import snapshot_download
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.utils.sglang_server import start_sglang_server

from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

register_cuda_ci(
    est_time=400,
    suite="stage-c-2-gpu-h200",
    labels=["sglang", "replay"],
    hardware=["hopper"],
)

_MODEL_ID = "Qwen/Qwen3-30B-A3B"
_MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
_DEFAULT_MODEL_PATH = "/root/models/Qwen3-30B-A3B"
_MODEL_PATH_OVERRIDE = os.environ.get("SGLANG_SESSION_ADDITION_R3_MODEL_PATH")
_MODEL_PATH = _MODEL_PATH_OVERRIDE or _DEFAULT_MODEL_PATH
_NUM_TURNS = 10
_NUM_LAYERS = 48
_ROUTED_EXPERTS_TOPK = 8
_ROW_BYTES = _NUM_LAYERS * _ROUTED_EXPERTS_TOPK * np.dtype(np.int32).itemsize
_SEED = 1234
_MAX_COMPLETION_TOKENS = 64
_HTTP_TIMEOUT_SECS = 120.0


@pytest.fixture(scope="module")
def sglang_server():
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 2
    assert "H200" in torch.cuda.get_device_name(0)
    if _MODEL_PATH_OVERRIDE is None:
        snapshot_download(_MODEL_ID, revision=_MODEL_REVISION, local_dir=_DEFAULT_MODEL_PATH)

    server = start_sglang_server(
        model_path=_MODEL_PATH,
        extra_args=[
            "--tp",
            "2",
            "--attention-backend",
            "fa3",
            "--enable-return-routed-experts",
            "--mem-fraction-static",
            "0.7",
        ],
    )
    try:
        yield server
    finally:
        server.stop()


@contextmanager
def _serve_session(backend_url: str) -> Iterator[str]:
    port = find_available_port(31000)
    args = SimpleNamespace(
        miles_router_timeout=_HTTP_TIMEOUT_SECS,
        hf_checkpoint=_MODEL_PATH,
        chat_template_path=None,
        apply_chat_template_kwargs={"enable_thinking": False},
        tito_model="qwen3",
        sglang_speculative_algorithm=None,
        use_session_server="v1",
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=False,
        session_server_instance_id="session-addition-r3-e2e",
        save_debug_trajectory_data=None,
        pause_generation_mode="in_place",
        num_layers=_NUM_LAYERS,
    )
    server = UvicornThreadServer(SessionServer(args, backend_url=backend_url).app, host="127.0.0.1", port=port)
    server.start()
    try:
        yield server.url
    finally:
        server.stop()


def _decode_r3(payload: str) -> bytes:
    assert isinstance(payload, str)
    return base64.b64decode(payload, validate=True)


def test_tito_session_addition_r3_matches_full_r3(sglang_server):
    with _serve_session(sglang_server.base_url) as session_url:
        create_response = requests.post(f"{session_url}/sessions", timeout=_HTTP_TIMEOUT_SECS)
        assert create_response.status_code == 200, create_response.text
        session_id = create_response.json()["session_id"]
        endpoint = f"{session_url}/sessions/{session_id}"

        messages = [
            {
                "role": "system",
                "content": "Reply to every user request with exactly the requested label and no other text.",
            }
        ]
        checkpoints = []
        snapshot = None
        for turn in range(1, _NUM_TURNS + 1):
            messages.append({"role": "user", "content": f"Reply with exactly the label T{turn:02d}."})
            chat_response = requests.post(
                f"{endpoint}/v1/chat/completions",
                json={
                    "model": _MODEL_PATH,
                    "messages": messages,
                    "temperature": 0,
                    "max_completion_tokens": _MAX_COMPLETION_TOKENS,
                    "seed": _SEED,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=_HTTP_TIMEOUT_SECS,
            )
            assert chat_response.status_code == 200, chat_response.text
            choice = chat_response.json()["choices"][0]
            assert choice["finish_reason"] == "stop"
            assert choice["message"]["role"] == "assistant"
            assert choice["message"]["content"]
            messages.append(choice["message"])

            snapshot_response = requests.get(endpoint, timeout=_HTTP_TIMEOUT_SECS)
            assert snapshot_response.status_code == 200, snapshot_response.text
            snapshot = snapshot_response.json()
            checkpoints.append(snapshot["metadata"]["accumulated_token_ids"])

        assert snapshot is not None
        records = snapshot["records"]
        assert len(records) == len(checkpoints) == _NUM_TURNS
        final_token_ids = checkpoints[-1]

        oracle_response = requests.post(
            f"{endpoint}/generate",
            json={
                "input_ids": final_token_ids[:-1],
                "sampling_params": {
                    "max_new_tokens": 1,
                    "temperature": 0,
                    "sampling_seed": _SEED,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "return_routed_experts": True,
            },
            timeout=_HTTP_TIMEOUT_SECS,
        )
        assert oracle_response.status_code == 200, oracle_response.text
        oracle_meta = oracle_response.json()["meta_info"]
        oracle_output_ids = [item[1] for item in oracle_meta["output_token_logprobs"]]
        assert oracle_output_ids == [final_token_ids[-1]]
        full_r3 = _decode_r3(oracle_meta["routed_experts"])
        assert len(full_r3) == (len(final_token_ids) - 1) * _ROW_BYTES

        covered_rows = 0
        for index, (record, checkpoint) in enumerate(zip(records, checkpoints, strict=True)):
            expected_roles = ["system", *(["user", "assistant"] * index), "user"]
            assert [message["role"] for message in record["request"]["messages"]] == expected_roles

            choice = record["response"]["choices"][0]
            assert choice["message"]["role"] == "assistant"
            assert record["request"]["return_routed_experts"] is True
            start = record["request"]["routed_experts_start_len"]
            end = len(record["request"]["input_ids"]) + len(choice["meta_info"]["output_token_logprobs"]) - 1
            assert start == covered_rows
            if index:
                assert start == len(checkpoints[index - 1]) - 1
                assert start > 0
            assert end == len(checkpoint) - 1

            patch = _decode_r3(choice["meta_info"]["routed_experts"])
            assert len(patch) == (end - start) * _ROW_BYTES
            assert patch == full_r3[start * _ROW_BYTES : end * _ROW_BYTES]
            covered_rows = end

        assert covered_rows == len(final_token_ids) - 1

        samples_response = requests.post(
            f"{endpoint}/samples",
            json={"max_seq_len": None},
            timeout=_HTTP_TIMEOUT_SECS,
        )
        assert samples_response.status_code == 200, samples_response.text
        samples_reply = decode_samples_and_merge_input_sample(samples_response.content, Sample())
        assert samples_reply.empty_reason is None
        assert len(samples_reply.samples) == 1
        sample = samples_reply.samples[0]
        assert sample.tokens == final_token_ids
        assert sample.rollout_routed_experts is not None
        assert sample.rollout_routed_experts.dtype == np.int32
        assert sample.rollout_routed_experts.shape == (
            len(final_token_ids) - 1,
            _NUM_LAYERS,
            _ROUTED_EXPERTS_TOPK,
        )
        assert sample.rollout_routed_experts.tobytes(order="C") == full_r3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
