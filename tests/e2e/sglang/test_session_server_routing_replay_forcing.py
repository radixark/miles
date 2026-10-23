"""Session server v1 with routing replay forcing, across a retract and a router-weight change.

A control session runs two turns under the original router weights. A second session is retracted during turn 2
while the router weights are perturbed, and turn 3 re-prefills its whole history (radix cache off). With forcing,
rows of tokens computed before the retract match the control's; an unforced prefill under the new weights does
not. Rows are compared as sorted expert sets, since a forced row may keep the current gate's order.
"""

import base64
import json
import os
import shutil
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import requests
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.utils.sglang_server import start_sglang_server
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

register_cuda_ci(
    est_time=900,
    suite="stage-c-2-gpu-h200",
    labels=["sglang", "replay"],
    hardware=["hopper", "blackwell"],
    num_gpus=2,
)

_MODEL_ID = "Qwen/Qwen3-30B-A3B"
_MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
_DEFAULT_MODEL_PATH = "/root/models/Qwen3-30B-A3B"
_MODEL_PATH_OVERRIDE = os.environ.get("SGLANG_SESSION_ROUTING_REPLAY_FORCING_MODEL_PATH")
_MODEL_PATH = _MODEL_PATH_OVERRIDE or _DEFAULT_MODEL_PATH
_NUM_LAYERS = 48
_ROUTED_EXPERTS_TOPK = 8
_SEED = 1234
_HTTP_TIMEOUT_SECS = 300.0
# About 2,300 tokens (Qwen3 tokenizes digits one by one), so the retract lands well inside turn 2's decode.
_LONG_TURN_PROMPT = "Write the integers from 1 to 600, one per line, and nothing else."
_LONG_TURN_TOKENS = 4096
_RETRACT_DELAY_SECS = 3.0
# Gaussian noise on each router weight, in units of that weight's std: enough to flip most routes.
_ROUTER_NOISE = 1.0


@pytest.fixture(scope="module")
def sglang_server():
    assert torch.cuda.device_count() >= 2
    if _MODEL_PATH_OVERRIDE is None:
        snapshot_download(_MODEL_ID, revision=_MODEL_REVISION, local_dir=_DEFAULT_MODEL_PATH)
    # Deterministic inference (on by default) picks its attention backend for the GPU generation.
    server = start_sglang_server(
        model_path=_MODEL_PATH,
        extra_args=[
            *("--tp", "2", "--mem-fraction-static", "0.7", "--moe-runner-backend", "triton"),
            *("--enable-return-routed-experts", "--enable-routed-experts-forcing"),
            # Every turn then re-prefills its whole history, the case forcing exists for.
            "--disable-radix-cache",
        ],
    )
    try:
        yield server
    finally:
        server.stop()


def _perturbed_router_checkpoint(out_dir: Path) -> str:
    """A partial checkpoint of only the MoE router weights, perturbed; update_weights_from_disk loads just these."""
    model_dir = Path(_MODEL_PATH)
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    names = [name for name in weight_map if name.endswith(".mlp.gate.weight")]
    assert len(names) == _NUM_LAYERS
    generator = torch.Generator().manual_seed(_SEED)
    tensors = {}
    for name in names:
        with safe_open(str(model_dir / weight_map[name]), framework="pt") as shard:
            weight = shard.get_tensor(name)
        noise = torch.randn(weight.shape, generator=generator) * weight.float().std() * _ROUTER_NOISE
        tensors[name] = (weight.float() + noise).to(weight.dtype)
    out_dir.mkdir(parents=True)
    shutil.copy(model_dir / "config.json", out_dir / "config.json")
    save_file(tensors, str(out_dir / "model.safetensors"))
    return str(out_dir)


@contextmanager
def _serve_session(backend_url: str) -> Iterator[str]:
    port = find_available_port(31000)
    config = make_session_server_config(
        port=port,
        instance_id="session-routing-replay-forcing-e2e",
        backend_url=backend_url,
        timeout=_HTTP_TIMEOUT_SECS,
        hf_checkpoint=_MODEL_PATH,
        tito_model="qwen3",
        apply_chat_template_kwargs={"enable_thinking": False},
        use_rollout_routing_replay=True,
        use_rollout_routing_replay_forcing=True,
        use_session_server="v1",
        pause_generation_mode="retract",
        num_layers=_NUM_LAYERS,
    )
    server = UvicornThreadServer(SessionServer(config).app, host="127.0.0.1", port=port)
    server.start()
    try:
        yield server.url
    finally:
        server.stop()


def _post(url: str, payload: dict) -> dict:
    response = requests.post(url, json=payload, timeout=_HTTP_TIMEOUT_SECS)
    assert response.status_code == 200, response.text
    return response.json()


def _chat(endpoint: str, messages: list[dict], *, max_tokens: int) -> dict:
    payload = dict(model=_MODEL_PATH, messages=messages, temperature=0, max_completion_tokens=max_tokens, seed=_SEED)
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    return _post(f"{endpoint}/v1/chat/completions", payload)["choices"][0]


def _two_turns(session_url: str, *, during_turn_2: Callable[[], None] = lambda: None) -> tuple[str, list[dict]]:
    """A short turn 1, then a long turn 2 with ``during_turn_2`` run while it decodes."""
    endpoint = f"{session_url}/sessions/{_post(f'{session_url}/sessions', {})['session_id']}"
    messages = [
        {"role": "system", "content": "Follow every instruction exactly."},
        {"role": "user", "content": "Reply with exactly the label T01."},
    ]
    messages += [_chat(endpoint, messages, max_tokens=64)["message"], {"role": "user", "content": _LONG_TURN_PROMPT}]
    with ThreadPoolExecutor(max_workers=1) as pool:
        turn_2 = pool.submit(_chat, endpoint, messages, max_tokens=_LONG_TURN_TOKENS)
        time.sleep(_RETRACT_DELAY_SECS)
        assert not turn_2.done(), "turn 2 finished too early; ask it for a longer answer"
        during_turn_2()
        return endpoint, [*messages, turn_2.result(timeout=_HTTP_TIMEOUT_SECS)["message"]]


def _switch_router_weights(base_url: str, checkpoint: str) -> None:
    """The weight sync Miles runs under --pause-generation-mode retract, with the router weights as the update."""
    _post(f"{base_url}/pause_generation", {"mode": "retract"})
    deadline = time.monotonic() + 60
    while requests.get(f"{base_url}/flush_cache", timeout=_HTTP_TIMEOUT_SECS).status_code != 200:
        assert time.monotonic() < deadline, "flush_cache did not succeed within 60 s"
        time.sleep(1)
    assert _post(f"{base_url}/update_weights_from_disk", {"model_path": checkpoint})["success"]
    _post(f"{base_url}/continue_generation", {})


def _rows(payload: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(payload, validate=True), dtype=np.int32).reshape(
        -1, _NUM_LAYERS, _ROUTED_EXPERTS_TOPK
    )


def _turn_rows(records: list[dict]) -> np.ndarray:
    """The rows the turns returned, concatenated: each turn returns the rows its session did not hold yet."""
    payloads = [record["response"]["choices"][0]["meta_info"]["routed_experts"] for record in records]
    return np.concatenate([_rows(payload) for payload in payloads])


def _unforced_prefill_rows(base_url: str, token_ids: list[int]) -> np.ndarray:
    """Rows of every position of ``token_ids`` but the last, from a request without a session key."""
    sampling_params = {"max_new_tokens": 1, "temperature": 0, "sampling_seed": _SEED, "ignore_eos": True}
    payload = {"input_ids": token_ids[:-1], "sampling_params": sampling_params, "return_routed_experts": True}
    return _rows(_post(f"{base_url}/generate", payload)["meta_info"]["routed_experts"])


def _leading_equal_rows(actual: np.ndarray, expected: np.ndarray) -> int:
    num_rows = min(len(actual), len(expected))
    differs = (actual[:num_rows] != expected[:num_rows]).any(axis=(1, 2))
    return int(np.argmax(differs)) if differs.any() else num_rows


def test_routing_replay_forcing_keeps_first_computed_routes_across_retract_and_weight_change(sglang_server, tmp_path):
    base_url = sglang_server.base_url
    server_info = requests.get(f"{base_url}/server_info", timeout=_HTTP_TIMEOUT_SECS).json()
    assert server_info["routed_experts_forcing"]["version"] >= 1
    w1 = _perturbed_router_checkpoint(tmp_path / "w1")

    with _serve_session(base_url) as session_url:
        control_endpoint, _ = _two_turns(session_url)
        endpoint, messages = _two_turns(session_url, during_turn_2=lambda: _switch_router_weights(base_url, w1))
        _chat(endpoint, [*messages, {"role": "user", "content": "Reply with exactly the label DONE."}], max_tokens=16)

        records = requests.get(endpoint, timeout=_HTTP_TIMEOUT_SECS).json()["records"]
        control_records = requests.get(control_endpoint, timeout=_HTTP_TIMEOUT_SECS).json()["records"]
        meta_infos = [record["response"]["choices"][0]["meta_info"] for record in records]
        starts = [record["request"]["routed_experts_start_len"] for record in records]
        ends = [
            len(record["request"]["input_ids"]) + len(meta_info["output_token_logprobs"]) - 1
            for record, meta_info in zip(records, meta_infos, strict=True)
        ]
        # Under retract each turn still asks only for the rows its session does not hold yet.
        assert starts == [0, ends[0], ends[1]]
        assert len(_turn_rows(records)) == ends[2]

        # Turn 2 was retracted mid-decode, and the engine kept the routes it had computed by then.
        assert meta_infos[1]["num_retractions"] >= 1
        assert meta_infos[1]["routed_experts_forcing"]["retract_snapshots"] > 0
        # Turn 3 recomputed every held token under w1 and forced all of them.
        turn_3 = meta_infos[2]["routed_experts_forcing"]
        assert turn_3["held_tokens"] == turn_3["forced_tokens"] == starts[2]
        assert all(meta_info["routed_experts_forcing"]["unforced_tokens"] == 0 for meta_info in meta_infos)

        # The rows turn 2 reports for its prompt and for the tokens it decoded before the retract are the w0 rows
        # the control computed, although the retract recomputed them under w1.
        held = np.sort(_turn_rows(records[:2]), axis=-1)
        kept = _leading_equal_rows(held, np.sort(_turn_rows(control_records), axis=-1))
        assert kept > len(records[1]["request"]["input_ids"]), f"{kept} rows kept, none decoded before the retract"
        # An unforced w1 prefill routes them differently, so forcing them was load-bearing.
        w1_rows = np.sort(_unforced_prefill_rows(base_url, records[2]["request"]["input_ids"][: kept + 1]), axis=-1)
        assert (w1_rows[starts[1] :] != held[starts[1] : kept]).any(axis=(1, 2)).mean() > 0.5

        samples_response = requests.post(f"{endpoint}/samples", json={"max_seq_len": None}, timeout=_HTTP_TIMEOUT_SECS)
        assert samples_response.status_code == 200, samples_response.text
        [sample] = decode_samples_and_merge_input_sample(samples_response.content, Sample()).samples
        assert np.array_equal(sample.rollout_routed_experts[: starts[2]], _turn_rows(records[:2]))
        assert sample.rollout_routed_experts.shape == (len(sample.tokens) - 1, _NUM_LAYERS, _ROUTED_EXPERTS_TOPK)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
