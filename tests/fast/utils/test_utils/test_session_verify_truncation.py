"""Truncate mode of the session verifier, against real session servers and the mock SGLang backend.

The mock honors ``max_tokens`` the way SGLang does for a cut reply: it keeps the
first tokens, drops the tool calls and answers ``finish_reason='length'``.
"""

import asyncio
import json
import uuid
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import requests
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.session_truncation import CUT_KINDS, cut_segments, load_tito_tokenizer
from miles.utils.test_utils.session_verify_agent import run_agent
from miles.utils.test_utils.session_verify_runner import assert_session_verify_metrics
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

_MODEL = "Qwen/Qwen3-0.6B"
_REPLY = (
    "<think>\nThe user asks about the weather, so I should call the tool for it.\n</think>\n\n"
    "Let me check that for you now.\n"
    '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Beijing"}}\n</tool_call>'
)
_V2_ARGS = {
    "use_session_server": "v2",
    "session_sample_picker_path": "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
    "session_sample_postprocessor_path": "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
}


@contextmanager
def _serve(version: str):
    original = MockSGLangServer._compute_chat_completions_response

    def honor_max_tokens(self, payload: dict) -> dict:
        response = original(self, payload)
        choice = response["choices"][0]
        entries = choice["meta_info"]["output_token_logprobs"]
        limit = payload.get("max_tokens")
        if limit is not None and len(entries) > limit:
            kept = entries[:limit]
            choice["meta_info"]["output_token_logprobs"] = kept
            choice["meta_info"]["completion_tokens"] = limit
            choice["logprobs"]["content"] = choice["logprobs"]["content"][:limit]
            text = self.tokenizer.decode([token_id for _, token_id in kept])
            choice["message"] = {"role": "assistant", "content": text, "tool_calls": None}
            choice["finish_reason"] = "length"
        return response

    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=honor_max_tokens):
        with with_mock_server(process_fn=lambda prompt: ProcessResult(text=_REPLY, finish_reason="stop")) as backend:
            config = make_session_server_config(
                backend_url=backend.url,
                instance_id=uuid.uuid4().hex,
                hf_checkpoint=_MODEL,
                tito_model="qwen3",
                pause_generation_mode="retract",
                **(_V2_ARGS if version == "v2" else {}),
            )
            port = find_available_port(31000)
            server = UvicornThreadServer(SessionServer(config).app, host="127.0.0.1", port=port)
            server.start()
            try:
                yield f"http://127.0.0.1:{port}"
            finally:
                server.stop()


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_truncate_mode_cuts_and_continues_every_kind(version, seed):
    with _serve(version) as url:
        session_id = requests.post(f"{url}/sessions", json={"temperature": 0.7}, timeout=5.0).json()["session_id"]
        result = asyncio.run(
            run_agent(
                f"{url}/sessions/{session_id}",
                prompt=None,
                request_kwargs={"seed": seed},
                metadata={
                    "tito_model": "qwen3",
                    "session_verify_cycles": 1,
                    "session_verify_truncate": True,
                    "hf_checkpoint": _MODEL,
                },
            )
        )

    # The first reply is cut in its tool call and continued by the TOOL_RESULT step; the turns before
    # user, system, force-final and assistant-input steps rotate from a seed-dependent start. Every cut
    # is continued and checked closed, and later tool calls still get results.
    rotated = [CUT_KINDS[(seed + i) % 3] for i in range(4)]
    assert [cut["kind"] for cut in result["truncation_cuts"]] == ["tool_call", *rotated]
    assert result["continued_cut_kinds"] == [cut["kind"] for cut in result["truncation_cuts"]]
    assert "continue_cut" in result["driver_events"]
    assert "append_tool" in result["driver_events"]


def test_cut_segments_hold_only_cuts_on_this_path():
    tito = load_tito_tokenizer(_MODEL, "qwen3")
    encode = tito.tokenizer.encode
    prompt = encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    cut = encode('<think>\nok\n</think>\n\n<tool_call>\n{"name"', add_special_tokens=False)
    closer = tito.close_truncated_turn(prompt + cut, len(prompt))
    tokens = prompt + cut + closer + encode("\n<|im_start|>user\ngo on<|im_end|>\n", add_special_tokens=False)
    on_path = {"start": len(prompt), "ids": cut, "kind": "tool_call"}
    other_branch = {"start": len(prompt), "ids": cut[:-1] + [cut[-1] + 1], "kind": "answer"}

    # Segments: <|im_start|>, "user\nhi", <|im_end|>, "\n", <|im_start|>, "assistant\n<think>...", ...
    assert cut_segments(tito, tokens, [on_path, other_branch]) == {5: "tool_call"}


def test_metrics_gate_requires_every_continued_cut_kind(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    entry = {"driver_events": ["append_tool"], "had_assistant_mismatch": False, "hard_mismatch_count": 0}
    metrics.write_text(
        json.dumps({**entry, "continued_cut_kinds": ["reasoning"]})
        + "\n"
        + json.dumps({**entry, "continued_cut_kinds": ["tool_call"]})
        + "\n"
    )

    with pytest.raises(AssertionError, match=r"no sample cut a reply in \['answer'\]"):
        assert_session_verify_metrics(metrics, assistant_text_threshold=0.0, required_cut_kinds=CUT_KINDS)
    assert_session_verify_metrics(metrics, assistant_text_threshold=0.0, required_cut_kinds=("reasoning", "tool_call"))
