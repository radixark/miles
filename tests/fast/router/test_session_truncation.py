"""Continuing a turn cut by finish_reason='length' through session servers v1 and v2.

``test_tito_truncation`` covers the closer of every family. These tests pin its
wiring: the next prompt starts with the closer, sample assembly masks the closer
and trains the cut turn's sampled tokens, the mismatch check compares a cut
final turn closed, and a family without a truncation grammar answers 409.
"""

import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import requests
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.router.test_sessions import _create_session, _post_chat

from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

_V2_ARGS = {
    "use_session_server": "v2",
    "session_sample_picker_path": "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
    "session_sample_postprocessor_path": "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
}
_HARD_MISMATCH_TYPES = {"special_token_count", "special_token_type", "non_assistant_text"}

USER = {"role": "user", "content": "What's the weather in Beijing?"}
FOLLOW_UP = {"role": "user", "content": "Go on."}
# Cut inside a tool call, so the closer ends the tool block and then the turn.
CUT = ('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Bei', "length")
CUT_IN_ANSWER = ("The weather in", "length")
# The mock returns the text's tokens as the completion; a complete turn's
# completion ends with its stop token, as SGLang returns it.
DONE = ("It is sunny.<|im_end|>", "stop")


@contextmanager
def _serve(version: str, tito_model: str, replies: list[tuple[str, str]]):
    """A session server of *version* over a mock backend that answers *replies* in order."""
    queue = [ProcessResult(text=text, finish_reason=finish_reason) for text, finish_reason in replies]

    def process_fn(prompt: str) -> ProcessResult:
        return queue.pop(0)

    with with_mock_server(process_fn=process_fn) as backend:
        config = make_session_server_config(
            backend_url=backend.url,
            instance_id=uuid.uuid4().hex,
            hf_checkpoint="Qwen/Qwen3-0.6B",
            apply_chat_template_kwargs={"enable_thinking": False},
            tito_model=tito_model,
            pause_generation_mode="retract",
            **(_V2_ARGS if version == "v2" else {}),
        )
        port = find_available_port(31000)
        server = UvicornThreadServer(SessionServer(config).app, host="127.0.0.1", port=port)
        server.start()
        try:
            yield SimpleNamespace(url=f"http://127.0.0.1:{port}", backend=backend)
        finally:
            server.stop()


def _chat(env, session_id: str, messages: list[dict]) -> dict:
    response = _post_chat(env.url, session_id, {"messages": messages})
    assert response.status_code == 200, response.text
    return response.json()


def _stored_token_ids(env, session_id: str) -> list[int]:
    """The latest generation's prompt and sampled tokens, as the session stores them."""
    return requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json()["metadata"]["accumulated_token_ids"]


def _hard_mismatches(mismatches: list[dict]) -> list[dict]:
    return [m for m in mismatches if m["type"] in _HARD_MISMATCH_TYPES]


@pytest.mark.parametrize("version", ["v1", "v2"])
class TestContinueCutTurn:
    def test_next_prompt_starts_with_closer(self, version):
        with _serve(version, "qwen3", [CUT, DONE]) as env:
            session_id = _create_session(env.url)
            first = _chat(env, session_id, [USER])
            cut_turn = _stored_token_ids(env, session_id)

            _chat(env, session_id, [USER, first["choices"][0]["message"], FOLLOW_UP])
            next_prompt = env.backend.request_log[-1]["input_ids"]

            assert next_prompt[: len(cut_turn)] == cut_turn
            next_text = env.backend.tokenizer.decode(next_prompt[len(cut_turn) :])
            assert next_text.startswith("\n</tool_call><|im_end|>\n<|im_start|>user\nGo on.<|im_end|>")

    def test_assembly_masks_closer_and_trains_cut_turn(self, version):
        with _serve(version, "qwen3", [CUT, CUT_IN_ANSWER]) as env:
            session_id = _create_session(env.url)
            first = _chat(env, session_id, [USER])
            first_prompt_len = len(env.backend.request_log[-1]["input_ids"])
            cut_turn = _stored_token_ids(env, session_id)
            _chat(env, session_id, [USER, first["choices"][0]["message"], FOLLOW_UP])
            trajectory = _stored_token_ids(env, session_id)
            second_prompt_len = len(env.backend.request_log[-1]["input_ids"])

            reply = requests.post(f"{env.url}/sessions/{session_id}/samples", json={}, timeout=10.0)
            assert reply.status_code == 200, reply.text

        (sample,) = decode_samples_and_merge_input_sample(reply.content, Sample()).samples
        assert sample.tokens == trajectory
        assert sample.status == Sample.Status.TRUNCATED
        cut_len = len(cut_turn) - first_prompt_len
        observation_len = second_prompt_len - len(cut_turn)
        assert sample.loss_mask == [1] * cut_len + [0] * observation_len + [1] * (len(trajectory) - second_prompt_len)
        assert sample.rollout_log_probs[cut_len : cut_len + observation_len] == [0.0] * observation_len

    def test_mismatch_has_no_special_token_differences(self, version):
        """A closer mid-path and a cut final turn both compare cleanly with the template render."""
        with _serve(version, "qwen3", [CUT, CUT_IN_ANSWER]) as env:
            session_id = _create_session(env.url)
            first = _chat(env, session_id, [USER])
            _chat(env, session_id, [USER, first["choices"][0]["message"], FOLLOW_UP])
            metadata = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json()["metadata"]

        assert _hard_mismatches(metadata["tito_session_mismatch"]) == []

    def test_retried_cut_turn_is_not_closed(self, version):
        """Retrying the cut first turn replaces it, so extending the retry adds no closer."""
        with _serve(version, "qwen3", [CUT, DONE, DONE]) as env:
            session_id = _create_session(env.url)
            _chat(env, session_id, [USER])
            retry = _chat(env, session_id, [USER])
            retry_turn = _stored_token_ids(env, session_id)

            _chat(env, session_id, [USER, retry["choices"][0]["message"], FOLLOW_UP])
            next_prompt = env.backend.request_log[-1]["input_ids"]

            assert next_prompt[: len(retry_turn)] == retry_turn
            next_text = env.backend.tokenizer.decode(next_prompt[len(retry_turn) :])
            assert next_text.startswith("\n<|im_start|>user\nGo on.")

    def test_family_without_grammar_is_409(self, version):
        with _serve(version, "default", [CUT]) as env:
            session_id = _create_session(env.url)
            first = _chat(env, session_id, [USER])

            extend = _post_chat(env.url, session_id, {"messages": [USER, first["choices"][0]["message"], FOLLOW_UP]})

            assert extend.status_code == 409
            assert "truncated generation cannot be extended" in extend.json()["error"]
            assert len(env.backend.request_log) == 1
