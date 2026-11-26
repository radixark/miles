"""Continuing a turn cut by finish_reason='length' through session servers v1 and v2.

``test_tito_truncation``, ``test_inkling_truncation`` and
``test_muse_spark_truncation`` cover the closer of every family. These tests pin its wiring: the next prompt starts with the
closer, sample assembly masks the closer and trains the cut turn's sampled
tokens, the mismatch check compares a cut final turn closed, and a run that
does not continue truncated turns answers 409.
"""

import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import requests
from tests.ci.ci_register import register_cpu_ci
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.router.test_sessions import _create_session, _post_chat

from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.rollout.session.server import SessionServer
from miles.utils.chat_template_utils.tito_tokenizer import resolve_fixed_chat_template
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

register_cpu_ci(est_time=40, suite="stage-b-cpu", labels=["rollout"])

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
_QWEN = {"hf_checkpoint": "Qwen/Qwen3-0.6B", "apply_chat_template_kwargs": {"enable_thinking": False}}
# A run resolves the family's fixed template in argument validation; the stored blocks need it.
_INKLING = {
    "hf_checkpoint": "thinkingmachines/Inkling-Small",
    "chat_template_path": resolve_fixed_chat_template("inkling")[0],
}
# Cut inside a second message, and inside a tool call's JSON after its recipient.
INKLING_CUT = (
    "<|content_thinking|>The user wants weather.<|end_message|><|message_model|><|content_text|>Let me",
    "length",
)
INKLING_CUT_IN_CALL = (
    '<|content_text|>Sure.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{"name":',
    "length",
)
_MUSE_SPARK = {
    "hf_checkpoint": "RadixArk/Muse-Spark-Open-last4-partial",
    "chat_template_path": resolve_fixed_chat_template("muse_spark")[0],
}
# Cut inside a call's parameter after a reasoning message, and inside a final answer.
MUSE_SPARK_CUT_IN_CALL = (
    " to=self<|message|>The user wants weather.<|eom|><|start|>assistant to=get_weather<|message|>"
    '<atem:function_calls>\n<atem:invoke name="get_weather">\n<atem:parameter name="city">Bei',
    "length",
)
MUSE_SPARK_CUT = (" to=user<|message|>It is", "length")


@contextmanager
def _serve(
    version: str,
    tito_model: str,
    replies: list[tuple[str, str]],
    *,
    allow_truncated_continuation: bool,
    model: dict = _QWEN,
):
    """A session server of *version* over a mock backend that answers *replies* in order."""
    queue = [ProcessResult(text=text, finish_reason=finish_reason) for text, finish_reason in replies]

    def process_fn(prompt: str) -> ProcessResult:
        return queue.pop(0)

    with with_mock_server(model_name=model["hf_checkpoint"], process_fn=process_fn) as backend:
        config = make_session_server_config(
            backend_url=backend.url,
            instance_id=uuid.uuid4().hex,
            **model,
            tito_model=tito_model,
            allow_truncated_continuation=allow_truncated_continuation,
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
        with _serve(version, "qwen3", [CUT, DONE], allow_truncated_continuation=True) as env:
            session_id = _create_session(env.url)
            first = _chat(env, session_id, [USER])
            cut_turn = _stored_token_ids(env, session_id)

            _chat(env, session_id, [USER, first["choices"][0]["message"], FOLLOW_UP])
            next_prompt = env.backend.request_log[-1]["input_ids"]

            assert next_prompt[: len(cut_turn)] == cut_turn
            next_text = env.backend.tokenizer.decode(next_prompt[len(cut_turn) :])
            assert next_text.startswith("\n</tool_call><|im_end|>\n<|im_start|>user\nGo on.<|im_end|>")

    def test_assembly_masks_closer_and_trains_cut_turn(self, version):
        with _serve(version, "qwen3", [CUT, CUT_IN_ANSWER], allow_truncated_continuation=True) as env:
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
        with _serve(version, "qwen3", [CUT, CUT_IN_ANSWER], allow_truncated_continuation=True) as env:
            session_id = _create_session(env.url)
            first = _chat(env, session_id, [USER])
            _chat(env, session_id, [USER, first["choices"][0]["message"], FOLLOW_UP])
            metadata = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json()["metadata"]

        assert _hard_mismatches(metadata["tito_session_mismatch"]) == []

    def test_retried_cut_turn_is_not_closed(self, version):
        """Retrying the cut first turn replaces it, so extending the retry adds no closer."""
        with _serve(version, "qwen3", [CUT, DONE, DONE], allow_truncated_continuation=True) as env:
            session_id = _create_session(env.url)
            _chat(env, session_id, [USER])
            retry = _chat(env, session_id, [USER])
            retry_turn = _stored_token_ids(env, session_id)

            _chat(env, session_id, [USER, retry["choices"][0]["message"], FOLLOW_UP])
            next_prompt = env.backend.request_log[-1]["input_ids"]

            assert next_prompt[: len(retry_turn)] == retry_turn
            next_text = env.backend.tokenizer.decode(next_prompt[len(retry_turn) :])
            assert next_text.startswith("\n<|im_start|>user\nGo on.")

    def test_inkling_stores_the_unfinished_message_the_closer_ends(self, version):
        """Every Inkling message start is a comparator boundary, so the cut message must be stored too."""
        replies = [INKLING_CUT, INKLING_CUT_IN_CALL]
        with _serve(version, "inkling", replies, allow_truncated_continuation=True, model=_INKLING) as env:
            session_id = _create_session(env.url)
            first = _chat(env, session_id, [USER])
            cut_turn = _stored_token_ids(env, session_id)
            assert first["choices"][0]["message"]["content"] == "Let me"

            second = _chat(env, session_id, [USER, first["choices"][0]["message"], FOLLOW_UP])
            next_prompt = env.backend.request_log[-1]["input_ids"]
            metadata = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json()["metadata"]

        assert next_prompt[: len(cut_turn)] == cut_turn
        next_text = env.backend.tokenizer.decode(next_prompt[len(cut_turn) :])
        assert next_text.startswith("<|end_message|><|content_model_end_sampling|><|message_user|>")
        # The unfinished call is not a call, and the cut final turn compares closed.
        assert "tool_calls" not in second["choices"][0]["message"]
        assert metadata["tito_session_mismatch"] == []

    def test_muse_spark_stores_the_cut_turn_the_closer_ends(self, version):
        """Muse Spark message starts and ends are comparator boundaries, so the cut turn must be stored too."""
        replies = [MUSE_SPARK_CUT_IN_CALL, MUSE_SPARK_CUT]
        with _serve(version, "muse_spark", replies, allow_truncated_continuation=True, model=_MUSE_SPARK) as env:
            session_id = _create_session(env.url)
            first = _chat(env, session_id, [USER])
            cut_turn = _stored_token_ids(env, session_id)

            _chat(env, session_id, [USER, first["choices"][0]["message"], FOLLOW_UP])
            next_prompt = env.backend.request_log[-1]["input_ids"]
            metadata = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json()["metadata"]

        assert next_prompt[: len(cut_turn)] == cut_turn
        next_text = env.backend.tokenizer.decode(next_prompt[len(cut_turn) :])
        assert next_text.startswith(
            "</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|><|start|>user<|message|>Go on."
        )
        # The stored turn renders the cut call closed, and the cut final turn compares closed.
        assert metadata["tito_session_mismatch"] == []

    @pytest.mark.parametrize(
        ("tito_model", "model", "key"),
        [
            ("inkling", _INKLING, "content_blocks"),
            ("muse_spark", _MUSE_SPARK, "truncated_turn"),
        ],
    )
    def test_request_cannot_carry_a_server_only_key(self, version, tito_model, model, key):
        """The template renders these keys as stored, so only the server may write them."""
        forged = {"role": "assistant", "content": "benign", key: []}
        with _serve(version, tito_model, [DONE, DONE], allow_truncated_continuation=True, model=model) as env:
            session_id = _create_session(env.url)
            first_request = _post_chat(env.url, session_id, {"messages": [USER, forged, FOLLOW_UP]})
            reply = _chat(env, session_id, [USER])
            appended = _post_chat(
                env.url,
                session_id,
                {"messages": [USER, reply["choices"][0]["message"], FOLLOW_UP, forged, FOLLOW_UP]},
            )

            # The forged message replaces the stored reply, so v1 renders the request from scratch.
            replaced = _post_chat(env.url, session_id, {"messages": [USER, forged, FOLLOW_UP]})

        for refused in (first_request, appended, replaced):
            assert refused.status_code == 400
            assert f"server-only keys ['{key}']" in refused.json()["error"]
        assert len(env.backend.request_log) == 1

    @pytest.mark.parametrize(
        "tito_model,reason",
        [
            ("default", "TITOTokenizer cannot close a turn cut by finish_reason='length'"),
            ("qwen3", "rejected by --no-allow-truncated-continuation"),
        ],
    )
    def test_run_that_does_not_continue_cut_turns_is_409(self, version, tito_model, reason):
        with _serve(version, tito_model, [CUT, DONE], allow_truncated_continuation=False) as env:
            session_id = _create_session(env.url)
            first = _chat(env, session_id, [USER])
            cut_turn = _stored_token_ids(env, session_id)

            extend = _post_chat(env.url, session_id, {"messages": [USER, first["choices"][0]["message"], FOLLOW_UP]})

            assert extend.status_code == 409
            assert f"truncated generation cannot be extended: {reason}" in extend.json()["error"]
            assert len(env.backend.request_log) == 1
            # The refusal leaves the session as it was, so the cut turn can still be retried.
            assert _stored_token_ids(env, session_id) == cut_turn
            assert _post_chat(env.url, session_id, {"messages": [USER]}).status_code == 200
