"""End-to-end tests for pretokenized token injection through the session proxy.

Verifies that SingleUserTurnTrajectoryManager correctly accumulates token state
across multi-turn tool-call conversations and injects pretokenized_token_ids
into downstream requests.

Validation strategy:
- The mock server verifies the chat template prefix invariant (prefix_str is a
  prefix of full_str) and uses pretokenized_token_ids for prompt construction.
- We check mock server request_log to verify pretokenized fields are injected
  on turn 2+.
- We verify pretokenized_token_ids is a valid prefix of the returned
  prompt_token_ids.

Tests are parametrized across multiple models and chat template combinations.
Models whose native templates satisfy the prefix invariant (Qwen3.5,
Qwen3-Next-Instruct) are tested with native templates.  Models that ship
broken templates (GLM-5, Qwen3, Qwen3-Thinking-2507, Qwen3-Next-Thinking) are
tested with fixed templates, and a separate negative test verifies that
their native templates indeed break the invariant.

NOTE: Qwen3-Coder-Next is NOT covered here because its template uses an
XML-style tool call format (<function=name>) and arguments|items (requiring
dict arguments), which are incompatible with the mock server's qwen25 tool
call parser and the OpenAI-format string arguments in mock trajectories.
Its prefix invariant is verified in the unit tests (test_pretokenized_chat.py).
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import requests

from miles.rollout.session.session_server import SessionServer
from miles.utils.chat_template_utils import try_get_fixed_chat_template
from miles.utils.http_utils import find_available_port
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer
from miles.utils.test_utils.mock_trajectories import (
    LongChainTrajectory,
    MultiTurnTrajectory,
    SequentialProcessFn,
    build_trajectory,
)
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

# ---------------------------------------------------------------------------
# Model + template configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTemplateConfig:
    hf_checkpoint: str
    chat_template_path: str | None  # None = model's native template


WORKING_CONFIGS: dict[str, ModelTemplateConfig] = {
    "qwen3-fixed": ModelTemplateConfig(
        "Qwen/Qwen3-0.6B",
        try_get_fixed_chat_template("Qwen/Qwen3-0.6B"),
    ),
    "qwen3-thinking2507-fixed": ModelTemplateConfig(
        "Qwen/Qwen3-4B-Thinking-2507",
        try_get_fixed_chat_template("Qwen/Qwen3-4B-Thinking-2507"),
    ),
    "qwen3.5-native": ModelTemplateConfig("Qwen/Qwen3.5-0.8B", None),
    "qwen3-next-instruct-native": ModelTemplateConfig("Qwen/Qwen3-Next-80B-A3B-Instruct", None),
    "qwen3-next-thinking-fixed": ModelTemplateConfig(
        "Qwen/Qwen3-Next-80B-A3B-Thinking",
        try_get_fixed_chat_template("Qwen/Qwen3-Next-80B-A3B-Thinking"),
    ),
}

BROKEN_CONFIGS: dict[str, ModelTemplateConfig] = {
    # TODO: GLM-5 is here because the mock server lacks a reasoning parser, so
    # mock trajectories have no thinking content, causing a <think>/<\/think>
    # mismatch with add_generation_prompt. Once the mock server supports
    # ReasoningParser, GLM-5 should move back to WORKING_CONFIGS.
    "glm5-native": ModelTemplateConfig("zai-org/GLM-5", None),
    "qwen3-native": ModelTemplateConfig("Qwen/Qwen3-0.6B", None),
    "qwen3-thinking2507-native": ModelTemplateConfig("Qwen/Qwen3-4B-Thinking-2507", None),
    "qwen3-next-thinking-native": ModelTemplateConfig("Qwen/Qwen3-Next-80B-A3B-Thinking", None),
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=WORKING_CONFIGS.values(),
    ids=WORKING_CONFIGS.keys(),
    scope="class",
)
def model_config(request) -> ModelTemplateConfig:
    return request.param


@pytest.fixture(scope="class")
def tokenizer(model_config):
    try:
        return load_tokenizer(
            model_config.hf_checkpoint,
            chat_template_path=model_config.chat_template_path,
            trust_remote_code=True,
        )
    except (ValueError, OSError) as exc:
        pytest.skip(f"Cannot load tokenizer for {model_config.hf_checkpoint}: {exc}")


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _make_router_env(tokenizer, model_config, trajectory_cls):
    """Create a SessionServer + mock backend wired for a given trajectory."""
    trajectory = build_trajectory(tokenizer, trajectory_cls)
    process_fn = SequentialProcessFn(trajectory)

    ctx = {}

    backend = MockSGLangServer(
        model_name=model_config.hf_checkpoint,
        process_fn=process_fn,
        host="127.0.0.1",
        port=find_available_port(30000),
        chat_template_path=model_config.chat_template_path,
    )
    backend.start()
    ctx["backend"] = backend

    args = SimpleNamespace(
        miles_router_timeout=30,
        hf_checkpoint=model_config.hf_checkpoint,
        chat_template_path=model_config.chat_template_path,
        tito_model="default",
        use_rollout_routing_replay=False,
    )
    session_server = SessionServer(args, backend_url=backend.url)

    port = find_available_port(31000)
    server = UvicornThreadServer(session_server.app, host="127.0.0.1", port=port)
    server.start()
    ctx["server"] = server

    url = f"http://127.0.0.1:{port}"

    return SimpleNamespace(url=url, backend=backend, trajectory=trajectory, process_fn=process_fn, _ctx=ctx)


def _teardown_router_env(env):
    env._ctx["server"].stop()
    env._ctx["backend"].stop()


# ---------------------------------------------------------------------------
# Trajectory runner and verifiers
# ---------------------------------------------------------------------------


def _get_followup_messages_after_assistant(full_messages: list[dict], assistant_idx: int) -> list[dict]:
    """Extract all non-assistant messages following an assistant message until the next assistant."""
    followup = []
    i = assistant_idx + 1
    while i < len(full_messages) and full_messages[i]["role"] != "assistant":
        followup.append(full_messages[i])
        i += 1
    return followup


def _remap_followup_messages(followup_msgs: list[dict], response_tool_calls: list[dict]) -> list[dict]:
    """Remap tool_call_ids in tool messages from actual response; pass through system messages as-is."""
    remapped = []
    tool_idx = 0
    for msg in followup_msgs:
        new_msg = dict(msg)
        if msg["role"] == "tool":
            if tool_idx < len(response_tool_calls):
                new_msg["tool_call_id"] = response_tool_calls[tool_idx]["id"]
            tool_idx += 1
        remapped.append(new_msg)
    return remapped


def _run_trajectory_e2e(env):
    """Run all turns of a trajectory through the session proxy.

    Dynamically constructs messages from actual responses: turn 1 uses
    pre-defined messages, subsequent turns use the actual assistant message
    from the previous response + remapped tool messages from the trajectory.
    """
    trajectory = env.trajectory

    env.process_fn.reset()
    env.backend.reset_stats()

    session_id = requests.post(f"{env.url}/sessions", timeout=5.0).json()["session_id"]

    assistant_indices = [i for i, m in enumerate(trajectory.full_messages) if m["role"] == "assistant"]

    responses = []
    accumulated_messages: list[dict] = []

    for turn_idx, turn in enumerate(trajectory.turns):
        if turn_idx == 0:
            messages = turn.request_messages
            accumulated_messages = list(messages)
        else:
            messages = list(accumulated_messages)

        payload = {"messages": messages}
        if trajectory.tools:
            payload["tools"] = trajectory.tools

        resp = requests.post(
            f"{env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200, f"Turn {turn_idx} failed: {resp.text}"
        resp_json = resp.json()
        responses.append(resp_json)

        assistant_msg = resp_json["choices"][0]["message"]
        accumulated_messages.append(assistant_msg)

        ass_idx = assistant_indices[turn_idx]
        followup_msgs = _get_followup_messages_after_assistant(trajectory.full_messages, ass_idx)
        response_tool_calls = assistant_msg.get("tool_calls") or []
        remapped = _remap_followup_messages(followup_msgs, response_tool_calls)
        accumulated_messages.extend(remapped)

    return session_id, responses


def _verify_pretokenized_injection(env, responses):
    """Verify input_ids in request_log and prompt consistency."""
    backend = env.backend

    for turn_idx in range(len(env.trajectory.turns)):
        req = backend.request_log[turn_idx]
        resp_prompt_ids = responses[turn_idx]["choices"][0]["prompt_token_ids"]

        if turn_idx == 0:
            assert "input_ids" not in req, "Turn 0 should not have input_ids"
        else:
            assert "input_ids" in req, f"Turn {turn_idx} missing input_ids"

            input_ids = req["input_ids"]
            assert len(input_ids) > 0, f"Turn {turn_idx} input_ids is empty"

            assert resp_prompt_ids == input_ids, (
                f"Turn {turn_idx}: input_ids does not match prompt_token_ids.\n"
                f"input_ids len={len(input_ids)}, prompt len={len(resp_prompt_ids)}\n"
                f"First diff at index {next((i for i, (a, b) in enumerate(zip(input_ids, resp_prompt_ids, strict=False)) if a != b), 'length')}"
            )


def _verify_pretokenized_NOT_prefix(env, responses):
    """Verify that input_ids does NOT match prompt on turn 2+.

    This is the inverse of _verify_pretokenized_injection: it proves that a
    broken native template causes token mismatch in the e2e flow.
    """
    backend = env.backend
    found_mismatch = False

    for turn_idx in range(1, len(env.trajectory.turns)):
        req = backend.request_log[turn_idx]
        if "input_ids" not in req:
            continue
        input_ids = req["input_ids"]
        resp_prompt_ids = responses[turn_idx]["choices"][0]["prompt_token_ids"]

        if resp_prompt_ids != input_ids:
            found_mismatch = True
            break

    assert found_mismatch, (
        "Expected prefix invariant to be violated with native template, "
        "but input_ids matched prompt_token_ids on all turns"
    )


# ===========================================================================
# Positive E2E tests (auto-parametrized across WORKING_CONFIGS)
# ===========================================================================


class TestMultiTurnE2E:
    """2-turn trajectory: sys, user -> ass(tool) -> tool -> ass(tool) -> tool"""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, tokenizer, model_config):
        env = _make_router_env(tokenizer, model_config, MultiTurnTrajectory)
        type(self)._env = env
        yield
        _teardown_router_env(env)

    def test_all_turns_succeed(self):
        session_id, responses = _run_trajectory_e2e(self._env)
        assert len(responses) == 2

    def test_pretokenized_injection(self):
        session_id, responses = _run_trajectory_e2e(self._env)
        _verify_pretokenized_injection(self._env, responses)

    def test_session_records(self):
        session_id, responses = _run_trajectory_e2e(self._env)
        get_resp = requests.get(f"{self._env.url}/sessions/{session_id}", timeout=5.0)
        records = get_resp.json()["records"]
        assert len(records) == 2
        assert all(r["status_code"] == 200 for r in records)
        assert all(r["path"] == "/v1/chat/completions" for r in records)


class TestLongChainE2E:
    """3-turn trajectory: sys, user -> ass(tool) -> tool -> ass(tool) -> tool -> ass(tool) -> tool"""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, tokenizer, model_config):
        env = _make_router_env(tokenizer, model_config, LongChainTrajectory)
        type(self)._env = env
        yield
        _teardown_router_env(env)

    def test_all_turns_succeed(self):
        session_id, responses = _run_trajectory_e2e(self._env)
        assert len(responses) == 3

    def test_pretokenized_injection(self):
        self._env.backend.reset_stats()
        session_id, responses = _run_trajectory_e2e(self._env)
        _verify_pretokenized_injection(self._env, responses)

    def test_pretokenized_grows_across_turns(self):
        self._env.backend.reset_stats()
        session_id, responses = _run_trajectory_e2e(self._env)

        req1 = self._env.backend.request_log[1]
        req2 = self._env.backend.request_log[2]
        ids1 = req1["input_ids"]
        ids2 = req2["input_ids"]
        assert len(ids2) > len(ids1), f"Turn 2 input_ids ({len(ids2)}) should be longer than turn 1 ({len(ids1)})"

        assert ids2[: len(ids1)] == ids1, "Turn 2 input_ids should have turn 1's as prefix"

    def test_session_records(self):
        session_id, responses = _run_trajectory_e2e(self._env)
        get_resp = requests.get(f"{self._env.url}/sessions/{session_id}", timeout=5.0)
        records = get_resp.json()["records"]
        assert len(records) == 3


# ===========================================================================
# Negative E2E tests (native templates that break the prefix invariant)
# ===========================================================================


@pytest.mark.parametrize("config", BROKEN_CONFIGS.values(), ids=BROKEN_CONFIGS.keys())
class TestNativeTemplatePrefixBreaks:
    """Verify that native (unfixed) templates break the prefix invariant in e2e flow.

    The mock server uses tito_tokenizer to build prompt tokens incrementally
    (matching real sglang behaviour), so all turns succeed with 200.  We detect
    the broken template by comparing the pretokenized_token_ids injected on
    turn 1+ against a fresh full re-tokenization of the same messages — the two
    must diverge if the native template is truly broken.
    """

    def test_prefix_invariant_violated(self, config):
        try:
            tok = load_tokenizer(config.hf_checkpoint, trust_remote_code=True)
        except (ValueError, OSError) as exc:
            pytest.skip(f"Cannot load tokenizer for {config.hf_checkpoint}: {exc}")

        try:
            env = _make_router_env(tok, config, MultiTurnTrajectory)
        except AssertionError:
            # Some broken templates fail the text-level prefix invariant at
            # trajectory build time (e.g. GLM-5's <think>/<\/think> mismatch).
            # That confirms the invariant is violated.
            return

        try:
            _session_id, _responses = _run_trajectory_e2e(env)

            found_mismatch = False
            for turn_idx in range(1, len(env.trajectory.turns)):
                req = env.backend.request_log[turn_idx]
                input_ids = req.get("input_ids")
                if input_ids is None:
                    continue
                messages = req["messages"]
                tools = req.get("tools")
                full_prompt_str = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                )
                full_prompt_ids = tok.encode(full_prompt_str, add_special_tokens=False)
                if full_prompt_ids != list(input_ids):
                    found_mismatch = True
                    break

            assert found_mismatch, (
                "Expected prefix invariant to be violated with native template, "
                "but input_ids matched full re-tokenization on all turns"
            )
        finally:
            _teardown_router_env(env)
