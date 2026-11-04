"""Offline tests for the tau2-bench example: no GPU, no network, no live model.

Each episode runs the production path: ``agentic_tool_call.generate`` with a v1
session server (TITO, ``--tito-model qwen3``) in front of a mock SGLang server,
and tau2's own orchestrator, environment and evaluator on its ``mock`` domain.
The policy is scripted through the mock server, the user simulator through the
LLM call tau2 makes for it.

    pytest tests/fast/examples/experimental/tau2_bench -q
"""

import asyncio
import importlib.util
import json
import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("tau2", reason="pip install -e <tau2-bench checkout> -- the example's one extra dependency")

import tau2_agent_function as taf
from tau2.data_model.message import AssistantMessage
from tau2.evaluator import evaluator_communicate, evaluator_nl_assertions
from tau2.runner.helpers import get_tasks
from tests.fast.fixtures.generation_fixtures import (
    GenerateEnv,
    compute_generate_args,
    make_args,
    run_generate,
    with_session_server,
)

from miles.utils.http_utils import find_available_port
from miles.utils.misc import SingletonMeta
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.types import Sample

from . import EXAMPLE_DIR

_MODEL_NAME = "Qwen/Qwen3-0.6B"
_TASK = {"domain": "mock", "task_id": "create_task_1"}
_REQUEST = "Please create a task called 'Important Meeting' for user_1."
_CONFIRMATION = "Your task 'Important Meeting' has been created."


def _tool_call(title: str) -> str:
    call = {"name": "create_task", "arguments": {"user_id": "user_1", "title": title}}
    return f"<tool_call>\n{json.dumps(call)}\n</tool_call>"


def _load_launcher():
    # Loaded by path: other example test packages put their own run.py on sys.path.
    spec = importlib.util.spec_from_file_location("tau2_bench_run", EXAMPLE_DIR / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _script_user(monkeypatch, *replies: str | Exception) -> None:
    """Replace the user simulator's LLM with ``replies``, one per user turn."""
    script = iter(replies)

    def generate(**kwargs):
        reply = next(script)
        if isinstance(reply, Exception):
            raise reply
        return AssistantMessage(role="assistant", content=reply)

    monkeypatch.setattr("tau2.user.user_simulator.generate", generate)


@contextmanager
def _session_env(*policy_replies: ProcessResult, max_seq_len: int = 32768) -> Iterator[GenerateEnv]:
    """A v1 session server over a mock SGLang server answering with ``policy_replies``, one per turn."""
    script = iter(policy_replies)
    SingletonMeta.clear_all_instances()
    try:
        with with_mock_server(model_name=_MODEL_NAME, process_fn=lambda prompt: next(script)) as mock_server:
            port = find_available_port(31000)
            args = make_args(
                variant="agentic_tool_call",
                router_port=port,
                extra_argv=[
                    "--custom-agent-function-path",
                    "tau2_agent_function.run",
                    "--use-session-server",
                    "v1",
                    "--max-seq-len",
                    str(max_seq_len),
                ],
            )
            with with_session_server(mock_server.url, args, port=port):
                yield GenerateEnv(args=compute_generate_args(args), mock_server=mock_server)
    finally:
        SingletonMeta.clear_all_instances()


def _play(env: GenerateEnv) -> tuple[Sample, list[dict]]:
    sample = Sample(prompt=[{"role": "user", "content": "tau2 mock task create_task_1"}], metadata=dict(_TASK))
    result = run_generate(env, sample, variant="agentic_tool_call")
    return result.sample, result.requests


def _trained_text(sample: Sample) -> str:
    """The response tokens the loss covers, decoded."""
    tokenizer = load_tokenizer(_MODEL_NAME, trust_remote_code=True)
    response = sample.tokens[len(sample.tokens) - sample.response_length :]
    return tokenizer.decode([token for token, mask in zip(response, sample.loss_mask, strict=True) if mask])


def test_solved_task_scores_one_and_trains_every_policy_turn(monkeypatch):
    _script_user(monkeypatch, _REQUEST, "###STOP###")
    with _session_env(ProcessResult(_tool_call("Important Meeting")), ProcessResult(_CONFIRMATION)) as env:
        sample, requests = _play(env)

    assert sample.metadata["reward"] == 1.0
    assert sample.metadata["termination_reason"] == "user_stop"
    assert asyncio.run(taf.reward_func(None, sample)) == 1.0
    # Both turns ask for the domain tools, and the second extends the first: the session
    # kept one linear trajectory, so both completions train and the tool result does not.
    assert len(requests) == 2
    assert {"create_task", "get_users"} <= {tool["function"]["name"] for tool in requests[0]["tools"]}
    trained = _trained_text(sample)
    assert '"title": "Important Meeting"' in trained and _CONFIRMATION in trained
    assert '"task_id"' in sample.response and '"task_id"' not in trained


def test_wrong_tool_call_scores_zero(monkeypatch):
    _script_user(monkeypatch, _REQUEST, "###STOP###")
    with _session_env(ProcessResult(_tool_call("Unimportant Meeting")), ProcessResult(_CONFIRMATION)) as env:
        sample, _ = _play(env)

    assert sample.metadata["reward"] == 0.0
    assert sample.metadata["termination_reason"] == "user_stop"


@pytest.mark.parametrize(
    "reply,max_seq_len,reason",
    [
        pytest.param(ProcessResult("Let me", finish_reason="length"), 32768, "length", id="max_tokens"),
        # The prompt (system prompt and tools) fits in 2000 tokens; the prompt and this reply do not.
        pytest.param(ProcessResult("word " * 2000), 2000, "context_window_exceeded", id="max_seq_len"),
    ],
)
def test_policy_ending_the_episode_early_scores_zero(monkeypatch, reply, max_seq_len, reason):
    _script_user(monkeypatch, _REQUEST)
    with _session_env(reply, max_seq_len=max_seq_len) as env:
        sample, requests = _play(env)

    assert len(requests) == 1
    assert sample.metadata["reward"] == 0.0
    assert sample.metadata["termination_reason"] == reason


def test_user_simulator_failure_leaves_no_reward(monkeypatch):
    _script_user(monkeypatch, _REQUEST, RuntimeError("user simulator rate limited"))
    with _session_env(ProcessResult(_tool_call("Important Meeting")), ProcessResult(_CONFIRMATION)) as env:
        sample, _ = _play(env)

    assert sample.metadata["reward"] is None
    assert sample.metadata["termination_reason"] == "infrastructure_error"
    assert asyncio.run(taf.reward_func(None, [sample])) == [None]


@pytest.mark.parametrize(
    "arguments",
    [pytest.param('{"title": ', id="invalid_json"), pytest.param('["user_1"]', id="not_an_object")],
)
def test_malformed_tool_call_is_an_agent_error(arguments):
    call = SimpleNamespace(id="call0", function=SimpleNamespace(name="create_task", arguments=arguments))
    with pytest.raises(taf.PolicyEnded, match="agent_error"):
        taf._to_tau2_reply(SimpleNamespace(content=None, tool_calls=[call]))


def test_empty_reply_is_an_agent_error():
    with pytest.raises(taf.PolicyEnded, match="agent_error"):
        taf._to_tau2_reply(SimpleNamespace(content="  ", tool_calls=None))


def test_task_file_names_each_task_of_the_split(tmp_path):
    path = tmp_path / "retail_test.jsonl"
    _load_launcher()._write_task_file("retail", "test", str(path))

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 40
    assert all(row["metadata"]["domain"] == "retail" for row in rows)
    assert len({row["metadata"]["task_id"] for row in rows}) == 40
    assert rows[0]["prompt"] == [{"role": "user", "content": f"tau2 retail task {rows[0]['metadata']['task_id']}"}]


@pytest.mark.parametrize(
    "domain,missing",
    [
        pytest.param("retail", "OPENAI_API_KEY", id="retail_nl_judge"),
        pytest.param("airline", "OPENAI_API_KEY", id="airline_communicate_judge"),
        pytest.param("telecom", None, id="telecom_no_judge"),
    ],
)
def test_preflight_requires_the_keys_of_the_domain_judges(monkeypatch, domain, missing):
    launcher = _load_launcher()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    monkeypatch.setenv("TAU2_USER_LLM", "hosted_vllm/user-simulator")

    if missing is None:
        assert launcher._episode_api_keys(domain) == {}
    else:
        with pytest.raises(RuntimeError, match=missing):
            launcher._episode_api_keys(domain)
        monkeypatch.setenv(missing, "key")
        assert launcher._episode_api_keys(domain) == {missing: "key"}


_JUDGE = "hosted_vllm/judge"
_JUDGE_ARGS = {"api_base": "http://judge:8000/v1", "temperature": 0.0}


class _JudgeCalled(Exception):
    pass


def _capture_judge_call(monkeypatch, module) -> dict:
    """Stop the judge at its LLM call and keep the model and kwargs it was called with."""
    call = {}

    def generate(model, messages, call_name=None, **kwargs):
        call.update(model=model, **kwargs)
        raise _JudgeCalled

    monkeypatch.setattr(module, "generate", generate)
    return call


def test_judge_endpoint_replaces_both_tau2_judges(monkeypatch):
    # Restored at teardown: _configure_judges rewrites these process-wide.
    for name in ("DEFAULT_LLM_NL_ASSERTIONS", "DEFAULT_LLM_NL_ASSERTIONS_ARGS"):
        monkeypatch.setattr(evaluator_nl_assertions, name, getattr(evaluator_nl_assertions, name))
    monkeypatch.setenv("TAU2_JUDGE_LLM", _JUDGE)
    monkeypatch.setenv("TAU2_JUDGE_LLM_ARGS", json.dumps(_JUDGE_ARGS))
    scoring = taf._configure_judges()

    nl_call = _capture_judge_call(monkeypatch, evaluator_nl_assertions)
    (retail_task,) = get_tasks("retail", task_ids=["2"])
    with pytest.raises(_JudgeCalled):
        evaluator_nl_assertions.NLAssertionsEvaluator.calculate_reward(task=retail_task, full_trajectory=[])
    assert nl_call == {"model": _JUDGE, **_JUDGE_ARGS}

    communicate_call = _capture_judge_call(monkeypatch, evaluator_communicate)
    (airline_task,) = get_tasks("airline", task_ids=["3"])
    with pytest.raises(_JudgeCalled):
        evaluator_communicate.CommunicateEvaluator.calculate_reward(
            task=airline_task, full_trajectory=[], scoring=scoring
        )
    assert communicate_call == {"model": _JUDGE, **_JUDGE_ARGS}


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_own_endpoints_need_no_openai_key(monkeypatch, domain):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TAU2_USER_LLM", "hosted_vllm/user-simulator")
    monkeypatch.setenv("TAU2_JUDGE_LLM", _JUDGE)
    monkeypatch.setenv("TAU2_JUDGE_LLM_ARGS", json.dumps(_JUDGE_ARGS))
    launcher = _load_launcher()

    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    assert launcher._episode_api_keys(domain) == {}
    # An endpoint that checks a key reads the provider's optional one, which the workers need too.
    monkeypatch.setenv("HOSTED_VLLM_API_KEY", "endpoint-key")
    assert launcher._episode_api_keys(domain) == {"HOSTED_VLLM_API_KEY": "endpoint-key"}


def test_execute_forwards_the_keys_to_the_ray_workers(monkeypatch):
    # The workers of an already-joined ray cluster see only the job's runtime env, not the launcher's environment.
    launcher = _load_launcher()
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("TAU2_USER_LLM", "hosted_vllm/user-simulator")
    submitted = {}
    args = launcher.ScriptArgs(domain="retail", num_nodes=1)
    args.create_backend = lambda: SimpleNamespace(execute_train=lambda **kwargs: submitted.update(kwargs))

    launcher.execute(args)

    assert submitted["extra_env_vars"]["OPENAI_API_KEY"] == "openai-key"  # retail's NL-assertion judge
    assert submitted["extra_env_vars"]["TAU2_USER_LLM"] == "hosted_vllm/user-simulator"
    argv = shlex.split(submitted["train_args"])
    assert argv[argv.index("--actor-num-nodes") + 1] == "1"
    assert launcher.ScriptArgs().num_nodes == launcher.POLICY_MODELS["Qwen3.8-27B"].num_nodes
