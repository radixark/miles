"""The text-to-SQL episode must keep tokens, loss mask and log-probs aligned across tool turns."""

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
from tests.fast.import_isolation_utils import REPO_ROOT

from miles.rollout.base_types import GenerateFnInput
from miles.utils.types import Sample

_EOS = "#"


class _CharTokenizer:
    """One token per character, so the expected ids of a framing can be read off its text."""

    eos_token = _EOS
    eos_token_id = ord(_EOS)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + (
            "<assistant>" if add_generation_prompt else ""
        )


class _UnsafeQuery(Exception):
    pass


def _recipe_stub() -> dict[str, ModuleType]:
    """The recipe modules the episode imports, reduced to a `CALL <sql>` tool syntax and an `UNSAFE` query."""
    parsing = ModuleType("text_to_sql_recipe.parsing")
    parsing.parse_tool_calls = lambda message: (
        [SimpleNamespace(name="sql", arguments={"query": message["content"][5:]})]
        if message["content"].startswith("CALL ")
        else []
    )
    parsing.has_tool_call_intent = lambda message: "CALL" in message["content"]
    parsing.MALFORMED_ACTION_PROMPT = "malformed"

    prompts = ModuleType("text_to_sql_recipe.prompts")
    prompts.SQL_TOOL_NAME = "sql"
    prompts.SQL_LIMIT_PROMPT = "answer now"
    prompts.SQL_TOOL_CONTEXT_ERROR_MAX_BYTES = 240
    prompts.SQL_TOOL_CONTEXT_ERROR_GUIDANCE = ""
    prompts.tool_result_prompt = lambda: ""

    reward = ModuleType("text_to_sql_recipe.reward")
    reward.observe_tool_calls = lambda events, message, **kwargs: None

    sqlite_exec = ModuleType("text_to_sql_recipe.sqlite_exec")
    sqlite_exec.SQLiteInfrastructureError = type("SQLiteInfrastructureError", (Exception,), {})
    sqlite_exec.SQL_TOOL_TERMINAL_ERRORS = (_UnsafeQuery,)
    sqlite_exec.tool_executor = lambda task: None

    def execute_sql_tool(task, executor, arguments):
        if arguments["query"] == "UNSAFE":
            raise _UnsafeQuery
        return [SimpleNamespace(format=lambda: "rows")]

    sqlite_exec.execute_sql_tool = execute_sql_tool

    config = ModuleType("text_to_sql_recipe.config")
    config.RewardConfig = lambda **kwargs: kwargs

    package = ModuleType("text_to_sql_recipe")
    modules = {"parsing": parsing, "prompts": prompts, "reward": reward, "sqlite_exec": sqlite_exec, "config": config}
    for name, module in modules.items():
        setattr(package, name, module)
    package.grading = ModuleType("text_to_sql_recipe.grading")
    return {"text_to_sql_recipe": package, **{f"text_to_sql_recipe.{k}": v for k, v in modules.items()}}


@pytest.fixture
def sql_episode(monkeypatch):
    for name, module in _recipe_stub().items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("sql_episode", REPO_ROOT / "examples/text_to_sql/sql_episode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "compute_routing_headers", lambda args, sample: {})
    return module


async def _run(sql_episode, monkeypatch, turns: list[str], *, max_context_len: int = 0) -> Sample:
    """Run one episode whose engine replies with `turns` in order, each sampled up to and including eos."""
    replies = iter(turns)

    async def fake_post(url, payload, headers):
        text = next(replies)
        return {
            "text": text,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.5, ord(c)] for c in text + _EOS],
            },
        }

    monkeypatch.setattr(sql_episode, "post", fake_post)
    args = SimpleNamespace(
        sglang_router_ip="router",
        sglang_router_port=1,
        text_to_sql_max_tool_turns=2,
        rollout_max_response_len=1000,
        rollout_max_context_len=max_context_len,
        rollout_top_logprobs_num=0,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_output_store=False,
        sglang=SimpleNamespace(common_value=lambda key: None),
    )
    monkeypatch.setattr(
        sql_episode,
        "compute_request_payload",
        lambda args, tokens, sampling_params, **kwargs: ({"input_ids": tokens}, None),
    )
    sample = Sample(prompt=[{"role": "user", "content": "q"}], metadata={"task": {"id": 1}})
    state = SimpleNamespace(args=args, tokenizer=_CharTokenizer(), processor=None)
    output = await sql_episode.generate(
        GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False)
    )
    return output.samples


def _response(sample: Sample) -> str:
    return "".join(chr(i) for i in sample.tokens[-sample.response_length :])


async def test_tool_turns_are_masked_and_framed_after_the_sampled_eos(sql_episode, monkeypatch):
    sample = await _run(sql_episode, monkeypatch, ["</think>CALL SELECT 1", "</think>CALL SELECT 2", "</think>done"])

    observation = "\n<user>Results for sql:\nrows\n\n\n\nYou have 1 turns left to complete the task.<assistant>"
    last_observation = (
        "\n<user>Results for sql:\nrows\n\n\n\nYou have 0 turns left to complete the task.<user>answer now<assistant>"
    )
    sampled = [f"</think>CALL SELECT 1{_EOS}", f"</think>CALL SELECT 2{_EOS}", f"</think>done{_EOS}"]
    assert _response(sample) == sampled[0] + observation + sampled[1] + last_observation + sampled[2]
    expected_mask = (
        [1] * len(sampled[0])
        + [0] * len(observation)
        + [1] * len(sampled[1])
        + [0] * len(last_observation)
        + [1] * len(sampled[2])
    )
    assert sample.loss_mask == expected_mask
    assert sample.rollout_log_probs == [-0.5 if m else 0.0 for m in expected_mask]
    episode = sample.metadata["text_to_sql"]
    assert episode["tool_calls"] == 2
    assert episode["final_text"] == "done"


def test_a_turn_cut_before_eos_gets_one_closed_before_the_observation(sql_episode):
    tokenizer = _CharTokenizer()
    messages = [{"role": "user", "content": "obs"}]

    after_eos = sql_episode._observation_ids(tokenizer, messages, last_token_id=tokenizer.eos_token_id)
    after_text = sql_episode._observation_ids(tokenizer, messages, last_token_id=ord("x"))

    assert tokenizer.decode(after_eos) == "\n<user>obs<assistant>"
    assert tokenizer.decode(after_text) == f"{_EOS}\n<user>obs<assistant>"


async def test_an_unsafe_query_ends_the_rollout_without_an_observation(sql_episode, monkeypatch):
    sample = await _run(sql_episode, monkeypatch, ["</think>CALL UNSAFE", "</think>never sampled"])

    assert sample.metadata["text_to_sql"]["unsafe"]
    assert _response(sample) == f"</think>CALL UNSAFE{_EOS}"
    assert sample.loss_mask == [1] * sample.response_length


async def test_an_observation_that_would_overflow_the_context_truncates_instead(sql_episode, monkeypatch):
    sample = await _run(
        sql_episode, monkeypatch, ["</think>CALL SELECT 1", "</think>never sampled"], max_context_len=60
    )

    assert sample.status == Sample.Status.TRUNCATED
    assert _response(sample) == f"</think>CALL SELECT 1{_EOS}"
