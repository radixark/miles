import json

import pytest
from tests.fast.fixtures.generation_fixtures import listify, make_sample, run_generate

from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.types import Sample

pytest.importorskip("pydantic_ai")

_TOOL_CALL = {"name": "calculate", "arguments": {"a": 1200, "op": "*", "b": 3}}
_RECIPE_ARGV = [
    "--custom-agent-function-path",
    "pydantic_ai_agent_function.run",
    "--session-message-matcher",
    "loose_tool_call",
]


def _engine(final_reply: str):
    def process_fn(prompt: str) -> ProcessResult:
        if "<tool_response>" not in prompt:
            return ProcessResult(text=f"<tool_call>\n{json.dumps(_TOOL_CALL)}\n</tool_call>")
        assert "3600" in prompt.rsplit("<tool_response>", 1)[1]
        return ProcessResult(text=final_reply)

    return process_fn


def _broken_engine(prompt: str) -> ProcessResult:
    raise RuntimeError("engine fault")


def _run(env) -> list[Sample]:
    sample = make_sample(prompt=[{"role": "user", "content": "What is 1200 * 3?"}])
    sample.metadata = {"answer": 3600}
    return listify(run_generate(env, sample, variant="agentic_tool_call").sample)


@pytest.mark.parametrize("variant", ["agentic_tool_call"])
@pytest.mark.parametrize("generation_env", [{"args_kwargs": {"extra_argv": _RECIPE_ARGV}}], indirect=True)
@pytest.mark.parametrize(("final_reply", "reward"), [("1200 * 3 = 3,600", 1.0), ("It is 3500.", 0.0)])
def test_episode_records_one_sample_with_the_verdict(variant, generation_env, final_reply, reward):
    generation_env.mock_server.process_fn = _engine(final_reply)

    [sample] = _run(generation_env)

    assert sample.status == Sample.Status.COMPLETED
    assert len(sample.metadata["leaf"]["path_node_ids"]) == 2
    assert final_reply in sample.response
    assert sample.reward == reward
    assert sample.metadata["exit_status"] == "Completed"


@pytest.mark.parametrize("variant", ["agentic_tool_call"])
@pytest.mark.parametrize("generation_env", [{"args_kwargs": {"extra_argv": _RECIPE_ARGV}}], indirect=True)
def test_engine_failure_discards_the_episode(variant, generation_env):
    generation_env.mock_server.process_fn = _broken_engine

    [sample] = _run(generation_env)

    assert sample.status == Sample.Status.ABORTED
    assert sample.metadata["exit_status"] == "ModelServerError"
