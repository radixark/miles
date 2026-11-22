import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from tests.fast.import_isolation_utils import REPO_ROOT

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample


@pytest.mark.parametrize("failed", [False, True])
async def test_strands_generation_uses_supplied_tokenizer_and_wraps_success_or_truncation(
    monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    """SDK outcomes preserve caller state, captured tokens, and cleanup through the v2 wrapper."""
    tokenizer = SimpleNamespace(decode=Mock(return_value="answer"))
    state = SimpleNamespace(
        args=SimpleNamespace(
            partial_rollout=False, hf_checkpoint="models/tiny", sglang_router_ip="router", sglang_router_port=1
        ),
        tokenizer=tokenizer,
    )
    model = SimpleNamespace(
        tokenizer=tokenizer,
        token_manager=SimpleNamespace(segments=[[10]], token_ids=[10, 20], loss_mask=[0, 1], logprobs=[0.0, -0.2]),
        format_request_messages=Mock(return_value=[{"role": "tool"}]),
        reset=Mock(),
    )
    agent = SimpleNamespace(invoke_async=AsyncMock(), messages=[], cleanup=Mock())
    if failed:
        agent.invoke_async.side_effect = RuntimeError("SDK failed")
    model_factory = Mock(return_value=model)
    for name, attributes in {
        "camel": {},
        "camel.interpreters": {"SubprocessInterpreter": Mock()},
        "strands": {"Agent": Mock(return_value=agent), "tool": lambda function: function},
        "strands_sglang": {
            "SGLangClient": SimpleNamespace(from_miles_args=Mock(return_value="client")),
            "SGLangModel": model_factory,
        },
        "strands_sglang.tool_limiter": {"ToolIterationLimiter": Mock(return_value=SimpleNamespace(iteration_count=2))},
    }.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "strands_contract", REPO_ROOT / "examples/experimental/strands_sglang/generate_with_strands.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sample = Sample(prompt="question")
    sampling = {"max_new_tokens": 5, "temperature": 0.5, "top_p": 0.9}

    output = await module.generate(
        GenerateFnInput(state=state, sample=sample, sampling_params=sampling, evaluation=False)
    )

    assert isinstance(output, GenerateFnOutput)
    assert output.samples is sample
    assert model_factory.call_args.kwargs == {
        "tokenizer": tokenizer,
        "client": "client",
        "model_id": "tiny",
        "params": sampling,
    }
    agent.invoke_async.assert_awaited_once_with("question")
    assert sample.tokens == [10, 20]
    assert sample.loss_mask == [1]
    assert sample.rollout_log_probs == [-0.2]
    assert sample.response == "answer"
    assert sample.response_length == 1
    assert sample.tool_iterations == 2
    assert sample.tool_call_count == 1
    assert sample.status == (Sample.Status.TRUNCATED if failed else Sample.Status.COMPLETED)
    model.reset.assert_called_once_with()
    agent.cleanup.assert_called_once_with()
