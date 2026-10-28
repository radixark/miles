import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from tests.fast.import_isolation_utils import REPO_ROOT

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample


@pytest.mark.parametrize("finish", ["abort", "length"])
async def test_search_generation_preserves_caller_sample_and_tokenizer(
    monkeypatch: pytest.MonkeyPatch, finish: str
) -> None:
    """Early abort and generated responses both use the v2 caller-owned generation state."""
    scoring = ModuleType("qa_em_format")
    scoring.compute_score_em = Mock()
    monkeypatch.setitem(sys.modules, "qa_em_format", scoring)
    spec = importlib.util.spec_from_file_location(
        "search_contract", REPO_ROOT / "examples/experimental/search-r1/generate_with_search.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = Mock(return_value={"input_ids": [10]})
    state = SimpleNamespace(
        args=SimpleNamespace(partial_rollout=False, sglang_router_ip="router", sglang_router_port=1234),
        tokenizer=tokenizer,
    )
    sample = Sample(prompt="question")
    sampling = {"temperature": 0.5}
    post = AsyncMock(
        return_value={
            "text": "answer",
            "meta_info": {"finish_reason": {"type": finish}, "output_token_logprobs": [[-0.2, 20]]},
        }
    )
    monkeypatch.setattr(module, "post", post)

    output = await module.generate(
        GenerateFnInput(state=state, sample=sample, sampling_params=sampling, evaluation=False)
    )

    assert isinstance(output, GenerateFnOutput)
    assert output.samples is sample
    tokenizer.assert_called_once_with("question", add_special_tokens=False)
    assert post.call_args.args[0] == "http://router:1234/generate"
    assert post.call_args.args[1]["sampling_params"] is sampling
    if finish == "abort":
        assert sample.status == Sample.Status.ABORTED
    else:
        assert sample.status == Sample.Status.TRUNCATED
        assert sample.tokens == [10, 20]
        assert sample.response_length == 1
        assert sample.response == "answer"
        assert sample.loss_mask == [1]
        assert sample.rollout_log_probs == [-0.2]
