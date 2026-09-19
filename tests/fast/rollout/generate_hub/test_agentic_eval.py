import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, Request

from miles.ray.rollout import metrics
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.checkpoint_eval import retarget_args
from miles.rollout.generate_hub import agentic_tool_call
from miles.rollout.inference_rollout import inference_rollout_common, inference_rollout_eval
from miles.rollout.session.samples.codec import SamplesReply
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.function_registry import function_registry
from miles.utils.lora import LORA_ADAPTER_NAME
from miles.utils.types import Sample


def _input(**overrides):
    args = SimpleNamespace(
        custom_agent_function_path="test.eval_agent",
        sglang_router_ip="rollout-router",
        sglang_router_port=30000,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        apply_chat_template_kwargs={"enable_thinking": False},
        partial_rollout=False,
        group_rm=False,
        use_session_server=overrides.pop("use_session_server", "v2"),
        max_seq_len=1,
        sglang_speculative_algorithm=None,
        **overrides,
    )
    state = SimpleNamespace(
        args=args,
        generate_fn_semaphore=asyncio.Semaphore(2),
        aborted=False,
        generate_function=agentic_tool_call.generate,
    )
    return GenerateFnInput(
        state=state,
        sample=Sample(index=7, prompt=[{"role": "user", "content": "hello"}], metadata={"task": "test"}),
        sampling_params={"temperature": 0.0, "top_p": 0.8, "top_k": -1, "max_new_tokens": 8, "no_stop_trim": True},
        evaluation=True,
    )


@pytest.fixture(autouse=True)
def reject_session_and_reward_calls(monkeypatch):
    monkeypatch.setattr(
        agentic_tool_call.OpenAIEndpointTracer, "create", AsyncMock(side_effect=AssertionError("unexpected session"))
    )
    monkeypatch.setattr(
        inference_rollout_common, "async_rm", AsyncMock(side_effect=AssertionError("unexpected reward model"))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_session_server", [False, True, "v2"])
@pytest.mark.parametrize("reward", [0.0, 1.0, {"accuracy": 1.0}])
async def test_eval_uses_retargeted_router_and_agent_verdict(use_session_server, reward):
    input = _input(use_session_server=use_session_server, lora_rank=8)
    input.state.args = retarget_args(input.args, "eval-router", 31000, 1, 1)
    original_sample = deepcopy(input.sample)
    original_sampling_params = deepcopy(input.sampling_params)
    report = {"reward": reward, "eval_report": {"passed": True}, "agent_metrics": {"turns": 3}}
    calls = []

    async def agent(**kwargs):
        calls.append(kwargs)
        kwargs["metadata"]["agent_note"] = "done"
        return report

    with function_registry.temporary("test.eval_agent", agent):
        sample = await inference_rollout_common.generate_and_rm(
            input.state, input.sample, input.sampling_params, evaluation=True
        )

    assert isinstance(sample, Sample)
    assert sample.index == 7
    assert sample.reward == reward
    assert sample.status == Sample.Status.COMPLETED
    assert sample.metadata == {"task": "test", "agent_note": "done", **report}
    assert sample.tokens == []
    assert sample.response == ""
    assert sample.loss_mask is sample.rollout_log_probs is sample.rollout_routed_experts is None
    assert input.sample == original_sample
    assert input.sampling_params == original_sampling_params
    assert len(calls) == 1
    assert calls[0]["base_url"] == "http://eval-router:31000"
    assert calls[0]["request_kwargs"] == {
        "temperature": 0.0,
        "top_p": 0.8,
        "top_k": -1,
        "max_tokens": 8,
        "no_stop_trim": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "lora_path": LORA_ADAPTER_NAME,
    }
    assert "max_seq_len" not in calls[0]["metadata"]
    assert "session_server_id" not in calls[0]["metadata"]


@pytest.mark.asyncio
async def test_eval_request_template_args_override_launch_defaults():
    input = _input()
    input.sampling_params["chat_template_kwargs"] = {"enable_thinking": True}

    async def agent(**kwargs):
        assert kwargs["request_kwargs"]["chat_template_kwargs"] == {"enable_thinking": True}
        assert "lora_path" not in kwargs["request_kwargs"]
        return {"reward": 1}

    with function_registry.temporary("test.eval_agent", agent):
        await agentic_tool_call.generate(input)
    assert input.args.apply_chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, {}, {"reward": None}, RuntimeError("agent unavailable")])
async def test_failed_eval_keeps_one_aborted_result_without_scoring_empty_text(result):
    input = _input()

    async def agent(**kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    with function_registry.temporary("test.eval_agent", agent):
        sample = await inference_rollout_common.generate_and_rm(
            input.state, input.sample, input.sampling_params, evaluation=True
        )

    assert isinstance(sample, Sample)
    assert sample.index == input.sample.index
    assert sample.status == Sample.Status.ABORTED
    assert sample.reward is None
    assert sample.tokens == []


@pytest.mark.asyncio
async def test_eval_cancellation_propagates():
    async def agent(**kwargs):
        raise asyncio.CancelledError

    with function_registry.temporary("test.eval_agent", agent), pytest.raises(asyncio.CancelledError):
        await agentic_tool_call.generate(_input())


@pytest.mark.asyncio
@pytest.mark.parametrize("use_session_server", [True, "v2"])
async def test_training_still_creates_and_collects_session(monkeypatch, use_session_server):
    input = _input(use_session_server=use_session_server, session_server_addrs=["session:32000"])
    input = GenerateFnInput(input.state, input.sample, input.sampling_params, evaluation=False)
    collected = Sample(tokens=[1, 2], response="answer", response_length=1, status=Sample.Status.COMPLETED)
    tracer = SimpleNamespace(
        base_url="http://session:32000/sessions/sid",
        session_id="sid",
        session_server_id="session:32000",
        session_server_instance_id=None,
        collect_samples=AsyncMock(return_value=SamplesReply([collected], {}, None)),
    )
    create = AsyncMock(return_value=tracer)
    monkeypatch.setattr(agentic_tool_call.OpenAIEndpointTracer, "create", create)

    async def agent(**kwargs):
        assert kwargs["base_url"] == tracer.base_url
        assert kwargs["metadata"]["max_seq_len"] == 1
        return {"reward": 1}

    with function_registry.temporary("test.eval_agent", agent):
        output = await agentic_tool_call.generate(input)

    create.assert_awaited_once_with(input.args)
    tracer.collect_samples.assert_awaited_once()
    assert tracer.collect_samples.call_args.kwargs["max_seq_len"] == 1
    assert output.samples == ([collected] if use_session_server == "v2" else collected)


@pytest.mark.asyncio
@pytest.mark.parametrize("reward_key", [None, "accuracy"])
async def test_eval_http_calls_and_trial_metrics_without_session(monkeypatch, reward_key):
    input = _input()
    args = input.args
    args.__dict__.update(
        hf_checkpoint="unused-cached-dataset",
        apply_chat_template=False,
        chat_template_path=None,
        sglang_router_policy="round_robin",
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        eval_reward_key=reward_key,
        reward_key="train_score" if reward_key else None,
        custom_eval_rollout_log_function_path=None,
        log_passrate=True,
        n_samples_per_eval_prompt=2,
    )
    input.state.args = retarget_args(args, "eval-router", 31000, 1, 1)
    args = input.args
    config = EvalDatasetConfig("agent", "unused", n_samples_per_eval_prompt=2, temperature=0, top_p=1, top_k=-1)
    cache_key = config.cache_key + (args.hf_checkpoint, args.apply_chat_template, args.chat_template_path)
    dataset = SimpleNamespace(samples=[Sample(prompt="success"), Sample(prompt="no verdict", metadata={"fail": True})])
    app = FastAPI()
    requests = []

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        requests.append(request)
        return {"choices": [{"message": {"role": "assistant", "content": "answer"}}]}

    async def agent(base_url, prompt, request_kwargs, metadata):
        assert base_url == "http://eval-router:31000"
        if metadata.get("fail"):
            return None
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app)) as client:
            for _ in range(2):
                response = await client.post(
                    f"{base_url}/v1/chat/completions", json={"messages": [], **request_kwargs}
                )
                response.raise_for_status()
        return {"reward": {"accuracy": 1.0} if reward_key else 1.0}

    with function_registry.temporary("test.eval_agent", agent):
        data = await inference_rollout_eval.eval_rollout_single_dataset(input.state, config, {cache_key: dataset})

    assert len(requests) == 4
    assert data["agent"]["rewards"] == [1.0, 1.0, None, None]
    assert len(data["agent"]["samples"]) == 4
    monkeypatch.setattr(metrics, "compute_rollout_step", lambda *_: 0)
    monkeypatch.setattr(metrics.tracking, "log", lambda *_args, **_kwargs: None)
    logged = metrics.log_eval_rollout_data(0, args, data)
    assert logged["eval/agent"] == 0.5
    assert logged["eval/agent-none_reward_ratio"] == 0.5
    assert logged["eval/agent-pass@2"] == 0.5
    assert not any("response_len" in key or "num_training_samples" in key for key in logged)
