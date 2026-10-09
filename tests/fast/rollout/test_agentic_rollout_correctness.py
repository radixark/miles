"""Small-batch data checks for the same production path as the rollout benchmark."""

import asyncio
import base64
import json
from argparse import Namespace

import httpx
import numpy as np
from fastapi import FastAPI, Request
from tests.ci.ci_register import register_cpu_ci
from tests.fast.fixtures.generation_fixtures import with_session_server
from tests.manual.session import _rollout_benchmark_agent as agent
from tests.manual.session.bench_agentic_rollout import NUM_LAYERS, TOPK, build_args
from tests.manual.session.bench_session_server_overhead import _completion_token_ids

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.rollout.inference_rollout.compatibility import load_rollout_function
from miles.utils import http_utils
from miles.utils.async_utils import run
from miles.utils.chat_template_utils import get_tito_tokenizer, resolve_fixed_chat_template
from miles.utils.misc import SingletonMeta
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

register_cpu_ci(est_time=30, suite="stage-b-cpu", labels=["rollout"])


def _trajectory(tokenizer, tito, case):
    history, requests, responses = [], [], []
    tokens, log_probs, mask, routed = [], [], [], []
    prompt_length = 0
    for turn in range(case + 2):
        messages = history + [{"role": "user", "content": f"case {case}, turn {turn}: 请计算 {case + turn}"}]
        assistant = {"role": "assistant", "content": f"answer {case}: {turn + 10}"}
        prompt = tito.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, template_args=tito.default_template_args(None)
        )
        completion = _completion_token_ids(tito, tokenizer, messages, assistant, None)
        if turn == 0:
            prompt_length = len(prompt)
        else:
            assert prompt[: len(tokens)] == tokens
            gap = len(prompt) - len(tokens)
            log_probs.extend([0.0] * gap)
            mask.extend([0] * gap)
        values = [-(1 + case * 64 + turn * 16 + i) / 256 for i in range(len(completion))]
        log_probs.extend(values)
        mask.extend([1] * len(completion))
        tokens = prompt + completion
        start = sum(len(chunk) for chunk in routed)
        count = len(tokens) - 1 - start
        r3 = (np.arange(count * NUM_LAYERS * TOPK, dtype=np.int32) % 97 + case * 1000 + turn * 100).reshape(
            count, NUM_LAYERS, TOPK
        )
        routed.append(r3)
        requests.append({"messages": messages})
        responses.append(
            {
                "id": f"case-{case}-turn-{turn}",
                "object": "chat.completion",
                "created": 0,
                "model": "synthetic",
                "choices": [
                    {
                        "index": 0,
                        "message": assistant,
                        "finish_reason": "stop",
                        "meta_info": {
                            "completion_tokens": len(completion),
                            "output_token_logprobs": list(zip(values, completion, strict=True)),
                            "routed_experts": base64.b64encode(r3.tobytes()).decode(),
                        },
                    }
                ],
            }
        )
        history = messages + [assistant]
    expected = dict(
        tokens=tokens, log_probs=log_probs, mask=mask, r3=np.concatenate(routed), prompt_length=prompt_length
    )
    return requests, responses, expected


def test_small_batch_rollout_preserves_data(tmp_path, monkeypatch):
    template, template_kwargs = resolve_fixed_chat_template("qwen3")
    tokenizer = load_tokenizer("Qwen/Qwen3-0.6B", chat_template_path=template, trust_remote_code=True)
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen3", chat_template_kwargs=template_kwargs)
    cases = [_trajectory(tokenizer, tito, case) for case in range(2)]
    app = FastAPI()
    arrived = set()
    both_started = asyncio.Event()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        case = int(payload["messages"][0]["content"].split(",")[0].split()[1])
        turn = (len(payload["messages"]) - 1) // 2
        assert payload["messages"] == cases[case][0][turn]["messages"]
        if turn == 0:
            arrived.add(case)
            if len(arrived) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=10)
        return cases[case][1][turn]

    port = http_utils.find_available_port(28000)

    @app.get("/list_workers")
    async def list_workers():
        return {"urls": [f"http://127.0.0.1:{port}"]}

    @app.post("/abort_request")
    async def abort():
        return {"status": "ok"}

    async def run_agent(base_url, prompt, **kwargs):
        case = int(prompt[0]["content"])
        async with httpx.AsyncClient(timeout=30) as client:
            for body, expected_reply in zip(cases[case][0], cases[case][1], strict=True):
                response = await client.post(base_url + "/v1/chat/completions", json=body)
                response.raise_for_status()
                assert response.json()["choices"][0]["message"] == expected_reply["choices"][0]["message"]
        return {"case": case, "turns_ok": len(cases[case][0])}

    monkeypatch.setattr(agent, "run_agent", run_agent)
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "".join(json.dumps({"prompt": [{"role": "user", "content": str(i)}], "label": ""}) + "\n" for i in range(2))
    )
    bench = Namespace(hf_checkpoint="Qwen/Qwen3-0.6B", sessions=2, repetitions=1)
    args = build_args(bench, prompt_path, port)
    args.apply_chat_template = False
    backend = UvicornThreadServer(app, host="127.0.0.1", port=port)
    SingletonMeta.clear_all_instances()
    try:
        backend.start()
        http_utils.init_http_client(args)
        with with_session_server(backend.url, args, port=http_utils.find_available_port(33000)):
            rollout = load_rollout_function(
                RolloutFnConstructorInput(args=args, data_source=RolloutDataSourceWithBuffer(args)),
                args.rollout_function_path,
            )
            output = run(asyncio.wait_for(rollout(RolloutFnTrainInput(rollout_id=0)), timeout=30))
        assert arrived == {0, 1}
        assert len(output.samples) == 2
        assert {group[0][0].index for group in output.samples} == {0, 1}
        for group in output.samples:
            assert len(group) == 1 and len(group[0]) == 1
            sample = group[0][0]
            case = sample.index
            expected = cases[case][2]
            assert sample.status == Sample.Status.COMPLETED
            assert sample.metadata["case"] == case
            assert sample.metadata["turns_ok"] == case + 2
            assert sample.reward == 1.0
            assert sample.tokens == expected["tokens"]
            assert sample.response_length == len(expected["tokens"]) - expected["prompt_length"]
            assert sample.rollout_log_probs == expected["log_probs"]
            assert sample.loss_mask == expected["mask"]
            assert sample.rollout_routed_experts.dtype == np.int32
            np.testing.assert_array_equal(sample.rollout_routed_experts, expected["r3"])
    finally:
        if http_utils._http_client is not None:
            run(http_utils._http_client.aclose())
            http_utils._http_client = None
        backend.stop()
        SingletonMeta.clear_all_instances()
