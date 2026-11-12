"""A checkpoint import must finish before a multi-LoRA client's first update."""

from types import SimpleNamespace

import pytest

from examples.multi_lora import run_multi_tenant_example as example


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint", [None, "tinker://pissa-{client}/weights/init"])
async def test_initial_checkpoint_load_precedes_training(monkeypatch, checkpoint):
    events = []
    losses = iter([2.0, 1.0])

    async def complete(value=None, event=None):
        if event:
            events.append(event)
        return value

    async def load(path):
        events.append(path)
        return complete(event="loaded")

    async def forward(*args, **kwargs):
        events.append("forward")
        return complete(SimpleNamespace(metrics={"loss:sum": next(losses)}))

    async def save(**kwargs):
        return complete(SimpleNamespace(path="tinker://model/sampler_weights/final"))

    tokenizer = SimpleNamespace(encode=lambda _: [1, 2], decode=lambda _: example.MARKERS[1])
    training = SimpleNamespace(
        load_state_async=load,
        get_tokenizer=lambda: tokenizer,
        forward_backward_async=forward,
        optim_step_async=lambda *args: complete(complete()),
        save_weights_for_sampler_async=save,
    )
    sampler = SimpleNamespace(sample_async=lambda **kwargs: complete(SimpleNamespace(sequences=[SimpleNamespace(tokens=[1])])))
    service = SimpleNamespace(
        create_lora_training_client_async=lambda **kwargs: complete(training),
        create_sampling_client_async=lambda **kwargs: complete(sampler),
    )
    monkeypatch.setattr(example.tinker, "ServiceClient", lambda **kwargs: service)
    monkeypatch.setattr(
        example,
        "build_datums",
        lambda *args: [SimpleNamespace(loss_fn_inputs={"target_tokens": SimpleNamespace(shape=(1,))})],
    )
    args = SimpleNamespace(
        base_url="http://localhost",
        base_model="test",
        lora_rank=4,
        initial_checkpoint=checkpoint,
        steps=2,
        lr=0.001,
        max_tokens=8,
    )
    assert (await example.run_client(1, args))["ok"]
    assert events == (["tinker://pissa-1/weights/init", "loaded"] if checkpoint else []) + ["forward", "forward"]


@pytest.mark.asyncio
async def test_gsm8k_preserves_sampled_history_and_total_response_budget():
    from examples.multi_lora import run_gsm8k_tool_repro as gsm8k

    prompts, budgets = [], []
    generated = "sampled<|im_end|>"
    messages = iter(
        [
            {"role": "assistant", "tool_calls": [SimpleNamespace(id=None, function=SimpleNamespace(name="calculator", arguments='{"operation":"add","a":1,"b":2}'))]},
            {"role": "assistant", "tool_calls": [SimpleNamespace(id=None, function=SimpleNamespace(name="answer", arguments='{"answer":3}'))]},
        ]
    )
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
    renderer = SimpleNamespace(
        tokenizer=tokenizer,
        create_conversation_prefix_with_tools=lambda *args, **kwargs: [],
        build_generation_prompt=lambda *args: gsm8k.types.ModelInput.from_ints([1]),
        get_stop_sequences=lambda: [151645],
        parse_response=lambda tokens: (next(messages), "clean"),
    )

    async def sample(**kwargs):
        prompts.append(bytes(kwargs["prompt"].to_ints()).decode())
        budgets.append(kwargs["sampling_params"].max_tokens)
        tokens = list(generated.encode())
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens, logprobs=[0.0] * len(tokens), stop_reason="stop")])

    args = SimpleNamespace(raw_question=False, neil_chat_template=True, max_turns=20, total_response_tokens=8192, max_tokens=8192, context_tokens=32736)
    import asyncio

    result = await gsm8k._episode(SimpleNamespace(sample_async=sample), renderer, {"question": "One plus two?", "answer": "#### 3"}, args, asyncio.Semaphore(1), 0.0, 0)
    assert result["reward"] == 1.0
    assert budgets == [8192, 8192 - len(generated)]
    assert prompts[1].startswith(prompts[0] + generated + "<|im_start|>user\n<tool_response>\n")
    assert '{"result": 3.0}' in prompts[1]
    assert prompts[0].endswith("Problem: One plus two?\n\nPlease solve this step-by-step.<|im_end|>\n<|im_start|>assistant\n")


@pytest.mark.asyncio
async def test_gsm8k_group_filter_refills_without_using_invalid_samples(monkeypatch):
    import asyncio
    import numpy as np
    from examples.multi_lora import run_gsm8k_tool_repro as gsm8k

    samples = iter(
        [
            *[{"reward": 0.0, "failure": None} for _ in range(4)],
            {"reward": 0.0, "failure": None},
            {"reward": 1.0, "failure": None},
            {"reward": 1.0, "failure": None},
            {"reward": 0.0, "failure": "token_limit"},
            {"reward": 0.0, "failure": None},
            {"reward": 1.0, "failure": None},
            {"reward": 0.0, "failure": "no_answer_tool"},
            {"reward": 0.0, "failure": "token_limit"},
        ]
    )

    async def episode(*args):
        return next(samples)

    monkeypatch.setattr(gsm8k, "_episode", episode)
    args = SimpleNamespace(batch_size=2, group_size=4, max_turns=20, filter_groups=True)
    groups, selected, requested, rejected, reward, failures = await gsm8k._collect_training_groups(None, None, np.array([{}, {}]), args, asyncio.Semaphore(1), 0, 0)
    assert [len(g) for g in groups] == [3, 2]
    assert all(e["failure"] is None for group in groups for e in group)
    assert requested == 12 and rejected == 1 and len(selected) == 2
    assert reward == pytest.approx(3 / 12)
    assert failures == {"answer_submitted": 9, "token_limit": 2, "no_answer_tool": 1}
