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
    sampler = SimpleNamespace(
        sample_async=lambda **kwargs: complete(SimpleNamespace(sequences=[SimpleNamespace(tokens=[1])]))
    )
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
