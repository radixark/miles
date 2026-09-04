"""Marker-memorization smoke client for the Tinker gateway.

Each client is its own tenant (distinct api_key): it trains a LoRA adapter to
memorize a private marker phrase, then samples greedily and checks the model
completes with its marker. In multi mode all clients train concurrently on the
same prompt with different markers, so a passing run demonstrates adapter
isolation end to end.

Needs only the official SDK: ``pip install tinker``.

Usage:
  python examples/multi_lora/client.py --base-model /root/models/Qwen3-30B-A3B --mode single
  python examples/multi_lora/client.py --base-model /root/models/Qwen3-30B-A3B --mode multi --clients 4
"""

import argparse
import asyncio

import tinker
from tinker import types

PROMPT = "The secret word is"

MARKERS = [
    "cobalt-lantern-73",
    "velvet-otter-19",
    "amber-glacier-58",
    "crimson-abacus-31",
    "silver-nebula-86",
    "indigo-walrus-44",
    "maroon-zeppelin-27",
    "golden-isotope-65",
]


def build_datums(tokenizer, marker: str) -> list[types.Datum]:
    texts = [
        f"{PROMPT} {marker}.",
        f"Remember this: the secret word is {marker}.",
        f"Q: What is the secret word? A: The secret word is {marker}.",
        f"Note for later. The secret word is {marker}.",
    ]
    datums = []
    for text in texts:
        tokens = tokenizer.encode(text)
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={"target_tokens": tokens[1:], "weights": [1.0] * (len(tokens) - 1)},
            )
        )
    return datums


async def run_client(index: int, args) -> dict:
    marker = MARKERS[index]
    tag = f"client-{index}"
    # tml- prefix required by the SDK; the gateway reads the key as the tenant id
    service = tinker.ServiceClient(base_url=args.base_url, api_key=f"tml-smoke-{tag}")
    training = await service.create_lora_training_client_async(base_model=args.base_model, rank=args.lora_rank)
    tokenizer = training.get_tokenizer()

    datums = build_datums(tokenizer, marker)
    target_tokens = sum(datum.loss_fn_inputs["target_tokens"].shape[0] for datum in datums)

    losses = []
    for _ in range(args.steps):
        fb_future = await training.forward_backward_async(datums, loss_fn="cross_entropy")
        optim_future = await training.optim_step_async(types.AdamParams(learning_rate=args.lr))
        fb = await fb_future
        await optim_future
        losses.append(fb.metrics["loss:sum"] / target_tokens)

    save_future = await training.save_weights_for_sampler_async(name="final")
    sampler_path = (await save_future).path
    sampler = await service.create_sampling_client_async(model_path=sampler_path)

    response = await sampler.sample_async(
        prompt=types.ModelInput.from_ints(tokenizer.encode(PROMPT)),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
    )
    completion = tokenizer.decode(response.sequences[0].tokens)

    ok = losses[-1] < losses[0] and marker in completion
    print(
        f"[{tag}] {'PASS' if ok else 'FAIL'} loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
        f"marker {marker!r}, completion {completion!r}"
    )
    return {"tag": tag, "ok": ok}


async def main(args):
    n_clients = 1 if args.mode == "single" else args.clients
    assert n_clients <= len(MARKERS), f"at most {len(MARKERS)} clients"
    results = await asyncio.gather(*(run_client(i, args) for i in range(n_clients)))
    failed = [r["tag"] for r in results if not r["ok"]]
    assert not failed, f"failed clients: {failed}"
    print(f"all {n_clients} client(s) passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:10613")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-tokens", type=int, default=24)
    asyncio.run(main(parser.parse_args()))
