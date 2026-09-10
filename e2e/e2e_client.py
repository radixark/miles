"""One tenant, three dependent RL-style steps against the multi-LoRA Tinker gateway.

Every step depends on the previous one:
  rollout (sample from the current adapter version)
    -> forward_backward on the sampled continuation
    -> optim_step
    -> save_weights_for_sampler (publish the new adapter version)
    -> the next step's rollout samples from that new version.
"""

import argparse
import asyncio
import math
import time

import tinker
from tinker import types

PROMPT = "The quick brown fox"


async def publish(training, service, name: str):
    save_future = await training.save_weights_for_sampler_async(name=name)
    path = (await save_future).path
    sampler = await service.create_sampling_client_async(model_path=path)
    return path, sampler


async def main(args) -> None:
    # tml- prefix required by the SDK; the gateway reads the key as the tenant id
    service = tinker.ServiceClient(base_url=args.base_url, api_key="tml-e2e-single-user")
    # the gateway's adapter layout has no unembedding LoRA; the SDK default would be rejected
    training = await service.create_lora_training_client_async(
        base_model=args.base_model, rank=args.lora_rank, train_unembed=False
    )
    tokenizer = training.get_tokenizer()
    prompt_tokens = tokenizer.encode(PROMPT)

    t0 = time.time()
    path, sampler = await publish(training, service, "step0")
    print(f"[e2e] published initial adapter version: {path} ({time.time() - t0:.1f}s)", flush=True)

    for step in range(1, args.steps + 1):
        t_step = time.time()
        # 1. rollout from the version published by the previous step
        response = await sampler.sample_async(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0),
        )
        sampled = list(response.sequences[0].tokens)
        assert sampled, f"step {step}: rollout returned no tokens"
        t_rollout = time.time()

        # 2. forward/backward on prompt + sampled continuation; only the continuation carries loss
        tokens = prompt_tokens + sampled
        weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(sampled)
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights},
        )
        fb_future = await training.forward_backward_async([datum], loss_fn="cross_entropy")
        fb = await fb_future
        loss = fb.metrics["loss:sum"] / len(sampled)
        assert math.isfinite(loss), f"step {step}: non-finite loss {loss}"
        t_fb = time.time()

        # 3. optimizer step
        optim_future = await training.optim_step_async(types.AdamParams(learning_rate=args.lr))
        await optim_future
        t_optim = time.time()

        # 4. publish the new weights; the next rollout must come from this version
        new_path, sampler = await publish(training, service, f"step{step}")
        assert new_path != path, f"step {step}: publish returned the previous version {path}"
        path = new_path
        t_publish = time.time()

        print(
            f"[e2e] step {step}: rollout {len(sampled)} tok {t_rollout - t_step:.1f}s | "
            f"fwd/bwd loss/tok {loss:.3f} {t_fb - t_rollout:.1f}s | optim {t_optim - t_fb:.1f}s | "
            f"published {path} {t_publish - t_optim:.1f}s | sample: {tokenizer.decode(sampled)!r}",
            flush=True,
        )

    # the final version must serve too: one rollout from the last published adapter
    response = await sampler.sample_async(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0),
    )
    final = list(response.sequences[0].tokens)
    assert final, "final rollout returned no tokens"
    print(f"[e2e] PASS: {args.steps} dependent steps, final version {path} serves ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:10639")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=16)
    asyncio.run(main(parser.parse_args()))
