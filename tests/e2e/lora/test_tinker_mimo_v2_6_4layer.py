"""MiMo-V2.6 multi-LoRA through the Tinker gateway on the 4-layer partial (source layers 0, 1, 5, 6: global+dense,
SWA+MoE, global+MoE, SWA+MoE), with adapters on q/k/v/o, the dense and expert MLP, and lm_head.

The partial is not a language model (its NLL is ~16), so the checks compare log-probs of fixed token sequences
instead of sampling text. Four tenants train concurrently, at ranks below and at the pool's maximum:

- trainer vs engine: once a tenant's adapter is trained, the gap between the trainer's forward log-probs and the
  engine's prompt log-probs under its sampler snapshot stays at the zero-adapter gap, while the adapter itself moves
  the engine's log-probs far more. A misplaced q/k/v row (V is 128 wide against 192 for Q/K, and global and SWA
  layers differ in KV heads) or a dropped expert, dense or lm_head adapter would widen the gap by the adapter's effect.
- isolation: each tenant's own snapshot fits its own sequence better than any other tenant's snapshot does.
"""

import asyncio
import os
import random
from statistics import fmean

from scripts.run_mimo_v2_6_flash_tinker import prepare
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import running_gateway

import tinker
from tinker import types

register_cuda_ci(
    est_time=500,
    suite="stage-c-8-gpu-h100",
    labels=["multi-lora"],
    hardware=["hopper"],
    nightly=True,
    num_gpus=8,
)

MODEL_NAME = "mimo26-p4-bf16"
# The gateway's default base model is its --hf-checkpoint path.
BASE_MODEL = f"/root/models/{MODEL_NAME}"
# Four slots of maximum rank 32 (the launcher's defaults): one 8-GPU node, TP4/EP4 training and a TP4 engine.
SERVE_CMD = f"python scripts/run_mimo_v2_6_flash_tinker.py serve --model-name {MODEL_NAME}"
RANKS = [8, 16, 8, 32]
SEQUENCE_LENGTH = 64
STEPS = 4
LEARNING_RATE = 1e-3
# The adapter's own effect must dwarf any widening of the trainer/engine gap it causes.
MIN_ADAPTER_EFFECT = 1.0
MAX_GAP_GROWTH_FRACTION = 0.1


def _sequences() -> list[list[int]]:
    rng = random.Random(0)
    return [[rng.randrange(1000, 100000) for _ in range(SEQUENCE_LENGTH)] for _ in RANKS]


def _datum(tokens: list[int]) -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": [1.0] * (len(tokens) - 1)},
    )


async def _trainer_logprobs(training, tokens: list[int]) -> list[float]:
    output = await (await training.forward_async([_datum(tokens)], loss_fn="cross_entropy"))
    return output.loss_fn_outputs[0]["logprobs"].tolist()


async def _engine_logprobs(sampler, tokens: list[int]) -> list[float]:
    # The first prompt token has no log-prob; the rest align with the trainer's targets.
    return (await sampler.compute_logprobs_async(types.ModelInput.from_ints(tokens)))[1:]


async def _sampler(service, training, name: str):
    path = (await (await training.save_weights_for_sampler_async(name=name))).path
    return await service.create_sampling_client_async(model_path=path)


def _mean_abs_diff(left: list[float], right: list[float]) -> float:
    return fmean(abs(a - b) for a, b in zip(left, right, strict=True))


async def _run_tenant(base_url: str, index: int, sequences: list[list[int]]) -> dict:
    # tml- prefix required by the SDK; the gateway reads the key as the tenant id
    service = tinker.ServiceClient(base_url=base_url, api_key=f"tml-mimo-tenant-{index}")
    training = await service.create_lora_training_client_async(base_model=BASE_MODEL, rank=RANKS[index])
    tokens = sequences[index]

    zero_sampler = await _sampler(service, training, "zero")
    trainer_zero = await _trainer_logprobs(training, tokens)
    engine_zero = await _engine_logprobs(zero_sampler, tokens)

    for _ in range(STEPS):
        backward = await training.forward_backward_async([_datum(tokens)], loss_fn="cross_entropy")
        step = await training.optim_step_async(types.AdamParams(learning_rate=LEARNING_RATE))
        await backward
        await step

    trained_sampler = await _sampler(service, training, "trained")
    trainer_trained = await _trainer_logprobs(training, tokens)
    engine_trained = [await _engine_logprobs(trained_sampler, sequence) for sequence in sequences]
    return {
        "zero_gap": _mean_abs_diff(trainer_zero, engine_zero),
        "trained_gap": _mean_abs_diff(trainer_trained, engine_trained[index]),
        "effect": _mean_abs_diff(engine_trained[index], engine_zero),
        "trainer_gain": fmean(trainer_trained) - fmean(trainer_zero),
        # mean log-prob of every tenant's sequence under this tenant's trained snapshot
        "fit": [fmean(logprobs) for logprobs in engine_trained],
    }


async def _run_tenants(base_url: str) -> list[dict]:
    sequences = _sequences()
    return await asyncio.gather(*(_run_tenant(base_url, index, sequences) for index in range(len(RANKS))))


def _check(results: list[dict]) -> None:
    for index, result in enumerate(results):
        print(
            f"[tenant {index} rank {RANKS[index]}] gap {result['zero_gap']:.4f} -> {result['trained_gap']:.4f}, "
            f"effect {result['effect']:.4f}, trainer gain {result['trainer_gain']:.4f}, "
            f"fit {[round(fit, 4) for fit in result['fit']]}"
        )
    for index, result in enumerate(results):
        assert result["trainer_gain"] > 0, f"tenant {index}: training did not raise its own log-probs"
        assert result["effect"] > MIN_ADAPTER_EFFECT, f"tenant {index}: the engine barely applies the adapter"
        growth = result["trained_gap"] - result["zero_gap"]
        assert growth < MAX_GAP_GROWTH_FRACTION * result["effect"], (
            f"tenant {index}: trainer/engine gap grew by {growth:.4f} with the adapter "
            f"(effect {result['effect']:.4f}); the served adapter differs from the trained one"
        )
        own_fit = result["fit"][index]
        others = [other["fit"][index] for other_index, other in enumerate(results) if other_index != index]
        assert own_fit > max(others), f"tenant {index}: another tenant's adapter fits its sequence better"


if __name__ == "__main__":
    prepare(model_name=MODEL_NAME)
    # The gateway and the Tinker clients talk over loopback.
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    with running_gateway(SERVE_CMD) as base_url:
        _check(asyncio.run(_run_tenants(base_url)))
