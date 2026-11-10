"""A tenant's checkpoint saved on one backend loads into the other: native LoRA and torchtitan gateways take turns on
one checkpoint root, each loading what the other saved.

An import saved back untouched equals its source's adapter_state bit for bit; its forward logprobs match the
source's within the two backends' kernel noise, and it trains and samples on.
"""

import asyncio
import os
import tempfile
from pathlib import Path

import torch
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import QWEN3_8B_BASE_MODEL, prepare_gateway, qwen3_8b_serve_cmd, running_gateway
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

import tinker
from tinker import types

register_cuda_ci(
    est_time=1200,
    suite="stage-c-8-gpu-h100",
    labels=["lora", "multi-lora", "torchtitan"],
    hardware=["hopper"],
    num_gpus=8,
)

SAVE_DIR, RUN_ID = "/root/tinker-import", "import"
MARKER, PROMPT = "cobalt-lantern-73", "The secret word is"
# a wrong layout moves logprobs by whole nats; the backends' bf16 kernels differ by a few hundredths on average
MAX_MEAN_LOGPROB_DIFF = 0.25


def _datums(tokenizer, texts: list[str]) -> list[types.Datum]:
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


async def _tenant(base_url: str, source: str | None) -> dict:
    """Train and save a fresh adapter, or load ``source`` and save it back untouched; then forward the eval datums."""
    service = tinker.ServiceClient(base_url=base_url, api_key="tml-import-tenant")
    training = await service.create_lora_training_client_async(base_model=QWEN3_8B_BASE_MODEL, rank=8)
    tokenizer = training.get_tokenizer()
    train = _datums(tokenizer, [f"{PROMPT} {MARKER}.", f"Remember this: the secret word is {MARKER}."])
    evals = _datums(tokenizer, [f"Q: What is the secret word? A: The secret word is {MARKER}.", "Paris is in France."])
    adam = types.AdamParams(learning_rate=1e-3)
    result = {}
    if source is None:
        for _ in range(8):
            await (await training.forward_backward_async(train, loss_fn="cross_entropy"))
            await (await training.optim_step_async(adam))
        result["path"] = (await (await training.save_state_async(name="trained"))).path
    else:
        await (await training.load_state_with_optimizer_async(source))
        result["path"] = (await (await training.save_state_async(name="imported"))).path
    forward = await (await training.forward_async(evals, loss_fn="cross_entropy"))
    result["logprobs"] = [output["logprobs"].tolist() for output in forward.loss_fn_outputs]
    if source is not None:
        await (await training.forward_backward_async(train, loss_fn="cross_entropy"))
        await (await training.optim_step_async(adam))
        sampler = await service.create_sampling_client_async(
            model_path=(await (await training.save_weights_for_sampler_async(name="after"))).path
        )
        response = await sampler.sample_async(
            prompt=types.ModelInput.from_ints(tokenizer.encode(PROMPT)),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=16, temperature=0.0),
        )
        result["completion"] = tokenizer.decode(response.sequences[0].tokens)
    return result


def _adapter_state(tinker_path: str) -> dict[str, torch.Tensor]:
    model_id, kind, name = tinker_path.removeprefix("tinker://").split("/")
    committed = Path(os.path.realpath(Path(SAVE_DIR) / RUN_ID / model_id / kind / name))
    out = Path(tempfile.mkdtemp()) / "adapter_state.pt"
    dcp_to_torch_save(committed / "adapter_state", out)
    return {key: value for key, value in torch.load(out, weights_only=False).items() if torch.is_tensor(value)}


def _assert_imported(source: dict, imported: dict, direction: str) -> None:
    saved, resaved = _adapter_state(source["path"]), _adapter_state(imported["path"])
    # Megatron alone keeps lm_head's rows past the HF vocab (its padded vocab); each backend reads only its own
    shared = saved.keys() & resaved.keys()
    assert len(shared) > 0.99 * max(len(saved), len(resaved)), f"{direction}: keys {saved.keys() ^ resaved.keys()}"
    unequal = sorted(key for key in shared if not torch.equal(saved[key], resaved[key]))
    assert not unequal, f"{direction}: the import saved back differs from its source in {unequal[:5]}"
    diffs = [
        abs(a - b)
        for ours, theirs in zip(imported["logprobs"], source["logprobs"], strict=True)
        for a, b in zip(ours, theirs, strict=True)
    ]
    mean_diff = sum(diffs) / len(diffs)
    print(f"{direction}: mean |logprob diff| {mean_diff:.4f}, max {max(diffs):.4f}; {imported['completion']!r}")
    assert mean_diff < MAX_MEAN_LOGPROB_DIFF, f"{direction}: mean |logprob diff| {mean_diff}"
    assert MARKER in imported["completion"], f"{direction}: {imported['completion']!r}"


def execute():
    root = f"--save-dir {SAVE_DIR} --run-id {RUN_ID}"
    with running_gateway(qwen3_8b_serve_cmd("megatron", root)) as base_url:
        native = asyncio.run(_tenant(base_url, None))
    with running_gateway(qwen3_8b_serve_cmd("torchtitan", root)) as base_url:
        titan_from_native = asyncio.run(_tenant(base_url, native["path"]))
        titan = asyncio.run(_tenant(base_url, None))
    with running_gateway(qwen3_8b_serve_cmd("megatron", root)) as base_url:
        native_from_titan = asyncio.run(_tenant(base_url, titan["path"]))
    _assert_imported(native, titan_from_native, "native -> torchtitan")
    _assert_imported(titan, native_from_titan, "torchtitan -> native")


if __name__ == "__main__":
    prepare_gateway(QWEN3_8B_BASE_MODEL)
    execute()
