"""Sampler weights load back into a fresh slot exactly: a trained tenant's save_weights_for_sampler export,
loaded weights-only into another tenant's slot and exported again, gives the same bytes, and the slot's next
optimizer step starts from the loaded weights (its FP32 masters were rebuilt from them).

Qwen3-30B-A3B under TP2/EP4 with adapters on attention, the experts and lm_head covers the export layouts the
load inverts: TP shards, the fused q/k/v and gate/up adapters, per-expert adapters across EP ranks, and the
vocab-split lm_head. Rank 8 in a rank-32 slot pool covers the rank padding.
"""

import asyncio
import os
from pathlib import Path

import safetensors.torch
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import running_gateway

import tinker
from miles.utils.external_utils import command_utils
from tinker import types

register_cuda_ci(est_time=480, suite="stage-c-8-gpu-h100", labels=["multi-lora"], hardware=["hopper"], num_gpus=8)

MODEL_NAME = "Qwen3-30B-A3B"
BASE_MODEL = f"Qwen/{MODEL_NAME}"
# model ids are unique per gateway, so earlier runs under the same root never collide
SAVE_DIR = "/tmp/tinker-sampler-roundtrip"
RUN_ID = "roundtrip"
SERVE_CMD = (
    "python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve "
    f"--hf-checkpoint /root/models/{MODEL_NAME} --save-dir {SAVE_DIR} --run-id {RUN_ID} "
    f'--extra-args "--tinker-base-model {BASE_MODEL}"'
)
RANK = 8
STEPS = 2
LEARNING_RATE = 1e-3
ADAM = types.AdamParams(learning_rate=LEARNING_RATE)
# One Adam step moves an element by about the learning rate. A slot whose gradients never reach its optimizer
# does not move; one stepping from its fresh FP32 masters lands on its random lora_A init, far further away.
MAX_STEP_CHANGE = 3 * LEARNING_RATE


def _adapter_dir(tinker_path: str) -> Path:
    model_id, kind, name = tinker_path.removeprefix("tinker://").split("/")
    return Path(SAVE_DIR) / RUN_ID / model_id / kind / name


async def _train(training, datum: types.Datum, steps: int) -> None:
    for _ in range(steps):
        backward = await training.forward_backward_async([datum], loss_fn="cross_entropy")
        step = await training.optim_step_async(ADAM)
        await backward
        await step


async def _save(training, name: str) -> str:
    return (await (await training.save_weights_for_sampler_async(name=name))).path


async def _roundtrip(base_url: str) -> tuple[str, str, str]:
    # tml- prefix required by the SDK; the gateway reads the key as the tenant id
    service = tinker.ServiceClient(base_url=base_url, api_key="tml-sampler-roundtrip")
    source = await service.create_lora_training_client_async(base_model=BASE_MODEL, rank=RANK)
    tokens = source.get_tokenizer().encode("The secret word is cobalt-lantern-73.")
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": [1.0] * (len(tokens) - 1)},
    )
    await _train(source, datum, STEPS)
    exported = await _save(source, "exported")

    reloaded = await service.create_lora_training_client_async(base_model=BASE_MODEL, rank=RANK)
    await (await reloaded.load_state_async(exported))
    reexported = await _save(reloaded, "reexported")
    await _train(reloaded, datum, 1)
    return exported, reexported, await _save(reloaded, "stepped")


def _load(tinker_path: str) -> dict:
    return safetensors.torch.load_file(_adapter_dir(tinker_path) / "adapter_model.safetensors")


def _check(exported: str, reexported: str, stepped: str) -> None:
    exported_dir, reexported_dir = _adapter_dir(exported), _adapter_dir(reexported)
    tensors = _load(exported)
    for layout in (".self_attn.q_proj.", ".mlp.experts.", "lm_head."):
        assert any(layout in name for name in tensors), f"the exported adapter has no {layout} tensors"
    assert any(
        "lora_B" in name and tensor.any() for name, tensor in tensors.items()
    ), "training left every lora_B at zero; the round trip would not exercise trained weights"
    for file in ("adapter_model.safetensors", "adapter_config.json"):
        assert (exported_dir / file).read_bytes() == (
            reexported_dir / file
        ).read_bytes(), f"{file} changed across load_state and re-export"
    step_change = max(
        (tensor.float() - tensors[name].float()).abs().max().item()
        for name, tensor in _load(stepped).items()
        if "lora_A" in name
    )
    assert 0 < step_change < MAX_STEP_CHANGE, f"one step after the load moved lora_A by {step_change:.3e}"
    print(f"round trip of {len(tensors)} adapter tensors is byte-identical; next step moves lora_A {step_change:.3e}")


def execute() -> None:
    # The gateway and the Tinker client talk over loopback.
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    with running_gateway(SERVE_CMD) as base_url:
        _check(*asyncio.run(_roundtrip(base_url)))


if __name__ == "__main__":
    command_utils.default_config().create_backend().exec_command_cpu(
        "python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py prepare"
    )
    execute()
