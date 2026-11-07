"""A slot checkpoint preserves FP32 Adam state and resumes into another slot exactly."""

import argparse
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, init_adapter_slot
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from torch.utils._pytree import tree_flatten, tree_map

from miles.backends.megatron_utils.lora.bridge_adapter_slots import BridgeAdapterSlots
from miles.backends.megatron_utils.lora.checkpoint import LayerWiseSlotMembers, load_slot
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer, adapter_slot_parameters, step_slot_optimizers
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import save_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.commit import (
    commit_adapter_dir,
    resolve_adapter_dir,
    tenant_digest,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import base_model_fingerprint
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

register_cuda_ci(est_time=120, suite="stage-b-2-gpu-h200", labels=["lora"], hardware=["hopper"], num_gpus=2)
register_rocm_ci(est_time=60, suite="nightly-stage-c-2-gpu-mi350", labels=["lora"])

ADAM = dict(learning_rate=3e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.01, grad_clip_norm=1.0)


class CheckpointModel(MegatronModule):
    def __init__(self, config):
        super().__init__(config)
        self.layers = torch.nn.ModuleList()
        for index, (input_size, output_size) in enumerate(((16, 32), (32, 16))):
            linear = ColumnParallelLinear(
                input_size,
                output_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                gather_output=False,
            )
            linear.requires_grad_(False)
            self.layers.append(MultiLoRALinear(linear, n_adapters=2, dim=4, alpha=8, full_name=f"layers.{index}"))


class CheckpointModelExport:
    """Stands in for Bridge's HF export of CheckpointModel (TP1): layer 0 is an up_proj, layer 1 a down_proj."""

    def __init__(self, model):
        self._layers = [m for chunk in model for m in chunk.modules() if isinstance(m, MultiLoRALinear)]

    def _named(self, adapters, rank: int | None = None):
        for index, (adapter, projection) in enumerate(zip(adapters, ("up_proj", "down_proj"), strict=True)):
            prefix = f"model.layers.{index}.mlp.{projection}"
            yield f"{prefix}.lora_A.weight", adapter.linear_in.weight[:rank]
            yield f"{prefix}.lora_B.weight", adapter.linear_out.weight[:, :rank]

    def iter_current_adapter(self):
        # inside expose_adapter_slot, which swaps each layer's slots for the exposed one
        yield from self._named([layer.adapter for layer in self._layers])

    def materialize_adapter(self, spec, *, materialize: bool = True) -> dict[str, torch.Tensor]:
        if not materialize:
            return {}
        adapters = [layer.adapters[spec.slot] for layer in self._layers]
        return {name: tensor.detach().clone() for name, tensor in self._named(adapters, spec.rank)}


def snapshot(model, optimizer):
    state = {
        "weights": [param for param in adapter_slot_parameters(model, optimizer.slot)],
        "optimizer": optimizer._inner.state_dict(),
    }
    return tree_map(lambda value: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value, state)


def assert_same_state(actual, expected):
    actual_values, actual_spec = tree_flatten(actual)
    expected_values, expected_spec = tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_value, expected_value in zip(actual_values, expected_values, strict=True):
        if isinstance(expected_value, torch.Tensor):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        else:
            assert actual_value == expected_value


def apply_step(model, optimizer, step):
    # Identical post-all-reduce gradients isolate checkpoint/Adam correctness from forward kernels.
    for index, param in enumerate(adapter_slot_parameters(model, optimizer.slot)):
        gradient = torch.arange(param.numel(), device=param.device, dtype=torch.float32).reshape(param.shape)
        param.main_grad.copy_(torch.sin(gradient + index + step) * 0.1)
    outcome = step_slot_optimizers({optimizer.slot: optimizer}, {optimizer.slot: ADAM})
    assert outcome[optimizer.slot]["grad_norm"] > 0


def run_worker(checkpoint_dir):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    assert dist.get_world_size() == 2
    parallel_state.initialize_model_parallel()
    init_gloo_group()
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
    )
    model = [
        DistributedDataParallel(
            config,
            DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False),
            CheckpointModel(config).cuda(),
        )
    ]
    args = Namespace(
        backend=Namespace(optimizer="adam", bf16=True, use_gloo_process_groups=True, lr=ADAM["learning_rate"])
    )
    for slot in (0, 1):
        init_adapter_slot(model, slot, rank=4, alpha=8)
    hf_config = {"model_type": "qwen3", "num_hidden_layers": 2}
    targets = get_hf_lora_targets(hf_config)
    slots = BridgeAdapterSlots(model, CheckpointModelExport(model), targets, multi_lora=True)
    optimizers = {}
    members = LayerWiseSlotMembers(slots, optimizers)
    run = slots.parametrization(
        base_model="toy",
        base_model_fingerprint=base_model_fingerprint(hf_config),
        lora_rank=4,
        lora_alpha=8.0,
        lora_type="lora",
        experts_shared_outer_loras=False,
    )
    source = optimizers[0] = SlotOptimizer(args, model, slot=0)
    for step in (1, 2, 3):
        apply_step(model, source, step)
    saved = snapshot(model, source)
    masters = [
        param
        for child in source._inner.chained_optimizers
        for group in child.fp32_from_float16_groups
        for param in group
    ]
    assert sum(param.numel() for param in masters) == 192, "DP=2 must own half of the slot's 384 parameters"
    assert any(
        not torch.equal(param, param.bfloat16().float()) for param in masters
    ), "warmup must create sub-BF16 precision"
    assert all(param.dtype == torch.float32 for param in masters)
    for child in source._inner.chained_optimizers:
        assert child.optimizer.state
        assert all(group["step"] == 3 for group in child.param_groups)
        for state in child.optimizer.state.values():
            for name in ("exp_avg", "exp_avg_sq"):
                assert state[name].dtype == torch.float32 and torch.count_nonzero(state[name]) > 0
    print(f"rank={dist.get_rank()} warmup complete; owned_masters=192; sub_bf16_precision=True", flush=True)

    written = save_adapter_dir(checkpoint_dir, "ck", members, 0, run, tenant_digest=tenant_digest("tenant"))
    if dist.get_rank() == 0:
        commit_adapter_dir(checkpoint_dir, "ck", written, overwrite=False)
    dist.barrier()
    checkpoint_dir = resolve_adapter_dir(checkpoint_dir, "ck", tenant="tenant").path
    resumed = optimizers[1] = SlotOptimizer(args, model, slot=1)
    load_slot(model, members, resumed, str(checkpoint_dir), run, targets, load_optimizer=True)
    # Adam's hyperparameters are not checkpointed: every step passes them
    resumed.apply_adam_params(ADAM)
    assert_same_state(snapshot(model, resumed), saved)
    assert_same_state(snapshot(model, source), saved)
    print(f"rank={dist.get_rank()} cross-slot restore is exact", flush=True)

    apply_step(model, source, 4)
    uninterrupted = snapshot(model, source)
    apply_step(model, resumed, 4)
    assert_same_state(snapshot(model, resumed), uninterrupted)
    assert_same_state(snapshot(model, source), uninterrupted)
    assert all(group["step"] == 4 for child in resumed._inner.chained_optimizers for group in child.param_groups)
    print(f"rank={dist.get_rank()} resumed Adam step is exact", flush=True)

    load_slot(model, members, resumed, str(checkpoint_dir), run, targets, load_optimizer=True)
    # round the FP32 masters to the bf16 model params
    resumed._inner.reload_model_params()
    with pytest.raises(AssertionError):
        assert_same_state(snapshot(model, resumed), saved)
    print(f"rank={dist.get_rank()} negative control: rounding masters is detected", flush=True)

    resumed = optimizers[1] = SlotOptimizer(args, model, slot=1)
    load_slot(model, members, resumed, str(checkpoint_dir), run, targets, load_optimizer=False)
    apply_step(model, resumed, 4)
    with pytest.raises(AssertionError):
        assert_same_state(snapshot(model, resumed), uninterrupted)
    print(f"rank={dist.get_rank()} negative control: weights-only restart diverges", flush=True)
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--worker-dir", type=Path)
    options = parser.parse_args()
    if options.worker_dir is not None:
        run_worker(options.worker_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="slot-checkpoint-", dir=options.checkpoint_root) as directory:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    __file__,
                    "--worker-dir",
                    directory + "/checkpoint",
                ],
                check=True,
            )
