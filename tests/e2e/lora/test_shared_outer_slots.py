"""Shared expert factors retain gradients and Adam state across independent slot steps."""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.bridge.peft.multi_lora_layers import ExpertSlotRouting, MultiLoRAGroupedExpertLinear, init_adapter_slot
from megatron.bridge.peft.utils import (
    allreduce_expert_parallel_replicated_grads,
    enable_expert_parallel_grad_sync_in_finalize,
)
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear, TERowParallelGroupedLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.test_slot_checkpoint_resume import ADAM, apply_step, assert_same_state, snapshot

from miles.backends.megatron_utils.lora.checkpoint import load_slot, save_slot
from miles.backends.megatron_utils.lora.optimizer import (
    SlotOptimizer,
    accumulate_gradients,
    adapter_slot_parameters,
    step_slot_optimizers,
)
from miles.utils.distributed_utils import init_gloo_group

register_cuda_ci(est_time=120, suite="stage-b-2-gpu-h200", labels=["lora", "multi-lora"], hardware=["hopper"])

logger = logging.getLogger(__name__)


class ExpertModel(MegatronModule):
    def __init__(self, config, canonical):
        super().__init__(config)
        self.layers = torch.nn.ModuleList()
        for index, (linear, inputs, outputs) in enumerate(
            ((TEColumnParallelGroupedLinear, 32, 64), (TERowParallelGroupedLinear, 32, 32))
        ):
            base = linear(
                num_gemms=2,
                input_size=inputs,
                output_size=outputs,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=True,
                is_expert=True,
            )
            base.requires_grad_(False)
            self.layers.append(
                MultiLoRAGroupedExpertLinear(
                    base,
                    n_adapters=3,
                    dim=8,
                    alpha=16,
                    full_name=f"layers.{index}.linear_fc{index + 1}",
                    num_local_experts=2,
                    experts_shared_outer_loras=True,
                    row_init_method="normal",
                    projection_targets={"linear_fc1_gate", "linear_fc1_up"} if canonical and index == 0 else None,
                )
            )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        metadata = {"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}
        return {
            key: value
            for index, layer in enumerate(self.layers)
            for key, value in layer.sharded_state_dict(f"{prefix}layers.{index}.", sharded_offsets, metadata).items()
        }

    def forward(self, inputs):
        # Tokens arrive in expert order with the two active slots interleaved.
        slots = torch.arange(inputs.shape[0], device=inputs.device) % 2
        experts = torch.arange(inputs.shape[0], device=inputs.device) // (inputs.shape[0] // 2)
        keys = slots * 2 + experts
        order = keys.argsort(stable=True)
        counts = torch.bincount(keys, minlength=6)
        routing = ExpertSlotRouting(
            order, order.argsort(), counts.cumsum(0, dtype=torch.int32), counts.view(3, 2).sum(1), inputs.shape[0]
        )
        for index, layer in enumerate(self.layers):
            layer.expert_slot_routing = routing
            inputs, _ = layer(inputs, [inputs.shape[0] // 2] * 2)
            if index == 0:
                gate, up = inputs.chunk(2, dim=-1)
                inputs = F.silu(gate) * up
        return inputs


def forward_backward(model, inputs):
    with accumulate_gradients(model):
        model[0](inputs).float().square().sum().backward()
        finalize_model_grads(model)
        allreduce_expert_parallel_replicated_grads(model)


def run_worker(directory, tp, canonical):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp, expert_model_parallel_size=2, expert_tensor_parallel_size=1
    )
    init_gloo_group()
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_moe_experts=4,
        gated_linear_unit=True,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=False,
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        params_dtype=torch.bfloat16,
        bf16=True,
        gradient_accumulation_fusion=False,
    )
    expert_model = ExpertModel(config, canonical).cuda()
    enable_expert_parallel_grad_sync_in_finalize(expert_model)
    model = [
        DistributedDataParallel(
            config, DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False), expert_model
        )
    ]
    for slot, rank in enumerate((8, 4, 8)):
        init_adapter_slot(model, slot, rank=rank, alpha=rank * 2, seed=100 + slot)
    args = Namespace(optimizer="adam", bf16=True, use_gloo_process_groups=True, lr=ADAM["learning_rate"])
    optimizers = {slot: SlotOptimizer(args, model, slot) for slot in range(3)}
    params = [p for slot in range(3) for p in adapter_slot_parameters(model, slot)]
    for layer in expert_model.layers:
        for slot in layer.adapters:
            adapters = slot.values() if isinstance(slot, torch.nn.ModuleDict) else (slot,)
            for adapter in adapters:
                shared = adapter.linear_out.weight if layer.input_is_parallel else adapter.linear_in.weight
                replicas = [torch.empty_like(shared) for _ in range(2)]
                dist.all_gather(replicas, shared, group=parallel_state.get_expert_model_parallel_group())
                torch.testing.assert_close(replicas[0], replicas[1], atol=0, rtol=0)

    inputs = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    forward_backward(model, inputs)
    once = [p.main_grad.clone() for p in params]
    forward_backward(model, inputs)
    for param, expected in zip(params, once, strict=True):
        torch.testing.assert_close(param.main_grad, expected * 2, atol=0, rtol=0)
    assert all(torch.count_nonzero(p.main_grad) == 0 for p in adapter_slot_parameters(model, 2))

    # The norm counts shared EP replicas once, while retaining every routed expert.
    expected_norm_sq = sum(
        p.main_grad.float().square().sum()
        for p in adapter_slot_parameters(model, 0)
        if not getattr(p, "shared", False)
    )
    expected_norm_sq /= parallel_state.get_expert_data_parallel_world_size()
    dist.all_reduce(expected_norm_sq)
    untouched = snapshot(model, optimizers[1])
    slot1_grads = [p.main_grad.clone() for p in adapter_slot_parameters(model, 1)]
    outcome = step_slot_optimizers(optimizers, {0: ADAM})
    torch.testing.assert_close(
        torch.tensor(outcome[0]["grad_norm"], device="cuda"), expected_norm_sq.sqrt(), rtol=1e-5, atol=1e-7
    )
    assert_same_state(snapshot(model, optimizers[1]), untouched)
    for param, expected in zip(adapter_slot_parameters(model, 1), slot1_grads, strict=True):
        torch.testing.assert_close(param.main_grad, expected, atol=0, rtol=0)
    forward_backward(model, inputs)
    assert all(torch.isfinite(p.main_grad).all() for p in params)

    saved = snapshot(model, optimizers[0])
    save_slot(model, optimizers[0], str(directory))
    load_slot(model, optimizers[2], str(directory), load_optimizer=True)
    assert_same_state(snapshot(model, optimizers[2]), saved)
    apply_step(model, optimizers[0], 7)
    apply_step(model, optimizers[2], 7)
    assert_same_state(snapshot(model, optimizers[2]), snapshot(model, optimizers[0]))
    logger.warning(
        "rank=%s: accumulation, EP replicas, norm, isolation, checkpoint and resumed Adam passed; peak_allocated=%s",
        dist.get_rank(),
        torch.cuda.max_memory_allocated(),
    )
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--canonical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpus", type=int, default=2)
    options = parser.parse_args()
    if options.worker_dir is not None:
        run_worker(options.worker_dir, options.tp, options.canonical)
    else:
        with tempfile.TemporaryDirectory(prefix="shared-outer-", dir=options.checkpoint_root) as directory:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={options.gpus}",
                    __file__,
                    "--worker-dir",
                    directory + "/checkpoint",
                    "--tp",
                    str(options.tp),
                    "--canonical" if options.canonical else "--no-canonical",
                ],
                check=True,
            )
