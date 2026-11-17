"""Routed native slots on a real TP2/EP2/ETP1 MoE, with recomputation and export.

    torchrun --nproc_per_node=2 tests/fast-gpu/test_lora_native_experts.py
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci
from tests.fast.backends.megatron_utils.lora.native.naming_stub import FixedPrefixNaming
from torch.utils.checkpoint import checkpoint

from miles.backends.megatron_utils.lora.native.checkpoint import NativeAdapterSlots
from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named
from miles.backends.megatron_utils.lora.native.modules.linear import (
    assert_every_adapter_slotted,
    iter_multi_lora_adapters,
    iter_slot_adapters,
)
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.backends.megatron_utils.lora.native.spec.moe import HFGroupedExpertsSpec
from miles.backends.megatron_utils.lora.utils import freeze_router_bias
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.lora.hf_lora_targets import HfLoraTargets

register_cuda_ci(
    est_time=180,
    suite="stage-b-2-gpu-h200",
    num_gpus=2,
    labels=["lora", "megatron"],
    hardware=["hopper", "blackwell"],
)


class _UngatedExpertsSpec(HFGroupedExpertsSpec):
    gate_up_hf_leaves = ("up_proj",)


def _case(gated, shared_outer):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=64,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        num_moe_experts=4,
        moe_ffn_hidden_size=64,
        moe_router_topk=2,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0.01,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=False,
        moe_shared_expert_overlap=False,
        add_bias_linear=False,
        gated_linear_unit=gated,
        bias_activation_fusion=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        perform_initialization=True,
        use_cpu_initialization=False,
    )
    spec = get_moe_module_spec(use_te=True, num_experts=4, moe_grouped_gemm=True)
    mlp = spec(config=config, layer_number=1, pg_collection=ProcessGroupCollection.use_mpu_process_groups()).to(
        torch.bfloat16
    )
    for parameter in mlp.parameters():
        parameter.requires_grad = False
    freeze_router_bias([mlp])
    before_bias = mlp.router.expert_bias.clone()
    mlp.router.local_tokens_per_expert.copy_(torch.arange(4, device="cuda"))
    _update_router_expert_bias([mlp], config, None)
    torch.testing.assert_close(mlp.router.expert_bias, before_bias, rtol=0, atol=0)
    context = AttachContext(
        lora=LoRAConfig(
            rank=32, alpha=32, dropout=0, target_modules=None, num_slots=3, experts_shared_outer_loras=shared_outer
        ),
        transformer_config=config,
        tp_size=2,
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
        naming=FixedPrefixNaming("model.layers.0.mlp.experts."),
    )
    expert_spec = HFGroupedExpertsSpec() if gated else _UngatedExpertsSpec()
    assert expert_spec.attach(mlp, None, context) == 2
    assert_every_adapter_slotted([mlp])
    wrappers = list(iter_multi_lora_adapters([mlp]))
    counts = (4, 4, 4)
    for wrapper in wrappers:
        wrapper.token_counts_by_slot = counts
        for slot, rank in enumerate((8, 16, 32)):
            wrapper.init_slot(slot, rank, 32)
            with torch.no_grad():
                for name, parameter in wrapper.adapters[slot].named_parameters():
                    if name.endswith("_B"):
                        parameter.normal_(std=0.05)
                        parameter[..., rank:].zero_()
    local_rank = context.tp_rank
    generator = torch.Generator(device="cuda").manual_seed(123 + local_rank)
    x = torch.randn(6, 1, 64, device="cuda", dtype=torch.bfloat16, generator=generator, requires_grad=True)
    output = checkpoint(lambda inp: mlp(inp)[0], x, use_reentrant=False)
    output.float().square().sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()
    for wrapper in wrappers:
        assert sum(sum(row) for row in wrapper.routing.counts) == len(wrapper.routing.permutation)
        for parameter in wrapper.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    with torch.no_grad():
        before = mlp(x)[0].clone()
        for wrapper in wrappers:
            for name, parameter in wrapper.adapters[0].named_parameters():
                if name.endswith("_B"):
                    parameter.add_(0.05)
        after = mlp(x)[0]
    # Slot 0 occupies global rows [0,4); sequence parallel rank 1 holds rows [6,12).
    if local_rank == 1:
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    else:
        assert not torch.equal(after[:4], before[:4])
        torch.testing.assert_close(after[4:], before[4:], rtol=0, atol=0)
    for slot in range(3):
        tensors = export_lora_hf_named(iter_slot_adapters([mlp], slot))
        for name, value in tensors:
            assert value.shape[0] in (1, 4), (name, value.shape)
            if not shared_outer:
                assert value.shape[0] == 4, (name, value.shape)
            gathered = [torch.empty_like(value) for _ in range(2)]
            dist.all_gather(gathered, value)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    targets = HfLoraTargets(
        attention=(),
        mlp=tuple(
            f"model.layers.*.mlp.experts.{leaf}"
            for leaf in (("gate_up_proj", "down_proj") if gated else ("*.up_proj", "*.down_proj"))
        ),
        unembed=("lm_head",),
    )
    slots = NativeAdapterSlots([mlp], None, targets, multi_lora=True)
    for slot, rank in enumerate((8, 16, 32)):
        specs = slots.get_local_shard_specs(slot)
        pieces = slots.local_pieces(slot, slots.slot_parameters(slot))
        assert len(specs) == len(pieces) > 0
        for spec, piece in zip(specs, pieces, strict=True):
            assert tuple(piece.shape) == spec.local_shape
            rank_axis = -2 if spec.key.endswith(".lora_A") else -1
            assert spec.global_shape[rank_axis] == rank
    dist.barrier()


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    init_gloo_group()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2, expert_model_parallel_size=2, expert_tensor_parallel_size=1
    )
    model_parallel_cuda_manual_seed(1234)
    for gated in (True, False):
        for shared_outer in (False, True):
            _case(gated, shared_outer)
    if dist.get_rank() == 0:
        print(
            "Native expert slots passed: gated/ungated, independent/shared factors, mixed ranks/scales, TP2/EP2, recompute, export"
        )
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
