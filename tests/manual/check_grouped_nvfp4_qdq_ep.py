"""Actual MCore MoELayer all-to-all + Miles QDQ + DDP regression.

Run with torchrun --nproc_per_node=4 and --ep 2 or 4, on SM10x GPUs.
Requires a Megatron checkout with the Miles fake-QAT hook plus the native
moe_use_grouped_tensor integration from NVIDIA/Megatron-LM#6000, and a TE
build exposing use_grouped_tensor and single_grouped_weight. No opfuser is used.
The ordinary Miles Megatron fork may still need that integration backported.

This is a small synthetic BF16 MoE correctness regression with a plain SGD
update; it does not measure throughput or exercise the distributed optimizer.
"""

import argparse
import contextlib
import json
import os
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
os.environ["OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG"] = "1"
os.environ["OPEN_TRAINING_INT4_FAKE_QAT_FLAG"] = "0"
os.environ["NVTE_USE_FAST_MATH"] = "0"

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_submodules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig

import miles.utils.fused_nvfp4_qdq as grouped
from miles.utils.fused_nvfp4_qdq import compute_nvfp4_amax, current_nvfp4_qdq_config, fused_nvfp4_qdq

E, H, FFN = 8, 128, 128
original_grouped = grouped.fused_grouped_nvfp4_qdq


def scalar_loop(x, amax, config):
    return torch.stack([fused_nvfp4_qdq(t, compute_nvfp4_amax(t), config) for t in x])


def checked_grouped(x, amax, config):
    output = original_grouped(x, amax, config)
    torch.testing.assert_close(output, scalar_loop(x, amax, config), rtol=0, atol=0)
    return output


@contextlib.contextmanager
def prequantized_te_reference(model):
    """Independent STE oracle: quantize TE leaf storage, run TE, restore full precision.

    No Miles adapter or custom autograd is involved in this reference. TE writes
    directly into its original parameter's main_grad and signals MCore DDP.
    """
    originals = []
    with torch.no_grad():
        for name in ("linear_fc1", "linear_fc2"):
            weight = getattr(model.module.experts, name).weight
            storage = weight.rowwise_data
            originals.append((storage, storage.clone()))
            quantized = scalar_loop(storage.view(weight.shape), None, current_nvfp4_qdq_config())
            storage.copy_(quantized.reshape_as(storage))
    try:
        with patch.dict(os.environ, {"OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG": "0"}):
            yield
    finally:
        with torch.no_grad():
            for storage, original in originals:
                storage.copy_(original)


def make_layer(packed, fused, ep, pg):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=H,
        num_attention_heads=4,
        num_moe_experts=E,
        moe_ffn_hidden_size=FFN,
        use_cpu_initialization=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_router_load_balancing_type="none",
        moe_router_topk=2,
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=packed,
        moe_single_grouped_weight=packed,
        use_transformer_engine_op_fuser=False,
        expert_model_parallel_size=ep,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=False,
        gradient_accumulation_fusion=fused,
    )
    spec = get_gpt_layer_with_transformer_engine_submodules(num_experts=E, moe_grouped_gemm=True).mlp
    layer = MoELayer(config, submodules=get_submodules(spec), pg_collection=pg)
    layer = Float16Module(config, layer).module.cuda()
    layer.set_layer_number(0)
    expected_first = dist.get_rank(pg.ep) * (E // ep)
    assert layer.local_expert_indices == list(range(expected_first, expected_first + E // ep))
    with torch.no_grad():
        layer.router.weight.zero_()
        for expert in range(E):
            layer.router.weight[expert, expert] = 1
        for fc_name, shape in [("linear_fc1", (2 * FFN, H)), ("linear_fc2", (H, FFN))]:
            fc = getattr(layer.experts, fc_name)
            values = []
            for expert in layer.local_expert_indices:
                generator = torch.Generator().manual_seed(90210 + expert * 17 + shape[0])
                values.append((torch.randn(shape, generator=generator) * (0.025 + expert * 0.002)).cuda().bfloat16())
            if packed:
                fc.weight.rowwise_data.view(len(values), *shape).copy_(torch.stack(values))
                assert not fc.weight.allreduce
            else:
                for i, value in enumerate(values):
                    getattr(fc, f"weight{i}").copy_(value)
    return DDP(
        config,
        DistributedDataParallelConfig(overlap_grad_reduce=True, grad_reduce_in_fp32=True),
        layer,
        pg_collection=pg,
    )


def named_values(layer, grad=False):
    values = {"router": layer.router.weight.main_grad if grad else layer.router.weight}
    for name in ("linear_fc1", "linear_fc2"):
        fc = getattr(layer.experts, name)
        if hasattr(fc, "weight"):
            p = fc.weight
            values[name] = (p.main_grad if grad else p.rowwise_data).reshape(p.shape)
        else:
            ps = [getattr(fc, f"weight{i}") for i in range(layer.num_local_experts)]
            values[name] = torch.stack([p.main_grad if grad else p for p in ps])
    return values


def compare(actual, expected, exact):
    a, b = actual.float(), expected.float()
    error = (a - b).abs()
    rms = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    peak = (error.max() / b.abs().max().clamp_min(1e-12)).item()
    if exact:
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        assert rms < 0.015 and peak < 0.02, (rms, peak, error.max().item())
    return {"relative_l2": rms, "relative_peak": peak, "max_abs": error.max().item()}


def update(ddp):
    changed = 0
    with torch.no_grad():
        for p in ddp.module.parameters():
            storage = p.rowwise_data if hasattr(p, "rowwise_data") else p
            before = storage.clone()
            storage.add_(p.main_grad.reshape(storage.shape).to(storage.dtype), alpha=-0.05)
            changed += int(torch.count_nonzero(storage != before))
    return changed


def run_case(mode, fused, ep, pg, rank, world):
    # Legacy discrete QDQ is the non-fused baseline. The fused baseline bypasses
    # Miles STE and runs TE directly on scalar-QDQ values in the original leaves.
    checked = Mock(wraps=checked_grouped)
    rows = []
    actual = make_layer(True, fused, ep, pg)
    reference = make_layer(fused, fused, ep, pg)
    initial_ids = [id(p) for p in actual.module.parameters()]
    for step in range(3 if fused else 1):
        route = "skew_empty" if step == 1 else "balanced"
        actual.zero_grad_buffer()
        reference.zero_grad_buffer()
        metrics = {}
        for microbatch in range(2):
            generator = torch.Generator(device="cuda").manual_seed(4000 + rank * 101 + step * 11 + microbatch)
            data = torch.randn(32 + rank * 3, 1, H, device="cuda", generator=generator).bfloat16() * 0.1
            data[..., :E] = -1
            primary = (
                ((torch.arange(data.shape[0], device="cuda") + rank) % E)
                if route == "balanced"
                else torch.zeros(data.shape[0], device="cuda", dtype=torch.long)
            )
            data[torch.arange(data.shape[0]), 0, primary] = 4
            data[torch.arange(data.shape[0]), 0, (primary + 1) % E] = 2
            x, y = data.clone().requires_grad_(), data.clone().requires_grad_()
            outputs = []
            for model, input_, function in ((actual, x, checked), (reference, y, scalar_loop)):
                sync = model.no_sync() if microbatch == 0 else contextlib.nullcontext()
                oracle = prequantized_te_reference(model) if model is reference and fused else contextlib.nullcontext()
                with sync, oracle, patch.object(grouped, "fused_grouped_nvfp4_qdq", function):
                    out, bias = model(input_)
                    assert bias is None
                    (out.float().square().sum() / 10).backward()
                    outputs.append(out)
            metrics[f"output{microbatch}"] = compare(*outputs, exact=fused)
            metrics[f"input_grad{microbatch}"] = compare(x.grad, y.grad, exact=fused)
        actual.finish_grad_sync()
        reference.finish_grad_sync()
        for name, grad in named_values(actual.module, grad=True).items():
            assert torch.isfinite(grad).all()
            metrics[f"grad_{name}"] = compare(grad, named_values(reference.module, grad=True)[name], exact=fused)
            group = pg.dp if name == "router" else pg.expt_dp
            replicas = [torch.empty_like(grad) for _ in range(dist.get_world_size(group))]
            dist.all_gather(replicas, grad.contiguous(), group=group)
            for replica in replicas:
                torch.testing.assert_close(grad, replica, rtol=0, atol=0)
            if name == "router":
                assert torch.count_nonzero(grad) > 0
            else:
                for local, expert in enumerate(actual.module.local_expert_indices):
                    if route == "skew_empty" and expert >= 2:
                        assert torch.count_nonzero(grad[local]) == 0
                    else:
                        assert torch.count_nonzero(grad[local]) > 0
        if fused:
            for fc in (actual.module.experts.linear_fc1, actual.module.experts.linear_fc2):
                assert fc.weight.grad_added_to_main_grad
        changed = update(actual)
        update(reference)
        for name, value in named_values(actual.module).items():
            metrics[f"updated_{name}"] = compare(value, named_values(reference.module)[name], exact=fused)
        assert [id(p) for p in actual.module.parameters()] == initial_ids
        # Router replicas update on every rank, even when a rank has no expert tokens.
        assert changed > 0
        row = dict(
            rank=rank,
            world=world,
            ep=ep,
            edp=world // ep,
            mode=mode,
            fused_wgrad=fused,
            step=step,
            route=route,
            experts=actual.module.local_expert_indices,
            changed_elements=changed,
            qdq_bitwise_checks=checked.call_count,
            metrics=metrics,
        )
        assert checked.call_count == 4 * (step + 1)
        rows.append(row)
        print("EP_CASE_PASS " + json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep", type=int, choices=(2, 4), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("grouped-qdq-ep-results"))
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4, "Launch this regression with torchrun --nproc_per_node=4"
    parallel_state.initialize_model_parallel(expert_model_parallel_size=args.ep)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    model_parallel_cuda_manual_seed(123)
    rows = []
    try:
        for mode in ("standard", "4over6"):
            os.environ["NVTE_NVFP4_4OVER6"] = "none" if mode == "standard" else "weights"
            os.environ["NVTE_NVFP4_4OVER6_ERR_MODE"] = "MSE"
            os.environ["NVTE_NVFP4_4OVER6_E4M3_USE_256"] = "all"
            os.environ["NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"] = "0"
            for fused in (False, True):
                rows.extend(run_case(mode, fused, args.ep, pg, rank, world))
                dist.barrier()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"ep{args.ep}-rank{rank}.json").write_text(json.dumps(rows, indent=2))
        dist.barrier()
        if rank == 0:
            print(f"EP_ALL_PASS ep={args.ep} edp={world//args.ep} cases_per_rank={len(rows)}", flush=True)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
