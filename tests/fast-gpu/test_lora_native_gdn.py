"""Native canonical LoRA on the head-sharded Qwen3.5 GDN layer against the HF-layout reference with the
exported adapter merged into its weights, at TP = world size with and without sequence parallelism.

Checks: a zero-B adapter leaves the layer bit-identical; output, input gradient and every adapter gradient
match the merged reference; the export is identical on every rank and loads back into a fresh layer
bit-exactly; the merged-weight view equals the reference's merged weights in the layer's layout.

    torchrun --nproc_per_node=2 tests/fast-gpu/test_lora_native_gdn.py
"""

import copy
import itertools
import os
import sys
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from safetensors.torch import save_file
from tests.ci.ci_register import register_cuda_ci
from tests.fast.backends.megatron_utils.lora.native.naming_stub import FixedPrefixNaming

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named, load_lora_adapter_hf
from miles.backends.megatron_utils.lora.native.merge import merge_lora_into_weights
from miles.backends.megatron_utils.lora.native.modules.linear import iter_adapters
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.backends.megatron_utils.lora.native.spec.gdn import Qwen35GDNSpec
from miles.backends.megatron_utils.lora.utils import reduce_marked_lora_grads
from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import (
    LinearAttnHeads,
    qkv_flat_to_group_major,
    qkv_group_major_to_flat,
)

sys.path.insert(0, os.path.dirname(__file__))
from linear_attn_reference import ReplicatedGDN, build_layer, packed, rel_err, shard  # noqa: E402

register_cuda_ci(est_time=120, suite="stage-b-2-gpu-h200", labels=["lora"], hardware=["hopper", "blackwell"], num_gpus=2)

HIDDEN = 256
HEADS = LinearAttnHeads(num_k_heads=4, num_v_heads=8, head_k_dim=64, head_v_dim=64)
RANK, ALPHA = 8, 16
PREFIX = "model.layers.0.linear_attn."
ALL = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
TARGET_SETS = {"all": ALL, "qkvz": ("in_proj_qkv", "in_proj_z"), "ba+out": ("in_proj_b", "in_proj_a", "out_proj")}
COMPONENTS = {"in_proj_q": "q", "in_proj_k": "k", "in_proj_v": "v", "in_proj_z": "z", "in_proj_b": "b", "in_proj_a": "a"}
# floor of the head-sharded layer against the fla HF reference (bf16: test_linear_attn_head_sharded.py)
HF_TOLERANCE = {torch.bfloat16: 2e-2, torch.float32: 1e-3}
# fp32 against the same layer holding the merged weights. The recurrence's backward amplifies a 4e-7 relative
# perturbation of its decay input into a 1.8e-3 change of dL/d(decay), so in_proj_a's factors reach 2.8e-3; a
# misplaced row is O(1) (adapter_effect)
TOLERANCE = 5e-3


class _Layer:
    layer_number = 1


# reduce_marked_lora_grads caches each model's params under id(model[0]); a freed layer's id reused on only
# some ranks would make them reduce different params, so no layer is freed while the test runs
_LIVE_LAYERS = []


def _attach(layer, config, targets):
    tp_group = parallel_state.get_tensor_model_parallel_group()
    context = AttachContext(
        lora=LoRAConfig(rank=RANK, alpha=ALPHA, dropout=0.0, target_modules=tuple(PREFIX + t for t in targets)),
        transformer_config=config,
        tp_size=tp_group.size(),
        tp_rank=tp_group.rank(),
        naming=FixedPrefixNaming(PREFIX),
    )
    for parameter in layer.parameters():
        parameter.requires_grad = False
    assert Qwen35GDNSpec().attach(layer, _Layer(), context) == ("out_proj" in targets) + any(
        t != "out_proj" for t in targets
    )
    return [adapter for adapter in iter_adapters([layer])]


def _randomize(adapters, tp_group):
    """Seeded full tensors, identical on every rank, sliced to this rank's shard."""
    generator = torch.Generator(device="cuda").manual_seed(1234)
    with torch.no_grad():
        for adapter in adapters:
            for name, parameter in sorted(adapter.named_parameters()):
                full_shape = list(parameter.shape)
                if parameter.tensor_model_parallel:
                    full_shape[parameter.partition_dim] *= tp_group.size()
                # B large enough that a misplaced delta row is far outside the bf16 floor
                std = 0.05 if name.endswith("_B") else 0.2
                full = torch.randn(full_shape, generator=generator, device="cuda") * std
                if parameter.tensor_model_parallel:
                    full = shard(full, parameter.partition_dim, tp_group)
                parameter.copy_(full)


def _full_factors(adapters, tp_group):
    """Every projection's full ``(A, B)`` in the parameters' own dtype; the export is bf16 by contract."""
    factors = {}
    for adapter in adapters:
        for export in adapter.exports():
            a, b = export.a.detach(), export.b.detach()
            if export.layout.value == "row":
                a = _gather_cat(a, 1, tp_group)
            elif export.layout.value == "column":
                b = _gather_cat(b, 0, tp_group)
            factors[export.hf_name.removeprefix(PREFIX)] = (a, b)
    return factors


def _merged_reference(ref, factors):
    """``ref`` with ``scale * B @ A`` of every HF projection added to its HF-layout weight (fp32)."""
    merged = copy.deepcopy(ref)
    scale = ALPHA / RANK
    delta = {name: scale * (b.float() @ a.float()) for name, (a, b) in factors.items()}
    qkv = [delta.pop(component, None) for component in ("in_proj_q", "in_proj_k", "in_proj_v")]
    if any(d is not None for d in qkv):
        delta["in_proj_qkv"] = torch.cat(qkv)
    with torch.no_grad():
        for module_name, d in delta.items():
            weight = getattr(merged, module_name).weight
            weight.copy_((weight.float() + d).to(weight.dtype))
    return merged


def _run(layer, x, cu_seqlens, grad_out):
    x = x.clone().requires_grad_()
    out, _ = layer(x, packed_seq_params=packed(cu_seqlens))
    out.backward(grad_out)
    return out, x.grad


def check(name, targets, config, tp_group):
    dtype = config.params_dtype
    torch.manual_seed(7)
    ref = ReplicatedGDN("qwen3_5", HIDDEN, HEADS, dtype).cuda()
    base = build_layer(ref, config)
    layer = build_layer(ref, config)
    adapters = _attach(layer, config, targets)
    sp, world = config.sequence_parallel, tp_group.size()

    torch.manual_seed(11)
    x_full = torch.randn(512, 1, HIDDEN, device="cuda", dtype=dtype)
    cu_seqlens = torch.tensor([0, 200, 512], device="cuda", dtype=torch.int32)
    x = shard(x_full, 0, tp_group) if sp else x_full
    torch.manual_seed(13)
    grad_full = torch.randn(512, 1, HIDDEN, device="cuda", dtype=dtype)
    grad_out = shard(grad_full, 0, tp_group) if sp else grad_full

    for parameter in base.parameters():
        parameter.requires_grad = False
    with torch.no_grad():
        noop = torch.equal(layer(x, packed_seq_params=packed(cu_seqlens))[0], base(x, packed_seq_params=packed(cu_seqlens))[0])

    _randomize(adapters, tp_group)
    out, dx = _run(layer, x, cu_seqlens, grad_out)
    _LIVE_LAYERS.append(layer)
    reduce_marked_lora_grads([layer])
    exported = dict(export_lora_hf_named(iter_adapters([layer])))
    factors = _full_factors(adapters, tp_group)
    assert {f"{PREFIX}{name}.lora_{ab}.weight" for name in factors for ab in "AB"} == set(exported)
    for name, (a, b) in factors.items():
        assert torch.equal(exported[f"{PREFIX}{name}.lora_A.weight"], a.to(torch.bfloat16))
        assert torch.equal(exported[f"{PREFIX}{name}.lora_B.weight"], b.to(torch.bfloat16))

    merged = _merged_reference(ref, factors)
    full = (lambda t: _gather_cat(t, 0, tp_group)) if sp else (lambda t: t)
    with torch.no_grad():
        out_hf = merged(x_full.transpose(0, 1), cu_seqlens)
        effect = rel_err(ref(x_full.transpose(0, 1), cu_seqlens), out_hf)

    # oracle: the same layer and kernels holding the merged weights, so only the merged weights' bf16 rounding differs
    merged_layer = build_layer(merged, config)
    merged_out, merged_dx = _run(merged_layer, x, cu_seqlens, grad_out)
    core = merged_layer.linear_attn
    qkv_grad = qkv_group_major_to_flat(_gather_cat(core.in_proj_qkv.weight.grad, 0, tp_group), HEADS)
    weight_grads = {
        "in_proj_q": qkv_grad[: HEADS.key_dim],
        "in_proj_k": qkv_grad[HEADS.key_dim : 2 * HEADS.key_dim],
        "in_proj_v": qkv_grad[2 * HEADS.key_dim :],
        "in_proj_z": _gather_cat(core.in_proj_z.weight.grad, 0, tp_group),
        "in_proj_b": _gather_cat(core.in_proj_b.weight.grad, 0, tp_group),
        "in_proj_a": _gather_cat(core.in_proj_a.weight.grad, 0, tp_group),
        "out_proj": _gather_cat(core.out_proj.weight.grad, 1, tp_group),
    }
    scale = ALPHA / RANK
    errors = {
        "noop_bit_identical": 0.0 if noop else 1.0,
        "out_vs_hf": rel_err(out_hf, full(out).transpose(0, 1)),
        "out": rel_err(full(merged_out), full(out)),
        "dx": rel_err(full(merged_dx), full(dx)),
        "adapter_effect": -effect,
    }
    local_grads = {}
    for adapter in adapters:
        for export in adapter.exports():
            local_grads[export.hf_name.removeprefix(PREFIX)] = (export.a, export.b, export.layout)
    for hf_name, (a, b, layout) in local_grads.items():
        a_full, b_full = (factor.float() for factor in factors[hf_name])
        grad_w = weight_grads[hf_name].float()
        a_grad = _gather_cat(a.grad, 1, tp_group) if layout.value == "row" else a.grad
        b_grad = _gather_cat(b.grad, 0, tp_group) if layout.value == "column" else b.grad
        errors[f"{hf_name}.A"] = rel_err(scale * b_full.t() @ grad_w, a_grad)
        errors[f"{hf_name}.B"] = rel_err(scale * grad_w @ a_full.t(), b_grad)

    flat = torch.cat([t.float().reshape(-1) for t in exported.values()])
    errors["export_rank_spread"] = max(rel_err(flat, other) for other in _gather(flat, tp_group))

    fresh = build_layer(ref, config)
    fresh_adapters = _attach(fresh, config, targets)
    with tempfile.TemporaryDirectory() as tmp:
        save_file({k: v.cpu().contiguous() for k, v in exported.items()}, os.path.join(tmp, "adapter_model.safetensors"))
        loaded = load_lora_adapter_hf([fresh], tmp)
    assert loaded == 2 * sum(len(a.projection_specs) for a in fresh_adapters)
    with torch.no_grad():
        reload_equal = torch.equal(
            fresh(x, packed_seq_params=packed(cu_seqlens))[0], layer(x, packed_seq_params=packed(cu_seqlens))[0]
        )
    if dtype == torch.bfloat16:  # the export is bf16, so only bf16 parameters round-trip exactly
        errors["reload_bit_identical"] = 0.0 if reload_equal else 1.0

    merged_view = merge_lora_into_weights([layer], dict(layer.named_parameters()))
    expected = {
        "linear_attn.in_proj_qkv.weight": qkv_flat_to_group_major(merged.in_proj_qkv.weight, HEADS),
        "linear_attn.in_proj_z.weight": merged.in_proj_z.weight,
        "linear_attn.in_proj_b.weight": merged.in_proj_b.weight,
        "linear_attn.in_proj_a.weight": merged.in_proj_a.weight,
    }
    for param_name, full_weight in expected.items():
        errors[f"merge.{param_name.split('.')[1]}"] = rel_err(shard(full_weight, 0, tp_group), merged_view[param_name])
    errors["merge.out_proj"] = rel_err(shard(merged.out_proj.weight, 1, tp_group), merged_view["linear_attn.out_proj.weight"])
    return errors


def _gather(t, group):
    parts = [torch.empty_like(t) for _ in range(group.size())]
    dist.all_gather(parts, t.contiguous(), group=group)
    return parts


def _gather_cat(t, dim, group):
    return torch.cat(_gather(t, group), dim=dim)


def _bad(name, value, dtype):
    if name in ("noop_bit_identical", "reload_bit_identical"):
        return value != 0.0
    if name == "adapter_effect":
        return -value < 0.1  # the adapter must move the output far beyond the tolerance
    if name.startswith("merge.") or name == "export_rank_spread":
        return value > 1e-3
    if name == "out_vs_hf":
        return value > HF_TOLERANCE[dtype]
    # bf16 rounds the rank-r intermediate and the merged weights, which the recurrence amplifies past any
    # bound that would still catch a misplaced row; fp32 separates the two
    return dtype == torch.float32 and value > TOLERANCE


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(1234)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    failures = {}
    for dtype, sp in itertools.product((torch.float32, torch.bfloat16), [False, True] if world > 1 else [False]):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=8,
            params_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            use_cpu_initialization=False,
            tensor_model_parallel_size=world,
            sequence_parallel=sp,
        )
        for name, targets in TARGET_SETS.items():
            label = f"{name} TP={world}{'+SP' if sp else ''} {str(dtype).removeprefix('torch.')}"
            errors = check(name, targets, config, tp_group)
            if rank == 0:
                print(f"[{label}] " + "  ".join(f"{k}={v:.2e}" for k, v in errors.items()))
            bad = {k: v for k, v in errors.items() if _bad(k, v, dtype)}
            if bad:
                failures[label] = bad
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    if rank == 0:
        print(f"TP={world} native canonical GDN LoRA matches the merged HF reference")


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__])
    main()
