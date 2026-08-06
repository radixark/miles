"""Numerical verification of native (raw-mode) LoRA against dense reference math.

Builds a small real megatron-core GPTModel, attaches adapters with
``apply_native_lora``, and checks the adapter branch against independently
computed dense math at TP1 / TP2 / TP2+sequence-parallel.

Run it directly (needs as many GPUs as --tp):

  PYTHONPATH=/root/Megatron-LM:. torchrun --nproc-per-node 1 \
      tests/manual/lora/verify_lora_native.py --tp 1
  PYTHONPATH=/root/Megatron-LM:. torchrun --nproc-per-node 2 \
      tests/manual/lora/verify_lora_native.py --tp 2 --sp
  PYTHONPATH=/root/Megatron-LM:. torchrun --nproc-per-node 2 \
      tests/manual/lora/verify_lora_native.py --tp 2 --sp --gate

``--mla`` builds multi-latent attention instead of GQA; ``--gate`` builds GQA with
``attention_output_gate`` (Qwen3.5 / Qwen3-Next), where the fused qkv carries a
second query slice per head.

Exits nonzero if any check fails. Checks, per configuration:

  1. no-op: a fresh adapter (B is zero-init) leaves the output bit-identical
  2. delta: the adapter branch equals scale * B @ (A @ x) computed densely from the
     TP-gathered adapter, for both a column-parallel (fc1) and a row-parallel (fc2)
     module -- the base GEMM is subtracted out, so this tests only our math. The
     fused qkv is checked after mcore's own per-group split, so the row permutation
     (and, when gated, the query/gate deinterleave) is verified where it is consumed
  3. export: TP shards gather to tensors identical on every rank
  4. round-trip: export -> load into a fresh model reproduces params and outputs
  5. grads: dL/dA == 0 while dL/dB != 0 for a fresh adapter (B zero-init), grads are
     nonzero once B is randomized, and replicated-param grads agree across TP after
     reduce_marked_lora_grads combined genuinely distinct per-rank partials
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig

from miles.backends.megatron_utils.lora_utils import reduce_marked_lora_grads
from miles_plugins.lora.distributed import rmsnorm
from miles_plugins.lora.lora import apply_native_lora, export_lora_hf_named, load_lora_adapter_hf
from miles_plugins.lora.modules.linear import NativeLoRAAdapter

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FAILS = []


def note(message):
    if dist.get_rank() == 0:
        print(f"[SKIP] {message}", flush=True)


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
    if dist.get_rank() == 0:
        print(f"[{tag}] {name} {detail}", flush=True)


def build(tp, seq_parallel, seed=1234, mla=False, output_gate=False):
    torch.manual_seed(seed)
    if mla:
        return _build_mla(tp, seq_parallel)
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=8,
        num_query_groups=4,
        ffn_hidden_size=512,
        kv_channels=32,
        use_cpu_initialization=False,
        tensor_model_parallel_size=tp,
        sequence_parallel=seq_parallel,
        bf16=False,
        params_dtype=torch.float32,
        attention_output_gate=output_gate,
        gated_linear_unit=True,
        add_bias_linear=False,
        normalization="RMSNorm",
        pipeline_dtype=torch.float32,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=None, moe_grouped_gemm=False)
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=512,
        max_sequence_length=64,
        pre_process=True,
        post_process=True,
    ).cuda()
    return model, cfg


def _build_mla(tp, seq_parallel):
    """Small DeepSeek/GLM/Kimi-style MLA model: compressed q and kv paths."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec as _spec_fn

    cfg = MLATransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=512,
        use_cpu_initialization=False,
        tensor_model_parallel_size=tp,
        sequence_parallel=seq_parallel,
        bf16=False,
        params_dtype=torch.float32,
        gated_linear_unit=True,
        add_bias_linear=False,
        normalization="RMSNorm",
        pipeline_dtype=torch.float32,
        multi_latent_attention=True,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        rotary_scaling_factor=1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
    )
    spec = _spec_fn(num_experts=None, moe_grouped_gemm=False, multi_latent_attention=True)
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=512,
        max_sequence_length=64,
        pre_process=True,
        post_process=True,
    ).cuda()
    return model, cfg


class Args:
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.0
    lora_A_init_method = "xavier"
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    mla_target_modules = [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_provider_path = None


def fwd(model, tokens, pos, mask):
    return model(input_ids=tokens, position_ids=pos, attention_mask=mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--sp", action="store_true")
    p.add_argument("--mla", action="store_true", help="build multi-latent attention instead of GQA")
    p.add_argument("--gate", action="store_true", help="GQA with attention_output_gate (Qwen3.5 / Qwen3-Next)")
    a = p.parse_args()
    assert not (a.mla and a.gate), "attention_output_gate is a fused-qkv concern; MLA has no fused qkv"

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ps.initialize_model_parallel(tensor_model_parallel_size=a.tp)
    model_parallel_cuda_manual_seed(1234)
    label = f"TP{a.tp}{'+SP' if a.sp else ''}{'+MLA' if a.mla else ''}{'+GATE' if a.gate else ''}"
    if a.mla:
        Args.target_modules = Args.mla_target_modules
    n_mods_per_layer, n_tensors_per_layer = (7, 16) if a.mla else (4, 14)

    torch.manual_seed(7)
    b, s = 2, 16
    tokens = torch.randint(0, 512, (b, s), device="cuda")
    pos = torch.arange(s, device="cuda").unsqueeze(0).expand(b, s)
    mask = torch.ones(b, 1, s, s, dtype=torch.bool, device="cuda").tril().logical_not()

    lora_model, _ = build(a.tp, a.sp, mla=a.mla, output_gate=a.gate)
    lora_model.eval()
    with torch.no_grad():
        out_base = fwd(lora_model, tokens, pos, mask).clone()

    layer0 = lora_model.decoder.layers[0]
    fc1_mod, fc2_mod = layer0.mlp.linear_fc1, layer0.mlp.linear_fc2
    torch.manual_seed(31337)
    x_fc1 = torch.randn(4, 1, fc1_mod.weight.shape[1], device="cuda")
    x_fc2 = torch.randn(4, 1, fc2_mod.weight.shape[1], device="cuda")
    qkv_pre = {}
    if not a.mla:
        qkv_mod = layer0.self_attention.linear_qkv
        qkv_pre["x"] = torch.randn(4, 1, qkv_mod.weight.shape[1], device="cuda")
    mla_pre = {}
    if a.mla:
        attn0 = layer0.self_attention
        kv_up, kv_down = attn0.linear_kv_up_proj, attn0.linear_kv_down_proj
        mla_pre["x_kv"] = torch.randn(4, 1, kv_up.weight.shape[1], device="cuda")
        mla_pre["x_h"] = torch.randn(4, 1, kv_down.weight.shape[1], device="cuda")
    with torch.no_grad():
        y0_fc1 = fc1_mod(x_fc1)[0].clone()
        y0_fc2 = fc2_mod(x_fc2)[0].clone()
        if not a.mla:
            qkv_pre["y0"] = qkv_mod(qkv_pre["x"])[0].clone()
        if a.mla:
            mla_pre["y0_kv_up"] = kv_up(mla_pre["x_kv"])[0].clone()
            mla_pre["y0_kv_down"] = kv_down(mla_pre["x_h"])[0].clone()

    apply_native_lora(lora_model, Args())
    with torch.no_grad():
        out_fresh = fwd(lora_model, tokens, pos, mask)
    check(
        f"{label} fresh adapter is exact no-op",
        torch.equal(out_base, out_fresh),
        f"max|d|={(out_base - out_fresh).abs().max().item():.3e}",
    )

    n_adapters = sum(1 for m in lora_model.modules() if isinstance(m, NativeLoRAAdapter))
    expect_mods = 2 * n_mods_per_layer
    check(f"{label} adapters attached", n_adapters == expect_mods, f"count={n_adapters} (expect {expect_mods})")

    trainable = [n for n, q in lora_model.named_parameters() if q.requires_grad]
    check(
        f"{label} only adapter params trainable",
        all("lora" in n for n in trainable) and len(trainable) > 0,
        f"n_trainable={len(trainable)}",
    )

    torch.manual_seed(99)
    for m in lora_model.modules():
        if isinstance(m, NativeLoRAAdapter):
            for _, prm in m.named_parameters(recurse=False):
                with torch.no_grad():
                    prm.normal_(0, 0.02)

    scale = Args.lora_alpha / Args.lora_rank
    tp_group = ps.get_tensor_model_parallel_group()

    def gather_cat(t, dim):
        parts = [torch.empty_like(t) for _ in range(a.tp)]
        dist.all_gather(parts, t.contiguous(), group=tp_group)
        return torch.cat(parts, dim=dim)

    ad1 = layer0.mlp.lora_fc1_adapter
    with torch.no_grad():
        got1 = fc1_mod(x_fc1)[0] - y0_fc1
        xn = rmsnorm(x_fc1, fc1_mod.layer_norm_weight, lora_model.config.layernorm_epsilon)
        if a.sp:
            xn = gather_cat(xn, 0)
        ref1 = scale * torch.cat(
            [F.linear(F.linear(xn, ad1.gate_A), ad1.gate_B), F.linear(F.linear(xn, ad1.up_A), ad1.up_B)], dim=-1
        )
    e1 = (got1 - ref1).abs().max().item()
    r1 = e1 / max(ref1.abs().max().item(), 1e-9)
    check(f"{label} column-parallel (fc1) delta == dense reference", r1 < 1e-5, f"max|d|={e1:.3e} rel={r1:.2e}")

    ad2 = layer0.mlp.lora_fc2_adapter
    with torch.no_grad():
        got2 = fc2_mod(x_fc2)[0] - y0_fc2
        s_full = F.linear(gather_cat(x_fc2, -1), gather_cat(ad2.down_A, 1))
        if a.sp:
            s_full = s_full.chunk(a.tp, dim=0)[ps.get_tensor_model_parallel_rank()]
        ref2 = scale * F.linear(s_full, ad2.down_B)
    e2 = (got2 - ref2).abs().max().item()
    r2 = e2 / max(ref2.abs().max().item(), 1e-9)
    check(f"{label} row-parallel (fc2) delta == dense reference", r2 < 1e-5, f"max|d|={e2:.3e} rel={r2:.2e}")

    if not a.mla:
        attn0 = layer0.self_attention
        ad_qkv = attn0.lora_qkv_adapter
        num_q = attn0.num_attention_heads_per_partition
        num_kv = attn0.num_query_groups_per_partition
        head_dim = attn0.hidden_size_per_attention_head
        q_per_group = num_q // num_kv
        q_slices = 2 if a.gate else 1
        with torch.no_grad():
            delta = qkv_mod(qkv_pre["x"])[0] - qkv_pre["y0"]
            xn = rmsnorm(qkv_pre["x"], qkv_mod.layer_norm_weight, lora_model.config.layernorm_epsilon)
            if a.sp:
                xn = gather_cat(xn, 0)
            hf_delta = {
                name: scale * F.linear(F.linear(xn, getattr(ad_qkv, f"{name}_A")), getattr(ad_qkv, f"{name}_B"))
                for name in ("q", "k", "v")
            }
            grouped = delta.view(*delta.shape[:-1], num_kv, -1)
            split = [q_per_group * head_dim] * q_slices + [head_dim, head_dim]
            blocks = torch.split(grouped, split, dim=-1)
            key_block, value_block = blocks[-2], blocks[-1]

            def hf_query_slice(slice_idx):
                rows = [
                    hf_delta["q"].narrow(-1, ((head * q_slices) + slice_idx) * head_dim, head_dim)
                    for head in range(num_q)
                ]
                return torch.cat(rows, dim=-1).view(*delta.shape[:-1], num_kv, q_per_group * head_dim)

            worst_qkv = 0.0
            for slice_idx in range(q_slices):
                worst_qkv = max(worst_qkv, (blocks[slice_idx] - hf_query_slice(slice_idx)).abs().max().item())
            for block, name in ((key_block, "k"), (value_block, "v")):
                want = hf_delta[name].view(*delta.shape[:-1], num_kv, head_dim)
                worst_qkv = max(worst_qkv, (block - want).abs().max().item())
        largest = max(t.abs().max().item() for t in hf_delta.values())
        rel_qkv = worst_qkv / max(largest, 1e-9)
        check(
            f"{label} fused qkv delta lands in mcore's per-group slots",
            rel_qkv < 1e-5,
            f"max|d|={worst_qkv:.3e} rel={rel_qkv:.2e} q_slices={q_slices}",
        )

    if a.mla:
        attn0 = layer0.self_attention
        ad_kvb, kv_up = attn0.lora_mla_kv_b_adapter, attn0.linear_kv_up_proj
        with torch.no_grad():
            got_kv = kv_up(mla_pre["x_kv"])[0] - mla_pre["y0_kv_up"]
            xin = gather_cat(mla_pre["x_kv"], 0) if a.sp else mla_pre["x_kv"]
            ref_kv = scale * F.linear(F.linear(xin, ad_kvb.b_A), ad_kvb.b_B)
        e_kv = (got_kv - ref_kv).abs().max().item()
        r_kv = e_kv / max(ref_kv.abs().max().item(), 1e-9)
        check(f"{label} MLA kv_b_proj delta == dense reference", r_kv < 1e-5, f"max|d|={e_kv:.3e} rel={r_kv:.2e}")

        ad_kva, kv_down = attn0.lora_mla_kv_a_adapter, attn0.linear_kv_down_proj
        with torch.no_grad():
            got_a = kv_down(mla_pre["x_h"])[0] - mla_pre["y0_kv_down"]
            ref_a = scale * F.linear(F.linear(mla_pre["x_h"], ad_kva.a_A), ad_kva.a_B)
        e_a = (got_a - ref_a).abs().max().item()
        r_a = e_a / max(ref_a.abs().max().item(), 1e-9)
        check(f"{label} MLA kv_a_proj delta == dense reference", r_a < 1e-5, f"max|d|={e_a:.3e} rel={r_a:.2e}")

    exported = export_lora_hf_named([lora_model])
    expect_tensors = 2 * n_tensors_per_layer
    check(
        f"{label} export covers all adapters",
        len(exported) == expect_tensors,
        f"n={len(exported)} (expect {expect_tensors})",
    )

    flat = torch.cat([t.float().reshape(-1) for _, t in exported])
    gathered = [torch.empty_like(flat) for _ in range(a.tp)]
    dist.all_gather(gathered, flat, group=ps.get_tensor_model_parallel_group())
    same = all(torch.equal(gathered[0], g) for g in gathered)
    check(f"{label} exported adapter identical on every TP rank", same)

    import json
    import tempfile

    from safetensors.torch import save_file

    tmp = tempfile.mkdtemp()
    if dist.get_rank() == 0:
        save_file({n: t.contiguous() for n, t in exported}, os.path.join(tmp, "adapter_model.safetensors"))
        json.dump({"r": Args.lora_rank}, open(os.path.join(tmp, "adapter_config.json"), "w"))
    obj = [tmp]
    dist.broadcast_object_list(obj, src=0)
    tmp = obj[0]
    dist.barrier()

    fresh, _ = build(a.tp, a.sp, mla=a.mla, output_gate=a.gate)
    apply_native_lora(fresh, Args())
    base_state = {k: v for k, v in lora_model.state_dict().items() if "lora" not in k}
    missing, unexpected = fresh.load_state_dict(base_state, strict=False)
    check(
        f"{label} base weights copied for round-trip",
        not unexpected and all("lora" in m for m in missing),
        f"missing={len(missing)} unexpected={len(unexpected)}",
    )
    n_loaded = load_lora_adapter_hf([fresh], tmp)
    check(f"{label} load consumed every adapter tensor", n_loaded == expect_tensors, f"loaded={n_loaded}")

    max_d = 0.0
    for (n1, p1), (n2, p2) in zip(
        sorted((n, q) for n, q in lora_model.named_parameters() if "lora" in n),
        sorted((n, q) for n, q in fresh.named_parameters() if "lora" in n),
        strict=True,
    ):
        assert n1 == n2, (n1, n2)
        max_d = max(max_d, (p1.float() - p2.float()).abs().max().item())
    check(f"{label} export->load round-trip preserves params", max_d < 2e-2, f"max|d|={max_d:.3e}")

    fresh.eval()
    lora_model.eval()
    with torch.no_grad():
        o1 = fwd(lora_model, tokens, pos, mask)
        o2 = fwd(fresh, tokens, pos, mask)
    d = (o1 - o2).abs().max().item()
    check(f"{label} round-tripped model reproduces outputs", d < 5e-2, f"max|d|={d:.3e}")

    fresh2, _ = build(a.tp, a.sp, mla=a.mla, output_gate=a.gate)
    apply_native_lora(fresh2, Args())
    fresh2.train()
    fwd(fresh2, tokens, pos, mask).square().mean().backward()
    a_grads, b_grads = [], []
    for n, q in fresh2.named_parameters():
        if not q.requires_grad or q.grad is None:
            continue
        (a_grads if n.endswith("_A") else b_grads).append(q.grad.abs().max().item())
    check(
        f"{label} fresh adapter: dL/dA == 0 (B is zero-init)",
        a_grads and max(a_grads) == 0.0,
        f"max|dA|={max(a_grads):.3e}",
    )
    check(
        f"{label} fresh adapter: dL/dB != 0",
        b_grads and min(b_grads) > 0.0,
        f"min|dB|={min(b_grads):.3e} max|dB|={max(b_grads):.3e}",
    )

    lora_model.train()
    out = fwd(lora_model, tokens, pos, mask)
    out.square().mean().backward()
    for prm in lora_model.parameters():
        if prm.requires_grad and prm.grad is not None:
            prm.main_grad = prm.grad
    tagged = [
        (n, q)
        for n, q in lora_model.named_parameters()
        if getattr(q, "_lora_grad_sum_group", None) == "tp" and q.grad is not None
    ]
    check(f"{label} tagged replicated params exist", len(tagged) > 0, f"n={len(tagged)}")
    pre = {n: q.main_grad.clone() for n, q in tagged}
    reduce_marked_lora_grads([lora_model])

    nonzero = all(q.main_grad.abs().max().item() > 0 for _, q in tagged)
    check(f"{label} adapter grads are nonzero (randomized B)", nonzero)

    ok, worst = True, 0.0
    for _, q in tagged:
        parts = [torch.empty_like(q.main_grad) for _ in range(a.tp)]
        dist.all_gather(parts, q.main_grad, group=tp_group)
        for pg in parts:
            worst = max(worst, (pg - parts[0]).abs().max().item())
        ok = ok and all(torch.allclose(pg, parts[0], atol=1e-5) for pg in parts)
    check(f"{label} replicated-param grads consistent across TP", ok, f"max spread={worst:.3e}")

    if a.tp > 1:
        changed = any(not torch.allclose(pre[n], q.main_grad) for n, q in tagged)
        check(f"{label} TP sum actually combined distinct partial grads", changed)

    if a.tp > 1 and not a.sp:
        captured: dict[str, torch.Tensor] = {}

        def _capture(_module, _inputs, output):
            output.register_hook(lambda g: captured.setdefault("grad", g.detach().clone()))

        upstream, _ = build(a.tp, a.sp, mla=a.mla, output_gate=a.gate)
        apply_native_lora(upstream, Args())
        for prm in upstream.parameters():
            if prm.requires_grad and prm.dim() == 2:
                torch.nn.init.normal_(prm, std=0.02)
        probe = upstream.embedding.register_forward_hook(_capture)
        upstream.train()
        fwd(upstream, tokens, pos, mask).square().mean().backward()
        probe.remove()

        if "grad" in captured:
            parts = [torch.empty_like(captured["grad"]) for _ in range(a.tp)]
            dist.all_gather(parts, captured["grad"].contiguous(), group=tp_group)
            spread = max((pg - parts[0]).abs().max().item() for pg in parts)
            check(
                f"{label} first-activation grad consistent across TP",
                spread < 1e-5,
                f"max spread={spread:.3e}",
            )
    elif a.tp > 1:
        note(f"{label} first-activation grad: skipped (sequence-parallel shards it per rank)")

    ddp_model, ddp_cfg = build(a.tp, a.sp, mla=a.mla, output_gate=a.gate)
    apply_native_lora(ddp_model, Args())
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import DistributedDataParallelConfig

    wrapped = DDP(
        ddp_cfg,
        DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False),
        ddp_model,
    )
    for prm in wrapped.parameters():
        if prm.requires_grad:
            with torch.no_grad():
                prm.normal_(0, 0.02)
    wrapped.zero_grad_buffer()
    wrapped.train()
    fwd(wrapped, tokens, pos, mask).float().square().mean().backward()

    adapters = [(n, q) for n, q in wrapped.named_parameters() if "lora" in n and q.requires_grad]
    with_buffer = [q for _, q in adapters if getattr(q, "main_grad", None) is not None]
    nonzero_grads = [q for q in with_buffer if q.main_grad.abs().max().item() > 0]
    check(
        f"{label} DDP builds a grad buffer for every adapter param",
        len(adapters) > 0 and len(with_buffer) == len(adapters),
        f"{len(with_buffer)}/{len(adapters)} have main_grad",
    )
    check(
        f"{label} adapter grads reach the DDP buffer",
        len(nonzero_grads) > 0,
        f"{len(nonzero_grads)}/{len(with_buffer)} nonzero",
    )

    dist.barrier()
    if dist.get_rank() == 0:
        print(f"\n=== {label}: {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)} ===", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if FAILS else 0)


main()
