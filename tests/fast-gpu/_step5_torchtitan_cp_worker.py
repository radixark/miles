"""Run a small random-weight torchtitan Step5 with and without context parallelism on one CP rank."""

import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.distributed.context_parallel import cp_shard

from miles.backends.torchtitan_utils.models.step5 import build_model_config
from miles.backends.torchtitan_utils.models.step5.layers import Step5RMSNorm
from miles.backends.torchtitan_utils.models.step5.packed_sequence import ContextParallelLayout

VOCAB = 512
# the first document passes position 4103, where the indexer starts scoring regions; 6144 splits into 2 * cp chunks
DOCUMENTS = [4500, 900, 333, 411]
TEXT_CONFIG = dict(
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=3,
    vocab_size=VOCAB,
    rms_norm_eps=1e-5,
    max_position_embeddings=1 << 20,
    num_attention_heads=64,
    num_attention_groups=4,
    swa_num_attention_heads=64,
    head_dim=192,
    sliding_window=512,
    layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    rope_theta=[1e4, 1e7, 1e4],
    partial_rotary_factors=[1.0, 1 / 3, 1.0],
    rope_scaling={
        "factor": 1.0,
        "original_max_position_embeddings": 1 << 20,
        "low_freq_factor": 1.0,
        "high_freq_factor": 32.0,
    },
    zero_centered=True,
    fp32_residual_connection=True,
    use_head_wise_attn_gate=True,
    moe_layer_list=[1, 2],
    moe_num_experts=8,
    moe_top_k=2,
    moe_intermediate_size=256,
    share_expert_dim=256,
    moe_router_scaling_factor=3.0,
    use_moe_router_bias=True,
    norm_expert_weight=True,
    swiglu_limits=[7.0] * 3,
    swiglu_limits_shared=[7.0] * 3,
    sparse_config={
        "enabled": True,
        "proxy_dim": 256,
        "sparse_indexer_rope_dim": 32,
        "sparse_indexer_num_heads": 16,
        "num_provider_groups": 4,
        "topk": 512,
        "region_block_size": 8,
        "apply_to_layer_types": ["full_attention"],
    },
)


def _model():
    torch.manual_seed(0)
    model = build_model_config(SimpleNamespace(text_config=SimpleNamespace(**TEXT_CONFIG))).build().cuda()
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.normal_(std=0.3 if "norm" in name else 0.05)
        for layer in model.layers.values():
            if layer.moe_enabled:
                # Bias steps far above the score spread pin each token's experts: the two layouts' attention
                # differs by BF16 rounding, and a random router flips near-tied choices on that noise.
                layer.moe.expert_bias_E.copy_(torch.arange(layer.moe.expert_bias_E.numel(), dtype=torch.float32))
    # what FSDP's mixed precision runs: bf16 compute weights, fp32 norm weights
    for module in model.modules():
        if not isinstance(module, Step5RMSNorm):
            for name, param in module.named_parameters(recurse=False):
                setattr(module, name, torch.nn.Parameter(param.detach().bfloat16(), requires_grad=param.requires_grad))
    return model


def _inputs():
    generator = torch.Generator(device="cuda").manual_seed(1)
    length = sum(DOCUMENTS)
    tokens = torch.randint(VOCAB, (1, length), device="cuda", generator=generator)
    positions = torch.cat([torch.arange(n, device="cuda") for n in DOCUMENTS]).unsqueeze(0)
    weights = torch.randn(1, length, VOCAB, device="cuda", generator=generator)
    return tokens, positions, weights


def _step(model, tokens, positions, weights) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    logits = model(tokens, positions)
    (logits.float() * weights).sum().backward()
    grads = {name: param.grad.float() for name, param in model.named_parameters() if param.grad is not None}
    return logits.detach().float(), grads


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))

    model = _model()
    tokens, positions, weights = _inputs()
    reference_logits, reference_grads = _step(model, tokens, positions, weights)

    model.enable_context_parallel(mesh, load_balancer="headtail")
    local_inputs, _ = cp_shard(mesh, (tokens, positions, weights), None, "headtail")
    logits, grads = _step(model, *local_inputs)
    for grad in grads.values():
        dist.all_reduce(grad)
    rows = ContextParallelLayout.build(mesh, load_balancer="headtail", seq_len=tokens.shape[1], device="cuda")

    errors = {"logits": _rel_err(logits[0], reference_logits[0, rows.local_indices])}
    errors |= {f"d_{name}": _rel_err(grads[name], reference_grads[name]) for name in reference_grads}
    assert grads.keys() == reference_grads.keys()
    if dist.get_rank() == 0:
        with open(args.output, "w") as f:
            json.dump(errors, f, indent=2)
        print("PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
