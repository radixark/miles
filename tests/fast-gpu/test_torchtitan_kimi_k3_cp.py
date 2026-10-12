"""CP2 decoder logits, parameter gradients and optimizer updates against CP1.

Run directly (the CI runner's mode) or with torchrun for additional CP degrees.
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tests.ci.ci_register import register_cuda_ci
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.func import functional_call
from torchtitan.distributed import utils as titan_dist_utils
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.context_parallel import cp_shard
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.trainer import Trainer

from miles.backends.torchtitan_utils.loss import RLLossAdapter
from miles.backends.torchtitan_utils.models.kimi_k3.model import KimiK3Model
from miles.backends.torchtitan_utils.models.kimi_k3.parallelize import parallelize_kimi_k3

register_cuda_ci(
    est_time=600, suite="stage-b-2-gpu-h200", num_gpus=2, labels=["torchtitan"], hardware=["hopper", "blackwell"]
)


def _relative_error(actual, expected):
    return ((actual - expected).float().norm() / expected.float().norm().clamp_min(1e-12)).item()


def _full(tensor):
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _parameters(model):
    for name, parameter in model.named_parameters():
        yield name.replace("_checkpoint_wrapped_module.", ""), parameter


def _check_decoder(mesh, boundaries, *, fsdp=False):
    torch.manual_seed(42)
    config = KimiK3Model.Config(
        dim=128,
        vocab_size=256,
        n_layers=4,
        n_heads=4,
        mla_heads=4,
        num_experts=4,
        top_k=2,
        latent_dim=64,
        dense_intermediate=256,
        moe_intermediate=128,
        q_lora_rank=64,
        kv_lora_rank=64,
    )
    model = config.build()
    model.init_states()
    model.cuda()
    # Keep top-k choices away from ties so this tests CP gradients, not discrete
    # routing changes from BF16 rounding. Router scores still receive gradients.
    for layer in model.layers.values():
        if layer.moe_enabled:
            layer.moe.gate.e_score_correction_bias.copy_(torch.arange(config.num_experts, device="cuda"))
    # Cast through autograd just as FSDP does, preserving FP32 master weights.
    fp32_modules = (".core.", ".q_conv1d.", ".k_conv1d.", ".v_conv1d.", ".o_norm.")
    reference_parameters = {
        name: p.bfloat16() if fsdp and not any(part in name for part in fp32_modules) else p
        for name, p in _parameters(model)
    }
    tokens = torch.arange(1024, device="cuda").remainder(config.vocab_size).reshape(1, -1)
    targets = (tokens * 13 + 7).remainder(config.vocab_size)
    cu = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    # Padding intentionally repeats position zero; explicit boundaries are authoritative.
    positions = torch.cat(
        [torch.arange(b - a, device="cuda") for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)]
    )[None]
    positions[:, -32:] = 0
    initial = {name: p.detach().clone() for name, p in _parameters(model)}
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference = functional_call(model, reference_parameters, (tokens,), {"positions": positions, "cu_seqlens": cu})
    F.cross_entropy(reference.float().flatten(0, 1), targets.flatten()).backward()
    gradients = {name: p.grad.clone() for name, p in _parameters(model) if p.grad is not None}
    optimizer.step()
    updates = {name: p.detach().clone() - initial[name] for name, p in _parameters(model)}
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name, p in _parameters(model):
            p.copy_(initial[name])
    if fsdp:
        dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=mesh.size(),
            tp=1,
            pp=1,
            ep=mesh.size(),
            world_size=mesh.size(),
            spmd_backend="partial_dtensor",
        )
        dims.build_mesh()
        training = Trainer.Config()
        model = parallelize_kimi_k3(
            model,
            parallel_dims=dims,
            training=training.training,
            parallelism=training.parallelism,
            compile_config=training.compile,
            ac_config=FullAC.Config(),
            dump_folder="/tmp",
        )
        mesh = dims.get_mesh("cp")
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.cp_mesh = mesh
    (local_tokens, local_targets, local_positions, indices), _ = cp_shard(
        mesh,
        (tokens, targets, positions, torch.arange(tokens.numel(), device="cuda")[None]),
        None,
        "headtail",
        input_seq_dim=1,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model(local_tokens, positions=local_positions, cu_seqlens=cu)
    expected = reference[:, indices.flatten()]
    logits_error = _relative_error(actual, expected)
    assert logits_error < 0.02, logits_error
    loss_fn = RLLossAdapter(RLLossAdapter.Config())
    loss_fn.set_context_parallel(mesh, "headtail")
    loss_fn.set_gradient_scale(1.0 / mesh.size())
    loss_fn.arm(
        [targets],
        lambda pred, labels: (F.cross_entropy(pred.float().flatten(0, 1), labels.flatten()), {}),
        is_training=True,
    )
    loss, _ = loss_fn(actual, torch.zeros_like(local_targets))
    loss.backward()
    worst = (0.0, "")
    for name, p in _parameters(model):
        if name not in gradients:
            assert p.grad is None or not _full(p.grad).count_nonzero(), name
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if not fsdp:
            dist.all_reduce(p.grad, group=mesh.get_group())
        grad = _full(p.grad)
        error = _relative_error(grad, gradients[name])
        # Near-zero gradients are compared using an absolute floor as well.
        if gradients[name].norm() > 1e-6:
            worst = max(worst, (error, name))
        assert error < 0.02 or (grad - gradients[name]).abs().max() < 2e-5, (name, error)
    optimizer.step()
    squared_error, squared_update = 0.0, 0.0
    for name, p in _parameters(model):
        update = _full(p.detach()) - initial[name]
        delta = update - updates[name]
        # A tiny SGD update can be only a few FP32 ULPs of the master weight.
        rounding = 2 * torch.finfo(torch.float32).eps * initial[name].abs().max()
        assert delta.abs().max() <= 0.02 * updates[name].abs().max() + learning_rate * 2e-5 + rounding, name
        squared_error += delta.float().square().sum().item()
        squared_update += updates[name].float().square().sum().item()
    update_error = (squared_error / squared_update) ** 0.5
    assert update_error < 0.02, update_error
    assert model.layers["0"].self_attn.q_proj.weight.grad.norm() > 0
    print(
        f"CP{mesh.size()} fsdp_ep_ac={fsdp} boundaries={boundaries} logits_rel={logits_error:.6g} "
        f"worst_grad={worst} update_rel={update_error:.6g}",
        flush=True,
    )


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    titan_dist_utils.set_spmd_backend("partial_dtensor")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))
    try:
        for boundaries in ([0, 1024], [0, 127, 128, 521, 992, 1024]):
            _check_decoder(mesh, boundaries)
        _check_decoder(mesh, [0, 127, 128, 521, 992, 1024], fsdp=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--standalone", "--nproc-per-node=2", __file__])
    main()
