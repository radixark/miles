"""Real-DistributedOptimizer roundtrip for the native LoRA dist checkpoint.

Requires 2 GPUs. Saves under TP=2 with a real MCore DistributedOptimizer
(bf16 params, fp32 mains, ``fully_reshardable`` param state), reloads under
TP=1, then takes one more identical-gradient Adam step on both sides: if
adapter weights, fp32 main params, or Adam moments were resharded
incorrectly, the post-step values diverge.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.ci.ci_register import register_cuda_ci
from tests.fast.miles_plugins.lora.test_checkpointing import _build_chunk, _fill_chunk

from miles_plugins.lora import checkpointing

register_cuda_ci(
    est_time=120,
    suite="stage-b-2-gpu-h200",
    labels=["lora-native"],
)

_LR = 0.05
_ITERATION = 3


def _init_parallel(rank: int, world_size: int, tmp_path_str: str, phase: str) -> None:
    from megatron.core import parallel_state

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{tmp_path_str}/{phase}_pg",
        rank=rank,
        world_size=world_size,
    )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)


def _build_training_state(tp_rank: int, tp_size: int):
    """A CUDA bf16 chunk with frozen base + trainable adapters, wrapped in MCore DDP + DistributedOptimizer."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.transformer import TransformerConfig

    chunk = _build_chunk(tp_rank=tp_rank, tp_size=tp_size)
    _fill_chunk(chunk, tp_rank=tp_rank, tp_size=tp_size)
    chunk = chunk.to(device="cuda", dtype=torch.bfloat16)
    chunk.decoder.base_linear.weight.requires_grad = False

    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )
    model = DistributedDataParallel(config=transformer_config, ddp_config=ddp_config, module=chunk)
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=_LR,
        min_lr=_LR,
        weight_decay=0.0,
        clip_grad=0.0,
        bf16=True,
        use_distributed_optimizer=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
    )
    optimizer = get_megatron_optimizer(optimizer_config, [model])
    return model, optimizer


def _adapter_parameters(model) -> dict[str, torch.nn.Parameter]:
    module = model.module
    return {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and "adapter" in name
    }


def _step_with_constant_grad(model, optimizer, grad_value: float) -> None:
    """One optimizer step with an identical elementwise gradient on every adapter param.

    Elementwise Adam with identical per-element grads is layout-independent, so
    TP=2 and TP=1 runs must produce bitwise-identical parameters.
    """
    for parameter in model.module.parameters():
        if parameter.requires_grad:
            parameter.main_grad.fill_(grad_value)
    model.start_grad_sync()
    model.finish_grad_sync()
    update_successful, _, _ = optimizer.step()
    assert update_successful


def _gather_full_parameters(model, tp_size: int) -> dict[str, torch.Tensor]:
    """TP-gather every adapter parameter to its full (layout-independent) tensor."""
    from megatron.core import parallel_state

    full: dict[str, torch.Tensor] = {}
    group = parallel_state.get_tensor_model_parallel_group() if tp_size > 1 else None
    for name, parameter in _adapter_parameters(model).items():
        data = parameter.data
        if tp_size == 1 or not getattr(parameter, "tensor_model_parallel", False):
            full[name] = data.detach().cpu()
            continue
        shards = [torch.empty_like(data) for _ in range(tp_size)]
        dist.all_gather(shards, data.contiguous(), group=group)
        full[name] = torch.cat(shards, dim=parameter.partition_dim).detach().cpu()
    return full


def _save_worker(rank: int, world_size: int, tmp_path_str: str) -> None:
    _init_parallel(rank, world_size, tmp_path_str, phase="save")
    try:
        model, optimizer = _build_training_state(tp_rank=rank, tp_size=world_size)

        _step_with_constant_grad(model, optimizer, grad_value=0.5)
        reference_after_save = _gather_full_parameters(model, tp_size=world_size)

        checkpointing.save_native_adapter_dist_checkpoint(
            [model],
            f"{tmp_path_str}/adapter/{checkpointing.NATIVE_DIST_CKPT_DIRNAME}",
            optimizer=optimizer,
            iteration=_ITERATION,
        )

        _step_with_constant_grad(model, optimizer, grad_value=0.25)
        reference_after_extra_step = _gather_full_parameters(model, tp_size=world_size)

        if rank == 0:
            torch.save(
                {"after_save": reference_after_save, "after_extra_step": reference_after_extra_step},
                f"{tmp_path_str}/reference.pt",
            )
        dist.barrier()
    finally:
        _destroy_parallel()


def _load_worker(rank: int, world_size: int, tmp_path_str: str) -> None:
    _init_parallel(rank, world_size, tmp_path_str, phase="load")
    try:
        model, optimizer = _build_training_state(tp_rank=rank, tp_size=world_size)
        reference = torch.load(f"{tmp_path_str}/reference.pt", weights_only=True)

        iteration = checkpointing.load_native_adapter_dist_checkpoint(
            [model],
            f"{tmp_path_str}/adapter/{checkpointing.NATIVE_DIST_CKPT_DIRNAME}",
            optimizer=optimizer,
        )
        assert iteration == _ITERATION, f"expected iteration {_ITERATION}, got {iteration}"

        loaded = _gather_full_parameters(model, tp_size=world_size)
        for name, expected in reference["after_save"].items():
            torch.testing.assert_close(loaded[name], expected, rtol=0, atol=0, msg=f"weights diverged: {name}")

        # One more identical step: wrong fp32 mains or Adam moments diverge here.
        _step_with_constant_grad(model, optimizer, grad_value=0.25)
        stepped = _gather_full_parameters(model, tp_size=world_size)
        for name, expected in reference["after_extra_step"].items():
            torch.testing.assert_close(
                stepped[name], expected, rtol=0, atol=0, msg=f"optimizer state diverged after resharded resume: {name}"
            )
    finally:
        _destroy_parallel()


def _destroy_parallel() -> None:
    from megatron.core import parallel_state

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_dist_optimizer_roundtrip_tp2_save_tp1_load(tmp_path):
    mp.spawn(_save_worker, args=(2, str(tmp_path)), nprocs=2, join=True)
    assert checkpointing.is_native_adapter_dist_checkpoint(
        tmp_path / "adapter" / checkpointing.NATIVE_DIST_CKPT_DIRNAME
    )
    mp.spawn(_load_worker, args=(1, str(tmp_path)), nprocs=1, join=True)
