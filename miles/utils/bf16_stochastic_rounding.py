"""Stochastic rounding of FP32 master weights into BF16 parameters (arXiv:2611.13545).

After each optimizer step Megatron casts the FP32 master weights into the BF16 parameters with
round-to-nearest, so small RL updates often leave the BF16 weights unchanged ("weight sticking").
With ``OPEN_TRAINING_BF16_STOCHASTIC_ROUNDING_FLAG=1``, Megatron's cast sites call into this module
instead, which rounds every BF16 shard stochastically on the GPU
(``miles.kernels.quant.stochastic_round``); rollout and training then use the rounded weights until
the next step:

- ``DistributedOptimizer._copy_main_params_to_model_params`` (FP32 main shards held by the
  distributed optimizer, with the optimizer on the GPU or on the host) calls
  :func:`stochastic_round_main_param`;
- ``HybridDeviceOptimizer``'s copy-back hooks (``--optimizer-cpu-offload
  --use-precision-aware-optimizer``, masters held by the offloaded optimizer) call
  :func:`stochastic_round_copy_back`. A master on the host is staged on the GPU one shard at a
  time, exactly like the round-to-nearest copy it replaces.

The Philox key is ``torch.cuda.initial_seed()``, the seed TE's stochastic rounding takes from
PyTorch's CUDA generator, with a hash of the optimizer's shard sizes in its high word
(:func:`philox_seed`). Where TE draws a fresh offset from that generator for every call, the offset
here is ``(optimizer step << 32) | shard index``, the shard index being the master's position in its
optimizer's parameter list. The bits are a pure function of (seed, shards, step, shard, element), so
a shard rebuilt from its master (``--rematerialize-param-from-master-weight``) or after a resume gets
the same BF16 weights, and replicas of a shard with identical masters stay identical.
"""

from __future__ import annotations

import logging
import os
import weakref
import zlib
from collections.abc import Callable, Iterator

import torch

from miles.kernels.quant.stochastic_round import stochastic_round_to_bf16

logger = logging.getLogger(__name__)

BF16_STOCHASTIC_ROUNDING_FLAG = "OPEN_TRAINING_BF16_STOCHASTIC_ROUNDING_FLAG"

# Per-optimizer values derived on first use; the optimizer's shards are fixed for its lifetime.
_shard_indices: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_philox_seeds: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def is_bf16_stochastic_rounding_enabled() -> bool:
    return os.getenv(BF16_STOCHASTIC_ROUNDING_FLAG, "0") == "1"


def philox_offset(*, step: int, shard_index: int) -> int:
    """Fresh bits for every optimizer step and shard, and the same bits whenever both repeat."""
    return (step << 32) | shard_index


def _optimizer_params(optimizer: torch.optim.Optimizer) -> Iterator[torch.Tensor]:
    for group in optimizer.param_groups:
        yield from group["params"]


def _shard_index(optimizer: torch.optim.Optimizer) -> dict[torch.Tensor, tuple[int, int]]:
    """(shard index, param group index) of each parameter of ``optimizer``, built once."""
    if (index := _shard_indices.get(optimizer)) is None:
        index, shard = {}, 0
        for group_index, group in enumerate(optimizer.param_groups):
            for param in group["params"]:
                index[param] = (shard, group_index)
                shard += 1
        _shard_indices[optimizer] = index
    return index


def philox_seed(shard_numels: list[int]) -> int:
    """The Philox key for an optimizer's shards: the CUDA generator seed, with a CRC32 of the shard
    sizes in its high word so that the optimizers of one rank (dense, expert), which all number their
    shards from 0, draw different bits. Ranks holding identical shards get identical keys."""
    sizes = zlib.crc32(",".join(map(str, shard_numels)).encode())
    return torch.cuda.initial_seed() ^ (sizes << 32)


def _philox_seed(optimizer: torch.optim.Optimizer) -> int:
    if (seed := _philox_seeds.get(optimizer)) is None:
        seed = philox_seed([param.numel() for param in _optimizer_params(optimizer)])
        _philox_seeds[optimizer] = seed
        logger.info(
            f"BF16 stochastic rounding of {len(_shard_index(optimizer))} {type(optimizer).__name__} shards, "
            f"Philox seed {seed}"
        )
    return seed


def _optimizer_step(optimizer: torch.optim.Optimizer, param: torch.Tensor, group_index: int) -> int:
    """Steps taken for ``param``: per-parameter state (torch, HDO) or the param group (TE FusedAdam)."""
    if (step := optimizer.state.get(param, {}).get("step")) is None:
        step = optimizer.param_groups[group_index].get("step", 0)
    return int(step)


def _write_master(param: torch.Tensor, master: torch.Tensor, philox_key: Callable[[], tuple[int, int]]) -> None:
    """``param`` <- FP32 ``master`` on the current stream: BF16 rounded stochastically, FP32 copied."""
    if master.dtype != torch.float32:
        raise TypeError(f"BF16 stochastic rounding expects FP32 master weights, got {master.dtype}")
    if param.dtype == torch.bfloat16:
        seed, offset = philox_key()
        # A host master is staged like HDO's round-to-nearest copy stages it: one shard, freed after use.
        source = master.detach() if master.is_cuda else master.detach().to(param.device, non_blocking=True)
        stochastic_round_to_bf16(source.view(-1), param.data.view(-1), seed=seed, offset=offset)
    elif param.dtype == torch.float32:
        param.data.copy_(master.data, non_blocking=True)
    else:
        raise TypeError(f"BF16 stochastic rounding does not support {param.dtype} parameters")


def _write_hdo_master(hdo, optimizer: torch.optim.Optimizer, master: torch.Tensor) -> None:
    param = hdo.inner_param_to_orig_param[master]
    if param is master:  # an FP32 parameter its GPU optimizer updated in place
        return

    def philox_key() -> tuple[int, int]:
        shard_index, _ = _shard_index(hdo)[param]
        _, group_index = _shard_index(optimizer)[master]
        step = _optimizer_step(optimizer, master, group_index)
        return _philox_seed(hdo), philox_offset(step=step, shard_index=shard_index)

    _write_master(param, master, philox_key)


@torch.no_grad()
def stochastic_round_copy_back(hdo, optimizer: torch.optim.Optimizer) -> None:
    """Body of both ``HybridDeviceOptimizer`` copy-back step hooks: write ``optimizer``'s masters.

    Same stream protocol as Megatron's round-to-nearest copy: the copies run on HDO's H2D stream after
    the current stream's pending work, and the current stream waits for them.
    """
    stream = hdo._h2d_stream
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for master in _optimizer_params(optimizer):
            _write_hdo_master(hdo, optimizer, master)
    stream.record_event().wait(torch.cuda.current_stream())


@torch.no_grad()
def replay_stochastic_round_copy_back(hdo) -> None:
    """Rebuild every parameter shard of ``hdo`` from its master with the bits of the latest copy-back."""
    for optimizer in hdo.sub_optimizers:
        for master in _optimizer_params(optimizer):
            _write_hdo_master(hdo, optimizer, master)


@torch.no_grad()
def stochastic_round_main_param(dist_opt, main_param: torch.Tensor, model_shard: torch.Tensor) -> None:
    """``DistributedOptimizer``'s main-to-model copy of one shard, rounding BF16 shards stochastically."""
    optimizer = dist_opt.optimizer

    def philox_key() -> tuple[int, int]:
        shard_index, group_index = _shard_index(optimizer)[main_param]
        step = _optimizer_step(optimizer, main_param, group_index)
        return _philox_seed(optimizer), philox_offset(step=step, shard_index=shard_index)

    _write_master(model_shard, main_param, philox_key)


def validate_bf16_stochastic_rounding_optimizer(optimizer) -> None:
    """Fail at startup unless Megatron's hooked cast sites write every BF16 parameter."""
    # Imported here: tests import this module without Megatron.
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    for dist_opt in optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]:
        if not isinstance(dist_opt, DistributedOptimizer):
            raise ValueError(
                f"{BF16_STOCHASTIC_ROUNDING_FLAG}=1 requires --use-distributed-optimizer, got {type(dist_opt).__name__}"
            )
        if dist_opt.is_stub_optimizer:
            continue
        if dist_opt.config.fp16 or not dist_opt.config.bf16:
            raise ValueError(f"{BF16_STOCHASTIC_ROUNDING_FLAG}=1 requires --bf16 parameters")
        if dist_opt.ddp_config.use_megatron_fsdp:
            raise ValueError(f"{BF16_STOCHASTIC_ROUNDING_FLAG}=1 does not support Megatron-FSDP")
        if dist_opt.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 and not isinstance(
            dist_opt.optimizer, HybridDeviceOptimizer
        ):
            raise ValueError(
                f"{BF16_STOCHASTIC_ROUNDING_FLAG}=1 does not support --use-precision-aware-optimizer without "
                "--optimizer-cpu-offload: TE FusedAdam keeps those master weights and casts them itself"
            )
    logger.info(f"{BF16_STOCHASTIC_ROUNDING_FLAG}=1: BF16 parameters are rounded stochastically from FP32 masters")


__all__ = [
    "BF16_STOCHASTIC_ROUNDING_FLAG",
    "is_bf16_stochastic_rounding_enabled",
    "philox_offset",
    "philox_seed",
    "replay_stochastic_round_copy_back",
    "stochastic_round_copy_back",
    "stochastic_round_main_param",
    "validate_bf16_stochastic_rounding_optimizer",
]
