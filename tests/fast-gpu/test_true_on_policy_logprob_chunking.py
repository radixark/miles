from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="stage-b-2-gpu-h200", labels=["megatron"])

import gc
import os
import socket
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.parallel import set_parallel_state
from miles.utils.ft_utils.process_group_utils import GroupInfo

_WORLD_SIZE = 2
_CORRECTNESS_ROWS = 5
_CORRECTNESS_CHUNK_SIZE = 2
_LOCAL_PADDED_VOCAB_SIZE = 4
_REAL_VOCAB_SIZE = 7

_MEMORY_ROWS = 8192
_MEMORY_CHUNK_SIZE = 1024
_MEMORY_REAL_VOCAB_SIZE = 151936
_MEMORY_LOCAL_PADDED_VOCAB_SIZE = 76032


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _init_worker(rank: int, world_size: int, port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return torch.device("cuda", rank)


def _correctness_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full_logits = torch.tensor(
        [
            [0.2, -1.1, 2.0, 0.7, -0.3, 1.4, -2.2, 80.0],
            [1.5, 0.1, -0.8, 2.3, -1.7, 0.9, 1.1, 90.0],
            [-0.5, 2.1, 0.4, -1.3, 1.7, -0.2, 0.8, 100.0],
            [2.4, -0.6, 1.2, 0.3, -1.0, 1.8, -2.5, 110.0],
            [-1.4, 0.6, 2.2, -0.1, 1.0, -2.0, 0.5, 120.0],
        ],
        device=device,
        dtype=torch.float64,
    )
    full_sampling_mask = torch.tensor(
        [
            [True, False, False, False, True, False, False, True],
            [False, True, True, False, False, True, False, True],
            [False, False, False, True, False, False, True, True],
            [True, False, False, False, False, True, True, True],
            [False, False, True, False, True, False, True, True],
        ],
        device=device,
        dtype=torch.bool,
    )
    tokens = torch.tensor([4, 1, 6, 5, 2], device=device, dtype=torch.long)
    return full_logits, full_sampling_mask, tokens


def _assert_correctness_case(rank: int, device: torch.device, *, chunk_size: int, objective: str) -> None:
    full_logits, full_sampling_mask, tokens = _correctness_inputs(device)
    shard_start = rank * _LOCAL_PADDED_VOCAB_SIZE
    shard_end = shard_start + _LOCAL_PADDED_VOCAB_SIZE
    local_logits = full_logits[:, shard_start:shard_end].detach().clone().requires_grad_(True)
    local_sampling_mask = full_sampling_mask[:, shard_start:shard_end]

    with_entropy = objective == "entropy"
    actual_log_probs, actual_entropy = calculate_log_probs_and_entropy(
        local_logits,
        tokens,
        dist.group.WORLD,
        with_entropy=with_entropy,
        entropy_requires_grad=True,
        chunk_size=chunk_size,
        true_on_policy=True,
        vocab_size=_REAL_VOCAB_SIZE,
        sampling_mask=local_sampling_mask,
        temperature=1.0,
    )

    reference_logits = full_logits[:, :_REAL_VOCAB_SIZE].detach().clone().requires_grad_(True)
    reference_mask = full_sampling_mask[:, :_REAL_VOCAB_SIZE]
    reference_masked_log_probs = torch.log_softmax(
        reference_logits.masked_fill(~reference_mask, float("-inf")), dim=-1
    )
    reference_log_probs = reference_masked_log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    reference_entropy_log_probs = torch.log_softmax(reference_logits, dim=-1)
    reference_entropy = -(reference_entropy_log_probs.exp() * reference_entropy_log_probs).sum(dim=-1)

    torch.testing.assert_close(actual_log_probs, reference_log_probs, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual_log_probs[-1], reference_log_probs[-1], rtol=1e-10, atol=1e-12)

    weights = torch.tensor([0.5, -1.0, 0.75, 1.25, -0.25], device=device, dtype=torch.float64)
    if objective == "log_prob":
        assert actual_entropy is None
        actual_value = (actual_log_probs * weights).sum()
        reference_value = (reference_log_probs * weights).sum()
    else:
        assert actual_entropy is not None
        torch.testing.assert_close(actual_entropy, reference_entropy, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(actual_entropy[-1], reference_entropy[-1], rtol=1e-10, atol=1e-12)

        finite_masked_log_probs = reference_masked_log_probs.masked_fill(~reference_mask, 0)
        masked_entropy = -(reference_masked_log_probs.exp() * finite_masked_log_probs).sum(dim=-1)
        assert not torch.allclose(reference_entropy, masked_entropy)
        actual_value = (actual_entropy * weights).sum()
        reference_value = (reference_entropy * weights).sum()

    (actual_gradient,) = torch.autograd.grad(actual_value, local_logits)
    (reference_gradient,) = torch.autograd.grad(reference_value, reference_logits)

    expected_local_gradient = torch.zeros_like(local_logits)
    real_shard_end = min(shard_end, _REAL_VOCAB_SIZE)
    if shard_start < real_shard_end:
        expected_local_gradient[:, : real_shard_end - shard_start] = reference_gradient[:, shard_start:real_shard_end]
    torch.testing.assert_close(actual_gradient, expected_local_gradient, rtol=1e-10, atol=1e-12)
    if rank == _WORLD_SIZE - 1:
        torch.testing.assert_close(actual_gradient[:, -1], torch.zeros_like(actual_gradient[:, -1]))


def _correctness_worker(rank: int, world_size: int, port: int) -> None:
    device = _init_worker(rank, world_size, port)
    try:
        for chunk_size in (-1, _CORRECTNESS_CHUNK_SIZE):
            _assert_correctness_case(rank, device, chunk_size=chunk_size, objective="log_prob")
            _assert_correctness_case(rank, device, chunk_size=chunk_size, objective="entropy")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _memory_args(chunk_size: int, vocab_size: int) -> Namespace:
    return Namespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=True,
        bf16=True,
        fp16=False,
        log_probs_chunk_size=chunk_size,
        vocab_size=vocab_size,
        allgather_cp=False,
    )


def _run_memory_forward(
    device: torch.device,
    *,
    rows: int,
    local_padded_vocab_size: int,
    real_vocab_size: int,
    chunk_size: int,
) -> tuple[int, torch.Tensor, dict[str, list[torch.Tensor]]]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    total_length = rows + 1
    logits = torch.zeros(
        (1, total_length, local_padded_vocab_size),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    tokens = torch.arange(total_length, device=device, dtype=torch.long).remainder_(real_vocab_size)
    args = _memory_args(chunk_size, real_vocab_size)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    result = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=[tokens],
        total_lengths=[total_length],
        response_lengths=[rows],
        with_entropy=True,
        entropy_requires_grad=True,
    )
    torch.cuda.synchronize(device)
    peak_delta = torch.cuda.max_memory_allocated(device) - baseline
    return peak_delta, logits, result


def _release_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _warm_up_memory_path(device: torch.device) -> None:
    for chunk_size in (-1, 4):
        _, logits, result = _run_memory_forward(
            device,
            rows=17,
            local_padded_vocab_size=128,
            real_vocab_size=255,
            chunk_size=chunk_size,
        )
        del result
        del logits
        _release_cuda_memory()


def _measure_memory_case(device: torch.device, chunk_size: int) -> int:
    peak_delta, logits, result = _run_memory_forward(
        device,
        rows=_MEMORY_ROWS,
        local_padded_vocab_size=_MEMORY_LOCAL_PADDED_VOCAB_SIZE,
        real_vocab_size=_MEMORY_REAL_VOCAB_SIZE,
        chunk_size=chunk_size,
    )
    log_probs = result["log_probs"][0]
    entropy = result["entropy"][0]
    assert log_probs.shape == (_MEMORY_ROWS,)
    assert entropy.shape == (_MEMORY_ROWS,)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()
    del entropy
    del log_probs
    del result
    del logits
    _release_cuda_memory()
    return peak_delta


def _memory_worker(rank: int, world_size: int, port: int) -> None:
    device = _init_worker(rank, world_size, port)
    try:
        tp = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD)
        cp = GroupInfo(rank=0, size=1, group=None)
        set_parallel_state(SimpleNamespace(tp=tp, cp=cp))

        _warm_up_memory_path(device)
        chunked_peak = _measure_memory_case(device, _MEMORY_CHUNK_SIZE)
        unchunked_peak = _measure_memory_case(device, -1)

        local_peaks = torch.tensor([unchunked_peak, chunked_peak], device=device, dtype=torch.int64)
        gathered_peaks = [torch.empty_like(local_peaks) for _ in range(world_size)]
        dist.all_gather(gathered_peaks, local_peaks)
        peaks_by_rank = [tuple(int(value) for value in peaks.cpu().tolist()) for peaks in gathered_peaks]

        bf16_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
        one_buffer_saving = (_MEMORY_ROWS - _MEMORY_CHUNK_SIZE) * _MEMORY_REAL_VOCAB_SIZE * bf16_bytes
        minimum_saving = one_buffer_saving // 2
        savings = [unchunked - chunked for unchunked, chunked in peaks_by_rank]
        assert min(savings) >= minimum_saving, (
            f"expected at least {minimum_saving} bytes of true-on-policy peak-memory saving on every TP rank; "
            f"peaks={peaks_by_rank}, savings={savings}"
        )
        if rank == 0:
            print(
                "true-on-policy TP2 peak memory: "
                f"peaks_by_rank={peaks_by_rank}, savings={savings}, minimum_saving={minimum_saving}",
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_true_on_policy_chunking_tp2_correctness_and_gradients() -> None:
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(_correctness_worker, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)


def test_true_on_policy_chunking_reduces_8192_row_peak_memory() -> None:
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(_memory_worker, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
