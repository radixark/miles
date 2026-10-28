"""``fused_lm_head_log_probs_and_entropy`` must equal ``fused_log_probs_and_entropy`` on ``hidden @ weight.T``,
alone, over NCCL vocab shards and through the losses, without holding more than one tile of logits."""

import sys

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=2
)

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.fast.backends.training_utils.loss.loss_test_utils import (
    make_args,
    make_batch,
    make_inputs,
    make_parallel_state,
)
from tests.fast.dist_utils import find_free_port

from miles.backends.training_utils.loss.hub.fused_log_probs import (
    fused_lm_head_log_probs_and_entropy,
    fused_log_probs_and_entropy,
)
from miles.backends.training_utils.loss.objective import loss_function

_WORLD_SIZE = 2


def _inputs(n_rows, hidden_size, vocab, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(n_rows, hidden_size, device="cuda", generator=g).to(dtype)
    weight = (torch.randn(vocab, hidden_size, device="cuda", generator=g) * hidden_size**-0.5 * 3).to(dtype)
    rows = torch.cat([torch.arange(3, n_rows // 2), torch.arange(n_rows // 2 + 5, n_rows)]).cuda()
    targets = torch.randint(0, vocab, (rows.numel(),), device="cuda", generator=g)
    upstream = torch.randn(2, rows.numel(), device="cuda", generator=g)
    return hidden, weight, rows, targets, upstream


def _run(op, hidden, weight, rows, targets, upstream, *, entropy_requires_grad=True, **kwargs):
    """Log-probs, entropy and the (hidden, weight) gradients of ``g . log_probs + c . entropy``."""
    h = hidden.clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    log_probs, entropy = op(h, w, rows, targets, entropy_requires_grad=entropy_requires_grad, **kwargs)
    g, c = upstream
    loss = (log_probs * g).sum() + ((entropy * c).sum() if entropy is not None and entropy_requires_grad else 0)
    loss.backward()
    return (
        log_probs.detach(),
        entropy.detach() if entropy is not None else None,
        h.grad.float(),
        w.grad.float(),
    )


def _lm_head(h, w, rows, targets, **kwargs):
    """The op with a plain matmul standing for the output layer."""
    return fused_lm_head_log_probs_and_entropy(h, lambda rows_hidden: rows_hidden @ w.t(), rows, targets, **kwargs)


def _materialized(h, w, rows, targets, **kwargs):
    kwargs.pop("tile_rows", None)
    return fused_log_probs_and_entropy(h @ w.t(), rows, targets, inplace_backward=False, **kwargs)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("entropy_requires_grad", [True, False])
@pytest.mark.parametrize("tile_rows", [64, 4096], ids=["uneven_tiles", "one_tile"])
def test_matches_the_fused_op_on_materialized_logits(dtype, temperature, entropy_requires_grad, tile_rows):
    inputs = _inputs(300, 512, 50_001, dtype)
    kwargs = dict(
        tp_group=None,
        temperature=temperature,
        with_entropy=True,
        entropy_requires_grad=entropy_requires_grad,
        tile_rows=tile_rows,
    )
    ref = _run(_materialized, *inputs, **kwargs)
    out = _run(_lm_head, *inputs, **kwargs)

    torch.testing.assert_close(out[0], ref[0], rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(out[1], ref[1], rtol=1e-5, atol=5e-5)
    # both round dlogits to the logits' dtype; the weight gradient here also sums tiles in fp32
    grad_tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out[2], ref[2], **grad_tol)
    torch.testing.assert_close(out[3], ref[3], **grad_tol)


def test_unscored_rows_get_no_hidden_gradient():
    hidden, weight, rows, targets, upstream = _inputs(300, 256, 10_000, torch.bfloat16)
    _, _, grad_hidden, _ = _run(
        _lm_head,
        hidden,
        weight,
        rows,
        targets,
        upstream,
        entropy_requires_grad=False,
        tp_group=None,
        tile_rows=50,
    )
    unscored = torch.ones(hidden.size(0), dtype=torch.bool, device="cuda")
    unscored[rows] = False
    assert (grad_hidden[unscored] == 0).all()
    assert (grad_hidden[rows] != 0).any(dim=1).all()


def test_no_rows_still_gives_zero_gradients():
    hidden, weight, _, _, _ = _inputs(64, 128, 1_000, torch.bfloat16)
    empty = torch.empty(0, dtype=torch.long, device="cuda")
    h = hidden.clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    log_probs, entropy = _lm_head(h, w, empty, empty, tp_group=None, with_entropy=True)
    assert log_probs.numel() == 0 and entropy.numel() == 0
    (log_probs.sum() + entropy.sum()).backward()
    assert h.grad is not None and (h.grad == 0).all()
    assert w.grad is not None and (w.grad == 0).all()


def test_frozen_weight_gets_no_gradient():
    hidden, weight, rows, targets, upstream = _inputs(128, 128, 2_000, torch.bfloat16)
    h = hidden.clone().requires_grad_(True)
    log_probs, _ = _lm_head(h, weight, rows, targets, tp_group=None)
    (log_probs * upstream[0]).sum().backward()
    assert h.grad is not None and weight.grad is None


def test_no_grad_log_probs_match_the_training_forward_bitwise():
    hidden, weight, rows, targets, _ = _inputs(300, 512, 50_001, torch.bfloat16)
    with torch.no_grad():
        no_grad = _lm_head(hidden, weight, rows, targets, tp_group=None, tile_rows=64)[0]
    trained = _lm_head(hidden.clone().requires_grad_(True), weight, rows, targets, tp_group=None, tile_rows=64)[0]
    assert torch.equal(no_grad, trained.detach())


def test_peak_memory_is_one_tile_not_the_logits():
    """At 16384 rows x 129280 vocab the logits are 4 GiB of bf16. Beyond the hidden gradient, the extra peak
    over forward and backward is a few tiles plus the weight gradient and one tile's contribution to it
    (with Megatron's output layer that goes straight into main_grad instead)."""
    n_rows, hidden_size, vocab, tile_rows = 16_384, 1_024, 129_280, 512
    hidden, weight, _, _, _ = _inputs(n_rows, hidden_size, vocab, torch.bfloat16)
    rows = torch.arange(n_rows, device="cuda")
    targets = torch.randint(0, vocab, (n_rows,), device="cuda")
    h = hidden.clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    logits_bytes = n_rows * vocab * 2
    tile_bytes = tile_rows * vocab * 2
    weight_buffers_bytes = w.numel() * (2 + 2)
    hidden_grad_bytes = h.numel() * h.element_size()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    log_probs, entropy = _lm_head(h, w, rows, targets, tp_group=None, with_entropy=True, tile_rows=tile_rows)
    (log_probs.sum() + entropy.sum()).backward()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    print(f"extra peak {extra / 2**20:.0f} MiB; logits would be {logits_bytes / 2**20:.0f} MiB", flush=True)
    assert extra < 3 * tile_bytes + weight_buffers_bytes + hidden_grad_bytes
    assert extra < 0.5 * logits_bytes


_LOSS_VOCAB, _LOSS_HIDDEN = 4096, 256


def _to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, list):
        return [_to_cuda(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_cuda(item) for key, item in value.items()}
    return value


@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
@pytest.mark.parametrize("loss_type", ["policy_loss", "sft_loss"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_losses_match_the_fused_backend_on_the_projected_logits(nccl_world, qkv_format, loss_type, dtype):
    """Through ``loss_function``: the model output is the hidden states and the output layer comes from the batch."""
    make_parallel_state()
    prompt_lens, response_lens = [5, 9, 3], [7, 4, 6]
    results = {}
    for backend in ("fused", "fused_lm_head"):
        args = make_args(
            loss_type=loss_type,
            true_on_policy_mode=False,
            log_probs_backend=backend,
            qkv_format=qkv_format,
            log_probs_chunk_size=8,  # several tiles per micro-batch
            rollout_temperature=0.8,
            entropy_coef=0.01,
        )
        inputs = _to_cuda(make_inputs(7, len(prompt_lens), prompt_lens, response_lens, _LOSS_VOCAB, args))
        gen = torch.Generator(device="cuda").manual_seed(3)
        hidden_shape = (*inputs["policy_logits"].shape[:-1], _LOSS_HIDDEN)
        hidden = torch.randn(hidden_shape, device="cuda", generator=gen).to(dtype).requires_grad_(True)
        weight = torch.randn(_LOSS_VOCAB, _LOSS_HIDDEN, device="cuda", generator=gen) * _LOSS_HIDDEN**-0.5 * 4
        weight = weight.to(dtype).requires_grad_(True)
        batch = make_batch(inputs, loss_type)
        if backend == "fused":
            model_output = hidden @ weight.t()
        else:
            model_output = hidden * 1
            batch["lm_head"] = SimpleNamespace(project=lambda rows, w=weight: rows @ w.t())
        loss, _, log = loss_function(args, batch, 1, model_output)
        loss.backward()
        metrics = dict(zip(log["keys"], log["values"][1:].tolist(), strict=True))
        results[backend] = (loss.detach(), metrics, hidden.grad.float(), weight.grad.float())

    (ref_loss, ref_log, ref_hidden_grad, ref_weight_grad) = results["fused"]
    (loss, log, hidden_grad, weight_grad) = results["fused_lm_head"]
    # the reference projects every row in one GEMM, this backend only the scored rows a tile at a time,
    # so in bf16 a logit can round one ulp apart
    tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(loss, ref_loss, **tol)
    for key, value in ref_log.items():
        assert log[key] == pytest.approx(value, rel=tol["rtol"], abs=tol["atol"]), key
    grad_tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(hidden_grad, ref_hidden_grad, **grad_tol)
    torch.testing.assert_close(weight_grad, ref_weight_grad, **grad_tol)


def _tp_worker(rank: int, world_size: int, port: int, padding: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        vocab = 129_280
        vocab_size = vocab - padding
        hidden, weight, rows, _, upstream = _inputs(300, 512, vocab, torch.bfloat16, seed=5)
        targets = torch.randint(
            0, vocab_size, (rows.numel(),), device="cuda", generator=torch.Generator(device="cuda").manual_seed(8)
        )
        kwargs = dict(vocab_size=vocab_size, temperature=0.9, with_entropy=True)
        width = vocab // world_size
        shard = slice(rank * width, (rank + 1) * width)

        ref = _run(_materialized, hidden, weight[shard], rows, targets, upstream, tp_group=dist.group.WORLD, **kwargs)
        out = _run(
            _lm_head,
            hidden,
            weight[shard],
            rows,
            targets,
            upstream,
            tp_group=dist.group.WORLD,
            tile_rows=64,
            **kwargs,
        )
        torch.testing.assert_close(out[0], ref[0], rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(out[1], ref[1], rtol=1e-5, atol=5e-5)
        # each rank's hidden gradient covers its shard; the model's TP region sums them
        for grads in (out, ref):
            dist.all_reduce(grads[2])
        torch.testing.assert_close(out[2], ref[2], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out[3], ref[3], rtol=2e-2, atol=2e-2)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("padding", [0, 1_000, 129_280 // _WORLD_SIZE], ids=["none", "partial", "whole_shard"])
def test_vocab_shards_over_nccl_match_the_fused_op(padding):
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(_tp_worker, args=(_WORLD_SIZE, find_free_port(), padding), nprocs=_WORLD_SIZE, join=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
