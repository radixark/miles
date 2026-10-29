"""``fused_log_probs_and_entropy`` must match log_softmax and the torch backend, alone and over NCCL vocab shards."""

import sys

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=2
)

register_rocm_ci(est_time=120, suite="nightly-stage-c-2-gpu-mi350", labels=["precision"])

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.fast.dist_utils import find_free_port

from miles.backends.training_utils.loss.hub.fused_log_probs import fused_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.math_utils import calculate_log_probs_and_entropy

_WORLD_SIZE = 2


def _reference(logits, rows, targets, temperature, vocab_size=None):
    """log_softmax over the first ``vocab_size`` columns (all of them by default)."""
    log_softmax = torch.log_softmax(logits.index_select(0, rows)[:, :vocab_size].float() / temperature, dim=-1)
    log_probs = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
    return log_probs, -(log_softmax.exp() * log_softmax).sum(dim=-1)


def _inputs(n_rows, vocab, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    logits = (torch.randn(n_rows, vocab, device=device, generator=g) * 3).to(dtype)
    rows = torch.cat([torch.arange(3, n_rows // 2), torch.arange(n_rows // 2 + 5, n_rows)]).to(device)
    targets = torch.randint(0, vocab, (rows.numel(),), device=device, generator=g)
    return logits, rows, targets


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [50_001, 129_280])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("entropy_requires_grad", [True, False])
@pytest.mark.parametrize("inplace_backward", [True, False])
def test_kernels_match_log_softmax(dtype, vocab, temperature, entropy_requires_grad, inplace_backward):
    logits, rows, targets = _inputs(300, vocab, dtype, "cuda")
    gen = torch.Generator(device="cuda").manual_seed(1)
    g = torch.randn(rows.numel(), device="cuda", generator=gen)
    c = torch.randn(rows.numel(), device="cuda", generator=gen)

    ref_leaf = logits.clone().requires_grad_(True)
    ref_log_probs, ref_entropy = _reference(ref_leaf, rows, targets, temperature)
    ((ref_log_probs * g).sum() + (ref_entropy * c).sum() * entropy_requires_grad).backward()

    leaf = logits.clone().requires_grad_(True)
    log_probs, entropy = fused_log_probs_and_entropy(
        leaf * 1,
        rows,
        targets,
        tp_group=None,
        temperature=temperature,
        with_entropy=True,
        entropy_requires_grad=entropy_requires_grad,
        inplace_backward=inplace_backward,
    )
    ((log_probs * g).sum() + ((entropy * c).sum() if entropy_requires_grad else 0)).backward()

    torch.testing.assert_close(log_probs, ref_log_probs.detach(), rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(entropy, ref_entropy.detach(), rtol=1e-5, atol=5e-5)
    grad_tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(leaf.grad.float(), ref_leaf.grad.float(), **grad_tol)
    unscored = torch.ones(logits.size(0), dtype=torch.bool, device="cuda")
    unscored[rows] = False
    assert (leaf.grad[unscored] == 0).all()


def test_forward_only_matches_log_softmax():
    logits, rows, targets = _inputs(257, 129_280, torch.bfloat16, "cuda")
    with torch.no_grad():
        log_probs, entropy = fused_log_probs_and_entropy(
            logits, rows, targets, tp_group=None, temperature=0.6, with_entropy=True
        )
    ref_log_probs, ref_entropy = _reference(logits, rows, targets, 0.6)
    torch.testing.assert_close(log_probs, ref_log_probs, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-5, atol=5e-5)


@pytest.mark.parametrize("inplace_backward", [True, False])
def test_true_vocab_bound_excludes_the_padding_columns(inplace_backward):
    """With ``vocab_size`` below the logits width, the softmax runs over the true vocabulary only and
    the padding columns get a zero gradient, as torch's log_softmax over the trimmed logits does."""
    width, vocab_size = 129_280, 129_280 - 1_000
    logits, rows, _ = _inputs(300, width, torch.bfloat16, "cuda", seed=23)
    gen = torch.Generator(device="cuda").manual_seed(24)
    targets = torch.randint(0, vocab_size, (rows.numel(),), device="cuda", generator=gen)
    g = torch.randn(rows.numel(), device="cuda", generator=gen)
    c = torch.randn(rows.numel(), device="cuda", generator=gen)

    ref_leaf = logits.clone().requires_grad_(True)
    ref_log_probs, ref_entropy = _reference(ref_leaf, rows, targets, 0.8, vocab_size)
    ((ref_log_probs * g).sum() + (ref_entropy * c).sum()).backward()

    leaf = logits.clone().requires_grad_(True)
    log_probs, entropy = fused_log_probs_and_entropy(
        leaf * 1,
        rows,
        targets,
        tp_group=None,
        vocab_size=vocab_size,
        temperature=0.8,
        with_entropy=True,
        inplace_backward=inplace_backward,
    )
    ((log_probs * g).sum() + (entropy * c).sum()).backward()

    torch.testing.assert_close(log_probs, ref_log_probs.detach(), rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(entropy, ref_entropy.detach(), rtol=1e-5, atol=5e-5)
    torch.testing.assert_close(leaf.grad.float(), ref_leaf.grad.float(), rtol=1e-2, atol=1e-3)
    assert (leaf.grad[:, vocab_size:] == 0).all()


@pytest.fixture
def deterministic_algorithms():
    """What --deterministic-mode turns on (miles/backends/megatron_utils/initialize.py)."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("inplace_backward", [True, False])
def test_deterministic_mode_repeats_bit_for_bit(deterministic_algorithms, inplace_backward):
    """Every op of the fused path must be allowed under torch.use_deterministic_algorithms, and two
    runs of the forward and backward must agree bit for bit."""
    logits, rows, targets = _inputs(257, 129_280, torch.bfloat16, "cuda", seed=21)
    gen = torch.Generator(device="cuda").manual_seed(22)
    g = torch.randn(rows.numel(), device="cuda", generator=gen)
    c = torch.randn(rows.numel(), device="cuda", generator=gen)

    def run():
        leaf = logits.clone().requires_grad_(True)
        log_probs, entropy = fused_log_probs_and_entropy(
            leaf * 1,
            rows,
            targets,
            tp_group=None,
            temperature=0.7,
            with_entropy=True,
            inplace_backward=inplace_backward,
        )
        ((log_probs * g).sum() + (entropy * c).sum()).backward()
        return log_probs.detach(), entropy.detach(), leaf.grad

    for first, second in zip(run(), run(), strict=True):
        assert torch.equal(first, second)


@pytest.mark.parametrize("no_grad_with_entropy", [False, True])
def test_no_grad_log_probs_match_the_training_forward_bitwise(no_grad_with_entropy):
    """The stored old log-probs come from a no-grad pass that usually skips the entropy, while the
    training forward may compute it (a different kernel specialization): they must still agree."""
    logits, rows, targets = _inputs(257, 129_280, torch.bfloat16, "cuda", seed=9)
    with torch.no_grad():
        old_log_probs, _ = fused_log_probs_and_entropy(
            logits, rows, targets, tp_group=None, temperature=0.7, with_entropy=no_grad_with_entropy
        )
    log_probs, _ = fused_log_probs_and_entropy(
        logits.clone().requires_grad_(True), rows, targets, tp_group=None, temperature=0.7, with_entropy=True
    )
    assert torch.equal(old_log_probs, log_probs.detach())


def test_confident_rows_are_as_accurate_as_torch():
    """With the target 10 above every other logit, p_y is close to 1 and 1 - p_y cancels. Against a
    float64 reference, the op must be no less accurate than torch's own float32 log_softmax."""
    logits, rows, targets = _inputs(64, 129_280, torch.float32, "cuda", seed=11)
    logits[rows, targets] = logits[rows].max(dim=-1).values + 10

    def log_probs_and_grad(x, dtype):
        leaf = x.to(dtype).clone().requires_grad_(True)
        lp = torch.log_softmax(leaf.index_select(0, rows) / 0.7, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        lp.sum().backward()
        return lp.detach().double(), leaf.grad.double()

    exact_lp, exact_grad = log_probs_and_grad(logits, torch.float64)
    torch_lp, torch_grad = log_probs_and_grad(logits, torch.float32)
    leaf = logits.clone().requires_grad_(True)
    fused_lp, _ = fused_log_probs_and_entropy(leaf * 1, rows, targets, tp_group=None, temperature=0.7)
    fused_lp.sum().backward()

    def error(x, exact):
        return (x.double() - exact).abs().max().item()

    assert error(fused_lp.detach(), exact_lp) <= 2 * error(torch_lp, exact_lp) + 1e-7
    assert error(leaf.grad, exact_grad) <= 2 * error(torch_grad, exact_grad) + 1e-7


def test_rows_beyond_int32_offsets():
    """Rows whose element offset passes 2**31 must still be read and written at the right place."""
    vocab = 129_280
    n_rows = 2**31 // vocab + 8  # 2.15e9 elements, 4.3 GB of bf16
    logits = torch.zeros(n_rows, vocab, device="cuda", dtype=torch.bfloat16)
    rows = torch.arange(n_rows - 4, n_rows, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(13)
    logits[rows] = (torch.randn(rows.numel(), vocab, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    targets = torch.randint(0, vocab, (rows.numel(),), device="cuda", generator=gen)

    ref_leaf = logits[rows].clone().requires_grad_(True)
    ref_log_probs, ref_entropy = _reference(ref_leaf, torch.arange(rows.numel(), device="cuda"), targets, 0.8)
    (ref_log_probs.sum() + ref_entropy.sum()).backward()

    leaf = logits.requires_grad_(True)
    log_probs, entropy = fused_log_probs_and_entropy(
        leaf, rows, targets, tp_group=None, temperature=0.8, with_entropy=True
    )
    (log_probs.sum() + entropy.sum()).backward()

    torch.testing.assert_close(log_probs, ref_log_probs.detach(), rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(entropy, ref_entropy.detach(), rtol=1e-5, atol=5e-5)
    torch.testing.assert_close(leaf.grad[rows].float(), ref_leaf.grad.float(), rtol=1e-2, atol=1e-3)
    assert leaf.grad[: n_rows - rows.numel()].abs().max().item() == 0


def test_inplace_backward_invalidates_the_logits():
    """The Triton kernels write the gradient into the logits; the version bump makes any second
    reader of those logits fail instead of reading the gradient."""
    logits, rows, targets = _inputs(64, 50_001, torch.bfloat16, "cuda")
    leaf = logits.clone().requires_grad_(True)
    log_probs, _ = fused_log_probs_and_entropy(leaf * 1, rows, targets, tp_group=None, inplace_backward=True)
    log_probs.sum().backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        log_probs.sum().backward()


@pytest.mark.parametrize("used", ["log_probs", "entropy"])
def test_a_single_used_output_gives_its_own_gradient(used):
    """A loss that reads only one output gets only that output's gradient."""
    logits, rows, targets = _inputs(64, 50_001, torch.float32, "cuda", seed=17)
    ref_leaf = logits.clone().requires_grad_(True)
    ref_log_probs, ref_entropy = _reference(ref_leaf, rows, targets, 0.7)
    (ref_log_probs if used == "log_probs" else ref_entropy).sum().backward()

    leaf = logits.clone().requires_grad_(True)
    log_probs, entropy = fused_log_probs_and_entropy(
        leaf * 1, rows, targets, tp_group=None, temperature=0.7, with_entropy=True, inplace_backward=True
    )
    (log_probs if used == "log_probs" else entropy).sum().backward()
    torch.testing.assert_close(leaf.grad, ref_leaf.grad, rtol=1e-5, atol=1e-5)


def test_every_row_scored_matches_log_softmax():
    """With every row scored the zeroing kernel is skipped, and the gradient still matches."""
    logits, _, _ = _inputs(64, 50_001, torch.float32, "cuda", seed=18)
    rows = torch.arange(logits.size(0), device="cuda")
    targets = torch.randint(0, logits.size(1), (rows.numel(),), device="cuda")
    ref_leaf = logits.clone().requires_grad_(True)
    _reference(ref_leaf, rows, targets, 1.0)[0].sum().backward()

    leaf = logits.clone().requires_grad_(True)
    log_probs, _ = fused_log_probs_and_entropy(leaf * 1, rows, targets, tp_group=None, inplace_backward=True)
    log_probs.sum().backward()
    torch.testing.assert_close(leaf.grad, ref_leaf.grad, rtol=1e-5, atol=1e-5)


def test_no_rows_still_gives_the_logits_a_gradient():
    """A rank that scores nothing must still backprop through the model, or CP collectives hang."""
    leaf = torch.randn(5, 50_001, device="cuda", requires_grad=True)
    empty = torch.empty(0, dtype=torch.long, device="cuda")
    log_probs, _ = fused_log_probs_and_entropy(leaf * 1, empty, empty, tp_group=None, inplace_backward=True)
    log_probs.sum().backward()
    assert leaf.grad is not None and (leaf.grad == 0).all()


def test_cpu_logits_are_rejected():
    rows = torch.tensor([0])
    with pytest.raises(ValueError, match="CUDA logits"):
        fused_log_probs_and_entropy(torch.randn(2, 8), rows, rows, tp_group=None)


def test_policy_gradient_matches_the_torch_backend(nccl_world):
    """The torch backend rounds its logits gradient to bf16 (Megatron's fused CE), so compare at bf16."""
    logits, rows, targets = _inputs(513, 129_280, torch.bfloat16, "cuda", seed=3)
    gen = torch.Generator(device="cuda").manual_seed(4)
    g = torch.randn(rows.numel(), device="cuda", generator=gen)
    c = torch.randn(rows.numel(), device="cuda", generator=gen)
    grads = {}
    for backend in ("torch", "fused"):
        leaf = logits.clone().requires_grad_(True)
        if backend == "fused":
            lp, ent = fused_log_probs_and_entropy(
                leaf * 1, rows, targets, tp_group=None, temperature=0.8, with_entropy=True, inplace_backward=True
            )
        else:
            lp, ent = calculate_log_probs_and_entropy(
                (leaf * 1).index_select(0, rows), targets, dist.group.WORLD, with_entropy=True, temperature=0.8
            )
            lp = lp.squeeze(-1)
        ((lp * g).sum() + (ent * c).sum()).backward()
        grads[backend] = (lp.detach(), ent.detach(), leaf.grad.float())
    (t_lp, t_ent, t_grad), (f_lp, f_ent, f_grad) = grads["torch"], grads["fused"]
    torch.testing.assert_close(f_lp, t_lp, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(f_ent, t_ent, rtol=1e-5, atol=5e-5)
    # the torch backend rounds the log-prob and the entropy gradient to bf16 separately, then
    # adds them, so it sits up to about two bf16 ulps from the exact gradient
    torch.testing.assert_close(f_grad, t_grad, rtol=2e-2, atol=5e-3)


def test_the_op_holds_no_vocab_sized_buffer(nccl_world):
    """At 8192 rows x 129280 vocab (2 GiB of bf16 logits), the fused op adds under 1% of the logits
    to the peak, while the torch path's fp32 copies and saved softmax add several times the logits."""
    n_rows, vocab = 8192, 129_280
    logits = torch.randn(n_rows, vocab, device="cuda").to(torch.bfloat16)
    rows = torch.arange(n_rows, device="cuda")
    targets = torch.randint(0, vocab, (n_rows,), device="cuda")
    logits_bytes = logits.numel() * logits.element_size()
    extra = {}
    for backend in ("torch", "fused"):
        model_logits = logits.clone().requires_grad_(True)  # stands for the model output
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        if backend == "fused":
            lp, ent = fused_log_probs_and_entropy(
                model_logits, rows, targets, tp_group=None, with_entropy=True, inplace_backward=True
            )
        else:
            lp, ent = calculate_log_probs_and_entropy(model_logits, targets, dist.group.WORLD, with_entropy=True)
        (lp.sum() + ent.sum()).backward(inputs=[model_logits])
        torch.cuda.synchronize()
        # the leaf's own .grad is one logits-sized buffer in both backends
        extra[backend] = torch.cuda.max_memory_allocated() - before - logits_bytes
        del model_logits, lp, ent
    print(f"extra peak beyond logits and their grad: {extra}", flush=True)
    assert extra["fused"] < 0.01 * logits_bytes
    assert extra["torch"] > 2 * logits_bytes


def _tp_worker(rank: int, world_size: int, port: int, padding: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        vocab = 129_280
        vocab_size = vocab - padding
        logits, rows, _ = _inputs(300, vocab, torch.bfloat16, "cuda", seed=5)
        targets = torch.randint(
            0, vocab_size, (rows.numel(),), device="cuda", generator=torch.Generator(device="cuda").manual_seed(8)
        )
        g = torch.randn(rows.numel(), device="cuda", generator=torch.Generator(device="cuda").manual_seed(6))
        c = torch.randn(rows.numel(), device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
        full = logits.clone().requires_grad_(True)
        ref_log_probs, ref_entropy = _reference(full, rows, targets, 0.9, vocab_size)
        ((ref_log_probs * g).sum() + (ref_entropy * c).sum()).backward()

        width = vocab // world_size
        shard = logits[:, rank * width : (rank + 1) * width].clone().requires_grad_(True)
        log_probs, entropy = fused_log_probs_and_entropy(
            shard * 1,
            rows,
            targets,
            tp_group=dist.group.WORLD,
            vocab_size=vocab_size,
            temperature=0.9,
            with_entropy=True,
            inplace_backward=True,
        )
        ((log_probs * g).sum() + (entropy * c).sum()).backward()

        torch.testing.assert_close(log_probs, ref_log_probs.detach(), rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(entropy, ref_entropy.detach(), rtol=1e-5, atol=5e-5)
        torch.testing.assert_close(
            shard.grad.float(), full.grad[:, rank * width : (rank + 1) * width].float(), rtol=1e-2, atol=1e-3
        )
    finally:
        dist.destroy_process_group()


# Vocab padding at the end of the last shard: none, part of the last shard, and the whole last shard.
@pytest.mark.parametrize("padding", [0, 1_000, 129_280 // _WORLD_SIZE], ids=["none", "partial", "whole_shard"])
def test_vocab_shards_over_nccl_combine_like_the_full_vocab(padding):
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(_tp_worker, args=(_WORLD_SIZE, find_free_port(), padding), nprocs=_WORLD_SIZE, join=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
