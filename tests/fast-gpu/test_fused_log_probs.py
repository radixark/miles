from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"])

"""``--log-probs-backend fused`` must match log_softmax and the torch backend, over TP shards, CP layouts and losses."""

import dataclasses
import os
import socket
import sys

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

from miles.backends.training_utils.data.context_parallel import all_gather_with_cp
from miles.backends.training_utils.loss.hub.fused_log_probs import fused_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

_WORLD_SIZE = 2
# Megatron's fused CE (the torch backend) rounds its logits gradient to bf16 even for fp32 logits
# (fused_cross_entropy.py, calculate_gradients), so gradients are compared with it at bf16 precision.
_TORCH_BACKEND_GRAD_TOL = dict(rtol=2e-2, atol=5e-3)


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


@pytest.fixture
def nccl_world(tmp_path):
    """A 1-rank NCCL group: the torch backend's fused CE reaches a collective."""
    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


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


def _to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, list):
        return [_to_cuda(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_cuda(item) for key, item in value.items()}
    return value


_LOSS_VOCAB = 4096


@pytest.mark.parametrize("loss_type", ["policy_loss", "sft_loss"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_losses_match_the_torch_backend(nccl_world, loss_type, dtype):
    make_parallel_state()
    prompt_lens, response_lens = [5, 9, 3], [7, 4, 6]
    results = {}
    for backend in ("torch", "fused"):
        args = make_args(
            loss_type=loss_type,
            true_on_policy_mode=False,
            log_probs_backend=backend,
            rollout_temperature=0.8,
            entropy_coef=0.01,
        )
        inputs = _to_cuda(make_inputs(7, len(prompt_lens), prompt_lens, response_lens, _LOSS_VOCAB, args))
        leaf = inputs["policy_logits"].to(dtype).requires_grad_(True)
        loss, _, log = loss_function(args, make_batch(inputs, loss_type), 1, leaf * 1)
        loss.backward()
        results[backend] = (loss.detach(), dict(zip(log["keys"], log["values"][1:].tolist(), strict=True)), leaf.grad)

    (torch_loss, torch_log, torch_grad), (fused_loss, fused_log, fused_grad) = results["torch"], results["fused"]
    torch.testing.assert_close(fused_loss, torch_loss, rtol=1e-5, atol=1e-5)
    for key, value in torch_log.items():
        assert fused_log[key] == pytest.approx(value, rel=1e-4, abs=1e-5), key
    torch.testing.assert_close(fused_grad.float(), torch_grad.float(), **_TORCH_BACKEND_GRAD_TOL)


# Zigzag CP=2 (thd): each sample is padded to 4 chunks and rank r holds chunks r and 3 - r. Rank 1
# holds chunks 1 and 2 of the first sample (tokens 25-74 of 100), which lie wholly in its 90-token
# prompt, so both of its halves score no response row.
_ZIGZAG_PROMPT_LENS, _ZIGZAG_RESPONSE_LENS = [90, 10], [10, 30]


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_zigzag_cp2_matches_the_torch_backend(nccl_world, cp_rank):
    """Zigzag CP needs no collective to score its rows, so one process can stand in for each rank."""
    set_parallel_state(dataclasses.replace(make_parallel_state(), cp=GroupInfo(rank=cp_rank, size=2, group=None)))
    total_lens = [p + r for p, r in zip(_ZIGZAG_PROMPT_LENS, _ZIGZAG_RESPONSE_LENS, strict=True)]
    local_rows = sum(2 * -(-total // 4) for total in total_lens)  # two chunks of ceil(total / 4) per sample
    gen = torch.Generator(device="cuda").manual_seed(cp_rank)
    logits = torch.randn(1, local_rows, _LOSS_VOCAB, device="cuda", generator=gen) * 4
    tokens = [torch.randint(0, _LOSS_VOCAB, (total,), device="cuda", generator=gen) for total in total_lens]

    outputs = {}
    for backend in ("torch", "fused"):
        args = make_args(true_on_policy_mode=False, log_probs_backend=backend, rollout_temperature=0.8)
        leaf = logits.clone().requires_grad_(True)
        res = get_log_probs_and_entropy(
            leaf * 1,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=total_lens,
            response_lengths=_ZIGZAG_RESPONSE_LENS,
            with_entropy=True,
        )
        log_probs, entropy = torch.cat(res["log_probs"]), torch.cat(res["entropy"])
        (log_probs.sum() + 0.1 * entropy.sum()).backward()
        outputs[backend] = ([lp.numel() for lp in res["log_probs"]], log_probs.detach(), entropy.detach(), leaf.grad)

    (torch_sizes, torch_lp, torch_ent, torch_grad), (fused_sizes, fused_lp, fused_ent, fused_grad) = (
        outputs["torch"],
        outputs["fused"],
    )
    assert fused_sizes == torch_sizes
    if cp_rank == 1:
        assert fused_sizes[0] == 0
    torch.testing.assert_close(fused_lp, torch_lp, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(fused_ent, torch_ent, rtol=1e-5, atol=5e-5)
    torch.testing.assert_close(fused_grad, torch_grad, **_TORCH_BACKEND_GRAD_TOL)


@pytest.mark.parametrize("with_entropy", [False, True])
def test_score_centering_log_probs_match_the_torch_backend(nccl_world, with_entropy):
    """Score centering normalizes over the true vocabulary: the torch backend through
    selected_log_probs_and_entropy, the fused one through its vocab bound. They must agree."""
    make_parallel_state()
    prompt_lens, response_lens = [5, 9, 3], [7, 4, 6]
    width, vocab_size = _LOSS_VOCAB, _LOSS_VOCAB - 96
    total_lens = [p + r for p, r in zip(prompt_lens, response_lens, strict=True)]
    gen = torch.Generator(device="cuda").manual_seed(25)
    logits = torch.randn(1, sum(total_lens), width, device="cuda", generator=gen) * 4
    tokens = [torch.randint(0, vocab_size, (total,), device="cuda", generator=gen) for total in total_lens]

    outputs = {}
    for backend in ("torch", "fused"):
        args = make_args(
            loss_type="score_centering",
            true_on_policy_mode=False,
            log_probs_backend=backend,
            rollout_temperature=0.8,
            vocab_size=vocab_size,
        )
        leaf = logits.clone().requires_grad_(True)
        res = get_log_probs_and_entropy(
            leaf * 1,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=total_lens,
            response_lengths=response_lens,
            with_entropy=with_entropy,
        )
        log_probs = torch.cat(res["log_probs"])
        loss = log_probs.sum() + (0.1 * torch.cat(res["entropy"]).sum() if with_entropy else 0)
        loss.backward()
        entropy = torch.cat(res["entropy"]).detach() if with_entropy else None
        outputs[backend] = (log_probs.detach(), entropy, leaf.grad)

    (torch_lp, torch_ent, torch_grad), (fused_lp, fused_ent, fused_grad) = outputs["torch"], outputs["fused"]
    torch.testing.assert_close(fused_lp, torch_lp, rtol=1e-5, atol=2e-5)
    if with_entropy:
        torch.testing.assert_close(fused_ent, torch_ent, rtol=1e-5, atol=5e-5)
    torch.testing.assert_close(fused_grad, torch_grad, rtol=1e-5, atol=1e-5)
    assert (fused_grad[..., vocab_size:] == 0).all()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


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
    mp.spawn(_tp_worker, args=(_WORLD_SIZE, _free_port(), padding), nprocs=_WORLD_SIZE, join=True)


# (prompt_lens, response_lens): totals divide by 2 * cp. In the second case rank 0's contiguous
# half holds no response logits.
_ALLGATHER_CP_CASES = [([16, 2], [8, 6]), ([40], [24])]


def _allgather_cp2_worker(rank: int, world_size: int, port: int, prompt_lens, response_lens) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        tp_group = [dist.new_group([r]) for r in range(world_size)][rank]
        trivial = GroupInfo(rank=0, size=1, group=None)
        cp = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD)
        set_parallel_state(
            ParallelState(
                intra_dp=trivial,
                intra_dp_cp=cp,
                cp=cp,
                tp=GroupInfo(rank=0, size=1, group=tp_group),
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
                is_pp_last_stage=True,
            )
        )
        base = make_args(loss_type="sft_loss", true_on_policy_mode=False, allgather_cp=True, rollout_temperature=0.7)
        inputs = _to_cuda(make_inputs(42, len(prompt_lens), prompt_lens, response_lens, _LOSS_VOCAB, base))
        t_local = inputs["policy_logits"].size(1) // world_size
        local = inputs["policy_logits"][:, rank * t_local : (rank + 1) * t_local]

        outputs = {}
        for backend in ("torch", "fused"):
            args = make_args(**{**vars(base), "log_probs_backend": backend})
            res = get_log_probs_and_entropy(
                local.clone().requires_grad_(True) * 1,
                args=args,
                unconcat_tokens=inputs["unconcat_tokens"],
                total_lengths=inputs["total_lens"],
                response_lengths=response_lens,
            )
            leaf = local.clone().requires_grad_(True)
            loss, _, _ = loss_function(args, make_batch(inputs, "sft_loss"), 1, leaf * 1)
            loss.backward()
            full = [
                all_gather_with_cp(lp.detach(), total_len, response_len)
                for lp, total_len, response_len in zip(
                    res["log_probs"], inputs["total_lens"], response_lens, strict=True
                )
            ]
            outputs[backend] = (full, loss.detach(), leaf.grad)

        for fused_lp, torch_lp in zip(outputs["fused"][0], outputs["torch"][0], strict=True):
            torch.testing.assert_close(fused_lp, torch_lp, rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(outputs["fused"][1], outputs["torch"][1], rtol=1e-5, atol=1e-5)
        # the backward reached this rank's logits, even the one that scores no response row
        assert outputs["fused"][2] is not None
        torch.testing.assert_close(outputs["fused"][2], outputs["torch"][2], **_TORCH_BACKEND_GRAD_TOL)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("prompt_lens", "response_lens"), _ALLGATHER_CP_CASES, ids=["split_responses", "empty_rank"])
def test_allgather_cp2_matches_the_torch_backend(prompt_lens, response_lens):
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(
        _allgather_cp2_worker,
        args=(_WORLD_SIZE, _free_port(), prompt_lens, response_lens),
        nprocs=_WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
