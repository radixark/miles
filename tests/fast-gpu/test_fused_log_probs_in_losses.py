"""``--log-probs-backend fused`` must match the torch backend through the losses, with and without CP."""

import sys

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"], num_gpus=2
)

import dataclasses
import os

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

from miles.backends.training_utils.data.context_parallel import all_gather_with_cp
from miles.backends.training_utils.loss.hub import score_centering_loss
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.utils.sampling_mask import RolloutSamplingMask

_WORLD_SIZE = 2
# Megatron's fused CE (the torch backend) rounds its logits gradient to bf16 even for fp32 logits
# (fused_cross_entropy.py, calculate_gradients), so gradients are compared with it at bf16 precision.
_TORCH_BACKEND_GRAD_TOL = dict(rtol=2e-2, atol=5e-3)


def _to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, list):
        return [_to_cuda(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_cuda(item) for key, item in value.items()}
    return value


_LOSS_VOCAB = 4096


@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
@pytest.mark.parametrize("loss_type", ["policy_loss", "sft_loss"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_losses_match_the_torch_backend(nccl_world, loss_type, dtype, qkv_format):
    make_parallel_state()
    prompt_lens, response_lens = [5, 9, 3], [7, 4, 6]
    results = {}
    for backend in ("torch", "fused"):
        args = make_args(
            loss_type=loss_type,
            qkv_format=qkv_format,
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_score_centering_candidates_match_the_torch_backend(nccl_world, dtype):
    """Score centering's training loss scores ``[sampled token, top-K candidates]`` per row: the fused
    backend in one call with K targets, the torch backend through selected_log_probs_and_entropy.
    Scores, entropy and the logits gradient must agree."""
    make_parallel_state()
    prompt_lens, response_lens, n_candidates = [5, 9, 3], [7, 4, 6], 8
    width, vocab_size = _LOSS_VOCAB, _LOSS_VOCAB - 96
    total_lens = [p + r for p, r in zip(prompt_lens, response_lens, strict=True)]
    gen = torch.Generator(device="cuda").manual_seed(29)
    logits = (torch.randn(1, sum(total_lens), width, device="cuda", generator=gen) * 4).to(dtype)
    tokens = [torch.randint(0, vocab_size, (total,), device="cuda", generator=gen) for total in total_lens]
    candidates = []
    for sample_tokens, response in zip(tokens, response_lens, strict=True):
        ids = torch.randint(0, vocab_size, (response, n_candidates), device="cuda", generator=gen)
        ids[:, 0] = sample_tokens[-response:]  # the sampled token is usually among the candidates
        ids[::2, -2:] = -1
        candidates.append(ids.cpu())
    batch = {
        "unconcat_tokens": tokens,
        "total_lengths": total_lens,
        "response_lengths": response_lens,
        "rollout_topk_token_ids": candidates,
    }

    outputs = {}
    for backend in ("torch", "fused"):
        args = make_args(
            loss_type="score_centering",
            true_on_policy_mode=False,
            log_probs_backend=backend,
            rollout_temperature=0.8,
            vocab_size=vocab_size,
            use_kl_loss=True,
            entropy_coef=0.01,
            observe_training_entropy=False,
            use_sampling_support_replay=False,
            recompute_loss_function=False,
        )
        leaf = logits.clone().requires_grad_(True)
        result = score_centering_loss._candidate_log_probs(args, batch, leaf * 1)
        selected, entropy = torch.cat(result["selected"]), torch.cat(result["entropy"])
        weights = torch.linspace(-1, 1, selected.numel(), device="cuda").view(selected.shape)
        ((selected * weights).sum() + 0.1 * entropy.sum()).backward()
        outputs[backend] = (selected.detach(), torch.cat(result["kl_log_probs"]).detach(), entropy.detach(), leaf.grad)

    *scores, grad = outputs["fused"]
    *reference_scores, reference_grad = outputs["torch"]
    for fused, reference in zip(scores, reference_scores, strict=True):
        torch.testing.assert_close(fused, reference, rtol=1e-5, atol=5e-5)
    # both backends round the gradient into the logits' dtype once
    grad_tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(grad.float(), reference_grad.float(), **grad_tol)


def _support_batch(prompt_lens, response_lens, vocab_size, max_support, seed):
    """Token sequences and, per response token, a sampling support holding the sampled token."""
    gen = torch.Generator().manual_seed(seed)
    total_lens = [p + r for p, r in zip(prompt_lens, response_lens, strict=True)]
    tokens = [torch.randint(0, vocab_size, (total,), generator=gen) for total in total_lens]
    supports = []
    for sample_tokens, response in zip(tokens, response_lens, strict=True):
        rows = []
        for token in sample_tokens[-response:].tolist():
            size = int(torch.randint(1, max_support + 1, (1,), generator=gen))
            others = [i for i in torch.randperm(vocab_size, generator=gen)[: size + 1].tolist() if i != token]
            rows.append([token, *others[: size - 1]])
        supports.append(rows)
    return total_lens, [t.cuda() for t in tokens], supports


def test_sampling_support_replay_matches_the_torch_backend(nccl_world):
    """Under top-p/top-k replay the policy scores each token over its sampling support: the torch
    backend masks the logits, the fused one reads only the support. Log-probs, the (full-vocabulary)
    entropy and the logits gradient must agree."""
    make_parallel_state()
    prompt_lens, response_lens = [5, 9, 3], [7, 4, 6]
    total_lens, tokens, supports = _support_batch(prompt_lens, response_lens, _LOSS_VOCAB, 24, seed=33)
    masks = [RolloutSamplingMask.from_mask_list(rows) for rows in supports]
    logits = torch.randn(
        1, sum(total_lens), _LOSS_VOCAB, device="cuda", generator=torch.Generator("cuda").manual_seed(34)
    )

    outputs = {}
    for backend in ("torch", "fused"):
        args = make_args(true_on_policy_mode=False, log_probs_backend=backend, rollout_temperature=0.8)
        leaf = (logits * 4).requires_grad_(True)
        res = get_log_probs_and_entropy(
            leaf * 1,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=total_lens,
            response_lengths=response_lens,
            with_entropy=True,
            rollout_sampling_mask=masks,
        )
        log_probs, entropy = torch.cat(res["log_probs"]), torch.cat(res["entropy"])
        (log_probs.sum() + 0.1 * entropy.sum()).backward()
        outputs[backend] = (log_probs.detach(), entropy.detach(), leaf.grad)

    (torch_lp, torch_ent, torch_grad), (fused_lp, fused_ent, fused_grad) = outputs["torch"], outputs["fused"]
    torch.testing.assert_close(fused_lp, torch_lp, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(fused_ent, torch_ent, rtol=1e-5, atol=5e-5)
    torch.testing.assert_close(fused_grad, torch_grad, **_TORCH_BACKEND_GRAD_TOL)


def test_score_centering_filtered_mode_matches_the_torch_backend(nccl_world):
    """Filtered score centering scores the sampled token and its whole support, normalized over it,
    with the entropy over the support: the fused support mode against _SupportLogProbs."""
    make_parallel_state()
    prompt_lens, response_lens, width = [5, 9, 3], [7, 4, 6], 24
    total_lens, tokens, supports = _support_batch(prompt_lens, response_lens, _LOSS_VOCAB, width, seed=35)
    candidates = [
        torch.tensor([row + [-1] * (width - len(row)) for row in rows], dtype=torch.long) for rows in supports
    ]
    batch = {
        "unconcat_tokens": tokens,
        "total_lengths": total_lens,
        "response_lengths": response_lens,
        "rollout_topk_token_ids": candidates,
    }
    logits = torch.randn(
        1, sum(total_lens), _LOSS_VOCAB, device="cuda", generator=torch.Generator("cuda").manual_seed(36)
    )

    outputs = {}
    for backend in ("torch", "fused"):
        args = make_args(
            loss_type="score_centering",
            true_on_policy_mode=False,
            log_probs_backend=backend,
            rollout_temperature=0.8,
            use_kl_loss=False,
            entropy_coef=0.01,
            observe_training_entropy=False,
            use_sampling_support_replay=True,
            recompute_loss_function=False,
        )
        leaf = (logits * 4).requires_grad_(True)
        result = score_centering_loss._candidate_log_probs(args, batch, leaf * 1)
        selected, entropy = torch.cat(result["selected"]), torch.cat(result["entropy"])
        weights = torch.linspace(-1, 1, selected.numel(), device="cuda").view(selected.shape)
        ((selected * weights).sum() + 0.1 * entropy.sum()).backward()
        outputs[backend] = (selected.detach(), entropy.detach(), leaf.grad)

    for fused, reference in zip(outputs["fused"], outputs["torch"], strict=True):
        torch.testing.assert_close(fused, reference, rtol=1e-5, atol=5e-5)


def _score_centering_batch(prompt_lens, response_lens, vocab_size, n_candidates, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total_lens = [p + r for p, r in zip(prompt_lens, response_lens, strict=True)]
    tokens = [torch.randint(0, vocab_size, (total,), device="cuda", generator=gen) for total in total_lens]
    candidates = []
    for sample_tokens, response in zip(tokens, response_lens, strict=True):
        ids = torch.randint(0, vocab_size, (response, n_candidates), device="cuda", generator=gen)
        ids[:, 0] = sample_tokens[-response:]  # the sampled token is usually among the candidates
        ids[::2, -2:] = -1
        candidates.append(ids.cpu())
    return {
        "unconcat_tokens": tokens,
        "total_lengths": total_lens,
        "response_lengths": response_lens,
        "rollout_topk_token_ids": candidates,
    }


def _score_centering_args(backend, vocab_size):
    return make_args(
        loss_type="score_centering",
        true_on_policy_mode=False,
        log_probs_backend=backend,
        rollout_temperature=0.8,
        vocab_size=vocab_size,
        use_kl_loss=True,
        entropy_coef=0.01,
        observe_training_entropy=False,
        use_sampling_support_replay=False,
        recompute_loss_function=False,
    )


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_score_centering_zigzag_cp2_matches_the_torch_backend(nccl_world, cp_rank):
    """Zigzag CP2 with the layout where rank 1 scores no response row: the fused backend's K
    targets per row must land on the same rows, and an empty rank must still run."""
    set_parallel_state(dataclasses.replace(make_parallel_state(), cp=GroupInfo(rank=cp_rank, size=2, group=None)))
    vocab_size = _LOSS_VOCAB - 96
    batch = _score_centering_batch(_ZIGZAG_PROMPT_LENS, _ZIGZAG_RESPONSE_LENS, vocab_size, 8, seed=40 + cp_rank)
    local_rows = sum(2 * -(-total // 4) for total in batch["total_lengths"])
    logits = torch.randn(1, local_rows, _LOSS_VOCAB, device="cuda", generator=torch.Generator("cuda").manual_seed(41))

    outputs = {}
    for backend in ("torch", "fused"):
        leaf = (logits * 4).requires_grad_(True)
        result = score_centering_loss._candidate_log_probs(_score_centering_args(backend, vocab_size), batch, leaf * 1)
        selected, entropy = torch.cat(result["selected"]), torch.cat(result["entropy"])
        weights = torch.linspace(-1, 1, selected.numel(), device="cuda").view(selected.shape)
        ((selected * weights).sum() + 0.1 * entropy.sum()).backward()
        outputs[backend] = ([s.size(0) for s in result["selected"]], selected.detach(), entropy.detach(), leaf.grad)

    (torch_sizes, *torch_outputs), (fused_sizes, *fused_outputs) = outputs["torch"], outputs["fused"]
    assert fused_sizes == torch_sizes
    if cp_rank == 1:
        assert fused_sizes[0] == 0
    for fused, reference in zip(fused_outputs, torch_outputs, strict=True):
        torch.testing.assert_close(fused, reference, rtol=1e-5, atol=5e-5)


def test_score_centering_rejects_a_candidate_outside_the_vocabulary(nccl_world):
    """Both backends reject such an id with the same readable error, before any kernel runs."""
    make_parallel_state()
    vocab_size = _LOSS_VOCAB - 96
    batch = _score_centering_batch([5, 9], [7, 4], vocab_size, 8, seed=42)
    batch["rollout_topk_token_ids"][1][2, 3] = vocab_size  # first id of Megatron's vocab padding
    logits = torch.randn(1, sum(batch["total_lengths"]), _LOSS_VOCAB, device="cuda")
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        score_centering_loss._candidate_log_probs(_score_centering_args("fused", vocab_size), batch, logits)


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_sampling_support_replay_zigzag_cp2_matches_the_torch_backend(nccl_world, cp_rank):
    """Replay under zigzag CP2, with the layout where rank 1 scores no response row: its supports
    pad to width 0, and it must still run and match the torch backend."""
    set_parallel_state(dataclasses.replace(make_parallel_state(), cp=GroupInfo(rank=cp_rank, size=2, group=None)))
    total_lens, tokens, supports = _support_batch(_ZIGZAG_PROMPT_LENS, _ZIGZAG_RESPONSE_LENS, _LOSS_VOCAB, 24, seed=43)
    masks = [RolloutSamplingMask.from_mask_list(rows) for rows in supports]
    local_rows = sum(2 * -(-total // 4) for total in total_lens)
    logits = torch.randn(1, local_rows, _LOSS_VOCAB, device="cuda", generator=torch.Generator("cuda").manual_seed(44))

    outputs = {}
    for backend in ("torch", "fused"):
        args = make_args(true_on_policy_mode=False, log_probs_backend=backend, rollout_temperature=0.8)
        leaf = (logits * 4).requires_grad_(True)
        res = get_log_probs_and_entropy(
            leaf * 1,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=total_lens,
            response_lengths=_ZIGZAG_RESPONSE_LENS,
            with_entropy=True,
            rollout_sampling_mask=masks,
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
        args=(_WORLD_SIZE, find_free_port(), prompt_lens, response_lens),
        nprocs=_WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
