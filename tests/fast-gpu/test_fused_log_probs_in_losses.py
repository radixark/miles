"""``--log-probs-backend fused`` must match the torch backend through the losses, with and without CP."""

import sys

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"], num_gpus=2
)
register_rocm_ci(est_time=90, suite="nightly-stage-c-2-gpu-mi350", labels=["megatron"])

import dataclasses
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.fast.backends.training_utils.loss.loss_test_utils import (
    args_to_dict,
    make_args,
    make_batch,
    make_inputs,
    make_parallel_state,
)
from tests.fast.dist_utils import find_free_port

from miles.backends.training_utils.data.context_parallel import all_gather_with_cp
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

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


@pytest.mark.parametrize("train_backend", ["megatron", "fsdp"])
def test_fused_normalization_uses_the_backend_vocabulary(nccl_world, train_backend: str) -> None:
    """Megatron padding is excluded while native backends retain their full vocabulary."""
    make_parallel_state()
    args = make_args(true_on_policy_mode=False, log_probs_backend="fused", vocab_size=7)
    args.train_backend = train_backend
    if train_backend != "megatron":
        del args.backend.vocab_size
    logits = torch.zeros(1, 3, 8, device="cuda", requires_grad=True)
    with torch.no_grad():
        logits[..., -1] = 10
    result = get_log_probs_and_entropy(
        logits * 1,
        args=args,
        unconcat_tokens=[torch.tensor([0, 1, 2], device="cuda")],
        total_lengths=[3],
        response_lengths=[2],
        with_entropy=False,
    )

    vocabulary = 7 if train_backend == "megatron" else 8
    expected = torch.log_softmax(logits[0, :2, :vocabulary], dim=-1)[[0, 1], [1, 2]]
    torch.testing.assert_close(result["log_probs"][0], expected)
    result["log_probs"][0].sum().backward()
    if train_backend == "megatron":
        assert torch.count_nonzero(logits.grad[..., -1]) == 0


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
            args = make_args(**{**args_to_dict(base), "log_probs_backend": backend})
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
