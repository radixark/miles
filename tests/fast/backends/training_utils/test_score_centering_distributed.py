"""TP=2, CP=2 parity on CPU/Gloo and opt-in four-GPU NCCL."""

import os
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean, slice_log_prob_with_cp, slice_with_cp
from miles.backends.training_utils.loss_hub.score_centering import score_centering_loss, selected_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _layout(parts: list[torch.Tensor], args: Namespace, cp_rank: int) -> torch.Tensor:
    if args.allgather_cp:
        full = torch.cat(parts)
        full = F.pad(full, (0, 0, 0, full.size(0) % 2))
        return full.chunk(2)[cp_rank].unsqueeze(0)
    pieces = [slice_with_cp(part, 0, args.qkv_format, 12) for part in parts]
    return torch.stack(pieces) if args.qkv_format == "bshd" else torch.cat(pieces).unsqueeze(0)


def _check_selected(tp: GroupInfo, dtype: torch.dtype, device: torch.device) -> None:
    full = (
        torch.randn(4, 8, generator=torch.Generator().manual_seed(31)).to(device=device, dtype=dtype).requires_grad_()
    )
    local = full.detach()[:, tp.rank * 4 : (tp.rank + 1) * 4].clone().requires_grad_()
    ids = torch.tensor([[0, 3, 6, -1], [1, 4, 4, 1], [6, 3, -1, -1], [5, 2, 4, 6]], device=device)
    actual, entropy = selected_log_probs_and_entropy(
        local, ids, group=tp.group, vocab_size=7, temperature=0.7, chunk_size=2, with_entropy=True
    )
    logp = (full.float()[:, :7] / 0.7).log_softmax(-1)
    reference = logp.gather(-1, ids.clamp_min(0)).masked_fill(ids < 0, 0)
    expected_entropy = -(logp.exp() * logp).sum(-1)
    torch.testing.assert_close(entropy, expected_entropy, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    weights = torch.randn(4, 4, generator=torch.Generator().manual_seed(41)).to(device)
    ((actual * weights).sum() - 0.2 * entropy.sum()).backward()
    ((reference * weights).sum() - 0.2 * expected_entropy.sum()).backward()
    expected = full.grad[:, tp.rank * 4 : (tp.rank + 1) * 4]
    torch.testing.assert_close(
        local.grad,
        expected,
        atol=0.01 if dtype == torch.bfloat16 else 1e-6,
        rtol=0.01 if dtype == torch.bfloat16 else 1e-6,
    )


def _check_loss(tp: GroupInfo, cp: GroupInfo, layout: str, mode: str, device: torch.device) -> None:
    args = Namespace(
        qkv_format="bshd" if layout == "bshd" else "thd",
        allgather_cp=layout == "allgather",
        true_on_policy_mode=False,
        rollout_temperature=0.7,
        log_probs_chunk_size=2,
        vocab_size=7,
        score_centering_is=mode,
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
        entropy_coef=0.03,
        observe_training_entropy=True,
        use_kl_loss=False,
    )
    generator = torch.Generator().manual_seed(19)
    totals, responses = [5, 9, 3], [2, 3, 0]
    parts = [torch.randn(total, 8, generator=generator).to(device).requires_grad_() for total in totals]
    tokens = [torch.randint(0, 7, (total,), generator=generator).to(device) for total in totals]
    masks = [torch.tensor(mask, device=device) for mask in ([1, 0], [1, 1, 1], [])]
    advantages = [torch.randn(response, generator=generator).to(device) for response in responses]
    distributions = [torch.randn(response, 7, generator=generator).to(device).softmax(-1) for response in responses]
    candidates = [q.topk(3, dim=-1) for q in distributions]
    sampled_q = [
        q.gather(-1, token[-response:, None] if response else token[:0, None]).squeeze(-1).log()
        for q, token, response in zip(distributions, tokens, responses, strict=True)
    ]
    expected_loss = torch.zeros((), device=device)
    for part, token, total, response, advantage, mask, head, q_sample in zip(
        parts, tokens, totals, responses, advantages, masks, candidates, sampled_q, strict=True
    ):
        logp = (part[total - response - 1 : total - 1, :7] / 0.7).log_softmax(-1)
        sampled = token[-response:] if response else token[:0]
        per_token, _ = score_centering_loss(
            logp.gather(-1, sampled[:, None]).squeeze(-1),
            logp.gather(-1, head.indices),
            q_sample,
            head.values.log(),
            torch.ones_like(head.indices, dtype=torch.bool),
            advantage,
            mode=mode,
        )
        entropy = -(logp.exp() * logp).sum(-1)
        expected_loss = expected_loss + (
            (per_token - args.entropy_coef * entropy) * mask
        ).sum() / mask.sum().clamp_min(1)
    expected_loss.backward()
    local = (
        _layout([part.detach() for part in parts], args, cp.rank)[..., tp.rank * 4 : (tp.rank + 1) * 4]
        .clone()
        .requires_grad_()
    )
    batch = dict(
        total_lengths=totals,
        response_lengths=responses,
        unconcat_tokens=tokens,
        loss_masks=masks,
        max_seq_lens=[12] * 3 if args.qkv_format == "bshd" else None,
        rollout_topk_token_ids=[head.indices.cpu().numpy().astype("int32") for head in candidates],
        rollout_topk_log_probs=[head.values.log().cpu().numpy() for head in candidates],
        rollout_log_probs=[
            slice_log_prob_with_cp(q, t, r, args.qkv_format, 12)
            for q, t, r in zip(sampled_q, totals, responses, strict=True)
        ],
        advantages=[
            slice_log_prob_with_cp(a, t, r, args.qkv_format, 12)
            for a, t, r in zip(advantages, totals, responses, strict=True)
        ],
    )
    reduce = get_sum_of_sample_mean(
        totals, responses, masks, qkv_format=args.qkv_format, max_seq_lens=batch["max_seq_lens"]
    )
    loss, metrics = score_centering_loss_function(args, batch, local, reduce)
    loss.backward()
    expected_grad = _layout([part.grad for part in parts], args, cp.rank)[..., tp.rank * 4 : (tp.rank + 1) * 4]
    torch.testing.assert_close(local.grad, expected_grad, atol=2e-6, rtol=2e-5)
    actual_loss = loss.detach().clone()
    dist.all_reduce(actual_loss, group=cp.group)
    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-6, rtol=2e-5)
    assert all(torch.isfinite(value) for value in metrics.values())


def _worker(rank: int, rendezvous: str, backend: str) -> None:
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=120))
    try:
        tp_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        cp_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        tp = GroupInfo(rank=rank % 2, size=2, group=tp_groups[rank // 2])
        cp = GroupInfo(rank=rank // 2, size=2, group=cp_groups[rank % 2])
        singleton = GroupInfo(rank=0, size=1, group=None)
        state = {name: singleton for name in ("intra_dp", "intra_dp_cp", "pp", "ep", "etp", "indep_dp")}
        set_parallel_state(ParallelState(**state, tp=tp, cp=cp))
        for dtype in (torch.float32, torch.bfloat16):
            _check_selected(tp, dtype, device)
        for layout in ("thd", "bshd", "allgather"):
            for mode in ("none", "tis", "mis"):
                _check_loss(tp, cp, layout, mode, device)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("MILES_TEST_DISTRIBUTED") != "1", reason="opt-in multiprocess CPU parity")
def test_tensor_and_context_parallel_gradients(tmp_path: Path) -> None:
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(), "gloo"), nprocs=4, join=True)


@pytest.mark.skipif(
    os.environ.get("MILES_TEST_CUDA_DISTRIBUTED") != "1" or torch.cuda.device_count() < 4,
    reason="opt-in four-GPU NCCL parity",
)
def test_tensor_and_context_parallel_cuda_gradients(tmp_path: Path) -> None:
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(), "nccl"), nprocs=4, join=True)
