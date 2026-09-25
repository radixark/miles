"""Compare sequence-statistic reduction with full-response reconstruction on Gloo."""

from functools import partial
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.cp_utils import (
    all_gather_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
    slice_with_cp,
)
from miles.backends.training_utils.loss_hub import losses
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

from .loss_test_utils import make_args


def _set_state(rank, size, tp_group):
    trivial = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=rank, size=size, group=dist.group.WORLD if size > 1 else None)
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
        )
    )


def _run_case(rank, world_size, port, *, qkv_format):
    init_gloo(rank, world_size, port=port)
    tp_group = [dist.new_group([r]) for r in range(world_size)][rank]
    _set_state(rank, world_size, tp_group)
    generator = torch.Generator().manual_seed(12)
    total_lengths = [33, 25, 33, 25]
    response_lengths = [16, 16, 16, 16]
    max_seq_lens = [40] * len(total_lengths) if qkv_format == "bshd" else [None] * len(total_lengths)
    masks = [torch.ones(n) for n in response_lengths]
    masks[0][2:7] = 0  # Masked observations inside a response.
    masks[2].zero_()  # A fully masked sequence has denominator one.
    full_logits = [torch.randn(n, 16, generator=generator) for n in total_lengths]
    tokens = [torch.randint(16, (n,), generator=generator) for n in total_lengths]
    # Keep both sides of the OPSM threshold and PPO clipping bounds represented.
    old_log_probs = [
        torch.log_softmax(logits, dim=-1)[n - r - 1 : n - 1].gather(1, tok[n - r :].unsqueeze(1)).squeeze(1) + offset
        for logits, tok, n, r, offset in zip(
            full_logits, tokens, total_lengths, response_lengths, [0.05, 0.3, -0.4, -0.1], strict=True
        )
    ]
    advantages = [torch.linspace(-1, 1, n) for n in response_lengths]

    def slice_response(xs):
        return [
            slice_log_prob_with_cp(x, n, r, qkv_format, m)
            for x, n, r, m in zip(xs, total_lengths, response_lengths, max_seq_lens, strict=True)
        ]

    local_logits = torch.cat(
        [slice_with_cp(x, 0, qkv_format, m) for x, m in zip(full_logits, max_seq_lens, strict=True)]
    ).unsqueeze(0)
    batch = dict(
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        loss_masks=masks,
        unconcat_tokens=tokens,
        log_probs=slice_response(old_log_probs),
        advantages=slice_response(advantages),
    )
    reducer = get_sum_of_sample_mean(
        total_lengths, response_lengths, masks, qkv_format=qkv_format, max_seq_lens=max_seq_lens
    )

    def reconstructed_kl(log_probs, old_probs, local_masks, loss_masks):
        full_current = [
            all_gather_with_cp(x, n, r, qkv_format, m)
            for x, n, r, m in zip(log_probs, total_lengths, response_lengths, max_seq_lens, strict=True)
        ]
        full_old = [
            all_gather_with_cp(x, n, r, qkv_format, m)
            for x, n, r, m in zip(old_probs, total_lengths, response_lengths, max_seq_lens, strict=True)
        ]
        return torch.stack(
            [
                ((old - new) * mask).sum() / mask.sum().clamp_min(1)
                for old, new, mask in zip(full_old, full_current, loss_masks, strict=True)
            ]
        )

    reduce_kl = losses.compute_sequence_kl
    for estimator, opsm in [("gspo", False), ("gspo", True), ("grpo", True)]:
        for reuse in [False, True]:
            args = make_args(
                advantage_estimator=estimator,
                use_opsm=opsm,
                qkv_format=qkv_format,
                skip_actor_forward_only=reuse,
                entropy_coef=0.0,
                observe_training_entropy=False,
            )
            results = []
            for sequence_kl_fn in [reconstructed_kl, reduce_kl]:
                logits = local_logits.detach().clone().requires_grad_()

                def checked_kl(*inputs, fn=sequence_kl_fn, requires_grad=estimator == "gspo"):
                    with patch.object(dist.nn, "all_reduce", wraps=dist.nn.all_reduce) as collective:
                        value = fn(*inputs)
                    if fn is reduce_kl:
                        assert collective.call_count == (1 if world_size > 1 else 0)
                        if world_size > 1:
                            assert collective.call_args.args[0].numel() == len(response_lengths)
                    assert value.requires_grad == requires_grad
                    return value

                with patch.object(losses, "compute_sequence_kl", checked_kl):
                    loss, metrics = losses.policy_loss_function(args, batch, logits, reducer)
                loss.backward()
                results.append((loss.detach(), metrics, logits.grad))
            torch.testing.assert_close(results[0], results[1], rtol=2e-5, atol=1e-6)
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
def test_sequence_reduction_matches_reconstructed_loss_and_gradients(world_size, qkv_format):
    run_multiprocess(partial(_run_case, qkv_format=qkv_format), world_size=world_size)
