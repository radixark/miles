"""CP>1 consistency tests for the policy losses that gather full responses (GSPO, OPSM).

Each case runs the same batch through cp=1 and cp=2 (zigzag layout: each rank holds chunks
``rank`` and ``2*cp-1-rank`` of every sequence) and asserts that the summed loss, ``ppo_kl``,
``pg_clipfrac`` and logits gradients all match the cp=1 run. bshd cuts chunks from each
sample's padded ``max_seq_len`` rather than its length, so it puts different tokens on each rank.
With ``allgather_cp`` each rank's logits are a contiguous piece of every padded row instead, and
the loss redistributes the response log-probs into the bshd zigzag layout.
"""

from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.data.context_parallel import slice_log_prob_with_cp, slice_with_cp
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, get_parallel_state, set_parallel_state

from .loss_test_utils import make_args

register_cpu_ci(est_time=70, suite="stage-b-cpu", labels=["precision"])

VOCAB_SIZE = 64
# (total_length, response_length) per sample
SHAPES = [(8, 5), (11, 6)]
# Sample 0's mask is uneven so a rotated response changes its masked mean; sample 1 skips a tool span.
LOSS_MASKS = [[1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]]
# old - new log-prob per response token. Sample 0's masked mean is 0.2 > opsm_delta=0.1, so OPSM masks it;
# rotated to [d1, d2, d3, d4, d0] the masked mean would be -0.2 and OPSM would keep it.
OLD_MINUS_NEW = [[0.0, 0.0, -0.6, 0.0, 0.6], [0.3] * 6]
ADVANTAGE = -1.0


def _set_parallel_state(rank: int, world_size: int, tp_group) -> None:
    trivial = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD if world_size > 1 else None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=cp,
            tp=GroupInfo(rank=0, size=1, group=tp_group),
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
            is_pp_last_stage=True,
        )
    )


def _response_log_probs(logits: torch.Tensor, tokens: torch.Tensor, response_length: int) -> torch.Tensor:
    """The log-prob of each response token under one sample's ``[total_length, V]`` logits."""
    total_length = tokens.numel()
    rows = logits[total_length - response_length - 1 : total_length - 1]
    return F.log_softmax(rows, dim=-1).gather(-1, tokens[-response_length:].unsqueeze(-1)).squeeze(-1)


def _make_samples() -> list[dict]:
    g = torch.Generator()
    g.manual_seed(11)
    samples = []
    for (total_length, response_length), loss_mask, old_minus_new in zip(
        SHAPES, LOSS_MASKS, OLD_MINUS_NEW, strict=True
    ):
        tokens = torch.randint(0, VOCAB_SIZE, (total_length,), generator=g)
        logits = torch.randn(total_length, VOCAB_SIZE, generator=g)
        new_log_probs = _response_log_probs(logits, tokens, response_length)
        samples.append(
            dict(
                tokens=tokens,
                logits=logits,
                total_length=total_length,
                response_length=response_length,
                loss_mask=torch.tensor(loss_mask),
                old_log_probs=new_log_probs + torch.tensor(old_minus_new),
                advantages=torch.full((response_length,), ADVANTAGE),
            )
        )
    return samples


def _batch(samples: list[dict], qkv_format: str, max_seq_len: int | None, local: bool) -> dict:
    """The loss batch; with `local`, per-token rollout tensors are this rank's CP shard, as get_rollout_data leaves them."""

    def shard(value: torch.Tensor, sample: dict) -> torch.Tensor:
        if not local:
            return value.clone()
        return slice_log_prob_with_cp(
            value.clone(), sample["total_length"], sample["response_length"], qkv_format, max_seq_len
        )

    batch = {
        "unconcat_tokens": [s["tokens"] for s in samples],
        "total_lengths": [s["total_length"] for s in samples],
        "response_lengths": [s["response_length"] for s in samples],
        "loss_masks": [s["loss_mask"].clone() for s in samples],
        "log_probs": [shard(s["old_log_probs"], s) for s in samples],
        "advantages": [shard(s["advantages"], s) for s in samples],
    }
    if qkv_format == "bshd":
        batch["max_seq_lens"] = [max_seq_len] * len(samples)
    return batch


def _local_bshd_row(row: torch.Tensor, max_seq_len: int, allgather_cp: bool) -> torch.Tensor:
    """This rank's part of one padded bshd row: a contiguous piece with allgather-CP, else its zigzag chunks."""
    if not allgather_cp:
        return slice_with_cp(row, 0.0, "bshd", max_seq_len)
    cp = get_parallel_state().cp
    piece_rows = max_seq_len // cp.size
    return F.pad(row, (0, 0, 0, max_seq_len - row.size(0)))[cp.rank * piece_rows : (cp.rank + 1) * piece_rows]


def _logits(
    samples: list[dict], qkv_format: str, max_seq_len: int | None, local: bool, allgather_cp: bool
) -> torch.Tensor:
    """Model logits: thd packs samples into ``[1, T, V]``; bshd pads each to a row of ``[B, S, V]``."""
    if local and qkv_format == "bshd":
        pieces = [_local_bshd_row(s["logits"], max_seq_len, allgather_cp) for s in samples]
    elif local:
        pieces = [slice_with_cp(s["logits"], 0.0, "thd") for s in samples]
    elif qkv_format == "bshd":
        pieces = [F.pad(s["logits"], (0, 0, 0, max_seq_len - s["total_length"])) for s in samples]
    else:
        pieces = [s["logits"] for s in samples]
    logits = torch.stack(pieces) if qkv_format == "bshd" else torch.cat(pieces).unsqueeze(0)
    return logits.clone().requires_grad_(True)


def _expected_local_grad(
    full_grad: torch.Tensor, samples: list[dict], qkv_format: str, max_seq_len: int | None, allgather_cp: bool
):
    """This rank's shard of the cp=1 logits gradient, laid out like `_logits(local=True)`."""
    if qkv_format == "bshd":
        return torch.stack([_local_bshd_row(row, max_seq_len, allgather_cp) for row in full_grad])
    offsets = torch.tensor([0] + [s["total_length"] for s in samples]).cumsum(0).tolist()
    pieces = [
        slice_with_cp(full_grad[0, start:end], 0.0, "thd")
        for start, end in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    return torch.cat(pieces).unsqueeze(0)


def _metric(logging: dict, key: str) -> torch.Tensor:
    return logging["values"][1 + logging["keys"].index(key)]


def _run_case(
    rank: int,
    world_size: int,
    port: int,
    *,
    estimator: str,
    use_opsm: bool,
    qkv_format: str,
    max_seq_len: int | None,
    allgather_cp: bool,
) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        tp_group = [dist.new_group([r]) for r in range(world_size)][rank]
        args = make_args(
            advantage_estimator=estimator,
            use_opsm=use_opsm,
            opsm_delta=0.1,
            qkv_format=qkv_format,
            true_on_policy_mode=False,
            entropy_coef=0.0,
            observe_training_entropy=False,
            global_batch_size=len(SHAPES),
        )
        samples = _make_samples()

        args.allgather_cp = False
        _set_parallel_state(rank=0, world_size=1, tp_group=tp_group)
        full_logits = _logits(samples, qkv_format, max_seq_len, local=False, allgather_cp=False)
        base_loss, _, base_log = loss_function(
            args, _batch(samples, qkv_format, max_seq_len, local=False), 1, full_logits
        )
        base_loss.backward()
        if use_opsm:
            # the case must exercise OPSM: sample 0 (and 1) are masked at cp=1
            assert _metric(base_log, "opsm_clipfrac") > 0

        args.allgather_cp = allgather_cp
        _set_parallel_state(rank=rank, world_size=world_size, tp_group=tp_group)
        local_logits = _logits(samples, qkv_format, max_seq_len, local=True, allgather_cp=allgather_cp)
        cp_loss, _, cp_log = loss_function(args, _batch(samples, qkv_format, max_seq_len, local=True), 1, local_logits)
        cp_loss.backward()

        # each rank reduces its own tokens with full-sample denominators; the CP group sums to cp=1
        for name, cp_value, base_value in [
            ("loss", cp_loss.detach(), base_loss.detach()),
            ("ppo_kl", _metric(cp_log, "ppo_kl"), _metric(base_log, "ppo_kl")),
            ("pg_clipfrac", _metric(cp_log, "pg_clipfrac"), _metric(base_log, "pg_clipfrac")),
        ]:
            summed = cp_value.clone()
            dist.all_reduce(summed, group=dist.group.WORLD)
            torch.testing.assert_close(summed, base_value, rtol=1e-6, atol=1e-6, msg=lambda m, n=name: f"{n}: {m}")

        expected_grad = _expected_local_grad(full_logits.grad, samples, qkv_format, max_seq_len, allgather_cp)
        torch.testing.assert_close(local_logits.grad, expected_grad, rtol=1e-6, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "qkv_format,max_seq_len,allgather_cp",
    # max_seq_len=12: sample 0 keeps its thd per-rank counts but holds different tokens (a thd gather rotates it);
    # max_seq_len=16: per-rank counts differ from thd (a thd gather fails its shape assert)
    [("thd", None, False), ("bshd", 12, False), ("bshd", 16, False), ("bshd", 12, True)],
    ids=["thd", "bshd_rows_moved", "bshd_row_counts_differ", "bshd_allgather_cp"],
)
@pytest.mark.parametrize("estimator,use_opsm", [("gspo", False), ("grpo", True)], ids=["gspo", "opsm"])
def test_full_response_policy_loss_cp2_matches_cp1(
    estimator: str, use_opsm: bool, qkv_format: str, max_seq_len: int | None, allgather_cp: bool
) -> None:
    run_multiprocess(
        partial(
            _run_case,
            estimator=estimator,
            use_opsm=use_opsm,
            qkv_format=qkv_format,
            max_seq_len=max_seq_len,
            allgather_cp=allgather_cp,
        )
    )
