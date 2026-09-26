"""CP>1 consistency tests for the Tinker (multi-LoRA) loss path.

Each case runs the same client batch through cp=1 and cp>1 (zigzag layout:
each rank holds chunks ``rank`` and ``2*cp-1-rank`` of every sequence) and
asserts that per-datum log-probs, the summed loss, logits gradients, and the
``per_datum`` outputs all match the cp=1 run.
"""

from functools import partial

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp, slice_with_cp
from miles.backends.training_utils.loss import loss_function
from miles.backends.training_utils.loss_hub import tinker_losses
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

from .loss_test_utils import make_args

VOCAB_SIZE = 64

# (total_length, response_length); (10, 6) leaves rank 0 with no response logits.
DATUM_SHAPES = [(12, 8), (7, 5), (10, 6)]
# A lone one-token response: rank 0 holds no response tokens for the whole
# microbatch and must still produce a loss connected to the graph.
EMPTY_RANK_SHAPES = [(9, 1)]


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


def _make_datums(shapes: list[tuple[int, int]]) -> list[dict]:
    g = torch.Generator()
    g.manual_seed(7)

    def randn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    datums = []
    loss_mask_zeros = {0: [2], 1: [0]}
    for i, (total_length, response_length) in enumerate(shapes):
        loss_mask = torch.ones(response_length, dtype=torch.float32)
        loss_mask[[z for z in loss_mask_zeros.get(i, []) if z < response_length]] = 0
        datums.append(
            dict(
                unconcat_tokens=torch.randint(0, VOCAB_SIZE, (total_length,), generator=g),
                target_tokens=torch.randint(0, VOCAB_SIZE, (response_length,), generator=g).tolist(),
                advantages=randn(response_length).tolist(),
                loss_weights=(randn(response_length).abs() + 0.5).tolist(),
                rollout_log_probs=randn(response_length) * 2 - 3,
                loss_mask=loss_mask,
                logits=randn(total_length, VOCAB_SIZE),
            )
        )
    return datums


def _batch(datums: list[dict], rollout_log_probs: list, loss_fn: str) -> dict:
    return {
        "loss_fn": loss_fn,
        "unconcat_tokens": [d["unconcat_tokens"] for d in datums],
        "target_tokens": [d["target_tokens"] for d in datums],
        "total_lengths": [d["unconcat_tokens"].numel() for d in datums],
        "response_lengths": [len(d["target_tokens"]) for d in datums],
        "advantages": [d["advantages"] for d in datums],
        "loss_weights": [d["loss_weights"] for d in datums],
        "rollout_log_probs": rollout_log_probs,
        "loss_masks": [d["loss_mask"] for d in datums],
        "sample_indices": list(range(len(datums))),
        "dynamic_global_batch_size": len(datums),
    }


def _run_case(
    rank: int,
    world_size: int,
    port: int,
    *,
    loss_fn: str,
    shapes: list[tuple[int, int]],
    recompute: bool = False,
) -> None:
    init_gloo(rank, world_size, port=port)
    tp_group = [dist.new_group([r]) for r in range(world_size)][rank]

    args = make_args(
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=64,
        allgather_cp=False,
        multi_lora=True,
        calculate_per_token_loss=False,
        recompute_loss_function=recompute,
        use_dynamic_global_batch_size=True,
        global_batch_size=len(shapes),
    )
    datums = _make_datums(shapes)
    total_lengths = [d["unconcat_tokens"].numel() for d in datums]
    response_lengths = [len(d["target_tokens"]) for d in datums]

    # cp=1 reference over the full packed logits.
    _set_parallel_state(rank=0, world_size=1, tp_group=tp_group)
    logits_full = torch.cat([d["logits"] for d in datums], dim=0).unsqueeze(0).requires_grad_(True)
    batch_full = _batch(datums, [d["rollout_log_probs"] for d in datums], loss_fn)
    base_loss, _, base_logging = loss_function(args, batch_full, 1, logits_full)
    base_log_probs = tinker_losses._target_logprobs(args, batch_full, logits_full)
    base_loss.backward()
    grad_full = logits_full.grad.squeeze(0)

    # cp>1: slice each sample's logits into this rank's zigzag shard and
    # pre-slice rollout_log_probs like get_rollout_data does.
    _set_parallel_state(rank=rank, world_size=world_size, tp_group=tp_group)
    logits_local = (
        torch.cat(
            [slice_with_cp(d["logits"], 0.0, "thd") for d in datums],
            dim=0,
        )
        .unsqueeze(0)
        .requires_grad_(True)
    )
    rollout_log_probs_local = [
        torch.as_tensor(
            slice_log_prob_with_cp(d["rollout_log_probs"], total, response),
            dtype=torch.float32,
        )
        for d, total, response in zip(datums, total_lengths, response_lengths, strict=True)
    ]
    batch_local = _batch(datums, rollout_log_probs_local, loss_fn)
    cp_loss, _, cp_logging = loss_function(args, batch_local, 1, logits_local)
    cp_log_probs = tinker_losses._target_logprobs(args, batch_local, logits_local)
    cp_loss.backward()
    grad_local = logits_local.grad.squeeze(0)

    # the client vectors are sliced per call, never in place: a recomputed
    # forward must see the full-length batch again.
    for key in ("advantages", "loss_weights", "loss_masks"):
        assert [len(v) for v in batch_local[key]] == response_lengths

    # (a) per-token log-probs: CE is row-wise, so the gather is exact.
    for i, (total, response) in enumerate(zip(total_lengths, response_lengths, strict=True)):
        full = all_gather_with_cp(cp_log_probs[i], total, response)
        torch.testing.assert_close(full, base_log_probs[i], rtol=0, atol=0)

    # (b) each rank's loss is its local shard sum; the cp group sums to cp=1.
    cp_loss_sum = cp_loss.detach().clone()
    dist.all_reduce(cp_loss_sum, group=dist.group.WORLD)
    torch.testing.assert_close(cp_loss_sum, base_loss.detach(), rtol=1e-6, atol=1e-6)

    # (c) logits gradients land on this rank's zigzag shard of grad_full.
    offset = 0
    expected_grad_chunks = []
    for d in datums:
        total = d["unconcat_tokens"].numel()
        expected_grad_chunks.append(slice_with_cp(grad_full[offset : offset + total], 0.0, "thd"))
        offset += total
    torch.testing.assert_close(grad_local, torch.cat(expected_grad_chunks, dim=0), rtol=1e-6, atol=1e-6)

    # (d) per_datum outputs are full-response and identical on every rank.
    base_per_datum = base_logging["per_datum"]
    cp_per_datum = cp_logging["per_datum"]
    assert [o["sample_index"] for o in cp_per_datum] == [o["sample_index"] for o in base_per_datum]
    for cp_output, base_output in zip(cp_per_datum, base_per_datum, strict=True):
        torch.testing.assert_close(cp_output["loss"], base_output["loss"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(cp_output["logprobs"], base_output["logprobs"], rtol=0, atol=0)


@pytest.mark.parametrize("shapes", [DATUM_SHAPES, EMPTY_RANK_SHAPES], ids=["ragged", "empty_rank"])
@pytest.mark.parametrize("loss_fn", sorted(tinker_losses.TINKER_LOSS_FUNCTIONS))
def test_tinker_cp2_matches_cp1(loss_fn: str, shapes: list[tuple[int, int]]) -> None:
    run_multiprocess(partial(_run_case, loss_fn=loss_fn, shapes=shapes))


@pytest.mark.parametrize("recompute", [False, True], ids=["direct", "recompute"])
@pytest.mark.parametrize("loss_fn", ["importance_sampling", "cross_entropy"])
def test_tinker_cp4_matches_cp1(loss_fn: str, recompute: bool) -> None:
    run_multiprocess(partial(_run_case, loss_fn=loss_fn, shapes=DATUM_SHAPES, recompute=recompute), world_size=4)
