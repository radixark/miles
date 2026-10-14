"""FlashREINFORCE: batch-mean advantages and the binary-KL sequence trust region.

The loss test checks Miles' policy loss against a dense reference of the paper's
objective, -1/B sum_i m_i A_i / T_i sum_t stopgrad(rho_it) log pi_it.
"""

import math
from functools import partial

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.data.context_parallel import slice_log_prob_with_cp
from miles.backends.training_utils.loss.hub.corrections import binary_kl_trust_region_function
from miles.backends.training_utils.loss.objective import compute_advantages_and_returns, loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

from .loss_test_utils import make_args, make_parallel_state

TRUST_REGION = "miles.backends.training_utils.loss.hub.corrections.binary_kl_trust_region_function"
# pi = [1, 0.5] vs mu = [1/3, 0.5]: the first token's binary KL is ~0.5.
FAR_TRAIN, FAR_ROLLOUT = [0.0, math.log(0.5)], [-math.log(3.0), math.log(0.5)]
# pi = 0.5 vs mu = 0.505: binary KL ~5e-5.
NEAR_TRAIN, NEAR_ROLLOUT = math.log(0.5), math.log(0.505)


def _args(**overrides):
    return make_args(
        **{
            "advantage_estimator": "flash_reinforce",
            "use_tis": True,
            "custom_tis_function_path": TRUST_REGION,
            "tis_binary_kl_threshold": 3e-3,
            **overrides,
        }
    )


def _trust_region(args, train, rollout, masks=None, *, parallel_state=None):
    train = [torch.tensor(x, dtype=torch.float32) for x in train]
    rollout = [torch.tensor(x, dtype=torch.float32) for x in rollout]
    lengths = [len(x) for x in train]
    return binary_kl_trust_region_function(
        args,
        pg_loss=torch.ones(sum(lengths)),
        train_log_probs=train,
        rollout_log_probs=rollout,
        loss_masks=[torch.ones(n, dtype=torch.int) for n in lengths] if masks is None else masks,
        total_lengths=[n + 4 for n in lengths],
        response_lengths=lengths,
        parallel_state=parallel_state or make_parallel_state(),
    )


# ------------------------------- advantages -------------------------------


def test_advantages_broadcast_the_centered_reward():
    make_parallel_state()
    args = make_args(advantage_estimator="flash_reinforce", kl_coef=0.0)
    rollout_data = dict(
        log_probs=[torch.zeros(3), torch.zeros(2)],
        rewards=[0.75, -0.25],
        response_lengths=[3, 2],
        loss_masks=[torch.ones(3, dtype=torch.int), torch.ones(2, dtype=torch.int)],
        total_lengths=[5, 4],
    )

    compute_advantages_and_returns(args, rollout_data)

    for advantage, reward in zip(rollout_data["advantages"], [0.75, -0.25], strict=True):
        torch.testing.assert_close(advantage, torch.full_like(advantage, reward))


# ------------------------------- trust region -------------------------------


def test_one_far_token_rejects_the_whole_sequence():
    pg_loss, masks, metrics = _trust_region(_args(), [FAR_TRAIN], [FAR_ROLLOUT])

    torch.testing.assert_close(pg_loss, torch.zeros(2))
    assert masks[0].tolist() == [1, 1]  # rejection is a zero weight, not a mask
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.ones(2))
    torch.testing.assert_close(metrics["tis"], torch.tensor([3.0, 1.0]))


def test_trust_region_is_two_sided():
    pg_loss, _, metrics = _trust_region(_args(), [FAR_ROLLOUT], [FAR_TRAIN])

    torch.testing.assert_close(pg_loss, torch.zeros(2))
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.ones(2))


def test_close_sequence_keeps_its_unclipped_importance_weight():
    pg_loss, _, metrics = _trust_region(_args(), [[NEAR_TRAIN] * 2], [[NEAR_ROLLOUT] * 2])

    torch.testing.assert_close(pg_loss, torch.full((2,), 0.5 / 0.505))
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.zeros(2))
    assert metrics["tis_binary_kl"].max() < 1e-4


def test_gate_thresholds_the_mean_not_the_sum():
    # pi = 0.5 vs mu = 0.52 is ~8e-4 per token: 10 tokens sum past delta, their mean does not.
    pg_loss, _, metrics = _trust_region(_args(), [[math.log(0.5)] * 10], [[math.log(0.52)] * 10])

    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.zeros(10))
    torch.testing.assert_close(pg_loss, torch.full((10,), 0.5 / 0.52))


def test_infinite_threshold_disables_the_gate():
    pg_loss, _, _ = _trust_region(_args(tis_binary_kl_threshold=math.inf), [FAR_TRAIN], [FAR_ROLLOUT])

    torch.testing.assert_close(pg_loss, torch.tensor([3.0, 1.0]))


def test_masked_tokens_stay_out_of_the_sequence_mean():
    train, rollout = [[NEAR_TRAIN, 0.0, NEAR_TRAIN]], [[NEAR_ROLLOUT, math.log(1e-9), NEAR_ROLLOUT]]

    pg_loss, _, metrics = _trust_region(_args(), train, rollout, [torch.tensor([1, 0, 1], dtype=torch.int)])

    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.zeros(3))
    torch.testing.assert_close(pg_loss[[0, 2]], torch.full((2,), 0.5 / 0.505))


def test_sequence_without_loss_tokens_is_kept():
    _, _, metrics = _trust_region(_args(), [FAR_TRAIN], [FAR_ROLLOUT], [torch.zeros(2, dtype=torch.int)])

    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.zeros(2))


@pytest.mark.parametrize("bad", [math.nan, -math.inf])
def test_non_finite_rollout_log_probs_are_rejected_with_finite_weights(bad):
    pg_loss, _, metrics = _trust_region(_args(), [[0.0, math.log(0.5)]], [[bad, math.log(0.5)]])

    torch.testing.assert_close(pg_loss, torch.zeros(2))
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.ones(2))
    assert torch.isfinite(metrics["tis"]).all()


def test_only_the_offending_sequence_is_rejected():
    pg_loss, _, metrics = _trust_region(_args(), [[NEAR_TRAIN] * 2, FAR_TRAIN], [[NEAR_ROLLOUT] * 2, FAR_ROLLOUT])

    torch.testing.assert_close(pg_loss, torch.tensor([0.5 / 0.505, 0.5 / 0.505, 0.0, 0.0]))
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_mismatch_metrics_alone_leave_the_loss_untouched():
    pg_loss, _, metrics = _trust_region(_args(use_tis=False, get_mismatch_metrics=True), [FAR_TRAIN], [FAR_ROLLOUT])

    torch.testing.assert_close(pg_loss, torch.ones(2))
    assert set(metrics) == {"tis", "tis_abs", "tis_binary_kl", "tis_seq_reject_frac"}


# ------------------------------- policy loss -------------------------------

VOCAB = 16
PROMPT_LENS, RESPONSE_LENS = [6, 5, 7, 4], [5, 3, 4, 6]
# Rollout 1 spans samples 1 and 2: one advantage, one sample-mean denominator.
ROLLOUT_IDS = [0, 1, 1, 2]
ADVANTAGES = [0.5, -0.25, -0.25, 0.75]
OFF_POLICY_SAMPLE = 3


def _target_log_probs(logits: torch.Tensor, tokens: list[torch.Tensor]) -> list[torch.Tensor]:
    out, offset = [], 0
    for prompt_len, response_len, tok in zip(PROMPT_LENS, RESPONSE_LENS, tokens, strict=True):
        rows = logits[0, offset + prompt_len - 1 : offset + prompt_len + response_len - 1]
        out.append(rows.log_softmax(-1).gather(-1, tok[prompt_len:, None]).squeeze(-1))
        offset += prompt_len + response_len
    return out


def _policy_batch() -> tuple[torch.Tensor, dict]:
    g = torch.Generator().manual_seed(7)
    total_lens = [p + r for p, r in zip(PROMPT_LENS, RESPONSE_LENS, strict=True)]
    tokens = [torch.randint(0, VOCAB, (n,), generator=g) for n in total_lens]
    logits = torch.randn(1, sum(total_lens), VOCAB, generator=g)
    current = [x.detach() for x in _target_log_probs(logits, tokens)]
    rollout = [x + 1e-3 * torch.randn(x.shape, generator=g) for x in current]
    # The sampler claims mu = 0.9 for one token whose trainer probability is far lower.
    rollout[OFF_POLICY_SAMPLE][1] = math.log(0.9)
    loss_masks = [torch.ones(n, dtype=torch.int) for n in RESPONSE_LENS]
    loss_masks[0][2] = loss_masks[2][0] = 0  # e.g. tool-response tokens
    mask_sums = {rid: 0 for rid in ROLLOUT_IDS}
    for rid, mask in zip(ROLLOUT_IDS, loss_masks, strict=True):
        mask_sums[rid] += int(mask.sum())
    batch = dict(
        unconcat_tokens=tokens,
        total_lengths=total_lens,
        response_lengths=list(RESPONSE_LENS),
        loss_masks=loss_masks,
        rollout_mask_sums=torch.tensor([mask_sums[rid] for rid in ROLLOUT_IDS], dtype=torch.float32),
        rollout_log_probs=rollout,
        advantages=[torch.full((n,), a) for n, a in zip(RESPONSE_LENS, ADVANTAGES, strict=True)],
    )
    return logits, batch


def _reference_loss(logits: torch.Tensor, batch: dict, threshold: float) -> torch.Tensor:
    """Eq. (11) with Miles' per-rollout sample mean over B = 3 rollouts."""
    loss = logits.new_zeros(())
    log_probs = _target_log_probs(logits, batch["unconcat_tokens"])
    for i, log_prob in enumerate(log_probs):
        mu, mask = batch["rollout_log_probs"][i], batch["loss_masks"][i].float()
        with torch.no_grad():
            rho = (log_prob - mu).clamp(-30, 30).exp()
            p, q = mu.exp().clamp(1e-6, 1 - 1e-6), log_prob.exp().clamp(1e-6, 1 - 1e-6)
            kl = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
            keep = float((kl * mask).sum() / mask.sum() <= threshold)
        term = -batch["advantages"][i] * keep * rho * log_prob
        loss = loss + (term * mask).sum() / batch["rollout_mask_sums"][i]
    return loss / len(set(ROLLOUT_IDS))


def _run_loss(args, batch, logits):
    # make_args defaults to true_on_policy_mode, whose log_softmax log-probs are exact; the fused
    # vocab-parallel CE backward is only accurate to ~1e-3 and would blur the comparison.
    logits = logits.clone().requires_grad_()
    loss, _, log = loss_function(args, batch, 1, logits, num_rollouts=len(set(ROLLOUT_IDS)))
    (grad,) = torch.autograd.grad(loss, logits)
    metrics = dict(zip(log["keys"], log["values"][1:], strict=True))
    return loss.detach(), grad, metrics


@pytest.mark.parametrize("threshold", [5e-3, math.inf])
def test_policy_loss_matches_the_flash_reinforce_objective(threshold):
    make_parallel_state()
    args = _args(
        tis_binary_kl_threshold=threshold,
        skip_actor_forward_only=True,
        kl_coef=0.0,
        entropy_coef=0.0,
        observe_training_entropy=False,
    )
    logits, batch = _policy_batch()

    _, grad, metrics = _run_loss(args, batch, logits)

    reference_logits = logits.clone().requires_grad_()
    (expected,) = torch.autograd.grad(_reference_loss(reference_logits, batch, threshold), reference_logits)
    torch.testing.assert_close(grad, expected, atol=1e-6, rtol=1e-5)
    # The training forward is the old policy: the PPO ratio is exactly 1, so nothing clips.
    assert metrics["ppo_kl"].item() == 0.0
    assert metrics["pg_clipfrac"].item() == 0.0

    offset = sum(PROMPT_LENS[:OFF_POLICY_SAMPLE]) + sum(RESPONSE_LENS[:OFF_POLICY_SAMPLE])
    off_policy_grad = grad[0, offset : offset + PROMPT_LENS[OFF_POLICY_SAMPLE] + RESPONSE_LENS[OFF_POLICY_SAMPLE]]
    if threshold == math.inf:
        assert off_policy_grad.count_nonzero() > 0
        assert metrics["tis_seq_reject_frac"].item() == 0.0
    else:
        # Sample 3 is rejected: zero gradient, yet it still counts in B. The metric is a per-rollout
        # sum here; the logger divides it by the rollout count.
        assert off_policy_grad.count_nonzero() == 0
        torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.tensor(1.0))
        assert grad.count_nonzero() > 0


def test_rejection_leaves_entropy_and_ppo_metrics_alone():
    make_parallel_state()
    common = dict(
        skip_actor_forward_only=True,
        kl_coef=0.0,
        entropy_coef=0.0,
        observe_training_entropy=True,
    )
    logits, batch = _policy_batch()

    _, _, gated = _run_loss(_args(tis_binary_kl_threshold=5e-3, **common), batch, logits)
    _, _, open_gate = _run_loss(_args(tis_binary_kl_threshold=math.inf, **common), batch, logits)

    for key in ("entropy_loss", "ppo_kl", "pg_clipfrac", "tis", "tis_binary_kl"):
        torch.testing.assert_close(gated[key], open_gate[key])
    assert gated["pg_loss"] != open_gate["pg_loss"]


# ------------------------------ context parallel ------------------------------

# Sample 0 sits just inside delta (~4.06e-3 per token vs 5e-3), and rank 0 holds only its masked
# token, so a rank-local mean would flip it. Sample 1 (T=8, R=1) has its only response token on
# rank 0; rank 1 owns none of it but must still join the all-reduce. Sample 2 is far off-policy.
CP_TOTALS, CP_RESPONSES = [8, 8, 8], [3, 1, 5]


def _cp_inputs():
    g = torch.Generator().manual_seed(5)
    train = [-0.7 - torch.randn(n, generator=g).abs() for n in CP_RESPONSES]
    rollout = [x + 0.002 * torch.randn(x.shape, generator=g) for x in train]
    train[0], rollout[0] = torch.full((3,), math.log(0.5)), torch.full((3,), math.log(0.545))
    train[2][1], rollout[2][1] = math.log(0.1), math.log(0.9)
    masks = [
        torch.tensor([1, 1, 0], dtype=torch.int),
        torch.tensor([1], dtype=torch.int),
        torch.ones(5, dtype=torch.int),
    ]
    return train, rollout, masks


def _cp_parallel_state(rank: int, world_size: int) -> ParallelState:
    trivial = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD if world_size > 1 else None)
    state = ParallelState(
        intra_dp=trivial,
        intra_dp_cp=cp,
        cp=cp,
        tp=trivial,
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
    )
    set_parallel_state(state)
    return state


def _cp_worker(rank: int, world_size: int, port: int, *, qkv_format: str, recompute: bool) -> None:
    init_gloo(rank, world_size, port=port)
    args = _args(tis_binary_kl_threshold=5e-3, qkv_format=qkv_format)
    max_seq_lens = list(CP_TOTALS) if qkv_format == "bshd" else None
    train, rollout, masks = _cp_inputs()

    def gate(pg_loss, train_log_probs, rollout_log_probs, parallel_state):
        out, _, metrics = binary_kl_trust_region_function(
            args,
            pg_loss=pg_loss,
            train_log_probs=train_log_probs,
            rollout_log_probs=rollout_log_probs,
            loss_masks=masks,
            total_lengths=CP_TOTALS,
            response_lengths=CP_RESPONSES,
            parallel_state=parallel_state,
            max_seq_lens=max_seq_lens,
        )
        return out, metrics["tis_seq_reject_frac"]

    full_pg, full_reject = gate(torch.ones(sum(CP_RESPONSES)), train, rollout, _cp_parallel_state(0, 1))

    parallel_state = _cp_parallel_state(rank, world_size)

    def local(values: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            slice_log_prob_with_cp(x, t, r, qkv_format, max_seq_lens[i] if max_seq_lens else None)
            for i, (x, t, r) in enumerate(zip(values, CP_TOTALS, CP_RESPONSES, strict=True))
        ]

    expected_pg = torch.cat(local(list(full_pg.split(CP_RESPONSES))))
    expected_reject = torch.cat(local(list(full_reject.split(CP_RESPONSES))))
    assert expected_reject.sum() > 0  # the off-policy sequence 2 is rejected, and both ranks hold part of it

    local_train, local_rollout = local(train), local(rollout)
    pg_loss = torch.ones(sum(len(x) for x in local_train), requires_grad=True)
    if recompute:
        # --recompute-loss-function reruns the hook, and its all-reduce, in backward.
        gated, reject = checkpoint(gate, pg_loss, local_train, local_rollout, parallel_state, use_reentrant=False)
    else:
        gated, reject = gate(pg_loss, local_train, local_rollout, parallel_state)
    gated.sum().backward()

    torch.testing.assert_close(gated.detach(), expected_pg)
    torch.testing.assert_close(reject, expected_reject)
    torch.testing.assert_close(pg_loss.grad, expected_pg)
    dist.destroy_process_group()


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
def test_cp2_gate_matches_full_sequences(qkv_format: str, recompute: bool) -> None:
    run_multiprocess(partial(_cp_worker, qkv_format=qkv_format, recompute=recompute))
