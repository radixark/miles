"""Tests for the ``off_policy_is_function`` importance-sampling correction and the
``cur_log_probs`` hook wiring that enables it.

``off_policy_is_function`` is truncated IS between the *current* policy and the
*actual rollout generator*: the (detached) weight is ``clip(pi_theta / pi_rollout)``.
On a plain REINFORCE base ``-A * log pi`` it reproduces the CISPO surrogate
(https://arxiv.org/abs/2506.13585). The correction needs the current grad-carrying
log-probs, which the policy-loss TIS hook now passes as ``cur_log_probs``.
"""

from argparse import Namespace

import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub.corrections import off_policy_is_function, vanilla_tis_function
from miles.backends.training_utils.loss_hub.losses import policy_loss_function

from .loss_test_utils import make_args, make_batch, make_inputs, make_parallel_state

# Modules under tests/fast with no explicit CI register call are implicitly
# assigned to the stage-a-cpu suite by collect_tests, so none is needed here.


def test_off_policy_is_function_clips_weight_and_passes_masks_through():
    # ratio = exp(cur - rollout); clip to [1-eps_clip, 1+eps_clip_high].
    # cur - rollout = ln(2) -> ratio 2 -> clamped to 1.2; = ln(0.5) -> 0.5 -> 0.8; = 0 -> 1.0.
    cur = torch.tensor([1.0, 1.0, 1.0])
    rollout = cur - torch.tensor([2.0, 0.5, 1.0]).log()
    pg_loss = torch.tensor([1.0, 1.0, 1.0])
    loss_masks = [torch.ones(3)]
    args = Namespace(eps_clip=0.2, eps_clip_high=0.2)

    out_loss, out_masks, metrics = off_policy_is_function(
        args, pg_loss=pg_loss, cur_log_probs=[cur], rollout_log_probs=[rollout], loss_masks=loss_masks
    )

    expected_w = torch.tensor([1.2, 0.8, 1.0])
    assert torch.allclose(out_loss, pg_loss * expected_w)
    assert torch.allclose(metrics["is_clipfrac"], torch.tensor([1.0, 1.0, 0.0]))
    assert out_masks is loss_masks  # no rejection-sampling masking


def test_off_policy_is_on_reinforce_base_equals_cispo_surrogate():
    # On a plain REINFORCE base (-A * log pi), off_policy_is_function reproduces the
    # CISPO surrogate exactly, with gradient flowing ONLY through log_probs.
    advantages = torch.tensor([2.0, -1.0, 0.5, 1.5])
    rollout = torch.tensor([-0.5, -0.2, -0.9, -0.3])  # behavior policy mu (frozen)
    log_probs = torch.tensor([-0.1, -0.4, -0.3, -0.8], requires_grad=True)
    args = Namespace(eps_clip=0.2, eps_clip_high=0.2)

    pg_loss = -advantages * log_probs  # plain REINFORCE base
    pg_loss, _, _ = off_policy_is_function(
        args, pg_loss=pg_loss, cur_log_probs=[log_probs], rollout_log_probs=[rollout], loss_masks=[torch.ones(4)]
    )

    ratio = torch.exp(log_probs.detach() - rollout)  # pi_theta / pi_rollout
    clipped = ratio.clamp(1 - args.eps_clip, 1 + args.eps_clip_high)
    assert torch.allclose(pg_loss, -clipped * advantages * log_probs.detach())

    pg_loss.sum().backward()
    # d/d log_probs [ -clip(ratio).detach() * A * log_probs ] = -clip(ratio) * A
    assert torch.allclose(log_probs.grad, -clipped * advantages)


def test_off_policy_is_single_sided_when_eps_clip_one():
    # Canonical CISPO: eps_clip=1.0 disables the lower bound (ratio >= 0 never clipped low).
    cur = torch.tensor([0.0, 0.0])
    rollout = cur - torch.tensor([10.0, 0.01]).log()  # ratios 10.0 (high) and ~0.01 (very low)
    pg_loss = torch.tensor([1.0, 1.0])
    args = Namespace(eps_clip=1.0, eps_clip_high=4.0)

    _, _, metrics = off_policy_is_function(
        args, pg_loss=pg_loss, cur_log_probs=[cur], rollout_log_probs=[rollout], loss_masks=[torch.ones(2)]
    )

    # high ratio 10.0 > 1+eps_clip_high=5.0 clipped; low ratio ~0.01 >= 1-eps_clip=0.0 NOT clipped
    assert torch.allclose(metrics["is_clipfrac"], torch.tensor([1.0, 0.0]))


def test_hook_passes_cur_log_probs_to_correction():
    # The policy-loss TIS hook must forward the current grad-carrying log-probs as
    # cur_log_probs. Wiring smoke test through the full path (grpo base): the
    # correction runs and reports its metrics. (reinforce + off_policy_is = CISPO is
    # covered by the unit equivalence test above.)
    make_parallel_state()
    args = make_args(
        advantage_estimator="grpo",
        entropy_coef=0.0,
        use_tis=True,
        custom_tis_function_path="miles.backends.training_utils.loss_hub.corrections.off_policy_is_function",
    )
    inputs = make_inputs(seed=1, batch_size=2, prompt_lens=[6, 4], response_lens=[5, 7], vocab_size=32, args=args)

    batch = make_batch(inputs, "policy_loss")
    logits = inputs["policy_logits"].requires_grad_(True)
    som = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"])

    loss, metrics = policy_loss_function(args, batch, logits, som)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert "is_weight" in metrics and "is_clipfrac" in metrics


def test_vanilla_tis_ignores_cur_log_probs_kwarg():
    # The hook now also passes cur_log_probs; existing corrections take **kwargs and
    # must be unaffected (vanilla TIS uses train_log_probs / rollout_log_probs only).
    train = [torch.tensor([-0.1, -0.2, -0.3])]
    rollout = [torch.tensor([-0.2, -0.2, -0.5])]
    cur = [torch.tensor([-0.05, -0.25, -0.35])]  # should be ignored
    pg_loss = torch.tensor([1.0, 1.0, 1.0])
    args = Namespace(tis_clip=1.5, tis_clip_low=0.5)

    with_cur, _, m_with = vanilla_tis_function(
        args,
        pg_loss=pg_loss.clone(),
        train_log_probs=train,
        rollout_log_probs=rollout,
        loss_masks=[torch.ones(3)],
        cur_log_probs=cur,
    )
    without_cur, _, m_without = vanilla_tis_function(
        args,
        pg_loss=pg_loss.clone(),
        train_log_probs=train,
        rollout_log_probs=rollout,
        loss_masks=[torch.ones(3)],
    )

    assert torch.allclose(with_cur, without_cur)
    assert torch.allclose(m_with["tis"], m_without["tis"])
