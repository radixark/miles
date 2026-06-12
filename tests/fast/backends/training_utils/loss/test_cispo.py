"""Unit tests for the CISPO advantage-estimator surrogate.

CISPO (MiniMax-M1, https://arxiv.org/abs/2506.13585) replaces PPO's ``min`` of
two ratio-weighted terms with an *additive* REINFORCE-style surrogate whose
importance-sampling weight is clipped and then stop-gradiented:

    L = -clip(ratio, 1-eps_clip, 1+eps_clip_high).detach() * A * log_probs

These tests cover the closed-form loss/clipfrac values and the load-bearing
invariant: gradient flows ONLY through ``log_probs`` (the IS weight is detached),
without needing the external loss snapshot artifacts.
"""

import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.backends.training_utils.loss_hub.math_utils import compute_cispo_loss

from .loss_test_utils import make_args, make_batch, make_inputs, make_parallel_state

# This module intentionally has no explicit CI registration call: modules under
# tests/fast with no register call are implicitly assigned to the stage-a-cpu
# suite by collect_tests, so no explicit registration is needed for this to run
# in CI.


def test_unclipped_matches_closed_form_reinforce_surrogate():
    # ratio = exp(-ppo_kl) = 1 (on-policy); clip is a no-op, so
    # pg_loss = -ratio * adv * log_probs = -1 * adv * log_probs.
    ppo_kl = torch.zeros(3)
    advantages = torch.tensor([2.0, -1.0, 0.5])
    log_probs = torch.tensor([-0.1, -0.2, -0.3])

    pg_loss, clipfrac = compute_cispo_loss(ppo_kl, advantages, log_probs, eps_clip=0.2, eps_clip_high=0.2)

    assert torch.allclose(pg_loss, -advantages * log_probs)
    # nothing clipped at ratio == 1
    assert torch.allclose(clipfrac, torch.zeros(3))


def test_is_weight_is_clipped():
    # ppo_kl = -ln(2) => ratio = 2, above 1 + eps_clip_high = 1.2 -> clamped to 1.2.
    # ppo_kl = -ln(0.5) => ratio = 0.5, below 1 - eps_clip = 0.8 -> clamped to 0.8.
    ratios = torch.tensor([2.0, 0.5, 1.0])
    ppo_kl = -ratios.log()
    advantages = torch.tensor([1.0, 1.0, 1.0])
    log_probs = torch.tensor([1.0, 1.0, 1.0])

    pg_loss, clipfrac = compute_cispo_loss(ppo_kl, advantages, log_probs, eps_clip=0.2, eps_clip_high=0.2)

    expected_clipped = torch.tensor([1.2, 0.8, 1.0])
    assert torch.allclose(pg_loss, -expected_clipped * advantages * log_probs)
    # first two tokens clipped, the on-policy one is not
    assert torch.allclose(clipfrac, torch.tensor([1.0, 1.0, 0.0]))


def test_gradient_flows_only_through_log_probs():
    # The defining CISPO property: the stop-gradient on the clipped IS weight means
    # backprop reaches `log_probs` but NOT `ppo_kl` / the IS ratio.
    ppo_kl = torch.tensor([0.3, -0.4, 0.0], requires_grad=True)
    advantages = torch.tensor([2.0, -1.0, 0.5])
    log_probs = torch.tensor([-0.1, -0.2, -0.3], requires_grad=True)

    pg_loss, _ = compute_cispo_loss(ppo_kl, advantages, log_probs, eps_clip=0.2, eps_clip_high=0.2)
    pg_loss.sum().backward()

    # gradient does NOT flow through the importance-sampling weight
    assert ppo_kl.grad is None or torch.allclose(ppo_kl.grad, torch.zeros_like(ppo_kl))
    # gradient DOES flow through the current-policy log_probs
    assert log_probs.grad is not None
    # d/d log_probs [ -clip(ratio) * adv * log_probs ] = -clip(ratio) * adv
    ratio = (-ppo_kl.detach()).exp()
    clipped = ratio.clamp(0.8, 1.2)
    assert torch.allclose(log_probs.grad, -clipped * advantages)


def test_inf_log_prob_at_masked_token_does_not_poison_loss():
    # Unlike PPO/GSPO (which only consume the sanitized ppo_kl), CISPO multiplies
    # raw current-policy log_probs into the loss. An inf at a loss-masked token
    # must be sanitized before the masked reduction, where 0 * inf = NaN.
    make_parallel_state()
    args = make_args(advantage_estimator="cispo", entropy_coef=0.0)
    inputs = make_inputs(seed=0, batch_size=1, prompt_lens=[4], response_lens=[4], vocab_size=16, args=args)

    inputs["loss_masks"][0][2] = 0.0
    # response token 2 is predicted from logits row total_len - response_len - 1 + 2 = 5;
    # -inf at its target logit makes the current log_prob -inf at that token.
    target_token = inputs["unconcat_tokens"][0][4 + 2]
    inputs["policy_logits"][0, 5, target_token] = float("-inf")

    batch = make_batch(inputs, "policy_loss")
    logits = inputs["policy_logits"].requires_grad_(True)
    som = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"])

    loss, _ = policy_loss_function(args, batch, logits, som)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_returns_match_compute_policy_loss_shapes():
    # CISPO must return the same (per_token_loss, per_token_clipfrac) shape as
    # compute_policy_loss so the caller's reduction / metric plumbing is reused.
    ppo_kl = torch.randn(5)
    advantages = torch.randn(5)
    log_probs = torch.randn(5)

    pg_loss, clipfrac = compute_cispo_loss(ppo_kl, advantages, log_probs, eps_clip=0.2, eps_clip_high=0.3)

    assert pg_loss.shape == ppo_kl.shape
    assert clipfrac.shape == ppo_kl.shape
    assert clipfrac.dtype == torch.float32
