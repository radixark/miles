"""Unit/integration tests for the ``reinforce`` advantage estimator.

``reinforce`` uses GRPO-style group-normalized advantages with the plain additive
surrogate ``-A * log pi_theta`` (no PPO/IS ratio, no clipping). Mirrors the
targeted-test style of ``test_opd.py`` (closed-form values + invariants through the
full ``policy_loss_function`` path) rather than the snapshot harness (whose artifacts
live in an external repo).
"""

import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.backends.training_utils.loss_hub.math_utils import compute_reinforce_loss

from .loss_test_utils import make_args, make_batch, make_inputs, make_parallel_state, make_rollout_data

# Modules under tests/fast with no explicit CI register call are implicitly
# assigned to the stage-a-cpu suite by collect_tests, so none is needed here.


def test_reinforce_loss_matches_closed_form():
    advantages = torch.tensor([2.0, -1.0, 0.5])
    log_probs = torch.tensor([-0.1, -0.2, -0.3])

    pg_loss, clipfrac = compute_reinforce_loss(advantages, log_probs)

    assert torch.allclose(pg_loss, -advantages * log_probs)
    assert torch.allclose(clipfrac, torch.zeros(3))


def test_reinforce_gradient_flows_only_through_log_probs():
    advantages = torch.tensor([2.0, -1.0, 0.5])
    log_probs = torch.tensor([-0.1, -0.2, -0.3], requires_grad=True)

    pg_loss, _ = compute_reinforce_loss(advantages, log_probs)
    pg_loss.sum().backward()

    # d/d log_probs [ -A * log_probs ] = -A
    assert torch.allclose(log_probs.grad, -advantages)


def test_reinforce_advantages_are_reward_broadcast():
    # `reinforce` reuses the GRPO returns path: the (group-normalized, upstream)
    # scalar reward is broadcast to every response token, and advantages == returns.
    make_parallel_state()
    args = make_args(advantage_estimator="reinforce", kl_coef=0.0)
    inputs = make_inputs(seed=3, batch_size=2, prompt_lens=[6, 4], response_lens=[5, 7], vocab_size=32, args=args)
    rollout_data = make_rollout_data(inputs)

    compute_advantages_and_returns(args, rollout_data)

    for i, reward in enumerate(inputs["rewards"]):
        expected = torch.full((inputs["response_lens"][i],), reward)
        assert torch.allclose(rollout_data["returns"][i], expected)
        assert torch.allclose(rollout_data["advantages"][i], rollout_data["returns"][i])


def test_reinforce_full_path_is_linear_in_advantages_and_zero_at_zero():
    # Integration property: with no IS ratio and entropy off, the reduced pg_loss is
    # linear in the advantages (loss = mean(-A * log pi)), and zero when A == 0.
    make_parallel_state()
    args = make_args(advantage_estimator="reinforce", entropy_coef=0.0)
    inputs = make_inputs(seed=2, batch_size=2, prompt_lens=[8, 5], response_lens=[6, 4], vocab_size=32, args=args)

    def run(scale: float) -> torch.Tensor:
        batch = make_batch(inputs, "policy_loss")
        batch["advantages"] = [scale * a for a in batch["advantages"]]
        logits = inputs["policy_logits"].clone().requires_grad_(True)
        som = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"])
        loss, metrics = policy_loss_function(args, batch, logits, som)
        return metrics["pg_loss"]

    base = run(1.0)
    assert torch.allclose(run(0.0), torch.zeros(()))
    assert torch.allclose(run(2.0), 2.0 * base, atol=1e-5)


def test_reinforce_full_path_masked_inf_logprob_does_not_poison_loss():
    # REINFORCE multiplies raw current-policy log_probs into the loss; an inf at a
    # loss-masked token must be sanitized before the masked reduction (0 * inf = NaN).
    make_parallel_state()
    args = make_args(advantage_estimator="reinforce", entropy_coef=0.0)
    inputs = make_inputs(seed=0, batch_size=1, prompt_lens=[4], response_lens=[4], vocab_size=16, args=args)

    inputs["loss_masks"][0][2] = 0.0
    # response token 2 is predicted from logits row total_len - response_len - 1 + 2 = 5.
    target_token = inputs["unconcat_tokens"][0][4 + 2]
    inputs["policy_logits"][0, 5, target_token] = float("-inf")

    batch = make_batch(inputs, "policy_loss")
    logits = inputs["policy_logits"].requires_grad_(True)
    som = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"])

    loss, metrics = policy_loss_function(args, batch, logits, som)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    # REINFORCE never clips
    assert torch.allclose(metrics["pg_clipfrac"], torch.zeros(()))
