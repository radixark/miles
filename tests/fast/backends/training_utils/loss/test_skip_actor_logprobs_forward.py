"""Tests for --skip-actor-logprobs-forward.

With the flag, the actor's standalone forward-only log-probs pass is skipped:
`policy_loss_function` uses the training forward's own detached log probs as
the old-policy log probs (the importance sampling ratio is identically 1), and
`compute_advantages_and_returns` builds the zero KL from the response loss
masks because rollout_data carries no "log_probs".
"""

from __future__ import annotations

import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.losses import policy_loss_function

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state, make_rollout_data

SEED = 1234


def _make_inputs(args):
    return make_inputs(
        seed=SEED, batch_size=3, prompt_lens=[20, 64, 40], response_lens=[10, 48, 32], vocab_size=128, args=args
    )


def _run_policy_loss(args, batch, inputs):
    logits = deep_clone(inputs["policy_logits"])
    logits.requires_grad_(True)
    som = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )
    loss, metrics = policy_loss_function(args, batch, logits, som)
    loss.backward()
    return loss.detach(), metrics, logits.grad.clone()


def test_policy_loss_matches_explicit_on_policy_old_log_probs():
    make_parallel_state()
    args_skip = make_args(skip_actor_logprobs_forward=True, kl_coef=0.0)
    inputs = _make_inputs(args_skip)

    batch_skip = make_batch(inputs, "policy_loss")
    del batch_skip["log_probs"]  # the forward-only pass never ran
    loss_skip, metrics_skip, grad_skip = _run_policy_loss(args_skip, batch_skip, inputs)

    # Baseline: what the recomputed old log probs equal in a single-step
    # on-policy run — the training forward's own log probs.
    args_base = make_args(kl_coef=0.0)
    train_log_probs = get_log_probs_and_entropy(
        deep_clone(inputs["policy_logits"]),
        args=args_base,
        unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
        total_lengths=list(inputs["total_lens"]),
        response_lengths=list(inputs["response_lens"]),
        with_entropy=False,
    )["log_probs"]
    batch_base = make_batch(inputs, "policy_loss")
    batch_base["log_probs"] = [x.detach() for x in train_log_probs]
    loss_base, metrics_base, grad_base = _run_policy_loss(args_base, batch_base, inputs)

    assert torch.equal(loss_skip, loss_base)
    assert torch.equal(grad_skip, grad_base)
    assert metrics_skip["ppo_kl"].item() == 0.0
    assert metrics_skip["pg_clipfrac"].item() == 0.0
    assert metrics_base["ppo_kl"].item() == 0.0


def test_advantages_without_actor_log_probs_match_baseline():
    make_parallel_state()
    args_skip = make_args(skip_actor_logprobs_forward=True, kl_coef=0.0)
    inputs = _make_inputs(args_skip)

    rollout_skip = make_rollout_data(inputs)
    for key in ("log_probs", "ref_log_probs", "values"):
        del rollout_skip[key]
    compute_advantages_and_returns(args_skip, rollout_skip)

    rollout_base = make_rollout_data(inputs)
    del rollout_base["values"]
    compute_advantages_and_returns(make_args(kl_coef=0.0), rollout_base)

    for a, b in zip(rollout_skip["advantages"], rollout_base["advantages"], strict=True):
        assert a.shape == b.shape
        assert torch.equal(a, b)
    for a, b in zip(rollout_skip["returns"], rollout_base["returns"], strict=True):
        assert torch.equal(a, b)


def test_skip_does_not_compute_on_intermediate_pp_stage():
    parallel_state = make_parallel_state()
    parallel_state.is_pp_last_stage = False
    try:
        args_skip = make_args(skip_actor_logprobs_forward=True, kl_coef=0.0)
        inputs = _make_inputs(args_skip)
        rollout_data = make_rollout_data(inputs)
        for key in ("log_probs", "ref_log_probs", "values"):
            del rollout_data[key]
        compute_advantages_and_returns(args_skip, rollout_data)
        assert "advantages" not in rollout_data
        assert "returns" not in rollout_data
    finally:
        make_parallel_state()
