"""Regression tests for KL and Tinker loss numerical stability in forward and backward."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.args_fixtures import ConfigNamespace

from miles.backends.training_utils.loss.hub import losses as loss_utils
from miles.backends.training_utils.loss.hub import response_layout as layout_utils
from miles.backends.training_utils.loss.hub import tinker_losses


def _make_policy_args(*, kl_loss_type: str, use_unbiased_kl: bool, kl_loss_coef: float = 0.5) -> Namespace:
    return ConfigNamespace(
        use_rollout_logprobs=False,
        use_sampling_support_replay=False,
        skip_actor_forward_only=False,
        use_opsm=False,
        advantage_estimator="ppo",
        get_mismatch_metrics=False,
        use_tis=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        dump_details=None,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        backend=Namespace(calculate_per_token_loss=False),
        qkv_format="thd",
        entropy_coef=0.0,
        use_kl_loss=True,
        use_unbiased_kl=use_unbiased_kl,
        kl_loss_type=kl_loss_type,
        kl_loss_coef=kl_loss_coef,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        true_on_policy_mode=False,
        allgather_cp=False,
        observe_training_entropy=False,
    )


def _patch_policy_loss_single_rank(monkeypatch: pytest.MonkeyPatch, *, log_probs_fn) -> None:
    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None), cp=SimpleNamespace(size=1)),
    )
    monkeypatch.setattr(
        layout_utils,
        "get_local_response_loss_masks",
        lambda total_lengths, response_lengths, loss_masks, qkv_format="thd", max_seq_lens=None: loss_masks,
    )
    monkeypatch.setattr(
        layout_utils,
        "compute_ess_ratio_contribution",
        lambda *, ppo_kl, **kwargs: ppo_kl.new_tensor(1.0),
    )
    monkeypatch.setattr(layout_utils, "get_log_probs_and_entropy", log_probs_fn)
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )


def _exact_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    *,
    kl_loss_type: str,
    use_unbiased_kl: bool,
) -> torch.Tensor:
    diff = log_probs - ref_log_probs
    if kl_loss_type == "k1":
        kl = diff
    elif kl_loss_type == "k2":
        kl = diff.square() / 2.0
    else:
        neg = -diff
        kl = neg.exp() - 1.0 - neg
    if use_unbiased_kl:
        kl = (log_probs - old_log_probs).exp() * kl
    if kl_loss_type == "low_var_kl":
        kl = kl.clamp(-10.0, 10.0)
    return kl


@pytest.mark.parametrize("kl_loss_type", ["k1", "k2", "k3", "low_var_kl"])
@pytest.mark.parametrize("use_unbiased_kl", [False, True])
def test_policy_loss_kl_extreme_and_nonfinite_inputs_keep_gradients_finite(
    monkeypatch: pytest.MonkeyPatch,
    kl_loss_type: str,
    use_unbiased_kl: bool,
) -> None:
    """Extreme log-ratios and -inf old/ref log-probs must not produce NaN gradients."""
    args = _make_policy_args(kl_loss_type=kl_loss_type, use_unbiased_kl=use_unbiased_kl, kl_loss_coef=0.5)
    raw_lp = torch.tensor([-100.0, -1.0, -0.5], dtype=torch.float32, requires_grad=True)
    old_lp = torch.tensor([-200.0, -1.1, float("-inf")], dtype=torch.float32)
    ref_lp = torch.tensor([0.0, -0.8, float("-inf")], dtype=torch.float32)

    batch = {
        "advantages": [torch.zeros(3, dtype=torch.float32)],
        "log_probs": [old_lp],
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4], dtype=torch.long)],
        "response_lengths": [3],
        "total_lengths": [4],
        "loss_masks": [torch.ones(3, dtype=torch.float32)],
        "ref_log_probs": [ref_lp],
    }

    _patch_policy_loss_single_rank(
        monkeypatch,
        log_probs_fn=lambda logits, *a, **kw: {"log_probs": [logits.squeeze(0)[:, 0]]},
    )

    logits = raw_lp.view(1, 3, 1)
    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=logits,
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["kl_loss"])
    assert raw_lp.grad is not None
    assert torch.isfinite(raw_lp.grad).all(), f"Non-finite grad: {raw_lp.grad}"

    # Token 1 is completely valid and must retain its exact analytical KL gradient.
    ref_token = torch.tensor([-1.0], dtype=torch.float32, requires_grad=True)
    expected_kl = _exact_kl(
        ref_token,
        torch.tensor([-0.8]),
        torch.tensor([-1.1]),
        kl_loss_type=kl_loss_type,
        use_unbiased_kl=use_unbiased_kl,
    )
    (0.5 * expected_kl.sum() / 3.0).backward()
    torch.testing.assert_close(raw_lp.grad[1:2], ref_token.grad, atol=1e-6, rtol=1e-6)
    # Token 2 has -inf ref/old log-probs and must contribute zero gradient.
    assert raw_lp.grad[2].item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("kl_loss_type", ["k1", "k2", "k3", "low_var_kl"])
@pytest.mark.parametrize("use_unbiased_kl", [False, True])
def test_policy_loss_kl_normal_inputs_match_exact_formula(
    monkeypatch: pytest.MonkeyPatch,
    kl_loss_type: str,
    use_unbiased_kl: bool,
) -> None:
    """In-range finite inputs must match the exact KL value and gradient."""
    args = _make_policy_args(kl_loss_type=kl_loss_type, use_unbiased_kl=use_unbiased_kl, kl_loss_coef=0.4)
    raw_lp = torch.tensor([-1.2, -0.7, -2.1], dtype=torch.float32, requires_grad=True)
    old_lp = torch.tensor([-1.0, -0.9, -1.8], dtype=torch.float32)
    ref_lp = torch.tensor([-1.5, -0.5, -2.5], dtype=torch.float32)

    batch = {
        "advantages": [torch.zeros(3, dtype=torch.float32)],
        "log_probs": [old_lp],
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4], dtype=torch.long)],
        "response_lengths": [3],
        "total_lengths": [4],
        "loss_masks": [torch.ones(3, dtype=torch.float32)],
        "ref_log_probs": [ref_lp],
    }

    _patch_policy_loss_single_rank(
        monkeypatch,
        log_probs_fn=lambda logits, *a, **kw: {"log_probs": [logits.squeeze(0)[:, 0]]},
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=raw_lp.view(1, 3, 1),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )
    loss.backward()

    lp_ref = raw_lp.detach().clone().requires_grad_(True)
    expected_kl = _exact_kl(
        lp_ref,
        ref_lp,
        old_lp,
        kl_loss_type=kl_loss_type,
        use_unbiased_kl=use_unbiased_kl,
    )
    expected_loss = 0.4 * expected_kl.mean()
    expected_loss.backward()

    torch.testing.assert_close(metrics["kl_loss"], expected_kl.mean().detach(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(loss, expected_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(raw_lp.grad, lp_ref.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("loss_name", ["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"])
def test_tinker_losses_masked_extreme_and_nonfinite_inputs_keep_loss_and_grads_finite(
    monkeypatch: pytest.MonkeyPatch,
    loss_name: str,
) -> None:
    """Masked (mask == 0) tokens with extreme or -inf inputs must not produce NaN loss or gradients."""
    args = Namespace(qkv_format="thd")
    raw_lp = torch.tensor([-1.0, -5.0, -100.0], dtype=torch.float32, requires_grad=True)
    sampling_lp = torch.tensor([-1.2, -200.0, float("-inf")], dtype=torch.float32)
    advantages = torch.tensor([1.5, 0.0, float("inf")], dtype=torch.float32)
    loss_weights = torch.tensor([2.0, 0.0, float("nan")], dtype=torch.float32)
    loss_mask = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    batch = {
        "unconcat_tokens": [torch.tensor([0, 1, 2, 3], dtype=torch.long)],
        "target_tokens": [torch.tensor([1, 2, 3], dtype=torch.long)],
        "total_lengths": [4],
        "response_lengths": [3],
        "loss_masks": [loss_mask],
        "loss_weights": [loss_weights],
        "rollout_log_probs": [sampling_lp],
        "advantages": [advantages],
        "sample_indices": [0],
    }

    monkeypatch.setattr(
        tinker_losses,
        "get_parallel_state",
        lambda: SimpleNamespace(cp=SimpleNamespace(size=1)),
    )
    monkeypatch.setattr(
        tinker_losses,
        "get_log_probs_and_entropy",
        lambda logits, **kw: {"log_probs": [logits.squeeze(0)[:, 0]]},
    )
    monkeypatch.setattr(
        tinker_losses,
        "get_local_response_loss_masks",
        lambda total_lengths, response_lengths, loss_masks, qkv_format="thd", max_seq_lens=None: loss_masks,
    )
    monkeypatch.setattr(
        tinker_losses,
        "slice_log_prob_with_cp",
        lambda values, *a, **kw: values,
    )

    loss_fn = tinker_losses.TINKER_LOSS_FUNCTIONS[loss_name]
    loss, outputs = loss_fn(args, batch, raw_lp.view(1, 3, 1), lambda x: x.sum())
    loss.backward()

    assert torch.isfinite(loss), f"Non-finite loss for {loss_name}: {loss}"
    assert raw_lp.grad is not None
    assert torch.isfinite(raw_lp.grad).all(), f"Non-finite grad for {loss_name}: {raw_lp.grad}"
    assert raw_lp.grad[1].item() == pytest.approx(0.0, abs=1e-7)
    assert raw_lp.grad[2].item() == pytest.approx(0.0, abs=1e-7)
    # Per-datum reported logprobs remain unmasked.
    torch.testing.assert_close(outputs["per_datum"][0]["logprobs"], raw_lp.detach(), atol=1e-6, rtol=1e-6)

    # Active token 0 must match the exact unmasked closed-form loss and gradient.
    lp0 = torch.tensor(-1.0, dtype=torch.float32, requires_grad=True)
    slp0 = torch.tensor(-1.2, dtype=torch.float32)
    adv0 = torch.tensor(1.5, dtype=torch.float32)
    if loss_name == "cross_entropy":
        ref = -(torch.tensor(2.0) * lp0)
    elif loss_name == "importance_sampling":
        ref = -(torch.exp(lp0 - slp0) * adv0)
    elif loss_name == "ppo":
        r = torch.exp(lp0 - slp0)
        ref = -torch.minimum(r * adv0, torch.clamp(r, 0.8, 1.2) * adv0)
    elif loss_name == "cispo":
        c = torch.clamp(torch.exp(lp0 - slp0), 0.0, 4.0).detach()
        ref = -(c * lp0 * adv0)
    else:  # dro
        ref = -(lp0 * adv0 - 0.5 * 0.05 * (lp0 - slp0) ** 2)
    ref.backward()

    torch.testing.assert_close(loss, ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(raw_lp.grad[0], lp0.grad, atol=1e-6, rtol=1e-6)
