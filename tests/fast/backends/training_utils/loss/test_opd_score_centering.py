"""Tests for how score centering composes with on-policy distillation (OPD).

`--opd-score-centering-mode=rl-only` centers only the RL advantage and adds the
OPD reverse-KL term back in as an uncentered policy-gradient contribution,
instead of centering the combined RL+OPD advantage as one term (the
`combined`, prior-behavior default). These tests exercise `policy_loss_function`
end to end so the composition in `losses.py` is verified against real score
centering + PPO-clip math, not just the advantage bookkeeping in `loss.py`.
"""

import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.loss_hub.losses import policy_loss_function

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state, make_rollout_data

# This module intentionally has no explicit CI registration call: modules under
# tests/fast are implicitly assigned to the stage-a-cpu suite by the CI collector
# (an explicit default-form call would be rejected by the AC-9 meta-test).

_VOCAB_SIZE = 8
_TOP_K = 3


def _fake_top_logprobs(response_lens: list[int]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build sampler top-k head data whose head mass is well below 1."""
    ids = torch.arange(_TOP_K, dtype=torch.long)
    logprobs = torch.tensor([-1.0, -1.5, -2.0], dtype=torch.float32)
    top_ids = [ids.unsqueeze(0).expand(rl, _TOP_K).clone() for rl in response_lens]
    top_logprobs = [logprobs.unsqueeze(0).expand(rl, _TOP_K).clone() for rl in response_lens]
    return top_ids, top_logprobs


def _build_batch(inputs: dict, args, top_ids, top_logprobs) -> dict:
    rollout_data = make_rollout_data(inputs)
    # OPD compares teacher log-probs against response-length student log-probs
    # (miles/backends/training_utils/loss_hub/opd.py enforces matching shapes);
    # `make_rollout_data`'s teacher_log_probs are prompt+response-length, sized
    # for other call sites, so build response-length ones here instead.
    g = torch.Generator().manual_seed(inputs["seed"] + 1)
    rollout_data["teacher_log_probs"] = [
        torch.randn(rl, generator=g, dtype=torch.float32) * 2 - 3 for rl in inputs["response_lens"]
    ]
    compute_advantages_and_returns(args, rollout_data)

    batch = make_batch(inputs, "policy_loss")
    batch["advantages"] = deep_clone(rollout_data["advantages"])
    batch["opd_reverse_kl"] = deep_clone(rollout_data["opd_reverse_kl"])
    if "rl_only_advantages" in rollout_data:
        batch["rl_only_advantages"] = deep_clone(rollout_data["rl_only_advantages"])
    batch["rollout_top_logprob_ids"] = deep_clone(top_ids)
    batch["rollout_top_logprobs"] = deep_clone(top_logprobs)
    return batch


def _run_policy_loss(args, batch, inputs):
    logits = deep_clone(inputs["policy_logits"]).requires_grad_(True)
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    return policy_loss_function(args, batch, logits, reducer)


def _make_case(opd_kl_coef: float, opd_score_centering_mode: str):
    make_parallel_state()
    common = dict(
        advantage_estimator="grpo",
        use_opd=True,
        opd_type="sglang",
        opd_kl_coef=opd_kl_coef,
        use_score_centering=True,
        score_centering_top_k=_TOP_K,
        normalize_advantages=False,
    )
    args = make_args(opd_score_centering_mode=opd_score_centering_mode, **common)
    inputs = make_inputs(
        seed=7, batch_size=2, prompt_lens=[2, 3], response_lens=[3, 4], vocab_size=_VOCAB_SIZE, args=args
    )
    top_ids, top_logprobs = _fake_top_logprobs(inputs["response_lens"])
    batch = _build_batch(inputs, args, top_ids, top_logprobs)
    loss, metrics = _run_policy_loss(args, batch, inputs)
    return loss, metrics


def test_rl_only_matches_combined_when_opd_contributes_nothing():
    # With opd_kl_coef=0, the OPD reverse-KL penalty is scaled to exactly zero,
    # so RL-only advantages equal the combined advantages and the extra
    # uncentered OPD policy-gradient term is exactly zero. The two composition
    # modes must therefore produce identical losses.
    loss_combined, metrics_combined = _make_case(opd_kl_coef=0.0, opd_score_centering_mode="combined")
    loss_rl_only, metrics_rl_only = _make_case(opd_kl_coef=0.0, opd_score_centering_mode="rl-only")

    torch.testing.assert_close(loss_rl_only, loss_combined)
    torch.testing.assert_close(metrics_rl_only["pg_loss"], metrics_combined["pg_loss"])


def test_rl_only_diverges_from_combined_when_opd_contributes():
    # With a nonzero coefficient the two modes route the OPD term through
    # different math (centered vs. plain PPO-clipped surrogate), so they must
    # not silently collapse to the same computation.
    loss_combined, _ = _make_case(opd_kl_coef=1.0, opd_score_centering_mode="combined")
    loss_rl_only, _ = _make_case(opd_kl_coef=1.0, opd_score_centering_mode="rl-only")

    assert not torch.allclose(loss_rl_only, loss_combined)
