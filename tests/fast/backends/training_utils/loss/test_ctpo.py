"""CTPO: raw prefix-ratio importance weight with a sqrt(t)-scaled clip band.

The importance weight is the prefix product rho_{0:t} = prod_{i<=t} rho_i, where
rho_i = pi_theta(a_i|s_i) / mu(a_i|s_i) and mu is the inference engine. miles
carries ppo_kl = old - new and forms ratio = exp(-ppo_kl), so the prefix weight is
produced by making ppo_kl a running sum rather than a per-token term.

log rho_{0:t} is a t-term random walk, so its spread grows as sqrt(t); the band
widens the same way, holding the clip rate roughly constant along the response
instead of clipping nearly everything at large t.
"""

import math

import pytest
import torch

from miles.backends.training_utils.loss_hub.math_utils import (
    compute_ctpo_clip_band,
    compute_ctpo_prefix_kl,
)


def _kl(old: list[float], new: list[float]) -> torch.Tensor:
    return torch.tensor(old, dtype=torch.float32), torch.tensor(new, dtype=torch.float32)


class TestPrefixRatio:
    def test_first_token_weight_is_the_local_ratio(self) -> None:
        """rho_{0:0} == rho_0: the prefix over one factor is that factor."""
        old, new = _kl([0.0, 0.0, 0.0], [0.5, 0.25, -0.75])
        mask = torch.ones(3)
        ppo_kl = compute_ctpo_prefix_kl([new], [old], [new], [mask])
        assert torch.exp(-ppo_kl)[0].item() == pytest.approx(math.exp(0.5), rel=1e-6)

    def test_weight_is_the_cumulative_product_of_token_ratios(self) -> None:
        old, new = _kl([0.0, 0.0, 0.0, 0.0], [0.5, 0.25, -0.75, 0.1])
        mask = torch.ones(4)
        ratio = torch.exp(-compute_ctpo_prefix_kl([new], [old], [new], [mask]))
        want = torch.cumprod(torch.exp(new - old), dim=0)
        assert torch.allclose(ratio, want, atol=1e-6)

    def test_prefix_does_not_bleed_across_samples(self) -> None:
        """The batch is one flat concatenated tensor; a naive cumsum would run
        straight through the boundary and corrupt every sample after the first."""
        old_a, new_a = _kl([0.0, 0.0], [1.0, 1.0])
        old_b, new_b = _kl([0.0, 0.0], [0.25, 0.25])
        masks = [torch.ones(2), torch.ones(2)]
        ppo_kl = compute_ctpo_prefix_kl([new_a, new_b], [old_a, old_b], [new_a, new_b], masks)
        ratio = torch.exp(-ppo_kl)
        # sample b restarts at its own first token, unaffected by sample a's +2.0
        assert ratio[2].item() == pytest.approx(math.exp(0.25), rel=1e-6)
        assert ratio[3].item() == pytest.approx(math.exp(0.50), rel=1e-6)

    def test_masked_tokens_do_not_contribute_to_the_prefix(self) -> None:
        old, new = _kl([0.0, 0.0, 0.0], [0.5, 99.0, 0.25])
        mask = torch.tensor([1.0, 0.0, 1.0])
        ratio = torch.exp(-compute_ctpo_prefix_kl([new], [old], [new], [mask]))
        # the masked middle token contributes nothing, so the third token's prefix
        # is exp(0.5 + 0.25), not exp(0.5 + 99.0 + 0.25)
        assert ratio[2].item() == pytest.approx(math.exp(0.75), rel=1e-6)


class TestClipBand:
    def test_first_response_token_gets_the_base_band(self) -> None:
        """t is 1-based, so the first generated token has band [1-eps_lo, 1+eps_hi]."""
        lo, hi = compute_ctpo_clip_band([torch.ones(3)], 0.025, 0.05)
        assert lo[0].item() == pytest.approx(0.025, rel=1e-6)
        assert hi[0].item() == pytest.approx(0.050, rel=1e-6)

    def test_band_scales_with_sqrt_of_position(self) -> None:
        lo, hi = compute_ctpo_clip_band([torch.ones(4)], 0.025, 0.05)
        for t in range(1, 5):
            assert lo[t - 1].item() == pytest.approx(0.025 * math.sqrt(t), rel=1e-6)
            assert hi[t - 1].item() == pytest.approx(0.050 * math.sqrt(t), rel=1e-6)

    def test_position_counts_only_masked_in_tokens(self) -> None:
        """A masked token must not advance t, or the band would widen for tokens
        that contributed no factor to the prefix product."""
        lo, _ = compute_ctpo_clip_band([torch.tensor([1.0, 0.0, 1.0])], 0.025, 0.05)
        assert lo[0].item() == pytest.approx(0.025 * math.sqrt(1), rel=1e-6)
        assert lo[2].item() == pytest.approx(0.025 * math.sqrt(2), rel=1e-6)

    def test_position_restarts_for_each_sample(self) -> None:
        lo, _ = compute_ctpo_clip_band([torch.ones(2), torch.ones(2)], 0.025, 0.05)
        assert lo[2].item() == pytest.approx(0.025 * math.sqrt(1), rel=1e-6)
        assert lo[3].item() == pytest.approx(0.025 * math.sqrt(2), rel=1e-6)


class TestBandAppliesElementwise:
    def test_clamp_uses_each_token_own_band(self) -> None:
        """compute_policy_loss must clamp per token, not with one scalar band."""
        from miles.backends.training_utils.loss_hub.math_utils import compute_policy_loss

        # one ratio, four different bands: 1.08 is outside the t=1 band [0.975, 1.05]
        # and inside the t=4 band [0.95, 1.10]. Deliberately not on an edge, so the
        # assertion does not turn on a float rounding.
        ppo_kl = -torch.log(torch.full((4,), 1.08))
        advantages = torch.ones(4)
        lo, hi = compute_ctpo_clip_band([torch.ones(4)], 0.025, 0.05)
        _, clipfrac = compute_policy_loss(ppo_kl, advantages, lo, hi)
        assert clipfrac[0].item() == 1.0
        assert clipfrac[3].item() == 0.0


class TestDispatch:
    """The estimator must actually be routed, not fall through to the token path."""

    @staticmethod
    def _run(estimator: str):
        import torch.distributed as dist

        from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
        from miles.backends.training_utils.loss_hub.losses import policy_loss_function
        from miles.backends.training_utils.parallel import set_parallel_state

        from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state

        set_parallel_state(make_parallel_state())
        args = make_args(
            advantage_estimator=estimator,
            loss_type="policy_loss",
            use_rollout_logprobs=True,
            eps_clip=0.025,
            eps_clip_high=0.05,
            entropy_coef=0.0,
            # not in the harness defaults; losses.py reads it unconditionally
            observe_training_entropy=False,
        )
        inputs = make_inputs(
            seed=7, batch_size=3, prompt_lens=[8, 12, 6], response_lens=[10, 16, 5],
            vocab_size=64, args=args,
        )
        batch = make_batch(inputs, "policy_loss")
        logits = deep_clone(inputs["policy_logits"])
        logits.requires_grad_(True)
        som = get_sum_of_sample_mean(
            batch["total_lengths"], batch["response_lengths"], batch["loss_masks"],
            args.calculate_per_token_loss, args.qkv_format, batch.get("max_seq_lens", None),
        )
        _, metrics = policy_loss_function(args, batch, logits, som)
        return metrics

    def test_ctpo_is_accepted_as_an_advantage_estimator(self) -> None:
        import argparse

        from miles.utils.arguments import get_miles_extra_args_provider

        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        # --rollout-batch-size is required by the miles parser regardless
        args = parser.parse_args(["--advantage-estimator", "ctpo", "--rollout-batch-size", "64"])
        assert args.advantage_estimator == "ctpo"

    def test_ctpo_ppo_kl_differs_from_the_token_level_estimator(self) -> None:
        """A missing branch falls through to `ppo_kl = old - new`, silently making
        ctpo a synonym for grpo. Prefix sums accumulate, so the reduced ppo_kl must
        not match the per-token one."""
        ctpo = self._run("ctpo")["ppo_kl"]
        grpo = self._run("grpo")["ppo_kl"]
        assert not torch.allclose(ctpo, grpo), (
            f"ctpo ppo_kl {ctpo.item()} == grpo {grpo.item()}: the ctpo branch is not being taken"
        )


class TestAdvantageDispatch:
    """compute_advantages dispatches SEPARATELY from the policy loss.

    A branch added only to policy_loss_function passes every loss test and then
    raises NotImplementedError at the first optimizer step of a real run.
    """

    @staticmethod
    def _advantages(estimator: str):
        from miles.backends.training_utils.loss_hub.advantages import compute_advantages

        from .loss_test_utils import make_args

        args = make_args(advantage_estimator=estimator)
        kl = [torch.zeros(4), torch.zeros(3)]
        return compute_advantages(
            args,
            kl=kl,
            rewards=[1.0, -0.5],
            log_probs=None,
            loss_masks=[torch.ones(4), torch.ones(3)],
            total_lengths=[9, 8],
            response_lengths=[4, 3],
        )[0]

    def test_ctpo_is_a_supported_advantage_estimator(self) -> None:
        self._advantages("ctpo")

    def test_ctpo_advantage_matches_grpo(self) -> None:
        """CTPO changes the importance weight, not the credit assignment: the
        broadcast group-normalised advantage is shared with GRPO."""
        for got, want in zip(self._advantages("ctpo"), self._advantages("grpo"), strict=True):
            assert torch.allclose(got, want)
