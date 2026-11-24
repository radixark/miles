"""Verify the BPO custom loss plugin math, precision handling, validation, and dispatch."""

import math
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from examples.experimental.bpo.bpo_loss import BpoConfig, bpo_loss_function, compute_bpo_policy_loss
from pydantic import ValidationError

BPO_LOSS_PATH = "examples.experimental.bpo.bpo_loss.bpo_loss_function"


def _make_bpo_pg_args(
    *,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
    eps_clip_c: float | None = None,
    bpo_smoothing: float = 0.1,
    bpo_weight_cap: float = 3.0,
    advantage_estimator: str = "grpo",
) -> SimpleNamespace:
    return SimpleNamespace(
        advantage_estimator=advantage_estimator,
        eps_clip=eps_clip,
        eps_clip_high=eps_clip_high,
        eps_clip_c=eps_clip_c,
        bpo_smoothing=bpo_smoothing,
        bpo_weight_cap=bpo_weight_cap,
    )


class TestBpoConfigAndMath:
    def test_bpo_config_defaults_and_json_roundtrip(self) -> None:
        """`BpoConfig` defaults match Appendix B values and survive JSON transport."""
        config = BpoConfig()
        assert (config.bpo_smoothing, config.bpo_weight_cap) == (0.1, 3.0)

        custom = BpoConfig(bpo_smoothing=0.25, bpo_weight_cap=2.5)
        restored = BpoConfig.model_validate_json(custom.model_dump_json())
        assert restored == custom

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            (dict(bpo_smoothing=0.0), "bpo-smoothing"),
            (dict(bpo_smoothing=-0.1), "bpo-smoothing"),
            (dict(bpo_smoothing=1e-50), "bpo-smoothing"),
            (dict(bpo_smoothing=1e100), "bpo-smoothing"),
            (dict(bpo_smoothing=float("nan")), "bpo-smoothing"),
            (dict(bpo_smoothing=float("inf")), "bpo-smoothing"),
            (dict(bpo_weight_cap=0.0), "bpo-weight-cap"),
            (dict(bpo_weight_cap=-1.0), "bpo-weight-cap"),
            (dict(bpo_weight_cap=1e-100), "bpo-weight-cap"),
            (dict(bpo_weight_cap=1e100), "bpo-weight-cap"),
            (dict(bpo_weight_cap=float("nan")), "bpo-weight-cap"),
            (dict(bpo_weight_cap=float("inf")), "bpo-weight-cap"),
        ],
        ids=[
            "smoothing-zero",
            "smoothing-neg",
            "smoothing-fp32-underflow",
            "smoothing-fp32-overflow",
            "smoothing-nan",
            "smoothing-inf",
            "cap-zero",
            "cap-neg",
            "cap-fp32-underflow",
            "cap-fp32-overflow",
            "cap-nan",
            "cap-inf",
        ],
    )
    def test_bpo_config_rejects_out_of_range_params(self, kwargs: dict[str, float], message: str) -> None:
        """Values outside positive normal fp32 range fail during `BpoConfig` validation."""
        with pytest.raises(ValidationError, match=message):
            BpoConfig(**kwargs)

    @pytest.mark.parametrize("smoothing,cap", [(0.1, 3.0), (0.05, 2.0)])
    def test_bpo_matches_scalar_formula_and_stops_weight_gradients(self, smoothing: float, cap: float) -> None:
        """Per-token loss matches Eqs. (14)-(15) and stops gradients through weight, old policy, and advantage."""
        probabilities = [0.1, 0.5, 0.8, 0.95, 0.999999, 1e-30]
        rollout_probabilities = [0.9, 0.4, 0.1, 0.1, 0.01, 1e-20]
        log_probs = torch.tensor(probabilities * 3, dtype=torch.float64).log().requires_grad_()
        old_log_probs = torch.tensor(rollout_probabilities * 3, dtype=torch.float64).log().requires_grad_()
        advantages = torch.tensor([1.0] * 6 + [-2.0] * 6 + [0.0] * 6, dtype=torch.float64, requires_grad=True)
        args = _make_bpo_pg_args(bpo_smoothing=smoothing, bpo_weight_cap=cap)

        pg_loss, pg_clipfrac = compute_bpo_policy_loss(
            args=args,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ppo_kl=old_log_probs - log_probs,
            advantages=advantages,
        )

        coefficients, expected_clipped = [], []
        for pi, mu, advantage in zip(probabilities * 3, rollout_probabilities * 3, advantages.tolist(), strict=True):
            weight = (1.0 + smoothing - mu) / (1.0 + smoothing - pi)
            masked = (advantage > 0 and weight > 1.28) or (advantage < 0 and weight < 0.8)
            coefficients.append(0.0 if masked else -advantage * min(weight, cap))
            expected_clipped.append(float(masked))
        expected = torch.tensor(coefficients, dtype=torch.float64)

        torch.testing.assert_close(pg_loss, expected * log_probs.detach())
        torch.testing.assert_close(pg_clipfrac, torch.tensor(expected_clipped, dtype=torch.float64))
        gradient, old_gradient, adv_gradient = torch.autograd.grad(
            pg_loss.sum(), (log_probs, old_log_probs, advantages), allow_unused=True
        )
        torch.testing.assert_close(gradient, expected)
        assert old_gradient is None and adv_gradient is None

    def test_bpo_masks_on_uncapped_weight_when_cap_below_clip_high(self) -> None:
        """Clipping mask evaluates uncapped `w_t` even when `bpo_weight_cap <= 1 + eps_clip_high`."""
        # Token 0: p=0.9, q=0.1, eps=0.1 -> w = 1.0 / 0.2 = 5.0 > 1.28 (must be masked, not capped to 1.1)
        # Token 1: p=0.5, q=0.38, eps=0.1 -> w = 0.72 / 0.6 = 1.2 in (1.1, 1.28] (kept, capped at 1.1)
        log_probs = torch.tensor([0.9, 0.5], dtype=torch.float64).log().requires_grad_()
        old_log_probs = torch.tensor([0.1, 0.38], dtype=torch.float64).log()
        advantages = torch.tensor([1.0, 1.0], dtype=torch.float64)
        args = _make_bpo_pg_args(eps_clip=0.2, eps_clip_high=0.28, bpo_smoothing=0.1, bpo_weight_cap=1.1)

        pg_loss, pg_clipfrac = compute_bpo_policy_loss(
            args=args,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ppo_kl=old_log_probs - log_probs,
            advantages=advantages,
        )

        torch.testing.assert_close(pg_clipfrac, torch.tensor([1.0, 0.0], dtype=torch.float64))
        expected_coeff = torch.tensor([0.0, -1.1], dtype=torch.float64)
        torch.testing.assert_close(pg_loss, expected_coeff * log_probs.detach())
        torch.testing.assert_close(torch.autograd.grad(pg_loss.sum(), log_probs)[0], expected_coeff)

    def test_bpo_mask_boundaries_small_smoothing_and_nonfinite_inputs(self) -> None:
        """Handles exact boundaries, tiny normal fp32 smoothing at `p=q=1`, valid `q=0` (`-inf`), and non-finite inputs."""
        # 0: p=q=1 with tiny normal smoothing -> w=1, grad=-1
        # 1: log_p=-inf with adv!=0 -> gated to 0 loss & 0 grad (no inf)
        # 2: log_p=NaN with adv=1 -> gated to 0 loss & 0 grad
        # 3: old_log_p=-inf (q=0, valid) with finite log_p=log(0.5), adv=-1 -> kept with w=(1+eps)/(0.5+eps)
        # 4: old_log_p=NaN with adv=1 -> gated to 0 loss & 0 grad
        # 5: old_log_p=+inf with adv=1 -> gated to 0 loss & 0 grad
        # 6,7: exact weight==1.0 on strict clip boundary eps_clip=eps_clip_high=0 -> unclipped
        log_probs = torch.tensor(
            [0.0, -torch.inf, torch.nan, math.log(0.5), math.log(0.5), math.log(0.5), math.log(0.5), math.log(0.5)],
            requires_grad=True,
        )
        old_log_probs = torch.tensor(
            [0.0, math.log(0.1), math.log(0.5), -torch.inf, torch.nan, torch.inf, math.log(0.5), math.log(0.5)]
        )
        advantages = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0])
        args = _make_bpo_pg_args(eps_clip=0.0, eps_clip_high=0.0, bpo_smoothing=1e-20, bpo_weight_cap=3.0)

        pg_loss, pg_clipfrac = compute_bpo_policy_loss(
            args=args,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ppo_kl=old_log_probs - log_probs,
            advantages=advantages,
        )

        assert torch.isfinite(pg_loss).all()
        assert torch.isfinite(pg_clipfrac).all()
        torch.testing.assert_close(pg_clipfrac.sum(), torch.tensor(0.0))
        grad = torch.autograd.grad(pg_loss.sum(), log_probs)[0]
        assert torch.isfinite(grad).all()
        expected_q0_weight = (1.0 + 1e-20) / (0.5 + 1e-20)
        torch.testing.assert_close(
            grad,
            torch.tensor([-1.0, 0.0, 0.0, expected_q0_weight, 0.0, 0.0, -1.0, 1.0]),
        )

    def test_bpo_empty_cp_slice_returns_empty_finite_tensors(self) -> None:
        """Empty local CP slice produces empty finite outputs without error."""
        log_probs = torch.empty(0, dtype=torch.float32, requires_grad=True)
        old_log_probs = torch.empty(0, dtype=torch.float32)
        advantages = torch.empty(0, dtype=torch.float32)
        args = _make_bpo_pg_args()

        pg_loss, pg_clipfrac = compute_bpo_policy_loss(
            args=args,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ppo_kl=old_log_probs - log_probs,
            advantages=advantages,
        )
        assert pg_loss.shape == (0,) and pg_clipfrac.shape == (0,)

    @pytest.mark.parametrize(
        "dtype,count_boundary,flip_log_p,flip_old_log_p",
        [
            (torch.bfloat16, 257, -0.69140625, -1.1015625),
            (torch.float16, 2049, -0.693359375, -1.103515625),
        ],
        ids=["bf16", "fp16"],
    )
    def test_bpo_low_precision_promotes_thresholds_and_clipfrac_count(
        self, dtype: torch.dtype, count_boundary: int, flip_log_p: float, flip_old_log_p: float
    ) -> None:
        """`bf16`/`fp16` inputs promote weight thresholds and `pg_clipfrac` to `fp32` so masks do not flip and counts stay exact."""
        args = _make_bpo_pg_args(bpo_smoothing=0.1, bpo_weight_cap=3.0, eps_clip=0.2, eps_clip_high=0.28)

        # Verify threshold-flipping pair: native dtype rounds weight/threshold and returns False, fp32 returns True
        log_p_flip = torch.tensor([flip_log_p], dtype=dtype, requires_grad=True)
        old_log_p_flip = torch.tensor([flip_old_log_p], dtype=dtype)
        adv_flip = torch.tensor([1.0], dtype=dtype)
        native_weight = (args.bpo_smoothing - old_log_p_flip.expm1()) / (args.bpo_smoothing - log_p_flip.expm1())
        assert not bool((native_weight > 1.0 + args.eps_clip_high).item())

        loss_flip, clip_flip = compute_bpo_policy_loss(
            args=args,
            log_probs=log_p_flip,
            old_log_probs=old_log_p_flip,
            ppo_kl=old_log_p_flip - log_p_flip,
            advantages=adv_flip,
        )
        assert clip_flip.dtype == torch.float32
        torch.testing.assert_close(clip_flip, torch.tensor([1.0]))
        torch.testing.assert_close(loss_flip, torch.tensor([0.0]))

        # Verify fp32 pg_clipfrac survives past native integer-precision boundary (256 for bf16, 2048 for fp16)
        log_probs_many = torch.full((count_boundary,), math.log(0.9), dtype=dtype, requires_grad=True)
        old_log_probs_many = torch.full((count_boundary,), math.log(0.1), dtype=dtype)
        advantages_many = torch.ones(count_boundary, dtype=dtype)
        int32_mask = torch.ones(count_boundary, dtype=torch.int32)
        pg_loss_many, pg_clipfrac_many = compute_bpo_policy_loss(
            args=args,
            log_probs=log_probs_many,
            old_log_probs=old_log_probs_many,
            ppo_kl=old_log_probs_many - log_probs_many,
            advantages=advantages_many,
        )
        assert pg_clipfrac_many.dtype == torch.float32 and pg_loss_many.dtype == torch.float32
        torch.testing.assert_close((pg_clipfrac_many * int32_mask).sum(), torch.tensor(float(count_boundary)))

        # Unclipped gradient differentiates cleanly back to the original low-precision dtype
        log_probs_low = torch.tensor([math.log(0.5)], dtype=dtype, requires_grad=True)
        old_log_probs_low = torch.tensor([math.log(0.4)], dtype=dtype)
        advantages_low = torch.tensor([1.0], dtype=dtype)
        loss_low, clip_low = compute_bpo_policy_loss(
            args=args,
            log_probs=log_probs_low,
            old_log_probs=old_log_probs_low,
            ppo_kl=old_log_probs_low - log_probs_low,
            advantages=advantages_low,
        )
        loss_fp32, clip_fp32 = compute_bpo_policy_loss(
            args=args,
            log_probs=log_probs_low.detach().float().requires_grad_(),
            old_log_probs=old_log_probs_low.float(),
            ppo_kl=old_log_probs_low.float() - log_probs_low.detach().float(),
            advantages=advantages_low.float(),
        )
        torch.testing.assert_close(loss_low, loss_fp32)
        torch.testing.assert_close(clip_low, clip_fp32)
        grad = torch.autograd.grad(loss_low.sum(), log_probs_low)[0]
        assert grad.dtype == dtype and torch.isfinite(grad).all()

    def test_bpo_preserves_float64_when_only_old_log_probs_or_advantages_is_float64(self) -> None:
        """`calc_dtype` promotes across all inputs rather than truncating `float64` advantages to `float32`."""
        log_probs = torch.tensor([math.log(0.5)], dtype=torch.float32, requires_grad=True)
        old_log_probs = torch.tensor([math.log(0.4)], dtype=torch.float32)
        advantages = torch.tensor([1.0], dtype=torch.float64)
        pg_loss, pg_clipfrac = compute_bpo_policy_loss(
            args=_make_bpo_pg_args(),
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ppo_kl=old_log_probs - log_probs,
            advantages=advantages,
        )
        assert pg_loss.dtype == torch.float64 and pg_clipfrac.dtype == torch.float64

    @pytest.mark.parametrize(
        "overrides,message",
        [
            (dict(advantage_estimator="gspo"), "advantage-estimator grpo"),
            (dict(eps_clip_c=3.0), "eps-clip-c"),
            (dict(eps_clip=1.0), "eps-clip"),
            (dict(eps_clip=-0.1), "eps-clip"),
            (dict(eps_clip_high=-0.1), "eps-clip-high"),
            (dict(eps_clip_high=None), "eps-clip-high"),
            (dict(bpo_smoothing=0.0), "bpo-smoothing"),
            (dict(bpo_smoothing=1e-50), "bpo-smoothing"),
            (dict(bpo_weight_cap=float("nan")), "bpo-weight-cap"),
        ],
        ids=[
            "non-grpo",
            "dual-clip-c",
            "eps-clip-one",
            "eps-clip-neg",
            "eps-clip-high-neg",
            "eps-clip-high-none",
            "smoothing-zero",
            "smoothing-underflow",
            "cap-nan",
        ],
    )
    def test_bpo_loss_function_rejects_invalid_args(self, overrides: dict[str, object], message: str) -> None:
        """Unsupported options like `--eps-clip-c` or non-GRPO estimators fail instead of being silently ignored."""
        args = _make_bpo_pg_args(**overrides)
        with pytest.raises(ValueError, match=message):
            bpo_loss_function(args, {}, torch.zeros(1, 2, 2), torch.sum)


class TestBpoPipelineIntegration:
    """End-to-end `custom_loss` dispatch through `objective.loss_function` and `policy_loss_function`."""

    @pytest.fixture(scope="class", autouse=True)
    def _process_group(self, tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
        from miles.backends.training_utils import parallel

        saved_state = parallel._parallel_state
        created_group = not dist.is_initialized()
        if created_group:
            rendezvous = tmp_path_factory.mktemp("bpo") / "rendezvous"
            dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
        try:
            yield
        finally:
            parallel._parallel_state = saved_state
            if created_group and dist.is_initialized():
                dist.destroy_process_group()

    @staticmethod
    def _make_custom_bpo_args(**overrides: object):
        from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

        from miles.utils.args.custom_function import CustomFunctionConfig
        from miles.utils.args.custom_view import ImmutableNamespace

        make_parallel_state()
        bpo_smoothing = overrides.pop("bpo_smoothing", 0.1)
        bpo_weight_cap = overrides.pop("bpo_weight_cap", 3.0)
        hook = CustomFunctionConfig(
            path=BPO_LOSS_PATH,
            config=BpoConfig(bpo_smoothing=bpo_smoothing, bpo_weight_cap=bpo_weight_cap),
        )
        return ImmutableNamespace(
            **dict(
                make_args(
                    loss_type="custom_loss",
                    custom_loss_function_path=hook,
                    eps_clip=overrides.pop("eps_clip", 0.2),
                    eps_clip_high=overrides.pop("eps_clip_high", 0.28),
                    **overrides,
                )
            )
        )

    @pytest.mark.parametrize("per_token", [False, True])
    @pytest.mark.parametrize("regularize", [False, True])
    @pytest.mark.parametrize("use_rollout_logprobs", [False, True])
    def test_bpo_dispatch_matches_dense_gradient_with_masks(
        self, per_token: bool, regularize: bool, use_rollout_logprobs: bool
    ) -> None:
        """Full `custom_loss` dispatch matches dense BPO gradients with masks, non-zero clipping, entropy, and KL."""
        from miles.backends.training_utils.loss import objective

        args = self._make_custom_bpo_args(
            use_rollout_logprobs=use_rollout_logprobs,
            true_on_policy_mode=False,
            observe_training_entropy=True,
            entropy_coef=0.03 if regularize else 0.0,
            use_kl_loss=regularize,
            kl_loss_coef=0.2,
            kl_loss_type="k2",
            calculate_per_token_loss=per_token,
        )
        # Token 0 has high model prob (~0.9) vs low old prob (0.05) and adv=+1 -> weight > 1.28 (BPO-clipped)
        logits = torch.randn((1, 6, 5), generator=torch.Generator().manual_seed(7))
        logits[0, 1] = torch.tensor([-2.0, -2.0, 3.0, -2.0, -2.0])
        logits.requires_grad_(True)
        old_log_probs = torch.tensor([math.log(0.05), torch.nan, math.log(0.4), math.log(0.2)], requires_grad=True)
        advantages = torch.tensor([1.0, torch.nan, -2.0, 0.5], requires_grad=True)
        batch = dict(
            unconcat_tokens=[torch.tensor([0, 1, 2, 3, 4, 0])],
            total_lengths=[6],
            response_lengths=[4],
            loss_masks=[torch.tensor([1, 0, 1, 1])],
            advantages=[advantages],
            rollout_log_probs=[old_log_probs if use_rollout_logprobs else torch.full((4,), -0.5)],
            log_probs=[torch.full((4,), -0.5) if use_rollout_logprobs else old_log_probs],
            ref_log_probs=[torch.full((4,), -1.5)],
        )

        dense_logits = logits.detach().clone().requires_grad_()
        loss, _, log = objective.loss_function(args, batch, 1, logits, num_rollouts=1, num_step_tokens=torch.tensor(3))
        metrics = dict(zip(log["keys"], log["values"][1:], strict=True))

        distribution = dense_logits[0, 1:5].log_softmax(-1)
        selected = distribution[torch.arange(4), torch.tensor([2, 3, 4, 0])]
        active = torch.tensor([True, False, True, True])
        coefficient = torch.zeros(4)
        masked = torch.zeros(4)
        for i in [0, 2, 3]:
            weight = (1.1 - old_log_probs[i].detach().exp()) / (1.1 - selected[i].detach().exp())
            advantage = advantages[i].item()
            reject = (advantage > 0 and weight > 1.28) or (advantage < 0 and weight < 0.8)
            coefficient[i] = 0.0 if reject else -advantage * min(weight.item(), 3.0)
            masked[i] = float(reject)

        assert masked[active].sum().item() >= 1.0, "fixture must exercise non-zero BPO clipping"
        divisor = 3 if not per_token else 1
        expected_pg_raw = (coefficient[active] * selected[active]).sum() / divisor
        expected_raw = expected_pg_raw
        if regularize:
            entropy = -(distribution.exp() * distribution).sum(-1)
            penalty = (selected + 1.5).square() / 2
            expected_raw = expected_raw + (-0.03 * entropy[active] + 0.2 * penalty[active]).sum() / divisor
        expected_scaled = expected_raw / (3 if per_token else 1)

        torch.testing.assert_close(loss, expected_scaled)
        torch.testing.assert_close(metrics["pg_loss"], expected_pg_raw.detach())
        torch.testing.assert_close(metrics["pg_clipfrac"], masked[active].sum() / divisor)
        torch.testing.assert_close(
            metrics["ppo_kl"], (old_log_probs[active].detach() - selected[active].detach()).sum() / divisor
        )
        assert {
            "loss",
            "pg_loss",
            "entropy_loss",
            "pg_clipfrac",
            "ppo_kl",
            "ess_ratio",
            "train_rollout_logprob_abs_diff",
            "train_rollout_kl",
        } <= metrics.keys()
        gradient, old_gradient, adv_gradient = torch.autograd.grad(
            loss, (logits, old_log_probs, advantages), allow_unused=True
        )
        torch.testing.assert_close(gradient, torch.autograd.grad(expected_scaled, dense_logits)[0])
        assert old_gradient is None and adv_gradient is None

    def test_active_token_nan_old_log_prob_stays_finite_through_policy_loss_function(self) -> None:
        """An active-token `NaN` in `old_log_probs` is sanitized to `w=1` fallback instead of turning the microbatch `NaN`."""
        from miles.backends.training_utils.loss import objective

        args = self._make_custom_bpo_args(
            use_rollout_logprobs=True,
            true_on_policy_mode=False,
            entropy_coef=0.0,
            observe_training_entropy=False,
        )
        logits = torch.randn((1, 5, 4), generator=torch.Generator().manual_seed(31), requires_grad=True)
        dense_logits = logits.detach().clone().requires_grad_()
        advantages = torch.tensor([1.0, -1.0, 0.5, 1.0])
        old_log_probs = torch.tensor([torch.nan, math.log(0.4), math.log(0.5), math.log(0.2)])
        batch = dict(
            unconcat_tokens=[torch.tensor([0, 1, 2, 3, 0])],
            total_lengths=[5],
            response_lengths=[4],
            loss_masks=[torch.ones(4)],
            advantages=[advantages],
            rollout_log_probs=[old_log_probs],
        )

        loss, _, _ = objective.loss_function(args, batch, 1, logits, num_rollouts=1)
        assert torch.isfinite(loss)
        grad = torch.autograd.grad(loss, logits)[0]
        assert torch.isfinite(grad).all()

        # Token 0 falls back to old_log_prob = current log_prob (w=1, unclipped -> coeff=adv=1.0)
        selected = dense_logits[0, 0:4].log_softmax(-1)[torch.arange(4), torch.tensor([1, 2, 3, 0])]
        clean_old = torch.where(old_log_probs.isnan(), selected.detach(), old_log_probs)
        weight = (1.1 - clean_old.exp()) / (1.1 - selected.detach().exp())
        clipped = ((advantages > 0) & (weight > 1.28)) | ((advantages < 0) & (weight < 0.8))
        expected = (-(advantages * weight.clamp(max=3.0) * (~clipped)) * selected).mean()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(grad, torch.autograd.grad(expected, dense_logits)[0])

    def test_opsm_rejected_sequence_zeroes_loss_and_gradient_without_nan(self) -> None:
        """OPSM rejection uses `torch.where` so even an infinite per-token `pg_loss` produces `0` loss and gradient, not `NaN`."""
        from miles.backends.training_utils.loss.hub.losses import policy_loss_function

        args = self._make_custom_bpo_args(
            use_rollout_logprobs=True,
            use_opsm=True,
            opsm_delta=0.05,
            true_on_policy_mode=False,
            entropy_coef=0.0,
            observe_training_entropy=False,
        )
        logits = torch.randn((1, 4, 4), generator=torch.Generator().manual_seed(37), requires_grad=True)
        # Negative advantage with positive seq_kl = mean(old_log_probs - log_probs) > opsm_delta triggers OPSM mask == 0
        batch = dict(
            unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
            total_lengths=[4],
            response_lengths=[3],
            loss_masks=[torch.ones(3)],
            advantages=[torch.full((3,), -1.0)],
            rollout_log_probs=[torch.zeros(3)],
        )

        # Synthetic hook returning -inf must still produce 0.0 (not NaN) after OPSM mask==0
        def _inf_pg_loss(*, log_probs: torch.Tensor, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.full_like(log_probs, -torch.inf) + 0.0 * log_probs, torch.zeros_like(log_probs)

        loss, metrics = policy_loss_function(args, batch, logits, torch.mean, pg_loss_fn=_inf_pg_loss)
        assert torch.isfinite(loss) and loss.item() == 0.0
        assert torch.isfinite(metrics["pg_loss"]) and metrics["pg_loss"].item() == 0.0
        grad = torch.autograd.grad(loss, logits)[0]
        assert torch.isfinite(grad).all() and grad.abs().max().item() == 0.0

    @pytest.mark.parametrize("per_token", [False, True])
    def test_bpo_keeps_policy_loss_aggregation(self, per_token: bool) -> None:
        """Multi-sample per-token vs per-sample aggregation matches `policy_loss_function`."""
        from miles.backends.training_utils.loss import objective

        args = self._make_custom_bpo_args(
            use_rollout_logprobs=True,
            true_on_policy_mode=False,
            entropy_coef=0.0,
            observe_training_entropy=False,
            calculate_per_token_loss=per_token,
        )
        logits = torch.randn((1, 6, 4), generator=torch.Generator().manual_seed(11), requires_grad=True)
        dense_logits = logits.detach().clone().requires_grad_()
        rollout_probs = torch.tensor([0.2, 0.4, 0.3, 0.5])
        advantages = torch.tensor([1.0, -1.0, 0.0, 2.0])
        batch = dict(
            advantages=list(advantages.split([3, 1])),
            rollout_log_probs=list(rollout_probs.log().split([3, 1])),
            unconcat_tokens=[torch.tensor([0, 1, 2, 3]), torch.tensor([0, 2])],
            total_lengths=[4, 2],
            response_lengths=[3, 1],
            loss_masks=[torch.tensor([1.0, 1.0, 0.0]), torch.ones(1)],
        )

        actual, _, _ = objective.loss_function(args, batch, 1, logits, num_rollouts=2, num_step_tokens=torch.tensor(3))

        # Sample 0 response tokens [1, 2, 3] predicted at logits[0, 0:3]; sample 1 token [2] at logits[0, 4:5]
        sample0_log_probs = dense_logits[0, 0:3].log_softmax(-1)[torch.arange(3), torch.tensor([1, 2, 3])]
        sample1_log_probs = dense_logits[0, 4:5].log_softmax(-1)[torch.arange(1), torch.tensor([2])]
        log_probs = torch.cat([sample0_log_probs, sample1_log_probs], dim=0)
        weight = (1.1 - rollout_probs) / (1.1 - log_probs.detach().exp())
        clipped = ((advantages > 0) & (weight > 1.28)) | ((advantages < 0) & (weight < 0.8))
        tokens = -(advantages * weight.clamp(max=3.0) * (~clipped)) * log_probs
        expected = (
            (tokens[0] + tokens[1] + tokens[3]) / 3 if per_token else ((tokens[0] + tokens[1]) / 2 + tokens[3]) / 2
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, logits)[0],
            torch.autograd.grad(expected, dense_logits)[0],
        )

    @pytest.mark.parametrize("use_rollout_logprobs,include_rollout", [(True, True), (False, True), (False, False)])
    def test_bpo_uses_configured_old_policy_source(self, use_rollout_logprobs: bool, include_rollout: bool) -> None:
        """`--use-rollout-logprobs` selects `rollout_log_probs` vs trainer `log_probs` as old policy."""
        from miles.backends.training_utils.loss import objective

        args = self._make_custom_bpo_args(
            use_rollout_logprobs=use_rollout_logprobs,
            true_on_policy_mode=False,
            entropy_coef=0.0,
            observe_training_entropy=False,
        )
        logits = torch.randn((1, 5, 4), generator=torch.Generator().manual_seed(19), requires_grad=True)
        dense_logits = logits.detach().clone().requires_grad_()
        rollout_probs = torch.tensor([0.8, 0.1, 0.4, 0.2])
        trainer_probs = torch.tensor([0.1, 0.8, 0.7, 0.9])
        rollout_log_probs = rollout_probs.log().requires_grad_()
        trainer_log_probs = trainer_probs.log().requires_grad_()
        advantages = torch.tensor([1.0, -2.0, -1.0, 2.0], requires_grad=True)
        batch = dict(
            unconcat_tokens=[torch.tensor([0, 1, 2, 3, 0])],
            total_lengths=[5],
            response_lengths=[4],
            loss_masks=[torch.ones(4)],
            advantages=[advantages],
            rollout_log_probs=[rollout_log_probs] if include_rollout else None,
            log_probs=[trainer_log_probs],
        )

        actual, _, log = objective.loss_function(args, batch, 1, logits, num_rollouts=1)
        metrics = dict(zip(log["keys"], log["values"][1:], strict=True))

        selected = dense_logits[0, 0:4].log_softmax(-1)[torch.arange(4), torch.tensor([1, 2, 3, 0])]
        old_probs = rollout_probs if use_rollout_logprobs else trainer_probs
        old_log_probs = rollout_log_probs if use_rollout_logprobs else trainer_log_probs
        weight = (1.1 - old_probs) / (1.1 - selected.detach().exp())
        clipped = ((advantages.detach() > 0) & (weight > 1.28)) | ((advantages.detach() < 0) & (weight < 0.8))
        expected = (-(advantages.detach() * weight.clamp(max=3.0) * (~clipped)) * selected).mean()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(metrics["ppo_kl"], (old_log_probs.detach() - selected.detach()).mean())
        grads = torch.autograd.grad(
            actual, (logits, rollout_log_probs, trainer_log_probs, advantages), allow_unused=True
        )
        torch.testing.assert_close(grads[0], torch.autograd.grad(expected, dense_logits)[0])
        assert all(g is None for g in grads[1:])

    def test_bpo_custom_config_values_flow_through_loss_dispatch(self) -> None:
        """Non-default `BpoConfig` values reach `bpo_loss_function` through `objective.loss_function`."""
        from miles.backends.training_utils.loss import objective

        logits = torch.randn((1, 4, 4), generator=torch.Generator().manual_seed(23), requires_grad=True)
        dense_logits = logits.detach().clone().requires_grad_()
        rollout_probs = torch.tensor([0.1, 0.4, 0.2])
        advantages = torch.tensor([1.0, 1.0, -1.0])
        batch = dict(
            unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
            total_lengths=[4],
            response_lengths=[3],
            loss_masks=[torch.ones(3)],
            advantages=[advantages],
            rollout_log_probs=[rollout_probs.log()],
        )
        args = self._make_custom_bpo_args(
            use_rollout_logprobs=True,
            true_on_policy_mode=False,
            entropy_coef=0.0,
            observe_training_entropy=False,
            bpo_smoothing=0.25,
            bpo_weight_cap=1.05,
            eps_clip=0.2,
            eps_clip_high=0.5,
        )

        actual, _, _ = objective.loss_function(args, batch, 1, logits, num_rollouts=1)

        selected = dense_logits[0, 0:3].log_softmax(-1)[torch.arange(3), torch.tensor([1, 2, 3])]
        weight = (1.25 - rollout_probs) / (1.25 - selected.detach().exp())
        clipped = ((advantages > 0) & (weight > 1.5)) | ((advantages < 0) & (weight < 0.8))
        expected = (-(advantages * weight.clamp(max=1.05) * (~clipped)) * selected).mean()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, logits)[0],
            torch.autograd.grad(expected, dense_logits)[0],
        )
