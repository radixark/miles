"""Unit tests for miles/utils/ppo_utils.py

These tests verify the correctness of PPO/GRPO utility functions without
requiring distributed training infrastructure or GPU hardware.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch


# Mock megatron.core.mpu before importing ppo_utils
@pytest.fixture(autouse=True)
def mock_megatron():
    """Mock megatron dependencies for unit testing."""
    mock_mpu = MagicMock()
    mock_mpu.get_context_parallel_world_size.return_value = 1

    with patch.dict("sys.modules", {"megatron": MagicMock(), "megatron.core": MagicMock(), "megatron.core.mpu": mock_mpu}):
        yield mock_mpu


class TestComputeApproxKL:
    """Tests for compute_approx_kl function."""

    def test_k1_estimator(self, mock_megatron):
        """Test k1 KL estimator (simple log ratio)."""
        from miles.utils.ppo_utils import compute_approx_kl

        log_probs = torch.tensor([-1.0, -2.0, -3.0])
        log_probs_base = torch.tensor([-1.5, -2.5, -3.5])

        kl = compute_approx_kl(log_probs, log_probs_base, "k1")

        # k1: KL = log_probs - log_probs_base
        expected = log_probs - log_probs_base
        torch.testing.assert_close(kl, expected)

    def test_k2_estimator(self, mock_megatron):
        """Test k2 KL estimator (squared log ratio / 2)."""
        from miles.utils.ppo_utils import compute_approx_kl

        log_probs = torch.tensor([-1.0, -2.0, -3.0])
        log_probs_base = torch.tensor([-1.5, -2.5, -3.5])

        kl = compute_approx_kl(log_probs, log_probs_base, "k2")

        # k2: KL = (log_ratio)^2 / 2
        log_ratio = log_probs - log_probs_base
        expected = log_ratio**2 / 2.0
        torch.testing.assert_close(kl, expected)

    def test_k3_estimator(self, mock_megatron):
        """Test k3 KL estimator (non-negative, unbiased)."""
        from miles.utils.ppo_utils import compute_approx_kl

        log_probs = torch.tensor([-1.0, -2.0, -3.0])
        log_probs_base = torch.tensor([-1.5, -2.5, -3.5])

        kl = compute_approx_kl(log_probs, log_probs_base, "k3")

        # k3: KL = exp(-log_ratio) - 1 - (-log_ratio) = exp(-log_ratio) - 1 + log_ratio
        log_ratio = -(log_probs - log_probs_base)
        expected = log_ratio.exp() - 1 - log_ratio
        torch.testing.assert_close(kl, expected)

    def test_low_var_kl_estimator(self, mock_megatron):
        """Test low_var_kl estimator (same as k3 with clamping)."""
        from miles.utils.ppo_utils import compute_approx_kl

        log_probs = torch.tensor([-1.0, -2.0, -3.0])
        log_probs_base = torch.tensor([-1.5, -2.5, -3.5])

        kl = compute_approx_kl(log_probs, log_probs_base, "low_var_kl")

        # low_var_kl: same as k3 but clamped to [-10, 10]
        log_ratio = -(log_probs - log_probs_base)
        expected = torch.clamp(log_ratio.exp() - 1 - log_ratio, min=-10, max=10)
        torch.testing.assert_close(kl, expected)

    def test_low_var_kl_clamping(self, mock_megatron):
        """Test that low_var_kl properly clamps extreme values."""
        from miles.utils.ppo_utils import compute_approx_kl

        # Create extreme log ratio that would produce values outside [-10, 10]
        log_probs = torch.tensor([0.0])
        log_probs_base = torch.tensor([-20.0])  # Very large difference

        kl = compute_approx_kl(log_probs, log_probs_base, "low_var_kl")

        assert kl.item() <= 10.0
        assert kl.item() >= -10.0

    def test_invalid_kl_type_raises(self, mock_megatron):
        """Test that invalid KL type raises ValueError."""
        from miles.utils.ppo_utils import compute_approx_kl

        log_probs = torch.tensor([-1.0])
        log_probs_base = torch.tensor([-1.5])

        with pytest.raises(ValueError, match="Unknown kl_loss_type"):
            compute_approx_kl(log_probs, log_probs_base, "invalid_type")

    def test_importance_ratio_applied(self, mock_megatron):
        """Test that importance ratio is correctly applied."""
        from miles.utils.ppo_utils import compute_approx_kl

        log_probs = torch.tensor([-1.0, -2.0])
        log_probs_base = torch.tensor([-1.5, -2.5])
        importance_ratio = torch.tensor([0.5, 2.0])

        kl = compute_approx_kl(log_probs, log_probs_base, "k1", importance_ratio)

        # With importance ratio: KL = importance_ratio * base_kl
        base_kl = log_probs - log_probs_base
        expected = importance_ratio * base_kl
        torch.testing.assert_close(kl, expected)

    def test_empty_tensor(self, mock_megatron):
        """Test handling of empty tensors."""
        from miles.utils.ppo_utils import compute_approx_kl

        log_probs = torch.tensor([])
        log_probs_base = torch.tensor([])

        kl = compute_approx_kl(log_probs, log_probs_base, "k1")

        assert kl.shape == torch.Size([0])


class TestComputePolicyLoss:
    """Tests for compute_policy_loss function."""

    def test_no_clipping_when_ratio_in_range(self, mock_megatron):
        """Test that loss equals -ratio * advantages when ratio is within clip range."""
        from miles.utils.ppo_utils import compute_policy_loss

        ppo_kl = torch.tensor([0.0, 0.0])  # ratio = exp(0) = 1
        advantages = torch.tensor([1.0, -1.0])
        eps_clip = 0.2
        eps_clip_high = 0.2

        pg_losses, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high)

        # When ratio=1, no clipping should occur
        expected_losses = -1.0 * advantages
        torch.testing.assert_close(pg_losses, expected_losses)
        assert clipfrac.sum() == 0.0

    def test_clipping_when_ratio_exceeds_range(self, mock_megatron):
        """Test that clipping occurs when ratio exceeds clip range."""
        from miles.utils.ppo_utils import compute_policy_loss

        # ratio = exp(-(-0.5)) = exp(0.5) > 1.2, should be clipped
        ppo_kl = torch.tensor([-0.5])
        advantages = torch.tensor([1.0])
        eps_clip = 0.2
        eps_clip_high = 0.2

        pg_losses, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high)

        # With positive advantage, clipped ratio should be 1 + eps_clip_high = 1.2
        expected_loss = -1.2 * advantages
        torch.testing.assert_close(pg_losses, expected_loss)
        assert clipfrac.item() == 1.0

    def test_dual_clip_ppo(self, mock_megatron):
        """Test dual-clip PPO with eps_clip_c parameter."""
        from miles.utils.ppo_utils import compute_policy_loss

        ppo_kl = torch.tensor([0.0])  # ratio = 1
        advantages = torch.tensor([-1.0])  # negative advantage
        eps_clip = 0.2
        eps_clip_high = 0.2
        eps_clip_c = 3.0

        pg_losses, _ = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c)

        # For negative advantages, dual-clip applies pg_losses3 = -eps_clip_c * advantages
        expected = torch.min(torch.tensor(-3.0 * -1.0), torch.tensor(-1.0 * -1.0))
        # The loss should be min(-3 * -1, -1 * -1) = min(3, 1) = 1
        torch.testing.assert_close(pg_losses, torch.tensor([1.0]))

    def test_eps_clip_c_must_be_greater_than_one(self, mock_megatron):
        """Test that eps_clip_c must be > 1.0."""
        from miles.utils.ppo_utils import compute_policy_loss

        ppo_kl = torch.tensor([0.0])
        advantages = torch.tensor([-1.0])

        with pytest.raises(AssertionError):
            compute_policy_loss(ppo_kl, advantages, 0.2, 0.2, eps_clip_c=0.5)


class TestGetGrpoReturns:
    """Tests for get_grpo_returns function."""

    def test_basic_grpo_returns(self, mock_megatron):
        """Test that GRPO returns broadcast rewards to match KL shapes."""
        from miles.utils.ppo_utils import get_grpo_returns

        rewards = torch.tensor([1.0, 2.0, 3.0])
        kl = [torch.zeros(5), torch.zeros(3), torch.zeros(7)]

        returns = get_grpo_returns(rewards, kl)

        assert len(returns) == 3
        torch.testing.assert_close(returns[0], torch.ones(5) * 1.0)
        torch.testing.assert_close(returns[1], torch.ones(3) * 2.0)
        torch.testing.assert_close(returns[2], torch.ones(7) * 3.0)


class TestGetReinforcePlusPlusBaselineAdvantages:
    """Tests for get_reinforce_plus_plus_baseline_advantages function."""

    def test_basic_advantages(self, mock_megatron):
        """Test REINFORCE++ baseline advantage computation."""
        from miles.utils.ppo_utils import get_reinforce_plus_plus_baseline_advantages

        rewards = torch.tensor([1.0, 2.0])  # Already baseline-subtracted
        kl = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])]
        loss_masks = [torch.ones(3), torch.ones(2)]
        kl_coef = 0.1

        advantages = get_reinforce_plus_plus_baseline_advantages(rewards, kl, loss_masks, kl_coef)

        assert len(advantages) == 2
        # advantage = reward - kl_coef * kl
        expected_0 = torch.ones(3) * 1.0 - 0.1 * kl[0]
        expected_1 = torch.ones(2) * 2.0 - 0.1 * kl[1]
        torch.testing.assert_close(advantages[0], expected_0)
        torch.testing.assert_close(advantages[1], expected_1)


class TestVanillaGAE:
    """Tests for vanilla_gae function."""

    def test_single_step_gae(self, mock_megatron):
        """Test GAE with single timestep."""
        from miles.utils.ppo_utils import vanilla_gae

        rewards = torch.tensor([[1.0]])
        values = torch.tensor([[0.5]])
        gamma = 0.99
        lambd = 0.95

        advantages, returns = vanilla_gae(rewards, values, gamma, lambd)

        # Single step: delta = r + gamma * 0 - V = 1 - 0.5 = 0.5
        # advantage = delta = 0.5
        # return = advantage + value = 0.5 + 0.5 = 1.0
        torch.testing.assert_close(advantages, torch.tensor([[0.5]]))
        torch.testing.assert_close(returns, torch.tensor([[1.0]]))

    def test_multi_step_gae(self, mock_megatron):
        """Test GAE with multiple timesteps."""
        from miles.utils.ppo_utils import vanilla_gae

        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[0.5, 1.0, 1.5]])
        gamma = 0.99
        lambd = 0.95

        advantages, returns = vanilla_gae(rewards, values, gamma, lambd)

        # Manual calculation for verification
        # delta[2] = r[2] + gamma * 0 - V[2] = 3 - 1.5 = 1.5
        # adv[2] = delta[2] = 1.5
        #
        # delta[1] = r[1] + gamma * V[2] - V[1] = 2 + 0.99 * 1.5 - 1 = 2.485
        # adv[1] = delta[1] + gamma * lambda * adv[2] = 2.485 + 0.99 * 0.95 * 1.5 = 3.89575
        #
        # ... similar for step 0

        assert advantages.shape == (1, 3)
        assert returns.shape == (1, 3)
        # Check that returns = advantages + values
        torch.testing.assert_close(returns, advantages + values)

    def test_batch_processing(self, mock_megatron):
        """Test GAE with batch dimension."""
        from miles.utils.ppo_utils import vanilla_gae

        batch_size = 4
        seq_len = 10
        rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        gamma = 0.99
        lambd = 0.95

        advantages, returns = vanilla_gae(rewards, values, gamma, lambd)

        assert advantages.shape == (batch_size, seq_len)
        assert returns.shape == (batch_size, seq_len)
        torch.testing.assert_close(returns, advantages + values)


class TestChunkedGAE:
    """Tests for chunked_gae function (FlashAttention-inspired algorithm)."""

    def test_matches_vanilla_gae(self, mock_megatron):
        """Test that chunked GAE produces same results as vanilla GAE."""
        from miles.utils.ppo_utils import chunked_gae, vanilla_gae

        torch.manual_seed(42)
        batch_size = 2
        seq_len = 256
        rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        gamma = 0.99
        lambd = 0.95

        vanilla_adv, vanilla_ret = vanilla_gae(rewards, values, gamma, lambd)
        chunked_adv, chunked_ret = chunked_gae(rewards, values, gamma, lambd, chunk_size=64)

        torch.testing.assert_close(chunked_adv, vanilla_adv, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(chunked_ret, vanilla_ret, rtol=1e-4, atol=1e-4)

    def test_different_chunk_sizes(self, mock_megatron):
        """Test chunked GAE with various chunk sizes."""
        from miles.utils.ppo_utils import chunked_gae, vanilla_gae

        torch.manual_seed(42)
        rewards = torch.randn(2, 200)
        values = torch.randn(2, 200)
        gamma = 0.99
        lambd = 0.95

        vanilla_adv, vanilla_ret = vanilla_gae(rewards, values, gamma, lambd)

        for chunk_size in [16, 32, 64, 128]:
            chunked_adv, chunked_ret = chunked_gae(rewards, values, gamma, lambd, chunk_size=chunk_size)
            torch.testing.assert_close(chunked_adv, vanilla_adv, rtol=1e-4, atol=1e-4)

    def test_non_divisible_sequence_length(self, mock_megatron):
        """Test chunked GAE when sequence length is not divisible by chunk size."""
        from miles.utils.ppo_utils import chunked_gae, vanilla_gae

        torch.manual_seed(42)
        # 137 is prime, not divisible by 64
        rewards = torch.randn(2, 137)
        values = torch.randn(2, 137)
        gamma = 0.99
        lambd = 0.95

        vanilla_adv, vanilla_ret = vanilla_gae(rewards, values, gamma, lambd)
        chunked_adv, chunked_ret = chunked_gae(rewards, values, gamma, lambd, chunk_size=64)

        torch.testing.assert_close(chunked_adv, vanilla_adv, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(chunked_ret, vanilla_ret, rtol=1e-4, atol=1e-4)

    def test_zero_discount(self, mock_megatron):
        """Test GAE with gamma=0 (no discounting)."""
        from miles.utils.ppo_utils import chunked_gae

        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[0.5, 1.0, 1.5]])
        gamma = 0.0
        lambd = 0.95

        advantages, returns = chunked_gae(rewards, values, gamma, lambd, chunk_size=2)

        # With gamma=0: delta = r - V, advantage = delta (no bootstrapping)
        expected_adv = rewards - values
        torch.testing.assert_close(advantages, expected_adv)


class TestComputeOpsmMask:
    """Tests for compute_opsm_mask function."""

    def test_opsm_mask_positive_advantage(self, mock_megatron):
        """Test OPSM mask with positive advantages (no masking)."""
        from miles.utils.ppo_utils import compute_opsm_mask

        args = Namespace(opsm_delta=0.5)
        full_log_probs = [torch.tensor([-1.0, -2.0])]
        full_old_log_probs = [torch.tensor([-1.5, -2.5])]
        advantages = [torch.tensor([1.0, 1.0])]  # Positive
        loss_masks = [torch.tensor([1.0, 1.0])]

        opsm_mask, clipfrac = compute_opsm_mask(
            args, full_log_probs, full_old_log_probs, advantages, loss_masks
        )

        # Positive advantage: mask should be all 1s (no masking)
        torch.testing.assert_close(opsm_mask, torch.tensor([1.0, 1.0]))

    def test_opsm_mask_negative_advantage_low_kl(self, mock_megatron):
        """Test OPSM mask with negative advantage but low KL (no masking)."""
        from miles.utils.ppo_utils import compute_opsm_mask

        args = Namespace(opsm_delta=1.0)
        # KL = (old - new) * mask / mask_sum = (0.5) / 1 = 0.5 < delta
        full_log_probs = [torch.tensor([-1.0])]
        full_old_log_probs = [torch.tensor([-0.5])]
        advantages = [torch.tensor([-1.0])]  # Negative
        loss_masks = [torch.tensor([1.0])]

        opsm_mask, clipfrac = compute_opsm_mask(
            args, full_log_probs, full_old_log_probs, advantages, loss_masks
        )

        # Negative advantage but KL < delta: mask should be 1
        torch.testing.assert_close(opsm_mask, torch.tensor([1.0]))

    def test_opsm_mask_negative_advantage_high_kl(self, mock_megatron):
        """Test OPSM mask with negative advantage and high KL (masking)."""
        from miles.utils.ppo_utils import compute_opsm_mask

        args = Namespace(opsm_delta=0.1)
        # KL = (old - new) * mask / mask_sum = 2.0 > delta
        full_log_probs = [torch.tensor([-1.0])]
        full_old_log_probs = [torch.tensor([1.0])]
        advantages = [torch.tensor([-1.0])]  # Negative
        loss_masks = [torch.tensor([1.0])]

        opsm_mask, clipfrac = compute_opsm_mask(
            args, full_log_probs, full_old_log_probs, advantages, loss_masks
        )

        # Negative advantage and KL > delta: mask should be 0
        torch.testing.assert_close(opsm_mask, torch.tensor([0.0]))


class TestComputeGspoKL:
    """Tests for compute_gspo_kl function."""

    def test_gspo_kl_basic(self, mock_megatron):
        """Test GSPO KL computation expands sequence-level KL to tokens."""
        from miles.utils.ppo_utils import compute_gspo_kl

        full_log_probs = [torch.tensor([-1.0, -2.0, -3.0])]
        full_old_log_probs = [torch.tensor([-1.5, -2.5, -3.5])]
        local_log_probs = [torch.tensor([-1.0, -2.0, -3.0])]
        loss_masks = [torch.tensor([1.0, 1.0, 1.0])]

        ppo_kl = compute_gspo_kl(full_log_probs, full_old_log_probs, local_log_probs, loss_masks)

        # Sequence KL = sum((old - new) * mask) / sum(mask)
        # = (-1.5 - (-1)) + (-2.5 - (-2)) + (-3.5 - (-3)) / 3
        # = (-0.5 - 0.5 - 0.5) / 3 = -1.5 / 3 = -0.5
        expected_kl = torch.tensor([-0.5, -0.5, -0.5])
        torch.testing.assert_close(ppo_kl, expected_kl)

    def test_gspo_kl_multiple_sequences(self, mock_megatron):
        """Test GSPO KL with multiple sequences."""
        from miles.utils.ppo_utils import compute_gspo_kl

        full_log_probs = [torch.tensor([-1.0, -2.0]), torch.tensor([-3.0])]
        full_old_log_probs = [torch.tensor([-1.5, -2.5]), torch.tensor([-3.5])]
        local_log_probs = [torch.tensor([-1.0, -2.0]), torch.tensor([-3.0])]
        loss_masks = [torch.tensor([1.0, 1.0]), torch.tensor([1.0])]

        ppo_kl = compute_gspo_kl(full_log_probs, full_old_log_probs, local_log_probs, loss_masks)

        # Should have 3 elements total (2 + 1)
        assert ppo_kl.shape == torch.Size([3])


# Edge case tests
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_batch(self, mock_megatron):
        """Test functions handle empty batches gracefully."""
        from miles.utils.ppo_utils import vanilla_gae

        rewards = torch.zeros(0, 10)
        values = torch.zeros(0, 10)

        advantages, returns = vanilla_gae(rewards, values, 0.99, 0.95)

        assert advantages.shape == (0, 10)
        assert returns.shape == (0, 10)

    def test_single_element_sequence(self, mock_megatron):
        """Test functions with single-element sequences."""
        from miles.utils.ppo_utils import vanilla_gae

        rewards = torch.tensor([[5.0]])
        values = torch.tensor([[2.0]])

        advantages, returns = vanilla_gae(rewards, values, 0.99, 0.95)

        # advantage = reward - value = 5 - 2 = 3
        # return = advantage + value = 3 + 2 = 5
        torch.testing.assert_close(advantages, torch.tensor([[3.0]]))
        torch.testing.assert_close(returns, torch.tensor([[5.0]]))

    def test_numerical_stability_large_values(self, mock_megatron):
        """Test numerical stability with large values."""
        from miles.utils.ppo_utils import compute_approx_kl

        log_probs = torch.tensor([100.0])
        log_probs_base = torch.tensor([100.5])

        # k1 should handle large values fine
        kl = compute_approx_kl(log_probs, log_probs_base, "k1")
        assert torch.isfinite(kl).all()

        # low_var_kl should clamp
        kl_lv = compute_approx_kl(log_probs, log_probs_base, "low_var_kl")
        assert torch.isfinite(kl_lv).all()
        assert kl_lv.abs().item() <= 10.0

    def test_gpu_compatibility(self, mock_megatron):
        """Test that functions work with GPU tensors when available."""
        from miles.utils.ppo_utils import compute_approx_kl

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        log_probs = torch.tensor([-1.0, -2.0]).cuda()
        log_probs_base = torch.tensor([-1.5, -2.5]).cuda()

        kl = compute_approx_kl(log_probs, log_probs_base, "k1")

        assert kl.device.type == "cuda"
        torch.testing.assert_close(kl, log_probs - log_probs_base)
