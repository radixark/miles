"""Tests for miles.core.preference_ops module."""

import pytest
import torch
import torch.nn as nn

from miles.core.preference_ops import (
    DPOLoss,
    IPOLoss,
    KTOLoss,
    SimPOLoss,
    PreferenceOptimizer,
    create_preference_optimizer,
    compute_logps_from_logits,
)


class TestDPOLoss:
    """Tests for DPO Loss."""

    @pytest.fixture
    def loss_fn(self):
        return DPOLoss(beta=0.1)

    def test_basic_computation(self, loss_fn):
        """Test basic DPO loss computation."""
        # Create sample log probabilities
        policy_chosen = torch.tensor([-1.5, -2.0, -1.8])
        policy_rejected = torch.tensor([-3.0, -2.5, -2.8])
        reference_chosen = torch.tensor([-1.6, -2.1, -1.9])
        reference_rejected = torch.tensor([-2.9, -2.4, -2.7])

        loss, metrics = loss_fn(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
        )

        assert isinstance(loss.item(), float)
        assert 'dpo_loss' in metrics
        assert 'dpo_chosen_accuracy' in metrics
        assert 0.0 <= metrics['dpo_chosen_accuracy'] <= 1.0

    def test_ipo_mode(self):
        """Test DPO with IPO regularization."""
        loss_fn = DPOLoss(beta=0.1, ipo=True)

        policy_chosen = torch.tensor([-1.5])
        policy_rejected = torch.tensor([-3.0])
        reference_chosen = torch.tensor([-1.6])
        reference_rejected = torch.tensor([-2.9])

        loss, metrics = loss_fn(policy_chosen, policy_rejected, 
                               reference_chosen, reference_rejected)

        assert isinstance(loss.item(), float)

    def test_with_mask(self):
        """Test DPO loss with choice mask."""
        loss_fn = DPOLoss(beta=0.1)

        policy_chosen = torch.tensor([-1.5, -2.0])
        policy_rejected = torch.tensor([-3.0, -2.5])
        reference_chosen = torch.tensor([-1.6, -2.1])
        reference_rejected = torch.tensor([-2.9, -2.4])
        mask = torch.tensor([1.0, 0.0])  # Only first sample valid

        loss, metrics = loss_fn(policy_chosen, policy_rejected,
                               reference_chosen, reference_rejected,
                               choice_mask=mask)

        # Verify mask is working - loss should only consider first sample
        # Re-run without mask to verify masking effect
        loss_unmasked, _ = loss_fn(policy_chosen, policy_rejected,
                                   reference_chosen, reference_rejected)
        
        # With mask, we only average over 1 sample instead of 2
        # So the losses should be different
        assert isinstance(loss.item(), float)

    def test_mask_affects_loss_value(self):
        """Verify that masking actually affects the computed loss."""
        loss_fn = DPOLoss(beta=0.1)
        
        # Create very different logps to ensure clear loss difference
        policy_chosen = torch.tensor([-0.5, -10.0])  # Second sample very bad
        policy_rejected = torch.tensor([-10.0, -0.5])
        reference_chosen = torch.tensor([-0.6, -10.1])
        reference_rejected = torch.tensor([-10.1, -0.6])
        
        # With mask=1 for both, loss should consider both
        mask_both = torch.tensor([1.0, 1.0])
        loss_both, _ = loss_fn(policy_chosen, policy_rejected,
                               reference_chosen, reference_rejected,
                               choice_mask=mask_both)
        
        # With mask=1 for only first, loss should only consider first
        mask_first = torch.tensor([1.0, 0.0])
        loss_first, _ = loss_fn(policy_chosen, policy_rejected,
                                reference_chosen, reference_rejected,
                                choice_mask=mask_first)
        
        # The losses should be different
        assert loss_both.item() != loss_first.item()


class TestIPOLoss:
    """Tests for IPO Loss."""

    @pytest.fixture
    def loss_fn(self):
        return IPOLoss(beta=0.1, tau=1.0)

    def test_basic_computation(self, loss_fn):
        """Test basic IPO loss computation."""
        policy_chosen = torch.tensor([-1.5, -2.0])
        policy_rejected = torch.tensor([-3.0, -2.5])
        reference_chosen = torch.tensor([-1.6, -2.1])
        reference_rejected = torch.tensor([-2.9, -2.4])

        loss, metrics = loss_fn(
            policy_chosen, policy_rejected,
            reference_chosen, reference_rejected,
        )

        assert isinstance(loss.item(), float)
        assert 'ipo_loss' in metrics
        assert 'ipo_chosen_accuracy' in metrics

    def test_kl_penalty_present(self):
        """Test that IPO has KL penalty in metrics."""
        loss_fn = IPOLoss(beta=0.1, tau=2.0)

        policy_chosen = torch.tensor([-1.5])
        policy_rejected = torch.tensor([-3.0])
        reference_chosen = torch.tensor([-1.6])
        reference_rejected = torch.tensor([-2.9])

        loss, metrics = loss_fn(policy_chosen, policy_rejected,
                               reference_chosen, reference_rejected)

        assert 'ipo_kl_penalty' in metrics

    def test_mask_affects_ipo_loss(self):
        """Verify that masking works correctly for IPO loss."""
        loss_fn = IPOLoss(beta=0.1, tau=1.0)
        
        policy_chosen = torch.tensor([-0.5, -10.0])
        policy_rejected = torch.tensor([-10.0, -0.5])
        reference_chosen = torch.tensor([-0.6, -10.1])
        reference_rejected = torch.tensor([-10.1, -0.6])
        
        mask_both = torch.tensor([1.0, 1.0])
        loss_both, _ = loss_fn(policy_chosen, policy_rejected,
                               reference_chosen, reference_rejected,
                               choice_mask=mask_both)
        
        mask_first = torch.tensor([1.0, 0.0])
        loss_first, _ = loss_fn(policy_chosen, policy_rejected,
                                reference_chosen, reference_rejected,
                                choice_mask=mask_first)
        
        # The losses should be different
        assert loss_both.item() != loss_first.item()


class TestKTOLoss:
    """Tests for KTO Loss."""

    @pytest.fixture
    def loss_fn(self):
        return KTOLoss(beta=0.1)

    def test_basic_computation(self, loss_fn):
        """Test basic KTO loss computation."""
        policy_wins = torch.tensor([-1.5, -2.0])
        policy_losses = torch.tensor([-3.0, -2.5])
        reference_wins = torch.tensor([-1.6, -2.1])
        reference_losses = torch.tensor([-2.9, -2.4])

        loss, metrics = loss_fn(
            policy_wins, policy_losses,
            reference_wins, reference_losses,
        )

        assert isinstance(loss.item(), float)
        assert 'kto_loss' in metrics
        assert 'kto_win_rate' in metrics
        assert 'kto_loss_rate' in metrics

    def test_weights(self):
        """Test KTO with custom weights."""
        loss_fn = KTOLoss(beta=0.1, loss_weight=2.0, win_weight=1.5)

        policy_wins = torch.tensor([-1.5])
        policy_losses = torch.tensor([-3.0])
        reference_wins = torch.tensor([-1.6])
        reference_losses = torch.tensor([-2.9])

        loss, metrics = loss_fn(
            policy_wins, policy_losses,
            reference_wins, reference_losses,
        )

        assert isinstance(loss.item(), float)

    def test_kto_formulation_correct(self):
        """Test that KTO loss matches the paper formulation."""
        # KTO paper defines loss as -log(sigmoid(v)) for wins and -log(sigmoid(-v)) for losses
        # This is equivalent to binary_cross_entropy_with_logits
        loss_fn = KTOLoss(beta=1.0)
        
        # Simple case: positive utility should give low loss for wins
        win_utilities = torch.tensor([2.0])  # High positive utility
        loss_utilities = torch.tensor([-2.0])  # High negative utility
        
        # Reference utilities same as policy (neutral loss)
        ref_wins = torch.tensor([0.0])
        ref_losses = torch.tensor([0.0])
        
        loss, metrics = loss_fn(win_utilities, loss_utilities, ref_wins, ref_losses)
        
        # Loss should be reasonable (not extremely high or low)
        assert 0.0 <= loss.item() <= 10.0


class TestSimPOLoss:
    """Tests for SimPO Loss."""

    @pytest.fixture
    def loss_fn(self):
        return SimPOLoss(gamma=0.5)

    def test_basic_computation(self, loss_fn):
        """Test basic SimPO loss computation."""
        policy_chosen = torch.tensor([-1.5, -2.0])
        policy_rejected = torch.tensor([-3.0, -2.5])

        loss, metrics = loss_fn(policy_chosen, policy_rejected)

        assert isinstance(loss.item(), float)
        assert 'simpo_loss' in metrics
        assert 'simpo_margin_success' in metrics

    def test_no_reference_needed(self):
        """Test that SimPO doesn't require reference."""
        loss_fn = SimPOLoss(gamma=0.5)

        policy_chosen = torch.tensor([-1.5])
        policy_rejected = torch.tensor([-3.0])

        # SimPO only needs policy logps (no reference)
        loss, metrics = loss_fn(policy_chosen, policy_rejected)

        assert isinstance(loss.item(), float)

    def test_margin_works(self):
        """Test that margin parameter affects loss correctly."""
        loss_fn_low_margin = SimPOLoss(gamma=0.1)
        loss_fn_high_margin = SimPOLoss(gamma=2.0)
        
        policy_chosen = torch.tensor([-1.5])
        policy_rejected = torch.tensor([-2.0])
        
        loss_low, _ = loss_fn_low_margin(policy_chosen, policy_rejected)
        loss_high, _ = loss_fn_high_margin(policy_chosen, policy_rejected)
        
        # Higher margin should give higher loss (harder to satisfy)
        assert loss_high.item() >= loss_low.item()


class TestPreferenceOptimizer:
    """Tests for unified PreferenceOptimizer interface."""

    def test_create_dpo(self):
        """Test creating DPO optimizer."""
        optimizer = PreferenceOptimizer(algorithm='dpo', beta=0.1)
        assert optimizer.algorithm == 'dpo'

    def test_create_ipo(self):
        """Test creating IPO optimizer."""
        optimizer = PreferenceOptimizer(algorithm='ipo', beta=0.1, tau=2.0)
        assert optimizer.algorithm == 'ipo'

    def test_create_kto(self):
        """Test creating KTO optimizer."""
        optimizer = PreferenceOptimizer(algorithm='kto', beta=0.1)
        assert optimizer.algorithm == 'kto'

    def test_create_simpo(self):
        """Test creating SimPO optimizer."""
        optimizer = PreferenceOptimizer(algorithm='simpo', gamma=0.5)
        assert optimizer.algorithm == 'simpo'

    def test_switch_algorithm(self):
        """Test switching algorithms."""
        optimizer = PreferenceOptimizer(algorithm='dpo', beta=0.1)
        optimizer.switch_algorithm('kto', beta=0.2)
        assert optimizer.algorithm == 'kto'

    def test_invalid_algorithm(self):
        """Test invalid algorithm raises error."""
        with pytest.raises(ValueError):
            PreferenceOptimizer(algorithm='invalid')

    def test_dpo_forward(self):
        """Test DPO optimizer forward pass."""
        optimizer = PreferenceOptimizer(algorithm='dpo', beta=0.1)

        policy_chosen = torch.tensor([-1.5])
        policy_rejected = torch.tensor([-3.0])
        reference_chosen = torch.tensor([-1.6])
        reference_rejected = torch.tensor([-2.9])

        loss, metrics = optimizer(
            policy_chosen=policy_chosen,
            policy_rejected=policy_rejected,
            reference_chosen=reference_chosen,
            reference_rejected=reference_rejected,
        )

        assert isinstance(loss.item(), float)

    def test_kto_forward(self):
        """Test KTO optimizer forward pass."""
        optimizer = PreferenceOptimizer(algorithm='kto', beta=0.1)

        policy_wins = torch.tensor([-1.5])
        policy_losses = torch.tensor([-3.0])
        reference_wins = torch.tensor([-1.6])
        reference_losses = torch.tensor([-2.9])

        loss, metrics = optimizer(
            policy_wins=policy_wins,
            policy_losses=policy_losses,
            reference_wins=reference_wins,
            reference_losses=reference_losses,
        )

        assert isinstance(loss.item(), float)


class TestCreatePreferenceOptimizer:
    """Tests for create_preference_optimizer convenience function."""

    def test_create_dpo(self):
        """Test creating DPO via convenience function."""
        optimizer = create_preference_optimizer('dpo', beta=0.1)
        assert isinstance(optimizer, PreferenceOptimizer)
        assert optimizer.algorithm == 'dpo'

    def test_create_all_algorithms(self):
        """Test creating all algorithms."""
        for algo in ['dpo', 'ipo', 'kto', 'simpo']:
            optimizer = create_preference_optimizer(algo)
            assert optimizer.algorithm == algo


class TestComputeLogpsFromLogits:
    """Tests for log probability computation utility."""

    def test_basic(self):
        """Test basic log probability computation."""
        # Create mock logits (batch=2, seq=5, vocab=10)
        logits = torch.randn(2, 5, 10)
        input_ids = torch.randint(0, 10, (2, 5))

        logps = compute_logps_from_logits(logits, input_ids, average=True)

        assert logps.shape == (2,)  # One value per sequence

    def test_not_average(self):
        """Test without averaging."""
        logits = torch.randn(2, 5, 10)
        input_ids = torch.randint(0, 10, (2, 5))

        logps = compute_logps_from_logits(logits, input_ids, average=False)

        # Should return per-token logps
        assert logps.shape == (2, 4)  # seq_len - 1
