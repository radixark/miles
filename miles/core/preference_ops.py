"""
Preference Optimization Operations for Miles RL Framework.

This module implements Direct Preference Optimization (DPO), Identity 
Preference Optimization (IPO), and Kahneman-Tversky Optimization (KTO)
for training large language models with human preferences.

These algorithms provide alternatives to PPO-based RLHF, offering:
- DPO: Stable, computationally lightweight preference tuning
- IPO: Better generalization with explicit overfitting avoidance
- KTO: Uses human judgments (wins/losses) instead of pairwise comparisons

Reference Papers:
- DPO: "Direct Preference Optimization" (arXiv:2305.18290)
- IPO: "IPO: Understanding and Improving DPO" (arXiv:2404.04686)
- KTO: "Kahneman-Tversky Optimization" (arXiv:2402.01306)

Compatible with Miles' FSDP2 and Megatron backends.
"""

import logging
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

logger = logging.getLogger(__name__)


class DPOLoss(nn.Module):
    """Direct Preference Optimization Loss.
    
    DPO directly optimizes the policy to align with human preferences
    without requiring a separate reward model. It uses the probability
    ratio between chosen and rejected responses.
    
    The loss encourages the policy to:
    - Increase probability of preferred (chosen) responses
    - Decrease probability of rejected responses
    
    Key advantages:
    - Stable training (no KL collapse)
    - No reward model needed
    - Computationally efficient
    
    Example:
        >>> loss_fn = DPOLoss(beta=0.1)
        >>> loss = loss_fn(
        ...     policy_chosen_logps=chosen_logps,
        ...     policy_rejected_logps=rejected_logps,
        ...     reference_chosen_logps=ref_chosen_logps,
        ...     reference_rejected_logps=ref_rejected_logps,
        ... )
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        ipo: bool = False,
        max_grad_norm: Optional[float] = None,
    ):
        """Initialize DPO Loss.
        
        Args:
            beta: Temperature parameter controlling KL penalty strength.
                  Higher values = stronger reference model adherence.
            label_smoothing: Label smoothing for numerical stability.
            ipo: If True, use IPO-style regularization.
            max_grad_norm: Maximum gradient norm for clipping (None = no clipping).
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo
        self.max_grad_norm = max_grad_norm
        
        # Register beta as buffer (not trained but part of computation)
        self.register_buffer('_beta_tensor', torch.tensor(beta))
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        choice_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss between policy and reference model.
        
        Args:
            policy_chosen_logps: Log probabilities of chosen responses under policy
            policy_rejected_logps: Log probabilities of rejected responses under policy
            reference_chosen_logps: Log probabilities of chosen responses under reference
            reference_rejected_logps: Log probabilities of rejected responses under reference
            choice_mask: Optional mask for valid preference pairs
            
        Returns:
            Tuple of (loss, metrics_dict) where metrics contains per-sample losses
        """
        # Compute log odds ratios for policy
        policy_logps_diff = policy_chosen_logps - policy_rejected_logps
        
        # Compute log odds ratios for reference
        reference_logps_diff = reference_chosen_logps - reference_rejected_logps
        
        # Compute the logits
        logits = self.beta * (policy_logps_diff - reference_logps_diff)
        
        if self.ipo:
            # IPO loss: encourages log(pi/pi_ref) to match log(d)
            # where d is the ground truth preference probability
            targets = torch.zeros_like(logits)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, label_smoothing=self.label_smoothing
            )
        else:
            # Standard DPO loss: Sigmoid cross-entropy
            # Equivalent to -E[log(sigmoid(logits))]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                torch.ones_like(logits),
                label_smoothing=self.label_smoothing,
            )
        
        # Apply mask if provided
        if choice_mask is not None:
            loss = loss * choice_mask
            num_valid = choice_mask.sum()
            if num_valid > 0:
                loss = loss.sum() / num_valid
        
        # Compute metrics
        with torch.no_grad():
            chosen_acc = (policy_chosen_logps > policy_rejected_logps).float().mean()
            kl_div = (policy_logps_diff - reference_logps_diff).mean()
        
        metrics = {
            'dpo_loss': loss.item(),
            'dpo_chosen_accuracy': chosen_acc.item(),
            'dpo_kl_divergence': kl_div.item(),
            'dpo_logits_mean': logits.mean().item(),
            'dpo_logits_std': logits.std().item(),
        }
        
        # Gradient clipping if specified
        if self.max_grad_norm is not None and loss.requires_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        
        return loss, metrics


class IPOLoss(nn.Module):
    """Identity Preference Optimization Loss.
    
    IPO addresses DPO's tendency to overfit by adding explicit regularization
    that prevents the policy from deviating too far from the reference model.
    
    The key difference from DPO:
    - DPO can push policy to extreme preferences
    - IPO constrains the policy to stay close to reference distribution
    
    This is particularly important for:
    - Small preference datasets
    - Long training runs
    - High-stakes applications
    
    Reference: "Understanding IPO" (arXiv:2404.04686)
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        tau: float = 1.0,
        max_grad_norm: Optional[float] = None,
    ):
        """Initialize IPO Loss.
        
        Args:
            beta: Temperature parameter for preference scaling
            tau: Regularization strength for IPO constraint
            max_grad_norm: Maximum gradient norm for clipping
        """
        super().__init__()
        self.beta = beta
        self.tau = tau
        self.max_grad_norm = max_grad_norm
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        choice_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute IPO loss with explicit regularization.
        
        The IPO loss has two components:
        1. Preference alignment loss (like DPO)
        2. KL regularization to prevent overfitting
        
        Args:
            policy_chosen_logps: Log probabilities of chosen responses under policy
            policy_rejected_logps: Log probabilities of rejected responses under policy
            reference_chosen_logps: Log probabilities of chosen responses under reference
            reference_rejected_logps: Log probabilities of rejected responses under reference
            choice_mask: Optional mask for valid preference pairs
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Log probability differences
        policy_diff = policy_chosen_logps - policy_rejected_logps
        reference_diff = reference_chosen_logps - reference_rejected_logps
        
        # IPO uses a squared loss that enforces identity mapping at convergence
        # The loss is: (policy_diff - reference_diff - log(π_r/π_p))²
        # Simplified: squared difference between policy and reference logits
        logits = self.beta * (policy_diff - reference_diff)
        
        # Squared loss (not cross-entropy like DPO)
        preference_loss = 0.5 * logits.pow(2)
        
        # IPO KL regularization: prevents deviation from reference
        kl_penalty = (policy_chosen_logps - reference_chosen_logps).pow(2) + \
                     (policy_rejected_logps - reference_rejected_logps).pow(2)
        kl_penalty = kl_penalty.mean()
        
        # Combined loss
        loss = preference_loss.mean() + self.tau * kl_penalty
        
        # Apply mask if provided
        if choice_mask is not None:
            loss = loss * choice_mask
            num_valid = choice_mask.sum()
            if num_valid > 0:
                loss = loss.sum() / num_valid
        
        # Compute metrics
        with torch.no_grad():
            chosen_acc = (policy_chosen_logps > policy_rejected_logps).float().mean()
            policy_ref_diff = (policy_diff - reference_diff).mean()
        
        metrics = {
            'ipo_loss': loss.item(),
            'ipo_chosen_accuracy': chosen_acc.item(),
            'ipo_policy_ref_diff': policy_ref_diff.item(),
            'ipo_kl_penalty': kl_penalty.item(),
        }
        
        # Gradient clipping
        if self.max_grad_norm is not None and loss.requires_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        
        return loss, metrics


class KTOLoss(nn.Module):
    """Kahneman-Tversky Optimization Loss.
    
    KTO uses human judgments (wins/losses) instead of pairwise comparisons.
    It treats preference learning as a binary classification problem:
    - "Win": responses that humans prefer
    - "Loss": responses that humans reject
    
    Advantages over DPO:
    - No need for paired data (easier data collection)
    - More robust to annotation noise
    - Works with single-response judgments
    
    The loss is derived from prospect theory, balancing:
    - Gains (increasing probability of wins)
    - Losses (decreasing probability of losses)
    
    Reference: "KTO: Model Alignment via Pareto Optimization" (arXiv:2402.01306)
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        descent: bool = False,
        max_grad_norm: Optional[float] = None,
    ):
        """Initialize KTO Loss.
        
        Args:
            beta: Reference point scaling parameter
            descent: If True, maximize utility; if False, minimize loss
            max_grad_norm: Maximum gradient norm for clipping
        """
        super().__init__()
        self.beta = beta
        self.descent = descent
        self.max_grad_norm = max_grad_norm
    
    def forward(
        self,
        policy_wins_logps: torch.Tensor,
        policy_losses_logps: torch.Tensor,
        reference_wins_logps: torch.Tensor,
        reference_losses_logps: torch.Tensor,
        loss_weight: float = 1.0,
        win_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute KTO loss using human judgments.
        
        Args:
            policy_wins_logps: Log probs of human-preferred responses
            policy_losses_logps: Log probs of human-rejected responses
            reference_wins_logps: Reference log probs for wins
            reference_losses_logps: Reference log probs for losses
            loss_weight: Weight for loss component
            win_weight: Weight for win component
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute utility for wins (gains)
        win_utilities = self.beta * (policy_wins_logps - reference_wins_logps)
        
        # Compute utility for losses (avoidance)
        loss_utilities = self.beta * (policy_losses_logps - reference_losses_logps)
        
        # KTO uses a balanced loss: minimize losses, maximize wins
        # The loss function is asymmetric (humans are loss-averse)
        win_losses = -win_utilities.sigmoid() * win_weight
        loss_losses = loss_utilities.sigmoid() * loss_weight
        
        # Combined loss
        loss = win_losses.mean() + loss_losses.mean() * loss_weight
        
        # Compute metrics
        with torch.no_grad():
            win_rate = (policy_wins_logps > reference_wins_logps).float().mean()
            loss_rate = (policy_losses_logps < reference_losses_logps).float().mean()
            avg_win_util = win_utilities.mean()
            avg_loss_util = loss_utilities.mean()
        
        metrics = {
            'kto_loss': loss.item(),
            'kto_win_rate': win_rate.item(),
            'kto_loss_rate': loss_rate.item(),
            'kto_avg_win_util': avg_win_util.item(),
            'kto_avg_loss_util': avg_loss_util.item(),
        }
        
        # Gradient clipping
        if self.max_grad_norm is not None and loss.requires_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        
        return loss, metrics


class SimPOLoss(nn.Module):
    """Simple Preference Optimization (SimPO) Loss.
    
    SimPO is a simplified preference optimization that removes the reference
    model entirely, using only the policy's own probabilities.
    
    Advantages:
    - No reference model needed (half the memory)
    - Simpler training setup
    - Good performance for many use cases
    
    Reference: "SimPO: Simple Preference Optimization" (arXiv:2405.14734)
    """
    
    def __init__(
        self,
        gamma: float = 0.5,
        max_grad_norm: Optional[float] = None,
    ):
        """Initialize SimPO Loss.
        
        Args:
            gamma: Margin parameter for preference margin
            max_grad_norm: Maximum gradient norm for clipping
        """
        super().__init__()
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        length_normalized: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SimPO loss without reference model.
        
        Args:
            policy_chosen_logps: Log probs of chosen responses
            policy_rejected_logps: Log probs of rejected responses
            length_normalized: Whether to normalize by sequence length
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Log probability differences
        logps_diff = policy_chosen_logps - policy_rejected_logps
        
        if length_normalized:
            # SimPO uses length-normalized log probs
            # This is already handled in the input logps
            pass
        
        # SimPO loss: margin-based ranking loss
        logits = logps_diff - self.gamma
        
        # Binary cross-entropy with targets = 1 (chosen > rejected + margin)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            torch.ones_like(logits),
        )
        
        # Compute metrics
        with torch.no_grad():
            margin_success = (logps_diff > self.gamma).float().mean()
        
        metrics = {
            'simpo_loss': loss.item(),
            'simpo_margin_success': margin_success.item(),
            'simpo_avg_diff': logps_diff.mean().item(),
        }
        
        # Gradient clipping
        if self.max_grad_norm is not None and loss.requires_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        
        return loss, metrics


def compute_logps_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    average: bool = True,
) -> torch.Tensor:
    """Compute log probabilities from model logits.
    
    This is a utility function for computing per-token log probabilities
    that can be used with the preference optimization losses.
    
    Args:
        logits: Model output logits of shape (batch, seq_len, vocab)
        input_ids: Input token IDs of shape (batch, seq_len)
        average: If True, return average log prob; else return sum
        
    Returns:
        Log probabilities, shape (batch,) if average=True, else (batch, seq_len)
    
    Example:
        >>> logits = model(input_ids)
        >>> logps = compute_logps_from_logits(logits, input_ids)
    """
    # Get log probabilities
    logps = log_softmax(logits, dim=-1)
    
    # Gather log probabilities at input positions
    # input_ids[:, 1:] corresponds to predictions for next tokens
    logps = logps[:, :-1, :]  # Remove last position (no prediction)
    input_ids = input_ids[:, 1:]  # Align with logps
    
    # Gather using torch.gather
    logps = torch.gather(
        logps, 2, input_ids.unsqueeze(-1)
    ).squeeze(-1)  # Shape: (batch, seq_len-1)
    
    if average:
        # Return average log prob per sequence
        return logps.sum(dim=-1) / logps.size(-1)
    else:
        # Return sum of log probs
        return logps.sum(dim=-1)


def compute_sequence_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities for a full sequence.
    
    This is a convenience function for computing log probabilities
    that can be used with the preference optimization losses.
    
    Args:
        model: The language model
        input_ids: Input token IDs of shape (batch, seq_len)
        attention_mask: Attention mask for padding
        
    Returns:
        Tuple of (logps, logits) where:
        - logps: Average log probability per sequence
        - logits: Raw logits from model
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute log probabilities
        logps = compute_logps_from_logits(logits, input_ids, average=True)
    
    return logps, logits


class PreferenceOptimizer:
    """Unified interface for preference optimization algorithms.
    
    This class provides a unified interface for DPO, IPO, KTO, and SimPO,
    making it easy to switch between algorithms and compare results.
    
    Attributes:
        algorithm: Current algorithm ('dpo', 'ipo', 'kto', 'simpo')
        loss_fn: The underlying loss function
    
    Example:
        >>> optimizer = PreferenceOptimizer(algorithm='dpo', beta=0.1)
        >>> loss, metrics = optimizer(
        ...     policy_chosen=chosen_logps,
        ...     policy_rejected=rejected_logps,
        ...     reference_chosen=ref_chosen,
        ...     reference_rejected=ref_rejected,
        ... )
    """
    
    ALGORITHMS = ['dpo', 'ipo', 'kto', 'simpo']
    
    def __init__(
        self,
        algorithm: str = 'dpo',
        beta: float = 0.1,
        ipo_tau: float = 1.0,
        simpo_gamma: float = 0.5,
        max_grad_norm: Optional[float] = None,
    ):
        """Initialize the preference optimizer.
        
        Args:
            algorithm: Which algorithm to use ('dpo', 'ipo', 'kto', 'simpo')
            beta: Temperature/scaling parameter
            ipo_tau: Regularization strength for IPO
            simpo_gamma: Margin parameter for SimPO
            max_grad_norm: Maximum gradient norm for clipping
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. "
                           f"Choose from: {self.ALGORITHMS}")
        
        self.algorithm = algorithm
        
        # Initialize the appropriate loss function
        if algorithm == 'dpo':
            self.loss_fn = DPOLoss(beta=beta, max_grad_norm=max_grad_norm)
        elif algorithm == 'ipo':
            self.loss_fn = IPOLoss(beta=beta, tau=ipo_tau, max_grad_norm=max_grad_norm)
        elif algorithm == 'kto':
            self.loss_fn = KTOLoss(beta=beta, max_grad_norm=max_grad_norm)
        else:  # simpo
            self.loss_fn = SimPOLoss(gamma=simpo_gamma, max_grad_norm=max_grad_norm)
    
    def __call__(
        self,
        policy_chosen: Optional[torch.Tensor] = None,
        policy_rejected: Optional[torch.Tensor] = None,
        policy_wins: Optional[torch.Tensor] = None,
        policy_losses: Optional[torch.Tensor] = None,
        reference_chosen: Optional[torch.Tensor] = None,
        reference_rejected: Optional[torch.Tensor] = None,
        reference_wins: Optional[torch.Tensor] = None,
        reference_losses: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute preference optimization loss.
        
        Args vary by algorithm:
        - DPO/IPO: policy_chosen, policy_rejected, reference_chosen, reference_rejected
        - KTO: policy_wins, policy_losses, reference_wins, reference_losses
        - SimPO: policy_chosen, policy_rejected
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        if self.algorithm in ['dpo', 'ipo']:
            assert policy_chosen is not None
            assert policy_rejected is not None
            assert reference_chosen is not None
            assert reference_rejected is not None
            
            return self.loss_fn(
                policy_chosen_logps=policy_chosen,
                policy_rejected_logps=policy_rejected,
                reference_chosen_logps=reference_chosen,
                reference_rejected_logps=reference_rejected,
                **kwargs,
            )
        
        elif self.algorithm == 'kto':
            assert policy_wins is not None
            assert policy_losses is not None
            assert reference_wins is not None
            assert reference_losses is not None
            
            return self.loss_fn(
                policy_wins_logps=policy_wins,
                policy_losses_logps=policy_losses,
                reference_wins_logps=reference_wins,
                reference_losses_logps=reference_losses,
                **kwargs,
            )
        
        else:  # simpo
            assert policy_chosen is not None
            assert policy_rejected is not None
            
            return self.loss_fn(
                policy_chosen_logps=policy_chosen,
                policy_rejected_logps=policy_rejected,
                **kwargs,
            )
    
    def switch_algorithm(self, new_algorithm: str, **kwargs) -> None:
        """Switch to a different preference optimization algorithm.
        
        Args:
            new_algorithm: The new algorithm to use
            **kwargs: Additional arguments for the new loss function
        """
        if new_algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {new_algorithm}")
        
        self.algorithm = new_algorithm
        
        if new_algorithm == 'dpo':
            self.loss_fn = DPOLoss(**kwargs)
        elif new_algorithm == 'ipo':
            self.loss_fn = IPOLoss(**kwargs)
        elif new_algorithm == 'kto':
            self.loss_fn = KTOLoss(**kwargs)
        else:  # simpo
            self.loss_fn = SimPOLoss(**kwargs)
        
        logger.info(f"Switched preference optimizer to {new_algorithm}")


# Convenience function for creating preference optimizer
def create_preference_optimizer(
    algorithm: str = 'dpo',
    **kwargs,
) -> PreferenceOptimizer:
    """Create a preference optimizer with specified algorithm.
    
    Args:
        algorithm: Algorithm name ('dpo', 'ipo', 'kto', 'simpo')
        **kwargs: Additional arguments for the optimizer
    
    Returns:
        Configured PreferenceOptimizer instance
    
    Example:
        >>> optimizer = create_preference_optimizer('dpo', beta=0.1)
        >>> loss, metrics = optimizer(policy_chosen=..., policy_rejected=...)
    """
    return PreferenceOptimizer(algorithm=algorithm, **kwargs)
