import pytest
import torch

from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy


class TestCalculateLogProbsAndEntropy:
    @pytest.mark.parametrize(("chunk_size", "expected_calls"), [(-1, 1), (2, 2)])
    def test_calculate_log_probs_propagates_the_debug_grad_contract_with_and_without_chunking(
        self,
        chunk_size: int,
        expected_calls: int,
        fused_cross_entropy_calls: list[tuple[bool, bool]],
    ) -> None:
        """Debug stored log probabilities use the grad contract with and without chunking."""
        with torch.no_grad():
            log_probs, entropy = calculate_log_probs_and_entropy(
                torch.randn(3, 8),
                torch.tensor([1, 2, 3]),
                None,
                chunk_size=chunk_size,
                debug_unified_grad_fused_logprob=True,
            )

        assert fused_cross_entropy_calls == [(True, True)] * expected_calls
        assert not log_probs.requires_grad
        assert entropy is None
