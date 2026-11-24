import pytest
import torch
from tests.fast.backends.training_utils.backend_owner_fixtures import native_backend_owner_state
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args

from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy

__all__ = ["native_backend_owner_state"]


class TestNativeVocabularyOwnership:
    @pytest.mark.parametrize("loss_type", ["policy_loss", "score_centering"])
    def test_native_scoring_uses_all_logits_without_a_megatron_vocab_field(
        self, native_backend_owner_state: None, loss_type: str
    ) -> None:
        """Native scoring retains its whole vocabulary without reading Megatron-only padding metadata."""
        args = make_args(loss_type=loss_type, true_on_policy_mode=loss_type != "score_centering")
        args.train_backend = "fsdp"
        del args.backend.vocab_size
        logits = torch.tensor([[[0.0, 0.0, 2.0, 2.0]] * 3])

        result = get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=[torch.tensor([0, 1, 0])],
            total_lengths=[3],
            response_lengths=[2],
            with_entropy=True,
        )

        log_probs = logits[0, :2].log_softmax(dim=-1)
        torch.testing.assert_close(result["log_probs"][0], log_probs[[0, 1], [1, 0]])
        torch.testing.assert_close(result["entropy"][0], -(log_probs.exp() * log_probs).sum(dim=-1))
