import math

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args

from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy


class TestLogprobVocabularyConfiguration:
    @pytest.mark.parametrize("loss_type,true_on_policy", [("score_centering", False), ("policy_loss", True)])
    def test_scoring_excludes_padded_vocabulary_logits_from_normalization(
        self, native_single_rank: None, loss_type: str, true_on_policy: bool
    ) -> None:
        """Configured true vocabulary excludes even dominant padded logits from scoring."""
        args = make_args(loss_type=loss_type, true_on_policy_mode=true_on_policy, vocab_size=2)
        logits = torch.tensor([[[0.0, 0.0, 100.0, 100.0]] * 3])

        result = get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=[torch.tensor([0, 1, 0])],
            total_lengths=[3],
            response_lengths=[2],
            with_entropy=True,
        )

        torch.testing.assert_close(result["log_probs"][0], torch.full((2,), -math.log(2.0)))
        torch.testing.assert_close(result["entropy"][0], torch.full((2,), math.log(2.0)))
