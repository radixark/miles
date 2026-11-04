from pathlib import Path

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args

from miles.backends.training_utils.loss.hub.losses import policy_loss_function


class TestPolicyConsumerConfiguration:
    @pytest.mark.parametrize("dual_clip,expected_loss", [(None, 4.0), (2.0, 2.0)])
    def test_configured_dual_clip_caps_negative_advantages_and_reaches_the_debug_dump(
        self, native_single_rank: None, tmp_path: Path, dual_clip: float | None, expected_loss: float
    ) -> None:
        """Dual clipping changes the real policy loss and its persisted diagnostics."""
        args = make_args(eps_clip_c=dual_clip, dump_details=tmp_path, entropy_coef=0.0, observe_training_entropy=False)
        logits = torch.zeros((1, 3, 2), requires_grad=True)
        batch = {
            "advantages": [torch.tensor([-1.0, -1.0])],
            "log_probs": [torch.log(torch.tensor([0.125, 0.125]))],
            "unconcat_tokens": [torch.tensor([0, 1, 0])],
            "response_lengths": [2],
            "total_lengths": [3],
            "loss_masks": [torch.ones(2)],
        }

        loss, metrics = policy_loss_function(args=args, batch=batch, logits=logits, sum_of_sample_mean=torch.mean)

        assert loss.item() == pytest.approx(expected_loss)
        assert metrics["pg_loss"].item() == pytest.approx(expected_loss)
        [dump_path] = (tmp_path / "policy_loss_debug").glob("rank_0_call_*.pt")
        payload = torch.load(dump_path, weights_only=False)
        torch.testing.assert_close(payload["pg_loss"], torch.full((2,), expected_loss))
        assert payload["samples"][0]["response_length"] == 2
