from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss.hub.logit_processors import get_responses


class TestResponsePrecision:
    @pytest.mark.parametrize(
        "true_on_policy,bf16,fp16,expected_dtype,expected_scale",
        [
            (True, True, False, torch.bfloat16, 2.0),
            (True, False, True, torch.float16, 2.0),
            (True, True, True, torch.bfloat16, 2.0),
            (True, False, False, torch.float32, 2.0),
            (False, True, False, torch.float32, 1.0),
        ],
    )
    def test_response_chunks_apply_temperature_before_configured_training_precision(
        self,
        native_single_rank: None,
        true_on_policy: bool,
        bf16: bool,
        fp16: bool,
        expected_dtype: torch.dtype,
        expected_scale: float,
    ) -> None:
        """On-policy chunks scale and cast logits while ordinary scoring preserves them."""
        logits = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])
        args = Namespace(
            true_on_policy_mode=true_on_policy,
            backend=Namespace(bf16=bf16, fp16=fp16),
            rollout_temperature=0.5,
            qkv_format="thd",
            allgather_cp=False,
        )

        [(response_logits, response_tokens)] = list(
            get_responses(
                logits,
                args=args,
                unconcat_tokens=[torch.tensor([0, 1, 0])],
                total_lengths=[3],
                response_lengths=[2],
            )
        )

        assert response_logits.dtype == expected_dtype
        torch.testing.assert_close(response_logits, (logits[0, :2] * expected_scale).to(expected_dtype))
        assert response_tokens.tolist() == [1, 0]
