from argparse import Namespace

import pytest
import torch
from tests.fast.backends.training_utils.backend_owner_fixtures import (
    native_backend_owner_state,
    policy_backend_args,
    single_token_policy_batch,
    uneven_policy_batch,
)

from miles.backends.training_utils.loss.hub.losses import policy_loss_function
from miles.backends.training_utils.loss.objective import loss_function
from miles.utils.args.custom_function import CustomFunctionConfig

__all__ = ["native_backend_owner_state", "policy_backend_args", "single_token_policy_batch", "uneven_policy_batch"]


class TestBackendLossNormalization:
    @pytest.mark.parametrize("per_token", [False, True])
    @pytest.mark.parametrize("megatron_scaling", [False, True])
    @pytest.mark.parametrize("dynamic_batch_size", [None, 4])
    def test_objective_uses_backend_token_mode_and_batch_size_through_real_tis(
        self,
        native_backend_owner_state: None,
        policy_backend_args: Namespace,
        uneven_policy_batch: dict,
        per_token: bool,
        megatron_scaling: bool,
        dynamic_batch_size: int | None,
    ) -> None:
        """Backend token mode controls TIS reduction, ESS, normalizer, metric count and Megatron scaling."""
        args = policy_backend_args
        args.backend.calculate_per_token_loss = per_token
        args.use_dynamic_global_batch_size = dynamic_batch_size is not None
        if dynamic_batch_size is not None:
            uneven_policy_batch["dynamic_global_batch_size"] = dynamic_batch_size

        loss, normalizer, metrics = loss_function(
            args=args,
            batch=uneven_policy_batch,
            num_microbatches=2,
            logits=torch.zeros((1, 10, 2), requires_grad=True),
            apply_megatron_loss_scaling=megatron_scaling,
            num_step_tokens=8,
        )

        if per_token:
            expected = -20.0 if megatron_scaling else -2.5
        elif megatron_scaling:
            expected = -2.0 if dynamic_batch_size is not None else -4.0
        else:
            expected = -1.0 if dynamic_batch_size is not None else -2.0
        assert loss.item() == pytest.approx(expected)
        assert normalizer.item() == (8 if per_token else 1)
        assert metrics["values"][0].item() == (8 if per_token else 2)
        assert metrics["values"][metrics["keys"].index("ess_ratio") + 1].item() == (8 if per_token else 2)

    @pytest.mark.parametrize("per_token,expected", [(False, -2.5), (True, -20.0)])
    def test_custom_policy_reducer_receives_the_backend_token_mode(
        self,
        native_backend_owner_state: None,
        policy_backend_args: Namespace,
        uneven_policy_batch: dict,
        per_token: bool,
        expected: float,
    ) -> None:
        """A configured plugin receives backend token normalization instead of requiring flat fields."""
        args = policy_backend_args
        args.backend.calculate_per_token_loss = per_token
        args.custom_pg_loss_reducer_function_path = CustomFunctionConfig(
            path="tests.fast.backends.training_utils.backend_owner_fixtures._per_token_reducer"
        )

        loss, _ = policy_loss_function(
            args=args,
            batch=uneven_policy_batch,
            logits=torch.zeros((1, 10, 2), requires_grad=True),
            sum_of_sample_mean=torch.mean,
        )

        assert loss.item() == pytest.approx(expected)

    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="Existing single-response-token log-probability squeeze drops the token dimension",
    )
    def test_single_response_token_preserves_a_concatenable_log_probability_dimension(
        self,
        native_backend_owner_state: None,
        policy_backend_args: Namespace,
        single_token_policy_batch: dict,
    ) -> None:
        """A one-token response retains its token dimension through real policy log-probability concatenation."""
        try:
            policy_loss_function(
                args=policy_backend_args,
                batch=single_token_policy_batch,
                logits=torch.zeros((1, 6, 2), requires_grad=True),
                sum_of_sample_mean=torch.mean,
            )
        except RuntimeError as error:
            assert str(error) == "zero-dimensional tensor (at position 0) cannot be concatenated"
            raise
