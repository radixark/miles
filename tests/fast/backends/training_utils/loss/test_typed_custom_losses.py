from collections.abc import Callable
from typing import Any

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import (
    make_args,
    make_batch,
    make_inputs,
    make_parallel_state,
)
from tests.fast.backends.training_utils.loss.test_training_logprob_reuse import process_group
from tests.fast.fixtures.hook_consumer_fixtures import _hook

from miles.backends.training_utils.loss import objective
from miles.backends.training_utils.loss.hub.losses import policy_loss_function
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.function_registry import function_registry

__all__ = ["process_group"]


class TestTypedCustomLoss:
    @pytest.mark.parametrize("recompute", [False, True])
    def test_loss_dispatch_merges_plugin_fields_in_both_sdk_checkpoint_branches(
        self, monkeypatch: pytest.MonkeyPatch, recompute: bool
    ) -> None:
        """Custom loss dispatch keeps plugin fields through ordinary and recomputed calls with real gradients."""
        make_parallel_state()
        hook = _hook()
        args = ImmutableNamespace(
            **dict(
                make_args(loss_type="custom_loss", custom_loss_function_path=hook, recompute_loss_function=recompute)
            )
        )
        batch = {"loss_masks": [torch.ones(1)], "response_lengths": [1], "total_lengths": [2]}
        logits = torch.tensor([2.0], requires_grad=True)
        checkpointed: list[bool] = []

        def sdk_checkpoint(function: Callable[..., Any], *values: Any) -> Any:
            checkpointed.append(True)
            return function(*values)

        def custom_loss(
            view: ImmutableNamespace, values: dict[str, Any], tensor: torch.Tensor, reducer: Any
        ) -> tuple[torch.Tensor, dict[str, float]]:
            assert view.hook_marker == 41
            assert values is batch
            assert tensor is logits
            return tensor.sum() * view.hook_marker, {"plugin": 7.0}

        monkeypatch.setattr(objective, "checkpoint", sdk_checkpoint)
        with function_registry.temporary(hook.path, custom_loss):
            loss, _, log = objective.loss_function(args, batch, 1, logits, num_rollouts=1)
        loss.backward()

        assert logits.grad.item() == 41
        assert checkpointed == ([True] if recompute else [])
        assert log["keys"] == ["plugin"]
        assert "hook_marker" not in dict(args)

    def test_policy_loss_supplies_typed_custom_tis_fields_to_the_actual_plugin_boundary(
        self, process_group: None
    ) -> None:
        """Policy loss forwards plugin configuration alongside real computed importance-sampling inputs."""
        make_parallel_state()
        hook = _hook()
        args = ImmutableNamespace(
            **dict(
                make_args(
                    custom_tis_function_path=hook,
                    use_tis=True,
                    true_on_policy_mode=False,
                    entropy_coef=0.0,
                    observe_training_entropy=False,
                    dump_details=None,
                )
            )
        )
        inputs = make_inputs(seed=29, batch_size=1, prompt_lens=[3], response_lens=[2], vocab_size=8, args=args)
        batch = make_batch(inputs, "policy_loss")

        class HookReached(Exception):
            pass

        def custom_tis(*, args: ImmutableNamespace, **kwargs: Any) -> Any:
            assert args.hook_marker == 41
            assert kwargs["pg_loss"].numel() == 2
            assert kwargs["loss_masks"] is batch["loss_masks"]
            assert kwargs["response_lengths"] == [2]
            raise HookReached

        with function_registry.temporary(hook.path, custom_tis), pytest.raises(HookReached):
            policy_loss_function(args, batch, inputs["policy_logits"], lambda tensor: tensor.mean())
