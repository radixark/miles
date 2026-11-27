from argparse import Namespace
from unittest.mock import Mock

from miles.backends.megatron_utils.lora import optimizer as slot_optimizer
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer


class TestSlotOptimizerBackendConfiguration:
    def test_constructor_copies_backend_hyperparameters_and_limits_adam_to_one_slot(
        self, optimizer_construction: object
    ) -> None:
        """Backend optimizer options configure a real Adam containing only the selected adapter slot."""
        args = Namespace(
            backend=Namespace(
                lr=0.03,
                adam_beta1=0.8,
                adam_beta2=0.9,
                adam_eps=1e-6,
                weight_decay=0.07,
                bf16=True,
                use_gloo_process_groups=True,
            )
        )
        model = optimizer_construction.model
        original_trainability = [parameter.requires_grad for parameter in model.parameters()]

        SlotOptimizer(args=args, model=[model], slot=1, slot_params=list(model.adapters[1].parameters()))

        assert optimizer_construction.use_gloo == [True]
        assert optimizer_construction.config.bf16 is True
        assert optimizer_construction.config.timers is None
        [child] = optimizer_construction.inner.chained_optimizers
        [group] = child.param_groups
        assert group["lr"] == 0.03
        assert group["betas"] == (0.8, 0.9)
        assert group["eps"] == 1e-6
        assert group["weight_decay"] == 0.07
        assert {id(parameter) for parameter in group["params"]} == {
            id(parameter) for parameter in model.adapters[1].parameters()
        }
        assert [parameter.requires_grad for parameter in model.parameters()] == original_trainability
        assert child.grad_stats_parallel_group is None

    def test_mtp_clipping_uses_an_independent_norm(self, monkeypatch) -> None:
        target = Namespace()
        draft = Namespace(grad_norm_group="mtp")
        child = Mock()
        child.get_parameters.return_value = [target, draft]
        child.get_grads_for_grad_norm.side_effect = [[3.0], [0.5]]
        clip = Mock()
        monkeypatch.setattr(slot_optimizer, "clip_grad_by_total_norm_fp32", clip)
        monkeypatch.setattr(slot_optimizer, "get_grad_norm_fp32", lambda grads, **kwargs: grads[0])
        optimizer = SlotOptimizer.__new__(SlotOptimizer)
        optimizer._trains_mtp = True
        optimizer._inner = Namespace(chained_optimizers=[child])

        assert optimizer.clip_and_step(1.0) == {"grad_norm": 3.0, "mtp_grad_norm": 0.5}
        assert [call.args for call in clip.call_args_list] == [
            ([target], 1.0, 3.0, False),
            ([draft], 1.0, 0.5, False),
        ]
        child.step_with_ready_grads.assert_called_once_with()
