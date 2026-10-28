from argparse import Namespace

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

        SlotOptimizer(args=args, model=[model], slot=1)

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
