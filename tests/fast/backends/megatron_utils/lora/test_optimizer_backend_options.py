from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.lora.optimizer import validate_multi_lora_optimizer_args
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace


class TestSlotOptimizerBackendOptions:
    def test_supported_options_use_backend_owner_and_preserve_configuration(self) -> None:
        """Slot optimizer validation accepts supported backend-owned Adam options without mutation."""
        backend = MegatronArgsNamespace(
            use_distributed_optimizer=False, bf16=True, fp16=False, optimizer="AdAm", optimizer_cpu_offload=False
        )
        args = SimpleNamespace(
            backend=backend, stream_optimizer_state_to_disk=False, rematerialize_param_from_master_weight=False
        )
        before = vars(backend).copy()

        validate_multi_lora_optimizer_args(args=args)

        assert vars(backend) == before

    @pytest.mark.parametrize(
        "backend_updates,shared_updates,message",
        [
            ({"use_distributed_optimizer": True}, {}, "use_distributed_optimizer=False"),
            ({"bf16": False}, {}, "require bf16"),
            ({"fp16": True}, {}, "require bf16"),
            ({"optimizer": "sgd"}, {}, "only implement Adam"),
            ({"optimizer_cpu_offload": True}, {}, "optimizer-cpu-offload"),
            ({}, {"stream_optimizer_state_to_disk": True}, "stream-optimizer-state-to-disk"),
            ({}, {"rematerialize_param_from_master_weight": True}, "rematerialize-param-from-master-weight"),
        ],
    )
    def test_unsupported_backend_or_shared_stateful_options_are_rejected(
        self, backend_updates: dict[str, object], shared_updates: dict[str, object], message: str
    ) -> None:
        """Each unsupported stateful option is rejected at its actual configuration owner."""
        backend = MegatronArgsNamespace(
            **(
                {
                    "use_distributed_optimizer": False,
                    "bf16": True,
                    "fp16": False,
                    "optimizer": "adam",
                    "optimizer_cpu_offload": False,
                }
                | backend_updates
            )
        )
        args = SimpleNamespace(
            **(
                {
                    "backend": backend,
                    "stream_optimizer_state_to_disk": False,
                    "rematerialize_param_from_master_weight": False,
                }
                | shared_updates
            )
        )

        with pytest.raises(AssertionError, match=message):
            validate_multi_lora_optimizer_args(args=args)
