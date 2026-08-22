from types import SimpleNamespace

import pytest

from miles.utils.multi_lora import uses_multi_lora_operation_executor, validate_multi_lora_args
from miles.utils.tinker import uses_explicit_training_operations, validate_tinker_args


def _args(tinker_backend: bool, n_adapters: int) -> SimpleNamespace:
    return SimpleNamespace(
        tinker_backend=tinker_backend,
        multi_lora_n_adapters=n_adapters,
        multi_lora=n_adapters > 0,
    )


class TestPredicateRoles:
    def test_operation_semantics_is_the_protocol_flag_alone(self):
        assert uses_explicit_training_operations(_args(True, 0))
        assert uses_explicit_training_operations(_args(True, 4))
        assert not uses_explicit_training_operations(_args(False, 4))
        assert not uses_explicit_training_operations(_args(False, 0))

    def test_executor_requires_protocol_and_slots(self):
        assert uses_multi_lora_operation_executor(_args(True, 4))
        assert not uses_multi_lora_operation_executor(_args(True, 0))
        assert not uses_multi_lora_operation_executor(_args(False, 4))


class TestValidationClosesTheGap:
    def _validate(self, args) -> None:
        validate_multi_lora_args(args)
        validate_tinker_args(args)

    def test_tinker_without_slots_is_rejected(self):
        with pytest.raises(AssertionError, match="--multi-lora-n-adapters"):
            self._validate(_args(True, 0))
