"""Refactor-equivalence witness for the protocol-mode / parameter-executor
predicate split (codex-rollout-fullparameter-design-0810 §3.2).

``train_one_step`` now keys its execution policy (retain accumulated grads,
no inline optimizer/scheduler step, no trailing grad clear) on
``uses_tinker_operation_semantics`` instead of ``is_multi_lora_enabled``.
That swap is behavior-preserving iff the two predicates agree on every
config that survives launch validation — which these tests prove by
exhausting the flag combinations: every combination where the predicates
would differ is rejected by ``validate_multi_lora_args`` or
``validate_tinker_args`` before a trainer can exist.
"""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

from miles.utils.multi_lora import is_multi_lora_enabled, validate_multi_lora_args
from miles.utils.tinker_backend import (
    is_tinker_enabled,
    uses_multi_lora_tinker_executor,
    uses_tinker_operation_semantics,
    validate_tinker_args,
)


def _args(tinker_backend: bool, n_adapters: int) -> SimpleNamespace:
    return SimpleNamespace(
        tinker_backend=tinker_backend,
        multi_lora_n_adapters=n_adapters,
        multi_lora=n_adapters > 0,
    )


class TestPredicateRoles:
    def test_operation_semantics_is_the_protocol_flag_alone(self):
        assert uses_tinker_operation_semantics(_args(True, 0))
        assert uses_tinker_operation_semantics(_args(True, 4))
        assert not uses_tinker_operation_semantics(_args(False, 4))
        assert not uses_tinker_operation_semantics(_args(False, 0))

    def test_executor_requires_protocol_and_slots(self):
        assert uses_multi_lora_tinker_executor(_args(True, 4))
        assert not uses_multi_lora_tinker_executor(_args(True, 0))
        assert not uses_multi_lora_tinker_executor(_args(False, 4))

    def test_is_tinker_enabled_is_unchanged(self):
        """Characterization: the legacy predicate keeps its exact truth table."""
        for tinker, n in [(True, 4), (True, 0), (False, 4), (False, 0)]:
            assert is_tinker_enabled(_args(tinker, n)) == (tinker and n > 0)


class TestValidationClosesTheGap:
    """Every flag combination either fails validation or makes the protocol
    predicate equal to the multi-LoRA one — so swapping the train_one_step
    policy gate cannot change any launched run."""

    def _validate(self, args) -> None:
        validate_multi_lora_args(args)
        validate_tinker_args(args)

    def test_multi_lora_without_tinker_is_rejected(self):
        with pytest.raises(AssertionError, match="requires --tinker-backend"):
            self._validate(_args(False, 4))

    def test_tinker_without_slots_is_rejected(self):
        with pytest.raises(AssertionError, match="--multi-lora-n-adapters"):
            self._validate(_args(True, 0))

    def test_predicates_agree_on_every_validated_config(self, monkeypatch):
        monkeypatch.setenv("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "1")
        for tinker, n in [(True, 4), (True, 0), (False, 4), (False, 0)]:
            args = _full_args(tinker, n)
            try:
                validate_multi_lora_args(args)
                validate_tinker_args(args)
            except AssertionError:
                continue  # rejected at launch: the trainer never sees this combo
            assert uses_tinker_operation_semantics(args) == is_multi_lora_enabled(args)
            assert uses_multi_lora_tinker_executor(args) == is_multi_lora_enabled(args)


def _full_args(tinker_backend: bool, n_adapters: int) -> SimpleNamespace:
    """Args rich enough to pass both validators when the combo is legal."""
    return SimpleNamespace(
        tinker_backend=tinker_backend,
        multi_lora_n_adapters=n_adapters,
        lora_rank=8,
        target_modules=["linear_qkv"],
        train_backend="megatron",
        pipeline_model_parallel_size=1,
        qkv_format="thd",
        experts_shared_outer_loras=False,
        optimizer="adam",
        colocate=False,
        indep_dp=False,
        ft_components=[],
        offload_train=False,
        enable_witness=False,
        sglang_tokenizer_worker_num=1,
        calculate_per_token_loss=False,
        disable_rollout_trim_samples=False,
        use_dynamic_global_batch_size=False,
        megatron_to_hf_mode="bridge",
        rollout_global_dataset=False,
        rollout_function_path=None,
        data_source_path="miles.rollout.data_source.RolloutDataSourceWithBuffer",
    )
