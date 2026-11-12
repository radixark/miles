import os
from unittest.mock import Mock

import pytest
import torch
from tests.fast.backends.fsdp_utils.init_backend_fixtures import (
    _InitBackendCapture,
    _make_fsdp_init_args,
    _OptimizerReached,
    init_backend_capture,
)

from miles.backends.fsdp_utils.actor import FSDPTrainRayActor, apply_fsdp2
from miles.backends.training_utils.parallel import ParallelState
from miles.utils.ft_utils.indep_dp import IndepDPInfo

__all__ = ["init_backend_capture"]


class TestInitBackendOwnership:
    def test_debug_initialization_seeds_from_the_backend(
        self, fsdp_three_rank_state: ParallelState, fsdp_debug_actor: FSDPTrainRayActor
    ) -> None:
        """The debug early return still initializes the backend's reproducible RNG stream."""
        previous = torch.get_rng_state()
        try:
            args = _make_fsdp_init_args(train_backend="fsdp", seed=173, debug_rollout_only=True, dumper_enable=False)
            expected = torch.rand(5, generator=torch.Generator().manual_seed(173))

            assert (
                fsdp_debug_actor.init(
                    args, "actor", indep_dp_info=IndepDPInfo.create_trivial(), indep_dp_store_addr=None
                )
                == 0
            )

            torch.testing.assert_close(torch.rand(5), expected, rtol=0, atol=0)
        finally:
            torch.set_rng_state(previous)

    @pytest.mark.parametrize(
        "deterministic,flash_env,expected_flash_env", [(False, None, None), (True, None, "1"), (True, "0", "0")]
    )
    def test_deterministic_mode_reaches_torch_and_flash_attention(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fsdp_three_rank_state: ParallelState,
        fsdp_debug_actor: FSDPTrainRayActor,
        deterministic: bool,
        flash_env: str | None,
        expected_flash_env: str | None,
    ) -> None:
        """Deterministic mode reaches torch and flash-attn, and leaves an explicit flash-attn setting alone."""
        use_deterministic_algorithms = Mock()
        monkeypatch.setattr(torch, "use_deterministic_algorithms", use_deterministic_algorithms)
        monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
        monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
        monkeypatch.setattr(torch.utils.deterministic, "fill_uninitialized_memory", True)
        monkeypatch.setenv("MILES_FILL_UNINIT", "0")
        # set before deleting, so that monkeypatch also removes a variable the initialization creates
        monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "0")
        if flash_env is None:
            monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC")
        previous = torch.get_rng_state()
        try:
            args = _make_fsdp_init_args(
                train_backend="fsdp", debug_rollout_only=True, dumper_enable=False, deterministic_mode=deterministic
            )

            assert (
                fsdp_debug_actor.init(
                    args, "actor", indep_dp_info=IndepDPInfo.create_trivial(), indep_dp_store_addr=None
                )
                == 0
            )
        finally:
            torch.set_rng_state(previous)

        if deterministic:
            use_deterministic_algorithms.assert_called_once_with(True, warn_only=False)
            assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
            assert not torch.utils.deterministic.fill_uninitialized_memory
        else:
            use_deterministic_algorithms.assert_not_called()
            assert not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
            assert torch.utils.deterministic.fill_uninitialized_memory
        assert os.environ.get("FLASH_ATTENTION_DETERMINISTIC") == expected_flash_env

    @pytest.mark.parametrize("cpu_offload", [False, True])
    @pytest.mark.parametrize("gradient_checkpointing", [False, True])
    def test_initialization_builds_adamw_and_model_policies_from_backend_values(
        self,
        fsdp_debug_actor: FSDPTrainRayActor,
        init_backend_capture: _InitBackendCapture,
        cpu_offload: bool,
        gradient_checkpointing: bool,
    ) -> None:
        """Real model loading and AdamW construction retain every configured optimizer and memory policy."""
        args = _make_fsdp_init_args(
            train_backend="fsdp",
            hf_checkpoint=str(init_backend_capture.checkpoint),
            dumper_enable=False,
            debug_rollout_only=False,
            offload_train=False,
            seed=173,
            optimizer_cpu_offload=cpu_offload,
            gradient_checkpointing=gradient_checkpointing,
            optimizer="adam",
            lr=0.007,
            adam_beta1=0.71,
            adam_beta2=0.93,
            adam_eps=0.00002,
            weight_decay=0.23,
            keep_fp32_master=False,
            attn_implementation="eager",
        )

        with pytest.raises(_OptimizerReached):
            fsdp_debug_actor.init(args, "actor", indep_dp_info=IndepDPInfo.create_trivial(), indep_dp_store_addr=None)

        [optimizer] = init_backend_capture.optimizers
        [group] = optimizer.param_groups
        assert group["lr"] == 0.007
        assert group["betas"] == (0.71, 0.93)
        assert group["eps"] == 0.00002
        assert group["weight_decay"] == 0.23
        assert fsdp_debug_actor.model.is_gradient_checkpointing is gradient_checkpointing
        assert fsdp_debug_actor.optimizer_cpu_offload is cpu_offload
        assert init_backend_capture.shard_options
        for options in init_backend_capture.shard_options:
            assert (options["offload_policy"] is not None) is cpu_offload
        assert {id(p) for p in group["params"]} == {id(p) for p in fsdp_debug_actor.model.parameters()}

    def test_unsupported_backend_optimizer_is_rejected(
        self, fsdp_debug_actor: FSDPTrainRayActor, init_backend_capture: _InitBackendCapture
    ) -> None:
        """An unsupported backend optimizer cannot silently fall back to AdamW."""
        args = _make_fsdp_init_args(
            train_backend="fsdp",
            hf_checkpoint=str(init_backend_capture.checkpoint),
            dumper_enable=False,
            debug_rollout_only=False,
            offload_train=False,
            optimizer="sgd",
            keep_fp32_master=False,
            attn_implementation="eager",
        )

        with pytest.raises(ValueError, match="Unsupported optimizer: sgd"):
            fsdp_debug_actor.init(args, "actor", indep_dp_info=IndepDPInfo.create_trivial(), indep_dp_store_addr=None)

        assert init_backend_capture.optimizers == []

    def test_combining_cpu_offload_with_train_offload_is_rejected(
        self, fsdp_three_rank_state: ParallelState, fsdp_debug_actor: FSDPTrainRayActor
    ) -> None:
        """Conflicting offload mechanisms fail before model or optimizer creation."""
        args = _make_fsdp_init_args(
            train_backend="fsdp",
            dumper_enable=False,
            debug_rollout_only=False,
            offload_train=True,
            optimizer_cpu_offload=True,
        )

        with pytest.raises(AssertionError):
            fsdp_debug_actor.init(args, "actor", indep_dp_info=IndepDPInfo.create_trivial(), indep_dp_store_addr=None)

        assert fsdp_debug_actor.optimizer_cpu_offload is True

    @pytest.mark.parametrize("fp16,expected", [(False, torch.bfloat16), (True, torch.float16)])
    def test_default_fsdp_wrapping_uses_backend_compute_precision(
        self, init_backend_capture: _InitBackendCapture, fp16: bool, expected: torch.dtype
    ) -> None:
        """The wrapping fallback derives parameter precision from the backend while reducing in fp32."""
        from transformers import GPT2Config, GPT2LMHeadModel

        model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=8, n_embd=8, n_layer=1, n_head=2))
        args = _make_fsdp_init_args(train_backend="fsdp", fp16=fp16)

        assert apply_fsdp2(model, args=args) is model

        assert init_backend_capture.shard_options
        for options in init_backend_capture.shard_options:
            assert options["mp_policy"].param_dtype is expected
            assert options["mp_policy"].reduce_dtype is torch.float32
