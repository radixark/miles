from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tests.fast.fixtures.args_fixtures import replace_config_values
from tests.fast.fixtures.inkling_provider_fixtures import (
    inkling_adapter_model,
    inkling_provider_env,
    inkling_reload_env,
)

from miles.utils.args.custom_function import CustomFunctionConfig
from miles_plugins.models.inkling.lora import InklingLoRAAdapter

_ = inkling_adapter_model, inkling_provider_env, inkling_reload_env


class TestNativeInklingSetup:
    def test_typed_provider_injects_trainable_adapters_before_model_wrapping(
        self, inkling_provider_env: SimpleNamespace
    ) -> None:
        """A typed Inkling provider still selects real native LoRA injection."""
        env = inkling_provider_env
        [model], optimizer, scheduler = env.module.setup_model_and_optimizer(env.args, role="actor")

        assert isinstance(model.lora_lm_head_adapter, InklingLoRAAdapter)
        assert model.lora_lm_head_adapter.head_A.requires_grad
        assert model.lora_lm_head_adapter.head_B.requires_grad
        assert not model.output_layer.weight.requires_grad
        assert optimizer is None and scheduler is None

    @pytest.mark.parametrize("provider", [None, CustomFunctionConfig(path="models.other.provider")])
    def test_other_providers_still_reject_unsupported_native_lora(
        self, inkling_provider_env: SimpleNamespace, provider: CustomFunctionConfig | None
    ) -> None:
        """The typed-provider fix does not accept unsupported native LoRA models."""
        env = inkling_provider_env
        args = replace_config_values(env.args, custom_model_provider_path=provider)

        with pytest.raises(AssertionError, match="Native LoRA injection is only implemented"):
            env.module.setup_model_and_optimizer(args, role="actor")

    @pytest.mark.parametrize("inkling", [True, False])
    def test_only_inkling_disables_muon_qkv_splitting(
        self, inkling_provider_env: SimpleNamespace, inkling: bool
    ) -> None:
        """Typed Inkling providers retain the fused-qkvr Muon safeguard."""
        env = inkling_provider_env
        provider = env.args.custom_model_provider_path
        if not inkling:
            provider = CustomFunctionConfig(path="models.other.provider")
        args = replace_config_values(
            env.args, custom_model_provider_path=provider, lora_rank=0, debug_disable_optimizer=False
        )
        _, optimizer, _ = env.module.setup_model_and_optimizer(args, role="actor")

        assert optimizer.config.muon_split_qkv is (not inkling)


class TestNativeInklingReload:
    @pytest.mark.parametrize("native_optimizer_restored", [False, True])
    def test_disk_adapter_reloads_only_when_native_optimizer_was_not_restored(
        self, inkling_reload_env: SimpleNamespace, tmp_path: Path, native_optimizer_restored: bool
    ) -> None:
        """Typed Inkling reloads adapter tensors and masters without overwriting native optimizer restores."""
        env = inkling_reload_env
        env.native_optimizer_restored = native_optimizer_restored
        adapter = env.model.lora_lm_head_adapter
        initial = adapter.head_B.detach().clone()
        save_file(
            {
                "language_model.lm_head.lora_A.weight": torch.full_like(adapter.head_A, 3),
                "language_model.lm_head.lora_B.weight": torch.full_like(adapter.head_B, 7),
            },
            str(tmp_path / "adapter_model.safetensors"),
        )
        args = replace_config_values(env.args, lora_adapter_path=str(tmp_path))

        env.module.load_model_state(
            args,
            model=[env.model],
            optimizer=env.optimizer,
            opt_param_scheduler=None,
            role="actor",
            checkpointing_context=None,
        )

        if native_optimizer_restored:
            torch.testing.assert_close(adapter.head_B, initial)
            assert env.optimizer.masters == {}
        else:
            torch.testing.assert_close(adapter.head_B, torch.full_like(adapter.head_B, 7))
            torch.testing.assert_close(env.optimizer.masters["lora_lm_head_adapter.head_B"], adapter.head_B)
