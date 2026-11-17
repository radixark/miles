from argparse import Namespace

import pytest

from miles.backends.megatron_utils.lora.bridge import _setup_lora_model_via_bridge


class TestLoraBridgeBackendConfiguration:
    @pytest.mark.parametrize(
        "optimizer,multi_lora,use_distributed", [("adam", False, True), ("Muon", False, False), ("adam", True, False)]
    )
    def test_provider_and_ddp_construction_use_backend_owned_training_options(
        self, bridge_args: Namespace, bridge_provider: object, optimizer: str, multi_lora: bool, use_distributed: bool
    ) -> None:
        """LoRA model construction forwards backend options to its actual provider and DDP boundary."""
        bridge_args.backend.optimizer = optimizer
        bridge_args.multi_lora = multi_lora

        [provider] = _setup_lora_model_via_bridge(bridge_args)

        assert provider is bridge_provider
        for field in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "sequence_parallel",
            "virtual_pipeline_model_parallel_size",
            "context_parallel_size",
            "gradient_accumulation_fusion",
            "recompute_granularity",
            "recompute_method",
            "recompute_num_layers",
            "recompute_modules",
            "distribute_saved_activations",
            "attention_backend",
            "moe_router_dtype",
            "moe_router_use_torch_mm",
        ):
            assert vars(provider)[field] == vars(bridge_args.backend)[field]
        assert provider.bias_activation_fusion is True
        assert provider.apply_rope_fusion is False
        assert provider.num_layers_in_first_pipeline_stage == 2
        assert provider.num_layers_in_last_pipeline_stage == 3
        assert provider.dsa_kernel_backend == "flash_mla"
        assert provider.ddp_config.use_distributed_optimizer is use_distributed
        assert provider.ddp_config.grad_reduce_in_fp32 is True
        assert provider.ddp_config.finalized is True
