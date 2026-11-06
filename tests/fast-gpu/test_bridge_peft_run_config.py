"""Megatron-Bridge's own LoRA / CanonicalLoRA, dumped as Bridge writes run_config.yaml, pass the PEFT checkpoint
reader's checks: a Bridge bump that renames or drops a field the checks require (not one read with a default) fails
here."""

from tests.ci.ci_register import register_cuda_ci

# CPU-only checks, on this stage because only the GPU image ships Megatron-Bridge's peft (it imports Transformer
# Engine)
register_cuda_ci(est_time=30, suite="stage-b-2-gpu-h200", labels=["lora"], hardware=["hopper"], num_gpus=2)

import pytest
import torch
import yaml
from megatron.bridge.peft.canonical_lora import CanonicalLoRA
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.utils.config_utils import _ConfigContainerBase
from megatron.bridge.utils.yaml_utils import safe_yaml_representers

from miles.backends.megatron_utils.lora.bridge_checkpoint_loader import assert_bridge_run_config_matches
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization

FP32 = {"_target_": "torch.float32", "_call_": False}


def _dumped(value) -> dict:
    """The run_config.yaml section Bridge's ConfigContainer.to_yaml writes for ``value``."""
    with safe_yaml_representers():
        return yaml.safe_load(yaml.safe_dump(_ConfigContainerBase._convert_value_to_dict(value)))


def _config(peft) -> dict:
    return {
        "peft": _dumped(peft),
        "optimizer": {"optimizer": "adam", "use_distributed_optimizer": True, "use_precision_aware_optimizer": False}
        | {field: FP32 for field in ("main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype")},
        "checkpoint": {"save_optim": True},
        "model": {"moe_grouped_gemm": True},
    }


def _run(lora_type: str) -> AdapterParametrization:
    return AdapterParametrization(
        base_model="base",
        lora_rank=8,
        lora_alpha=32.0,
        lora_type=lora_type,
        target_modules=("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.experts.down_proj"),
        experts_shared_outer_loras=False,
    )


@pytest.mark.parametrize("peft_class, lora_type", [(LoRA, "lora"), (CanonicalLoRA, "canonical_lora")])
def test_bridges_own_run_config_of_the_same_lora_is_accepted(peft_class, lora_type):
    config = _config(peft_class(dim=8, alpha=32, share_expert_adapters=False))
    assert_bridge_run_config_matches(config, _run(lora_type), params_dtype=torch.bfloat16, load_optimizer=True)
