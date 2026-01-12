import logging

import torch.nn as nn

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError as err:
    raise ImportError("peft library required for LoRA. Install with: pip install peft") from err

logger = logging.getLogger(__name__)

LORA_ADAPTER_NAME = "miles_lora"
LORA_SUBDIR = "tmp_lora"


def apply_lora_to_model(model: nn.Module, args) -> nn.Module:
    if args.lora_adapter_path:
        logger.info(f"Loading LoRA adapter from {args.lora_adapter_path}")
        model = PeftModel.from_pretrained(model, args.lora_adapter_path, is_trainable=True)
        peft_config = model.peft_config["default"]
        if isinstance(peft_config.task_type, str):
            peft_config.task_type = TaskType.CAUSAL_LM
        model.print_trainable_parameters()
        return model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info(f"Applied LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    return model


def is_lora_model(module: nn.Module) -> bool:
    unwrapped = getattr(module, "_fsdp_wrapped_module", module)
    return hasattr(unwrapped, "peft_config")


def get_lora_config(module: nn.Module) -> dict[str, any]:
    """Extract LoRA config from PEFT model."""
    peft_config = module.peft_config["default"]
    config_dict = {
        "peft_type": "LORA",
        "r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "target_modules": list(peft_config.target_modules),
        "bias": peft_config.bias,
    }
    return config_dict
