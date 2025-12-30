import logging
import os
import shutil
from pathlib import Path

import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

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

    model = get_peft_model(model, lora_config)  # autocast_adapter_dtype=False)
    model.print_trainable_parameters()
    logger.info(f"Applied LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    return model


def is_lora_model(module: nn.Module) -> bool:
    unwrapped = getattr(module, "_fsdp_wrapped_module", module)
    return hasattr(unwrapped, "peft_config")


def save_lora_to_disk(module: nn.Module, save_dir: str) -> str:
    """Save LoRA adapter to disk with file lock mechanism."""
    # TODO: All gather lora layers not full layers
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_state_dict = get_model_state_dict(module, options=options)

    lora_state_dict = {name: param for name, param in full_state_dict.items() if "lora_" in name}

    if dist.get_rank() == 0:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        module.save_pretrained(str(save_path), state_dict=lora_state_dict)

        # TODO: check if file lock is needed or better way to do it
        os.sync()

        logger.info(f"Saved LoRA adapter to {save_path}")
    return save_dir


def delete_lora_from_disk(save_dir: str) -> None:
    """Delete LoRA adapter files from disk."""
    save_path = Path(save_dir)
    if save_path.exists():
        shutil.rmtree(save_path)
        logger.info(f"Deleted LoRA adapter from {save_path}")


def get_lora_weights_and_config(module: nn.Module) -> tuple[dict[str, any], dict[str, any]]:
    """Extract LoRA weights and config from PEFT model for tensor-based sync."""
    # TODO: only gather lora weights, or gather lora weights in bucket logic i.e., layered summon
    # options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    options = StateDictOptions(full_state_dict=True, cpu_offload=False)
    full_state_dict = get_model_state_dict(module, options=options)

    state_dict = {name: param for name, param in full_state_dict.items() if "lora_" in name}
    if dist.get_rank() == 0:
        logger.info(f"Extracted {len(state_dict)} LoRA weight tensors")

    for name in list(state_dict.keys()):
        key = name.replace(".default.weight", ".weight")  # .replace("base_model.model.", "")
        state_dict[key] = state_dict.pop(name)

    peft_config = module.peft_config["default"]
    config_dict = {
        "peft_type": "LORA",
        "r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "target_modules": list(peft_config.target_modules),
        "bias": peft_config.bias,
    }

    return state_dict, config_dict
