##############################
###########lora###############
##############################
# to-do(yusheng): this should be moved to utils or split into hf_weight_iterator_bridge.py

"""LoRA utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu

logger = logging.getLogger(__name__)

LORA_ADAPTER_NAME = "miles_lora"
LORA_SUBDIR = "tmp_lora"


def is_lora_enabled(args: Namespace) -> bool:
    """Check if LoRA is enabled."""
    return args.lora_rank > 0 or args.lora_adapter_path is not None


# def apply_lora_to_megatron_model(
#     model: Sequence[torch.nn.Module],
#     args: Namespace,
# ) -> Sequence[torch.nn.Module]:
#     """Apply LoRA to Megatron model using Megatron-Bridge PEFT integration.
    
#     This uses the Megatron-Bridge's PEFT support from:
#     https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/peft
    
#     Note: in this version implementation, we use this Megatron-Bridge branch: https://github.com/yushengsu-thu/Megatron-Bridge/tree/merged-megatron-0.16.0rc0
     
#     Args:
#         model: Megatron model (DDP wrapped)
#         args: Training arguments with LoRA config
        
#     Returns:
#         LoRA-wrapped model
#     """
#     # from megatron.bridge.peft import apply_lora_adapter, LoraConfig
#     from megatron.bridge.peft.lora import LoRA
    
#     if args.lora_adapter_path:
#         # TODO: Loading existing LoRA adapter needs separate implementation
#         # Megatron-Bridge may have different API for loading
#         # Refer to this one: https://github.com/volcengine/verl/pull/4063/files#diff-10d5abfbdb508c9478018ad08f295686a960701639fc4e3f3c24a4bdc2f0b711
#         raise NotImplementedError("Loading existing LoRA adapter is not yet implemented")
#     else:
#         # Determine lora_dtype from args
#         if hasattr(args, 'bf16') and args.bf16:
#             lora_dtype = torch.bfloat16
#         elif hasattr(args, 'fp16') and args.fp16:
#             lora_dtype = torch.float16
#         else:
#             lora_dtype = None  # Will use model's dtype
        
#         # Get exclude_modules as list
#         exclude_modules = []
#         if hasattr(args, 'exclude_modules') and args.exclude_modules:
#             if isinstance(args.exclude_modules, str):
#                 exclude_modules = [m.strip() for m in args.exclude_modules.split(",")]
#             else:
#                 exclude_modules = list(args.exclude_modules)
        
#         # Create new LoRA adapter using Megatron-Bridge LoRA dataclass
#         # There are different lora_type, I just use the classic one (speed and acc might not the optimal)
#         # https://github.com/volcengine/verl/pull/4063/files#diff-10d5abfbdb508c9478018ad08f295686a960701639fc4e3f3c24a4bdc2f0b711
#         lora = LoRA(
#             target_modules=args.target_modules,                    # e.g., ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
#             exclude_modules=exclude_modules,                       # Modules to exclude from LoRA
#             dim=args.lora_rank,                                    # LoRA rank (called 'dim' in Megatron-Bridge)
#             alpha=args.lora_alpha,                                 # LoRA alpha scaling factor
#             dropout=args.lora_dropout,                             # LoRA dropout rate
#             dropout_position=getattr(args, 'lora_dropout_position', 'pre'),  # 'pre' or 'post'
#             lora_A_init_method=getattr(args, 'lora_A_init_method', 'xavier'),  # Initialization for LoRA A matrix
#             lora_B_init_method=getattr(args, 'lora_B_init_method', 'zero'),    # Initialization for LoRA B matrix
#             a2a_experimental=getattr(args, 'lora_a2a_experimental', False),    # Experimental All-to-All communication
#             lora_dtype=lora_dtype,                                 # Parameter data type for LoRA weights
#         )
#         logger.info(f"Applying LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, "
#                    f"dropout={args.lora_dropout}, target_modules={args.target_modules}, "
#                    f"exclude_modules={exclude_modules}, lora_dtype={lora_dtype}")
        
#         # Apply LoRA to each model chunk
#         # The LoRA class is callable - calling it applies the transformation
#         for model_chunk in model:
#             # lora(model_chunk.module, training=True) applies LoRA and freezes base model
#             lora(model_chunk.module, training=True)
    
#     # Print trainable parameters info
#     _print_trainable_parameters(model)
    
#     return model


def _print_trainable_parameters(model: Sequence[torch.nn.Module]) -> None:
    """Print trainable parameters statistics."""
    total_params = 0
    trainable_params = 0
    trainable_param_names = []
    
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_param_names.append((name, param.numel()))
    
    if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
        logger.info(
            f"LoRA trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        # if trainable_param_names:
        #     logger.info(f"\nTrainable layers ({len(trainable_param_names)} parameters):")
        #     for name, num_params in trainable_param_names:
        #         logger.info(f"  ✓ {name}: {num_params:,} params")
        # else:
        #     logger.warning("⚠️ NO TRAINABLE PARAMETERS! LoRA may not be applied correctly.")


def is_lora_model(model: Sequence[torch.nn.Module]) -> bool:
    """Check if model has LoRA layers applied."""

    for model_chunk in model:
        if hasattr(model_chunk.module, "peft_config"):
            return True
        # Check for LoRA layers in parameters
        for name, _ in model_chunk.named_parameters():
            if "lora_" in name or "adapter" in name:
                return True
    return False


# def get_lora_state_dict(
#     model: Sequence[torch.nn.Module],
#     args: Namespace,
# ) -> dict[str, torch.Tensor]:
#     """Extract LoRA weights from model.
    
#     Returns only the LoRA adapter weights, not the base model weights.
#     """
#     from miles.backends.megatron_utils.update_weight.common import named_params_and_buffers
    
#     lora_state_dict = {}
    
#     for name, param in named_params_and_buffers(args, model, convert_to_global_name=True):
#         if "lora_" in name or ".adapter." in name:
#             lora_state_dict[name] = param
    
#     return lora_state_dict


# def get_lora_weights_and_config(
#     model: Sequence[torch.nn.Module],
#     args: Namespace,
# ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
#     """Extract LoRA weights and config for tensor-based sync.
    
#     This is used for efficient weight sync to SGLang engines.
#     """
#     lora_state_dict = get_lora_state_dict(model, args)
    
#     # Convert Megatron names to HF-compatible names for SGLang
#     hf_state_dict = {}
#     for name, param in lora_state_dict.items():
#         # Convert megatron naming to HF naming
#         hf_name = _convert_megatron_to_hf_lora_name(name)
#         hf_state_dict[hf_name] = param
    
#     config_dict = {
#         "peft_type": "LORA",
#         "r": args.lora_rank,
#         "lora_alpha": args.lora_alpha,
#         "target_modules": list(args.target_modules),
#         "bias": "none",
#     }
    
#     if mpu.get_data_parallel_rank() == 0:
#         logger.info(f"Extracted {len(hf_state_dict)} LoRA weight tensors for sync")
    
#     return hf_state_dict, config_dict


# def _convert_megatron_to_hf_lora_name(name: str) -> str:
#     """Convert Megatron LoRA parameter name to HuggingFace format.
    
#     Megatron: module.module.decoder.layers.0.self_attention.linear_qkv.lora_A.weight
#     HF: model.layers.0.self_attn.q_proj.lora_A.weight
#     """
#     # This mapping should match your specific model architecture
#     replacements = [
#         ("module.module.decoder.layers.", "model.layers."),
#         (".self_attention.linear_qkv.lora_", ".self_attn.q_proj.lora_"),
#         (".self_attention.linear_proj.lora_", ".self_attn.o_proj.lora_"),
#         (".mlp.linear_fc1.lora_", ".mlp.gate_proj.lora_"),
#         (".mlp.linear_fc2.lora_", ".mlp.down_proj.lora_"),
#     ]
    
#     result = name
#     for old, new in replacements:
#         result = result.replace(old, new)
    
#     return result


# def save_lora_checkpoint(
#     model: Sequence[torch.nn.Module],
#     args: Namespace,
#     save_dir: str,
# ) -> str:
#     """Save LoRA adapter checkpoint to disk.
    
#     Args:
#         model: Megatron model with LoRA
#         args: Training arguments
#         save_dir: Directory to save checkpoint
        
#     Returns:
#         Path to saved checkpoint
#     """
#     from megatron.bridge.peft import save_lora_adapter
    
#     save_path = Path(save_dir)
#     save_path.mkdir(parents=True, exist_ok=True)
    
#     # Use Megatron-Bridge's save function
#     if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
#         for model_chunk in model:
#             save_lora_adapter(model_chunk.module, str(save_path))
#         os.sync()
#         logger.info(f"Saved LoRA adapter to {save_path}")
    
#     dist.barrier()
#     return str(save_path)


## to-do (yusheng): need to confirm usage
def save_lora_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
) -> str:
    """Save LoRA adapter checkpoint to disk in HuggingFace PEFT format.
    
    Since Megatron-Bridge doesn't have a save_lora_adapter function,
    we manually extract adapter weights and convert to PEFT format.
    """
    import json
    from pathlib import Path
    from megatron.bridge.peft.lora_layers import LoRALinear, LinearAdapter, TELinearAdapter
    from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
    
    save_path = Path(save_dir)
    
    # Only rank 0 saves (other ranks just return)
    if not (mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0):
        return str(save_path)
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    lora_state_dict = {}
   
    for model_chunk in model:
        for name, module in model_chunk.named_modules():
            linear_in = None
            linear_out = None
            
            # LoRALinear (wraps base layer with adapter)
            if isinstance(module, AdapterWrapper) and hasattr(module, 'adapter'):
                adapter = module.adapter
                if hasattr(adapter, 'linear_in') and hasattr(adapter, 'linear_out'):
                    linear_in = adapter.linear_in
                    linear_out = adapter.linear_out
            # LinearAdapter/TELinearAdapter (extends nn.Linear with lora)
            elif isinstance(module, (LinearAdapter, TELinearAdapter)):
                if hasattr(module, 'linear_in') and hasattr(module, 'linear_out'):
                    linear_in = module.linear_in
                    linear_out = module.linear_out
            
            if linear_in is not None and linear_out is not None:
                # Convert Megatron naming to HF PEFT naming
                base_name = name.replace("module.module.", "base_model.model.")
                base_name = base_name.replace(".decoder.layers.", ".model.layers.")
                base_name = base_name.replace(".self_attention.linear_qkv", ".self_attn.q_proj")
                base_name = base_name.replace(".self_attention.linear_proj", ".self_attn.o_proj")
                base_name = base_name.replace(".mlp.linear_fc1", ".mlp.gate_proj")
                base_name = base_name.replace(".mlp.linear_fc2", ".mlp.down_proj")
                
                lora_state_dict[f"{base_name}.lora_A.weight"] = linear_in.weight.data.cpu()
                lora_state_dict[f"{base_name}.lora_B.weight"] = linear_out.weight.data.cpu()
    
    # Save weights
    torch.save(lora_state_dict, save_path / "adapter_model.bin")
    
    # Save PEFT config
    config = {
        "peft_type": "LORA",
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": list(args.target_modules) if args.target_modules else ["q_proj", "o_proj", "gate_proj", "down_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    with open(save_path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    os.sync()
    logger.info(f"Saved LoRA adapter to {save_path} with {len(lora_state_dict)} tensors")
    
    return str(save_path)




def load_lora_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    load_dir: str,
) -> None:
    """Load LoRA adapter checkpoint from disk.
    
    Args:
        model: Megatron model
        args: Training arguments
        load_dir: Directory containing checkpoint
    """
    from megatron.bridge.peft import load_lora_adapter
    
    load_path = Path(load_dir)
    if not load_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found at {load_path}")
    
    logger.info(f"Loading LoRA adapter from {load_path}")
    
    for model_chunk in model:
        load_lora_adapter(model_chunk.module, str(load_path))
    
    dist.barrier()


# ## to-do (yusheng): need to confirm usage
# def load_lora_checkpoint(
#     model: Sequence[torch.nn.Module],
#     args: Namespace,
#     load_dir: str,
# ) -> None:
#     """Load LoRA adapter checkpoint from disk.
    
#     Note: This loads PEFT-format checkpoints into Megatron-Bridge LoRA layers.
#     The checkpoint must be in HuggingFace PEFT format (adapter_model.bin + adapter_config.json).
#     """
#     import json
#     from pathlib import Path
#     from megatron.bridge.peft.lora_layers import LoRALinear, LinearAdapter, TELinearAdapter
#     from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
    
#     load_path = Path(load_dir)
#     if not load_path.exists():
#         raise FileNotFoundError(f"LoRA checkpoint not found at {load_path}")
    
#     # Load state dict
#     state_dict_path = load_path / "adapter_model.bin"
#     if not state_dict_path.exists():
#         raise FileNotFoundError(f"adapter_model.bin not found in {load_path}")
    
#     lora_state_dict = torch.load(state_dict_path, map_location="cpu")
    
#     logger.info(f"Loading LoRA adapter from {load_path} with {len(lora_state_dict)} tensors")
    
#     # Build reverse name mapping (HF -> Megatron)
#     def hf_to_megatron_name(hf_name: str) -> str:
#         name = hf_name.replace("base_model.model.", "module.module.")
#         name = name.replace(".model.layers.", ".decoder.layers.")
#         name = name.replace(".self_attn.q_proj", ".self_attention.linear_qkv")
#         name = name.replace(".self_attn.o_proj", ".self_attention.linear_proj")
#         name = name.replace(".mlp.gate_proj", ".mlp.linear_fc1")
#         name = name.replace(".mlp.down_proj", ".mlp.linear_fc2")
#         return name
    
#     # Load weights into model
#     for model_chunk in model:
#         for name, module in model_chunk.named_modules():
#             linear_in = None
#             linear_out = None
            
#             if isinstance(module, AdapterWrapper) and hasattr(module, 'adapter'):
#                 adapter = module.adapter
#                 if hasattr(adapter, 'linear_in') and hasattr(adapter, 'linear_out'):
#                     linear_in = adapter.linear_in
#                     linear_out = adapter.linear_out
#             elif isinstance(module, (LinearAdapter, TELinearAdapter)):
#                 if hasattr(module, 'linear_in') and hasattr(module, 'linear_out'):
#                     linear_in = module.linear_in
#                     linear_out = module.linear_out
            
#             if linear_in is not None and linear_out is not None:
#                 # Find corresponding HF name
#                 base_name = name.replace("module.module.", "base_model.model.")
#                 base_name = base_name.replace(".decoder.layers.", ".model.layers.")
#                 base_name = base_name.replace(".self_attention.linear_qkv", ".self_attn.q_proj")
#                 base_name = base_name.replace(".self_attention.linear_proj", ".self_attn.o_proj")
#                 base_name = base_name.replace(".mlp.linear_fc1", ".mlp.gate_proj")
#                 base_name = base_name.replace(".mlp.linear_fc2", ".mlp.down_proj")
                
#                 lora_a_key = f"{base_name}.lora_A.weight"
#                 lora_b_key = f"{base_name}.lora_B.weight"
                
#                 if lora_a_key in lora_state_dict and lora_b_key in lora_state_dict:
#                     linear_in.weight.data.copy_(lora_state_dict[lora_a_key].to(linear_in.weight.device))
#                     linear_out.weight.data.copy_(lora_state_dict[lora_b_key].to(linear_out.weight.device))
    
#     dist.barrier()
#     logger.info(f"Successfully loaded LoRA adapter from {load_path}")




####!!!! (to-do) yusheng: need to based on different Lora to provide the different mapping
from typing import Union, Type, TYPE_CHECKING
if TYPE_CHECKING:
    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA

def convert_target_modules_to_megatron(
    hf_modules: list[str], 
    lora_type: Union[Type, object, None] = None
) -> list[str]:
    """Convert HuggingFace LoRA target module names to Megatron format.
    
    HF: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    
    Megatron (Standard LoRA): linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron (CanonicalLoRA): linear_q, linear_k, linear_v, linear_proj, 
                              linear_fc1_up, linear_fc1_gate, linear_fc2
    
    Special cases:
    - "all", "all-linear", "all_linear" -> returns all standard Megatron linear modules
    
    Args:
        hf_modules: List of HuggingFace module names or single string
        lora_type: LoRA class or instance (LoRA, CanonicalLoRA, etc.)
                   If None, defaults to CanonicalLoRA format
    
    Returns:
        List of Megatron module names
    
    If input is already in Megatron format, returns as-is without conversion.
    """
    # Get the class name whether lora_type is a class or an instance
    if lora_type is not None:
        if isinstance(lora_type, type):
            # It's a class
            class_name = lora_type.__name__
        else:
            # It's an instance
            class_name = type(lora_type).__name__
        
        logger.info(f"Converting target modules for {class_name}")
    else:
        # Default to CanonicalLoRA if not specified
        class_name = "CanonicalLoRA"
        logger.info(f"Converting target modules (defaulting to CanonicalLoRA)")
    
    # Handle special cases for "all" variants
    if isinstance(hf_modules, str):
        if hf_modules in ["all", "all-linear", "all_linear"]:
            if class_name == "CanonicalLoRA":
                return ["linear_q", "linear_k", "linear_v", "linear_proj", 
                        "linear_fc1_up", "linear_fc1_gate", "linear_fc2"]
            else:  # Standard LoRA
                return ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
        # Convert single string to list
        hf_modules = [hf_modules]
    elif isinstance(hf_modules, list) and len(hf_modules) == 1:
        if hf_modules[0] in ["all", "all-linear", "all_linear"]:
            if class_name == "CanonicalLoRA":
                return ["linear_q", "linear_k", "linear_v", "linear_proj",
                        "linear_fc1_up", "linear_fc1_gate", "linear_fc2"]
            else:  # Standard LoRA
                return ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    
    # Define module name sets based on LoRA type
    if class_name == "CanonicalLoRA":
        megatron_modules_set = {
            "linear_q", "linear_k", "linear_v", "linear_proj",
            "linear_fc1_up", "linear_fc1_gate", "linear_fc2"
        }
    else:  # Standard LoRA
        megatron_modules_set = {"linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"}
    
    hf_modules_set = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    
    # Check if all modules are already in Megatron format (or wildcards)
    all_megatron_format = True
    for module in hf_modules:
        # Skip wildcard patterns (e.g., "*.layers.0.*.linear_qkv")
        if "*" in module:
            continue
        # Check if it's a known HF module name
        if module in hf_modules_set:
            all_megatron_format = False
            break
    
    # If already in Megatron format, return as-is
    if all_megatron_format:
        return hf_modules
    
    # Otherwise, perform conversion based on LoRA type
    if class_name == "CanonicalLoRA":
        # CanonicalLoRA: Split Q/K/V and up/gate
        hf_to_megatron = {
            "q_proj": "linear_q",
            "k_proj": "linear_k",
            "v_proj": "linear_v",
            "o_proj": "linear_proj",
            "gate_proj": "linear_fc1_gate",
            "up_proj": "linear_fc1_up",
            "down_proj": "linear_fc2",
        }
    else:  # Standard LoRA
        # Standard LoRA: Merged Q/K/V and merged up/gate
        hf_to_megatron = {
            "q_proj": "linear_qkv",
            "k_proj": "linear_qkv",
            "v_proj": "linear_qkv",
            "o_proj": "linear_proj",
            "gate_proj": "linear_fc1",
            "up_proj": "linear_fc1",
            "down_proj": "linear_fc2",
        }
    
    megatron_modules = []
    for module in hf_modules:
        if module in hf_to_megatron:
            megatron_name = hf_to_megatron[module]
            if megatron_name not in megatron_modules:
                megatron_modules.append(megatron_name)
        else:
            # Keep as-is if not in mapping (might already be Megatron format or wildcard)
            if module not in megatron_modules:
                megatron_modules.append(module)
    
    return megatron_modules



# def convert_target_modules_to_megatron(hf_modules: list[str]) -> list[str]:
#     """Convert HuggingFace LoRA target module names to Megatron format.
    
#     HF: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
#     Megatron: linear_qkv, linear_proj, linear_fc1, linear_fc2
    
#     Special cases:
#     - "all", "all-linear", "all_linear" -> returns all standard Megatron linear modules
    
#     If input is already in Megatron format, returns as-is without conversion.
#     """
#     # Handle special cases for "all" variants
#     if isinstance(hf_modules, str):
#         if hf_modules in ["all", "all-linear", "all_linear"]:
#             return ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
#         # Convert single string to list
#         hf_modules = [hf_modules]
#     elif isinstance(hf_modules, list) and len(hf_modules) == 1:
#         if hf_modules[0] in ["all", "all-linear", "all_linear"]:
#             return ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    
#     # Define Megatron and HF module name sets
#     megatron_modules_set = {"linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"}
#     hf_modules_set = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    
#     # Check if all modules are already in Megatron format (or wildcards)
#     all_megatron_format = True
#     for module in hf_modules:
#         # Skip wildcard patterns (e.g., "*.layers.0.*.linear_qkv")
#         if "*" in module:
#             continue
#         # Check if it's a known HF module name
#         if module in hf_modules_set:
#             all_megatron_format = False
#             break
    
#     # If already in Megatron format, return as-is
#     if all_megatron_format:
#         return hf_modules
    
#     # Otherwise, perform conversion
#     hf_to_megatron = {
#         "q_proj": "linear_qkv",
#         "k_proj": "linear_qkv",
#         "v_proj": "linear_qkv",
#         "o_proj": "linear_proj",
#         "gate_proj": "linear_fc1",
#         "up_proj": "linear_fc1",
#         "down_proj": "linear_fc2",
#     }
    
#     megatron_modules = []
#     for module in hf_modules:
#         if module in hf_to_megatron:
#             megatron_name = hf_to_megatron[module]
#             if megatron_name not in megatron_modules:
#                 megatron_modules.append(megatron_name)
#         else:
#             # Keep as-is if not in mapping (might already be Megatron format or wildcard)
#             if module not in megatron_modules:
#                 megatron_modules.append(module)
    
#     return megatron_modules




def freeze_base_model(model: Sequence[torch.nn.Module]) -> None:
    """Freeze base model parameters, only keep LoRA trainable."""
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            if "lora_" not in name and "adapter" not in name:
                param.requires_grad = False


def get_trainable_params_for_optimizer(
    model: Sequence[torch.nn.Module],
) -> list[torch.nn.Parameter]:
    """Get only trainable parameters for optimizer (LoRA params only)."""
    trainable_params = []
    for model_chunk in model:
        for param in model_chunk.parameters():
            if param.requires_grad:
                trainable_params.append(param)
    return trainable_params
##############################
##############################
##############################