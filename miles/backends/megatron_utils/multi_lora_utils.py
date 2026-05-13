import dataclasses
import logging
import json
import os
from argparse import Namespace
from typing import Tuple
from pathlib import Path
from collections.abc import Mapping

import ray
import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.ray.multi_lora_controller import get_multi_lora_controller
from miles.utils.adapter_config import AdapterConfig

logger = logging.getLogger(__name__)


def is_multi_lora_enabled(args: Namespace) -> bool:
    return getattr(args, "multi_lora", False)


def create_multi_lora(args: Namespace):
    """Create a MultiLoRA instance from training args."""
    from megatron.bridge.peft.multi_lora import MultiLoRA

    from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_megatron

    lora_type_name = getattr(args, "lora_type", "lora").lower()
    if lora_type_name == "canonical_lora":
        from megatron.bridge.peft.canonical_lora import CanonicalLoRA
        lora_cls = CanonicalLoRA
    else:
        from megatron.bridge.peft.lora import LoRA
        lora_cls = LoRA

    return MultiLoRA(
        target_modules=convert_target_modules_to_megatron(args.target_modules, lora_type=lora_cls),
        n_adapters=args.multi_lora_n_adapters,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=getattr(args, "lora_dropout", 0.0),
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )


def build_multi_lora_model(args: Namespace):
    """Build Megatron model with MultiLoRA layers via megatron-bridge.

    Returns DDP-wrapped model chunks. Does NOT register adapters or load checkpoints —
    that happens after the optimizer is created.
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.config import DistributedDataParallelConfig
    from transformers import AutoConfig

    from miles.backends.megatron_utils.bridge_lora_helpers import _make_value_model_hook

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)

    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.expert_model_parallel_size = args.expert_model_parallel_size
    provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    provider.sequence_parallel = args.sequence_parallel
    provider.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    provider.context_parallel_size = args.context_parallel_size
    provider.variable_seq_lengths = True
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_router_load_balancing_type = "none"
    provider.finalize()

    multi_lora = create_multi_lora(args)

    def apply_hook(model_chunks):
        transformed = multi_lora(model_chunks, training=True)
        multi_lora.set_params_to_save(transformed)
        return transformed

    provider.register_pre_wrap_hook(apply_hook)

    is_value_model = (
        "ForTokenClassification" in hf_config.architectures[0]
        or "ForSequenceClassification" in hf_config.architectures[0]
    )
    if is_value_model:
        hidden_size = hf_config.text_config.hidden_size if hasattr(hf_config, "text_config") else hf_config.hidden_size
        provider.register_pre_wrap_hook(_make_value_model_hook(hidden_size, provider.sequence_parallel))

    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    ddp_config.finalize()

    if args.offload_train:
        from miles.backends.megatron_utils.lora_utils import patch_param_grad_buffer_for_colocate_mode_lora
        patch_param_grad_buffer_for_colocate_mode_lora()

    model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)
    return model, multi_lora


def initialize_multi_lora_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
):
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    from miles.backends.megatron_utils.checkpoint import load_checkpoint
    from miles.backends.megatron_utils.ci_utils import check_model_hashes, check_peak_gpu_memory_after_load
    from miles.backends.megatron_utils.model import get_optimizer_param_scheduler
    from miles.utils.memory_utils import clear_memory

    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from miles.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync

    model, multi_lora = build_multi_lora_model(args)
    model[0].role = role

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None
    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)

    # Hide adapter params so the bridge's conversion-task walk doesn't see them
    # while loading the base checkpoint.
    from megatron.bridge.peft.multi_lora_layers import hide_adapters

    clear_memory()
    with hide_adapters(model):
        iteration, _ = load_checkpoint(
            model, optimizer, opt_param_scheduler,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
    check_peak_gpu_memory_after_load(args)
    clear_memory()
    check_model_hashes(args, model, iteration)
    opt_param_scheduler.step(increment=iteration * args.global_batch_size)

    load_pending_adapters(args, model, optimizer)

    return model, optimizer, opt_param_scheduler, iteration

def all_megatron_checkpoints_exist(step_dir: Path, tp_size, pp_size) -> bool:
    return all(
        (step_dir / f"adapter_megatron_tp{tp}_pp{pp}.pt").exists()
        for tp in range(tp_size)
        for pp in range(pp_size)
    )

def find_latest_checkpoint(ckpt_dir: Path) -> Tuple[Path | None, int]:
    if not ckpt_dir.exists():
        return None, 0

    from megatron.core import mpu

    tp_size = mpu.get_tensor_model_parallel_world_size()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    get_step = lambda d: int(d.name.split("_")[1])

    step_dirs = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=get_step,
        reverse=True,
    )
    for step_dir in step_dirs:
        step = get_step(step_dir)
        if all_megatron_checkpoints_exist(step_dir, tp_size, pp_size):
            return step_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt", step

    return None, 0


def zero_optimizer_state_for_adapter(optimizer, model, idx: int) -> None:
    from megatron.bridge.peft.multi_lora_layers import (
        MultiLoRALinear,
        _iter_multi_lora_modules,
    )

    target_main_params = set()
    for module in _iter_multi_lora_modules(model):
        if not isinstance(module, MultiLoRALinear):
            continue
        adapter = module.adapters[idx]
        for param in adapter.parameters():
            main = getattr(param, "main_param", None)
            target_main_params.add(id(main if main is not None else param))

    chained = getattr(optimizer, "chained_optimizers", [optimizer])
    for chained_optimizer in chained:
        inner = getattr(chained_optimizer, "optimizer", chained_optimizer)
        for param, state in inner.state.items():
            if id(param) not in target_main_params:
                continue
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].zero_()

def slice_lora_to_rank(hf_name: str, tensor: torch.Tensor, adapter_rank: int) -> torch.Tensor:
    if "lora_A" in hf_name and adapter_rank < tensor.shape[0]:
        remainder = tensor[adapter_rank:]
        assert remainder.abs().max() == 0, (
            f"lora_A padded dims are non-zero: {hf_name}, "
            f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
        )
        return tensor[:adapter_rank]
    if "lora_B" in hf_name and adapter_rank < tensor.shape[1]:
        remainder = tensor[:, adapter_rank:]
        assert remainder.abs().max() == 0, (
            f"lora_B padded dims are non-zero: {hf_name}, "
            f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
        )
        return tensor[:, :adapter_rank]
    return tensor


def save_multi_lora_checkpoints(
    args,
    model,
    adapter_steps: Mapping[str, int],
    adapter_configs: Mapping[str, AdapterConfig],
):
    """Save per-adapter checkpoints in two formats per adapter.

    Layout (per adapter)::

        {adapter.dir}/checkpoints/step_{iteration}/
        ├── adapter_megatron_tp{tp}_pp{pp}.pt   ← per-rank shard, fast resume
        ├── adapter_model.safetensors           ← gathered HF, inference / external
        └── adapter_config.json                 ← HF PEFT metadata (r, alpha, ...)
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot
    from megatron.core import mpu
    from safetensors.torch import save_file as save_safetensors

    from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_hf
    from miles.utils import megatron_bridge_utils

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    is_dp_rank_0 = get_parallel_state().intra_dp.rank == 0
    is_global_writer = is_dp_rank_0 and tp_rank == 0 and pp_rank == 0

    target_modules_hf = (
        convert_target_modules_to_hf(list(args.target_modules))
        if args.target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

    for adapter_name, config in adapter_configs.items():
        log_prefix = f"[multilora] ({adapter_name})"
        iteration = adapter_steps[adapter_name]

        final_dir = config.dir / "checkpoints" / f"step_{iteration}"
        tmp_dir = config.dir / "checkpoints" / f"_tmp_step_{iteration}"
        if is_dp_rank_0:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        with expose_adapter_slot(model, config.slot):
            # Megatron checkpoints
            if is_dp_rank_0:
                shard: dict[str, torch.Tensor] = {
                    name: param.data.cpu()
                    for chunk in model
                    for name, param in chunk.named_parameters()
                    if ".adapter." in name
                }
                native_path = tmp_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
                torch.save(shard, native_path)
                logger.info(
                    f"{log_prefix} saved Megatron shard "
                    f"({len(shard)} tensors) to {native_path}"
                )

            hf_state: dict[str, torch.Tensor] = {}
            with megatron_bridge_utils.patch_megatron_model(model):
                for hf_name, weight, _megatron_name in bridge.export_adapter_weights(
                    model, cpu=True, show_progress=False,
                ):
                    # Safetensors format can't save aliased tensors, so need clone()
                    hf_state[hf_name] = weight.clone()

        if is_global_writer:
            save_safetensors(
                hf_state,
                str(tmp_dir / "adapter_model.safetensors"),
                metadata={"format": "pt"},
            )
            adapter_config_json = {
                "peft_type": "LORA",
                "r": config.rank,
                "lora_alpha": config.alpha,
                "target_modules": target_modules_hf,
                "lora_dropout": getattr(args, "lora_dropout", 0.0),
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
            with open(tmp_dir / "adapter_config.json", "w") as f:
                json.dump(adapter_config_json, f, indent=2)
            os.sync()
            logger.info(
                f"{log_prefix} saved HF PEFT to {tmp_dir} "
                f"({len(hf_state)} tensors)"
            )

        if dist.is_initialized():
            dist.barrier()

        # to avoid partially complete checkpoints, move the checkpoint to the
        # actual directory after everything is complete
        #
        # TODO(mathewjhan): it could be nice to have a callback that checks when all
        # checkpoints are available in final_dir and writes a final marker file to the folder,
        # useful for distributed file systems and verifying that the checkpoint is complete
        # Currently, this only guarantees that the trainer processes have written everything,
        # but doesn't account for the actual visibility of each checkpoint shard on every
        # node due to network latency, consistency semantics, etc
        if is_global_writer:
            if final_dir.exists():
                import shutil
                shutil.rmtree(final_dir)
            os.replace(tmp_dir, final_dir)
            logger.info(f"{log_prefix} promoted checkpoint to {final_dir}")
        if dist.is_initialized():
            dist.barrier()


def _register_adapter(name: str, config: AdapterConfig, model) -> None:
    """Install one PENDING adapter on this rank's local model shard.
    """
    from miles.backends.megatron_utils.initialize import is_megatron_main_rank
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot, load_adapter

    log_prefix = f"[multilora] ({name})"

    ckpt_root = config.dir / "checkpoints"
    ckpt, step = find_latest_checkpoint(ckpt_root)

    if is_megatron_main_rank():
        ray.get(get_multi_lora_controller().set_train_step.remote(name, step))

    if ckpt is None:
        logger.info(f"{log_prefix} no checkpoint under {ckpt_root}, starting from random init")
    else:
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        loaded = load_adapter(model, config.slot, state_dict)
        assert loaded > 0, (
            f"{log_prefix} loaded 0 tensors from {ckpt} "
            f"(state_dict has {len(state_dict)} entries) — name mismatch?"
        )
        logger.info(f"{log_prefix} loaded from {ckpt} ({loaded} tensors)")

    init_adapter_slot(model, config.slot, rank=config.rank, alpha=config.alpha)
    logger.info(f"{log_prefix} installed at slot {config.slot}")


def _deregister_adapter(name: str, config: AdapterConfig, rollout_id: int, args, model, optimizer) -> None:
    """Model-side cleanup for one DRAINED adapter.
    """
    from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot

    log_prefix = f"[multilora] ({name})"

    train_steps = ray.get(get_multi_lora_controller().adapter_train_steps.remote())
    step = train_steps[name]

    # Save the checkpoint
    save_multi_lora_checkpoints(args, model, {name: step}, {name: config})
    logger.info(f"{log_prefix} saved final checkpoint")

    # Clear out the multilora slot in the multilora layer in the Megatron model
    clear_adapter_slot(model, config.slot)
    logger.info(f"{log_prefix} cleared adapter slot {config.slot}")

    # Zero out the optimizer state to prevent future adapters from reusing previous adapter
    # momentum, etc
    zero_optimizer_state_for_adapter(optimizer, model, config.slot)
    optimizer.reload_model_params()
    logger.info(f"{log_prefix} cleared optimizer state for slot {config.slot}")


def _adapters_in_state(state):
    configs = ray.get(get_multi_lora_controller().adapter_configs.remote())
    return [(n, c) for n, c in configs.items() if c.state == state]


def load_pending_adapters(args, model, optimizer) -> int:
    from miles.backends.megatron_utils.initialize import is_megatron_main_rank
    from miles.utils.adapter_config import AdapterState
    from miles.utils.distributed_utils import get_gloo_group

    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    pending = _adapters_in_state(AdapterState.PENDING)
    if not pending:
        return 0

    for name, config in pending:
        _register_adapter(name, config, model)

    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())

    if is_megatron_main_rank():
        for name, _ in pending:
            ray.get(get_multi_lora_controller().update_adapter_state.remote(name, AdapterState.ACTIVE))
    optimizer.reload_model_params()
    return len(pending)


def unload_drained_adapters(args, model, optimizer, rollout_id: int) -> int:
    """DRAINED adapters model-side cleanup.
    """
    from miles.backends.megatron_utils.initialize import is_megatron_main_rank
    from miles.utils.adapter_config import AdapterState
    from miles.utils.distributed_utils import get_gloo_group

    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    drained = _adapters_in_state(AdapterState.DRAINED)
    if not drained:
        return 0
    for name, config in drained:
        _deregister_adapter(name, config, rollout_id, args, model, optimizer)
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    if is_megatron_main_rank():
        for name, _ in drained:
            ray.get(get_multi_lora_controller().mark_removed.remote(name))
    return len(drained)
