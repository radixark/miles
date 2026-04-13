"""Multi-LoRA weight sync: export per-adapter weights and send to SGLang engines.

Uses expose_adapter_slot to make the bridge see one adapter at a time,
then reuses the existing single-LoRA weight sync machinery.
"""

import logging

import ray
import torch
import torch.distributed as dist

from miles.backends.megatron_utils.lora_utils import build_lora_sync_config, is_lora_weight_name
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

_loaded_adapters: set[str] = set()


def slice_lora_to_rank(hf_name: str, tensor: torch.Tensor, adapter_rank: int) -> torch.Tensor:
    """Slice a LoRA weight tensor from max_rank to adapter_rank for export."""
    if "lora_A" in hf_name:
        return tensor[:adapter_rank]
    if "lora_B" in hf_name:
        return tensor[:, :adapter_rank]
    return tensor




def sync_multi_lora_weights(
    args,
    model,
    adapter_configs: dict[str, dict],
    rollout_engines,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    active_slots: set[int] | None = None,
):
    """Sync all adapter weights to SGLang engines.

    For each adapter slot, temporarily exposes it as a single-LoRA adapter
    and uses the existing single-LoRA weight sync path (HfWeightIteratorBridge
    + _send_hf_params pattern) to export and send weights.
    """
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

    from miles.utils.megatron_bridge_utils import patch_megatron_model
    from .common import post_process_weights
    from .update_weight_from_tensor import _send_to_colocated_engine
    from .hf_weight_iterator_bridge import HfWeightIteratorBridge

    rank = dist.get_rank()

    # Pause generation
    if rank == 0:
        mode = args.pause_generation_mode
        ray.get([engine.pause_generation.remote(mode=mode) for engine in rollout_engines])
        ray.get([engine.flush_cache.remote() for engine in rollout_engines])
    dist.barrier(group=get_gloo_group())

    for adapter_name, cfg in adapter_configs.items():
        idx = cfg["slot"]
        adapter_rank = cfg.get("rank", args.lora_rank)

        if active_slots is not None and idx not in active_slots:
            logger.info(f"Skipping weight sync for adapter '{adapter_name}' (slot {idx}) — not trained this step")
            continue

        lora_config = build_lora_sync_config(args)
        lora_config["r"] = adapter_rank
        lora_config["lora_alpha"] = cfg.get("alpha", args.lora_alpha)

        # Use the same HfWeightIteratorBridge as single LoRA, inside expose_adapter_slot
        with expose_adapter_slot(model, idx):
            iterator = HfWeightIteratorBridge(args=args, model=model, model_name=None, quantization_config=None, is_lora=True)
            for hf_named_tensors in iterator.get_hf_weight_chunks({}):
                weight_tensors = [
                    (n, slice_lora_to_rank(n, t, adapter_rank))
                    for n, t in hf_named_tensors if is_lora_weight_name(n)
                ]
                if not weight_tensors:
                    continue

                refs, long_lived = _send_to_colocated_engine(
                    hf_named_tensors=weight_tensors,
                    ipc_engine=ipc_engine,
                    ipc_gather_src=ipc_gather_src,
                    ipc_gather_group=ipc_gather_group,
                    lora_config=lora_config,
                    lora_name=adapter_name,
                    lora_loaded=adapter_name in _loaded_adapters,
                )
                if refs:
                    results = ray.get(refs)
                del long_lived

        _loaded_adapters.add(adapter_name)
        logger.info(f"Synced adapter '{adapter_name}' weights to SGLang")

    dist.barrier(group=get_gloo_group())

    # Resume generation
    if rank == 0:
        post_process_weights(
            rollout_engines=rollout_engines,
            restore_weights_before_load=False,
            post_process_quantization=True,
        )
        ray.get([engine.continue_generation.remote() for engine in rollout_engines])
    dist.barrier(group=get_gloo_group())


def save_multi_lora_checkpoints(
    args,
    model,
    iteration: int,
    adapter_configs: dict[str, dict],
):
    """Save per-adapter checkpoints to each adapter's directory."""
    from pathlib import Path

    from megatron.core import mpu
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

    from .hf_weight_iterator_bridge import HfWeightIteratorBridge

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    for adapter_name, cfg in adapter_configs.items():
        idx = cfg["slot"]
        ckpt_dir = Path(cfg["dir"]) / "checkpoints" / f"step_{iteration}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        adapter_state = {}
        with expose_adapter_slot(model, idx):
            iterator = HfWeightIteratorBridge(args=args, model=model, model_name=None, quantization_config=None, is_lora=True)
            for hf_named_tensors in iterator.get_hf_weight_chunks({}):
                for hf_name, weight in hf_named_tensors:
                    adapter_state[hf_name] = weight.cpu()

        native_path = ckpt_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
        torch.save(adapter_state, native_path)
        logger.info(f"Saved adapter '{adapter_name}' checkpoint ({len(adapter_state)} tensors) to {native_path}")


def deregister_adapter(
    name: str,
    rollout_id: int,
    args,
    model,
    optimizer,
    controller,
    ipc_engine=None,
    ipc_gather_src=None,
):
    """Full cleanup for an exhausted adapter: save, unload, reset, deregister."""
    from megatron.bridge.peft.multi_lora_layers import unregister_adapter

    from ..multi_lora import zero_optimizer_state_for_adapter

    adapter_configs = ray.get(controller.active_runs.remote())
    if name not in adapter_configs:
        return

    cfg = adapter_configs[name]
    idx = cfg["slot"]

    # 1. Save final checkpoint
    save_multi_lora_checkpoints(args, model, rollout_id, {name: cfg})
    logger.info(f"Saved final checkpoint for adapter '{name}'")

    # 2. Unload from SGLang
    if ipc_engine is not None and dist.get_rank() == ipc_gather_src:
        try:
            ray.get(ipc_engine.unload_lora_adapter.remote(lora_name=name))
        except Exception:
            pass
    logger.info(f"Unloaded adapter '{name}' from SGLang")

    # 3. Reset layer weights
    unregister_adapter(model, idx)
    logger.info(f"Reset layer weights for adapter '{name}' (slot {idx})")

    # 4. Zero optimizer state and sync reset weights to fp32 main params
    zero_optimizer_state_for_adapter(optimizer, model, idx)
    optimizer.reload_model_params()

    # 5. Deregister from controller (frees slot)
    ray.get(controller.deregister_run.remote(name))
    _loaded_adapters.discard(name)
    logger.info(f"Fully deregistered adapter '{name}'")
