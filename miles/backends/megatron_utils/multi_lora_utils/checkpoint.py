"""Per-slot sidecar checkpoints (bf16 weights plus each slot child optimizer's
state_dict — fp32 masters, Adam moments, both step counters — and rank/alpha)
backing slot swap-out/swap-in and per-adapter saves; parameter names are
slot-stripped and optimizer entries positional, so state restores into any slot.
Every rank writes its shard atomically and rank 0 commits a manifest; the LR
scheduler is rebuilt from the restored optimizer step, never serialized."""

import logging
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

FORMAT = "miles-multi-lora-slot-v3"
_SLOT_INDEX = re.compile(r"\.adapters\.(\d+)\.")


def stable_slot_param_name(name: str, slot: int) -> str:
    """``...adapters.{slot}.`` -> ``...adapter.``: the exposed-slot naming that
    ``load_adapter`` consumes, so a sidecar loads into any slot."""
    return _SLOT_INDEX.sub(lambda m: ".adapter." if int(m.group(1)) == slot else m.group(0), name)


def named_adapter_slot_parameters(model, slot: int):
    """Yield (stable_name, model_param) for one slot, in deterministic
    module-traversal order across chunks."""
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

    marker = f".adapters.{slot}."
    seen: set[int] = set()
    model_chunks = model if isinstance(model, (list, tuple)) else [model]
    for model_chunk in model_chunks:
        for module_name, module in model_chunk.named_modules():
            if not isinstance(module, MultiLoRALinear):
                continue
            for param_name, param in module.named_parameters(prefix=module_name):
                if marker in param_name and id(param) not in seen:
                    seen.add(id(param))
                    yield stable_slot_param_name(param_name, slot), param


def sidecar_dir(adapter) -> Path | None:
    save = adapter.config.save
    return Path(save) / "slot_state" if save is not None else None


def named_state_dir(adapter, tag: str) -> Path | None:
    """Immutable named training-state checkpoint (thinker save_state): same
    shard format as the swap sidecar, but never overwritten by swaps."""
    save = adapter.config.save
    return Path(save) / "states" / tag if save is not None else None


def _shard_path(base: Path, rank: int) -> Path:
    return base / f"shard_rank{rank:05d}.pt"


def save_slot_state(args, model, optimizer, adapter, *, reason: str = "swap", base: Path | None = None) -> Path | None:
    """Write-through sidecar: the durable record of the slot's full
    training state. Returns the manifest path (rank 0) or the shard path.
    ``base`` overrides the destination (named immutable states); the default
    is the swap sidecar dir."""
    base = base if base is not None else sidecar_dir(adapter)
    if base is None:
        # Adapters without a save dir are not swap-eligible; callers
        # must pin them instead. Reaching here is a caller bug for swaps.
        logger.warning(f"[multilora] ({adapter.name}) no save dir; slot state NOT persisted ({reason})")
        return None
    base.mkdir(parents=True, exist_ok=True)

    from miles.backends.megatron_utils.multi_lora_utils.optimizer import _slot_children

    slot = adapter.slot
    weights = {name: param.detach().cpu() for name, param in named_adapter_slot_parameters(model, slot)}
    # Each child state_dict carries the fp32 masters, Adam moments, and both
    # step counters; entries are positional across the slot's children, so a
    # sidecar saved from slot A restores into slot B.
    optimizer_state = [child.state_dict() for child in _slot_children(optimizer, slot)]

    rank = dist.get_rank() if dist.is_initialized() else 0
    payload = {
        "format": FORMAT,
        "name": adapter.name,
        "registration_id": adapter.registration_id,
        "rank_lora": adapter.config.rank,
        "alpha": adapter.config.alpha,
        "weights": weights,
        "optimizer_state": optimizer_state,
        "clocks": {"optimizer_step": adapter.step, "serving_version": adapter.version},
        "topology": {
            "rank": rank,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        },
        "reason": reason,
    }
    shard = _shard_path(base, rank)
    tmp = shard.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, shard)  # atomic per shard: a crash never leaves a torn file

    if dist.is_initialized():
        dist.barrier()
    manifest = base / "manifest.pt"
    if rank == 0:
        # Committed only after every rank's shard landed; the loader treats a
        # missing/older manifest as "no valid sidecar".
        tmp_manifest = manifest.with_suffix(".tmp")
        torch.save(
            {
                "format": FORMAT,
                "name": adapter.name,
                "registration_id": adapter.registration_id,
                "optimizer_step": adapter.step,
                "world_size": payload["topology"]["world_size"],
            },
            tmp_manifest,
        )
        os.replace(tmp_manifest, manifest)
    if dist.is_initialized():
        dist.barrier()
    logger.info(f"[multilora] ({adapter.name}) slot state saved at step {adapter.step} ({reason})")
    return manifest if rank == 0 else shard


def find_slot_state(adapter, base: Path | None = None) -> Path | None:
    """The sidecar base dir, only if a committed manifest matches this
    registration's world topology."""
    base = base if base is not None else sidecar_dir(adapter)
    if base is None or not (base / "manifest.pt").exists():
        return None
    manifest = torch.load(base / "manifest.pt", map_location="cpu", weights_only=True)
    if manifest.get("format") != FORMAT or manifest.get("name") != adapter.name:
        return None
    world = dist.get_world_size() if dist.is_initialized() else 1
    if manifest.get("world_size") != world:
        logger.warning(
            f"[multilora] ({adapter.name}) sidecar world_size {manifest.get('world_size')} != {world}; ignoring"
        )
        return None
    return base


def load_slot_state(args, model, optimizer, adapter, *, base: Path | None = None) -> int | None:
    """Restore a slot from its sidecar (weights -> rank/alpha -> optimizer
    children, in that order). Returns the restored optimizer step, or None when
    no sidecar exists — a real step-0 sidecar must not be re-initialized.
    ``base`` restores from a named state instead of the swap sidecar."""
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot, load_adapter

    from miles.backends.megatron_utils.multi_lora_utils.optimizer import _slot_children

    base = find_slot_state(adapter, base)
    if base is None:
        return None
    rank = dist.get_rank() if dist.is_initialized() else 0
    shard = _shard_path(base, rank)
    payload = torch.load(shard, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT or payload.get("name") != adapter.name:
        raise ValueError(f"[multilora] ({adapter.name}) sidecar shard mismatch at {shard}")

    slot = adapter.slot
    loaded = load_adapter(model, slot, payload["weights"])
    assert loaded > 0, f"[multilora] ({adapter.name}) sidecar restored 0 weight tensors"
    init_adapter_slot(model, slot, rank=payload["rank_lora"], alpha=payload["alpha"])

    children = _slot_children(optimizer, slot)
    saved_states = payload["optimizer_state"]
    if len(saved_states) != len(children):
        raise ValueError(
            f"[multilora] ({adapter.name}) sidecar has {len(saved_states)} optimizer children "
            f"but slot {slot} has {len(children)}; refusing partial restore"
        )
    for child, state in zip(children, saved_states, strict=True):
        # MCore copies fp32 masters and Adam state in place (main_param links
        # survive) and takes group hyperparams — including step — from the save.
        child.load_state_dict(state)
        for group in child.param_groups:
            group["miles_multi_lora_slot"] = slot  # the save carries the SOURCE slot's tag

    restored_step = int(payload["clocks"]["optimizer_step"])
    logger.info(f"[multilora] ({adapter.name}) slot state restored at step {restored_step}")
    return restored_step


def swap_out(args, model, optimizer, adapter) -> None:
    """Persist the tenant's full state, then vacate the slot for the next
    tenant: optimizer state and retained grads must never leak across."""
    from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot

    from miles.backends.megatron_utils.multi_lora_utils.optimizer import zero_adapter_slot_grads
    from miles.backends.megatron_utils.multi_lora_utils.scheduler import drop_slot_scheduler
    from miles.backends.megatron_utils.multi_lora_utils.utils import zero_optimizer_state_for_adapter

    save_slot_state(args, model, optimizer, adapter, reason="swap")
    clear_adapter_slot(model, adapter.slot)
    zero_optimizer_state_for_adapter(optimizer, model, adapter.slot)
    zero_adapter_slot_grads(model, adapter.slot)
    drop_slot_scheduler(optimizer, adapter.slot)


def swap_in(args, model, optimizer, adapter) -> int:
    """Bind a tenant into a (vacated) slot: sidecar restore when one exists,
    otherwise the weights-only registration path. Installs the scheduler at
    the restored step (the scheduler clock IS the optimizer step count)."""
    from miles.backends.megatron_utils.multi_lora_utils.scheduler import install_slot_scheduler

    restored_step = load_slot_state(args, model, optimizer, adapter)
    if restored_step is None:
        from miles.backends.megatron_utils.multi_lora_utils.utils import _register_adapter

        restored_step = _register_adapter(adapter, model)
        from miles.backends.megatron_utils.multi_lora_utils.optimizer import reload_adapter_slot_model_params

        reload_adapter_slot_model_params(optimizer, adapter.slot)
    install_slot_scheduler(args, optimizer, adapter, restored_step)
    return restored_step
