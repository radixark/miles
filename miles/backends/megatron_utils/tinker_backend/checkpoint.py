"""Per-slot training-state serialization for the tinker-compatible backend.

One artifact carries a slot's full training state — bf16 adapter weights plus
each slot child optimizer's state_dict (fp32 masters, Adam moments, both step
counters) and rank/alpha — for named save_state/load_state checkpoints and the
retirement final state. Parameter names are slot-stripped and optimizer
entries positional, so state saved from one slot restores into any slot.
Every rank writes its shard atomically and rank 0 commits a manifest after a
barrier; loading fences on FORMAT, world topology, and LoRA shape — never on
the adapter's display name, so a new registration may restore another run's
state (create-from-checkpoint)."""

import logging
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

FORMAT = "miles-tinker-slot-v1"
_SLOT_INDEX = re.compile(r"\.adapters\.(\d+)\.")


def stable_slot_param_name(name: str, slot: int) -> str:
    """``...adapters.{slot}.`` -> ``...adapter.``: the exposed-slot naming that
    ``load_adapter`` consumes, so a saved state loads into any slot."""
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


def _slot_children(optimizer, slot: int):
    """The chained optimizer children owning one slot's parameters (tagged by
    the tinker optimizer builder)."""
    return [optimizer.chained_optimizers[i] for i in optimizer.miles_slot_child_indices[slot]]


def sidecar_dir(adapter) -> Path | None:
    """Default state location (retirement final state and resume)."""
    save = adapter.config.save
    return Path(save) / "slot_state" if save is not None else None


def named_state_dir(adapter, tag: str) -> Path | None:
    """Immutable named training-state checkpoint (tinker save_state): same
    shard format, at ``states/{tag}`` under the adapter's save dir."""
    save = adapter.config.save
    return Path(save) / "states" / tag if save is not None else None


def _shard_path(base: Path, rank: int) -> Path:
    return base / f"shard_rank{rank:05d}.pt"


def save_slot_state(
    args,
    model,
    optimizer,
    adapter,
    *,
    reason: str = "state",
    base: Path | None = None,
    ttl_seconds: int | None = None,
) -> Path | None:
    """Write one slot's full training state. Returns the manifest path
    (rank 0) or the shard path. ``base`` overrides the destination (named
    states); ``ttl_seconds`` is recorded in the manifest for a later reaper."""
    base = base if base is not None else sidecar_dir(adapter)
    if base is None:
        logger.warning(f"[tinker] ({adapter.name}) no save dir; slot state NOT persisted ({reason})")
        return None
    base.mkdir(parents=True, exist_ok=True)

    slot = adapter.slot
    weights = {name: param.detach().cpu() for name, param in named_adapter_slot_parameters(model, slot)}
    # Each child state_dict carries the fp32 masters, Adam moments, and both
    # step counters; entries are positional across the slot's children, so a
    # state saved from slot A restores into slot B.
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
        # missing/older manifest as "no valid state".
        tmp_manifest = manifest.with_suffix(".tmp")
        torch.save(
            {
                "format": FORMAT,
                "name": adapter.name,
                "rank_lora": adapter.config.rank,
                "alpha": adapter.config.alpha,
                "optimizer_step": adapter.step,
                "world_size": payload["topology"]["world_size"],
                "ttl_seconds": ttl_seconds,
            },
            tmp_manifest,
        )
        os.replace(tmp_manifest, manifest)
    if dist.is_initialized():
        dist.barrier()
    logger.info(f"[tinker] ({adapter.name}) slot state saved at step {adapter.step} ({reason}) -> {base}")
    return manifest if rank == 0 else shard


def find_slot_state(adapter, base: Path | None = None) -> Path | None:
    """The state base dir, only if a committed manifest matches this
    deployment's shape: FORMAT, world topology, and LoRA rank/alpha. The
    display name is informational — a new registration may load another
    run's state, but never a state of a different shape."""
    base = base if base is not None else sidecar_dir(adapter)
    if base is None or not (base / "manifest.pt").exists():
        return None
    manifest = torch.load(base / "manifest.pt", map_location="cpu", weights_only=True)
    if manifest.get("format") != FORMAT:
        return None
    world = dist.get_world_size() if dist.is_initialized() else 1
    if manifest.get("world_size") != world:
        logger.warning(f"[tinker] ({adapter.name}) state world_size {manifest.get('world_size')} != {world}; ignoring")
        return None
    if manifest.get("rank_lora") != adapter.config.rank or manifest.get("alpha") != adapter.config.alpha:
        logger.warning(
            f"[tinker] ({adapter.name}) state shape rank/alpha "
            f"{manifest.get('rank_lora')}/{manifest.get('alpha')} != "
            f"{adapter.config.rank}/{adapter.config.alpha}; ignoring"
        )
        return None
    return base


def load_slot_state(args, model, optimizer, adapter, *, base: Path | None = None) -> int | None:
    """Restore a slot from a saved state (weights -> rank/alpha -> optimizer
    children, in that order). Returns the restored optimizer step, or None
    when no loadable state exists — a real step-0 state must not be
    re-initialized."""
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot, load_adapter

    base = find_slot_state(adapter, base)
    if base is None:
        return None
    rank = dist.get_rank() if dist.is_initialized() else 0
    shard = _shard_path(base, rank)
    payload = torch.load(shard, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"[tinker] ({adapter.name}) state shard format mismatch at {shard}")
    if payload.get("rank_lora") != adapter.config.rank or payload.get("alpha") != adapter.config.alpha:
        raise ValueError(f"[tinker] ({adapter.name}) state shard shape mismatch at {shard}")

    slot = adapter.slot
    loaded = load_adapter(model, slot, payload["weights"])
    assert loaded > 0, f"[tinker] ({adapter.name}) state restored 0 weight tensors"
    init_adapter_slot(model, slot, rank=payload["rank_lora"], alpha=payload["alpha"])

    children = _slot_children(optimizer, slot)
    saved_states = payload["optimizer_state"]
    if len(saved_states) != len(children):
        raise ValueError(
            f"[tinker] ({adapter.name}) state has {len(saved_states)} optimizer children "
            f"but slot {slot} has {len(children)}; refusing partial restore"
        )
    for child, state in zip(children, saved_states, strict=True):
        # MCore copies fp32 masters and Adam state in place (main_param links
        # survive) and takes group hyperparams — including step — from the save.
        child.load_state_dict(state)
        for group in child.param_groups:
            group["miles_multi_lora_slot"] = slot  # the save carries the SOURCE slot's tag

    restored_step = int(payload["clocks"]["optimizer_step"])
    logger.info(f"[tinker] ({adapter.name}) slot state restored at step {restored_step} from {base}")
    return restored_step
