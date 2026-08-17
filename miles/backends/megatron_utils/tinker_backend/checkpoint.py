"""Per-slot training-state serialization for the tinker-compatible backend.

One artifact carries a slot's full training state — bf16 adapter weights plus
each slot child optimizer's state_dict (fp32 masters, Adam moments, both step
counters) and rank/alpha — for named save_state/load_state checkpoints and the
retirement final state. Parameter names are slot-stripped and optimizer
entries positional, so state saved from one slot restores into any slot —
fenced by each rank's recorded per-child parameter names: LayerWise assigns
dense and expert parameters across their respective ownership groups, so two
slots' per-rank ownership patterns can differ and a blind positional restore
would silently load the wrong parameters.
Every rank writes its shard atomically and rank 0 commits a manifest after a
barrier; shards and manifest share a save token so a torn (interrupted) save
can never restore silently. Loading fences on FORMAT, world topology, and
LoRA shape — never on the adapter's display name, so a new registration may
restore another run's state (create-from-checkpoint)."""

import hashlib
import logging
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group

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


def _slot_child_param_names(model, optimizer, slot: int) -> list[list[str | None]]:
    """Per child, the stable (slot-stripped) names of this rank's owned params
    in group/param order — the exact order positional optimizer-state entries
    map to. LayerWise DP sharding narrows each child to this rank's shard, so
    the lists are the rank's ownership signature for the slot; a saved state
    restores positionally only into a slot with the identical signature."""
    names_by_param: dict[int, str] = {}
    for name, param in named_adapter_slot_parameters(model, slot):
        names_by_param[id(param)] = name
        # fp16/bf16 children hold the fp32 masters in their param groups.
        if (main := getattr(param, "main_param", None)) is not None:
            names_by_param[id(main)] = name
    return [
        [names_by_param.get(id(param)) for group in child.param_groups for param in group["params"]]
        for child in _slot_children(optimizer, slot)
    ]


def _save_token(adapter, reason: str) -> str:
    """Deterministic id every rank of one save agrees on (no collective):
    a registration writes any given (destination, reason, step) at most once,
    and a mixed-generation (torn) directory can never carry matching tokens."""
    return hashlib.sha256(f"{adapter.registration_id}:{adapter.step}:{reason}".encode()).hexdigest()[:16]


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
    # state saved from slot A restores into slot B — the recorded per-child
    # param names fence the restore to an identical ownership signature.
    optimizer_state = [child.state_dict() for child in _slot_children(optimizer, slot)]

    rank = dist.get_rank() if dist.is_initialized() else 0
    save_id = _save_token(adapter, reason)
    payload = {
        "format": FORMAT,
        "save_id": save_id,
        "name": adapter.name,
        "registration_id": adapter.registration_id,
        "rank_lora": adapter.config.rank,
        "alpha": adapter.config.alpha,
        "weights": weights,
        "optimizer_state": optimizer_state,
        "optimizer_param_names": _slot_child_param_names(model, optimizer, slot),
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
                "save_id": save_id,
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
    children, in that order, with every fence checked BEFORE anything
    mutates). Returns the restored optimizer step, or None when no loadable
    state exists — a real step-0 state must not be re-initialized."""
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot, load_adapter

    base = find_slot_state(adapter, base)
    if base is None:
        return None
    rank = dist.get_rank() if dist.is_initialized() else 0
    shard = _shard_path(base, rank)
    payload = torch.load(shard, map_location="cpu", weights_only=True)
    manifest = torch.load(base / "manifest.pt", map_location="cpu", weights_only=True)

    slot = adapter.slot
    children = _slot_children(optimizer, slot)
    saved_states = payload.get("optimizer_state") or []
    # Shard fences are PER RANK (a torn save can mix generations across
    # shards; ownership follows LayerWise DP sharding), so one rank can fail
    # while another passes — the verdict must be unanimous BEFORE any rank
    # mutates, or a lone refusal would leave the slot half-restored across
    # ranks (and desync the gloo collectives below).
    problem = None
    if payload.get("format") != FORMAT:
        problem = f"[tinker] ({adapter.name}) state shard format mismatch at {shard}"
    elif payload.get("rank_lora") != adapter.config.rank or payload.get("alpha") != adapter.config.alpha:
        problem = f"[tinker] ({adapter.name}) state shard shape mismatch at {shard}"
    elif payload.get("save_id") != manifest.get("save_id"):
        problem = (
            f"[tinker] ({adapter.name}) state at {base} is torn: shard and manifest come from "
            "different saves (interrupted write); refusing to restore a mixed generation"
        )
    elif len(saved_states) != len(children):
        problem = (
            f"[tinker] ({adapter.name}) state has {len(saved_states)} optimizer children "
            f"but slot {slot} has {len(children)}; refusing partial restore"
        )
    elif payload.get("optimizer_param_names") != _slot_child_param_names(model, optimizer, slot):
        # Positional entries follow LayerWise DP ownership; a different
        # signature would silently restore the wrong parameters' state.
        problem = (
            f"[tinker] ({adapter.name}) state at {base} was sharded with a different per-rank "
            f"parameter ownership than slot {slot} (mismatch on rank {rank}); cross-slot restore "
            "requires an identical ownership signature"
        )
    if dist.is_initialized():
        problems = [None] * dist.get_world_size(get_gloo_group())
        dist.all_gather_object(problems, problem, group=get_gloo_group())
        problem = next((p for p in problems if p is not None), None)
    if problem is not None:
        raise ValueError(problem)

    loaded = load_adapter(model, slot, payload["weights"])
    assert loaded > 0, f"[tinker] ({adapter.name}) state restored 0 weight tensors"
    init_adapter_slot(model, slot, rank=payload["rank_lora"], alpha=payload["alpha"])

    for child, state in zip(children, saved_states, strict=True):
        # MCore copies fp32 masters and Adam state in place (main_param links
        # survive) and takes group hyperparams — including step — from the save.
        child.load_state_dict(state)
        for group in child.param_groups:
            group["miles_multi_lora_slot"] = slot  # the save carries the SOURCE slot's tag

    restored_step = int(payload["clocks"]["optimizer_step"])
    logger.info(f"[tinker] ({adapter.name}) slot state restored at step {restored_step} from {base}")
    return restored_step
