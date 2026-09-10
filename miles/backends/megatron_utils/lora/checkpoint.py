"""Per-slot checkpoints: adapter weights plus the slot's optimizer state.

Weight shards are (tp, pp, ep)-addressed and slot-agnostic (saved under
expose_adapter_slot). Optimizer state is per global rank because LayerWise
scatters whole params across ranks; resume requires the same world topology.
"""

import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import MegatronOptimizer

from miles.backends.megatron_utils.lora.optimizer import _slot_children
from miles.backends.megatron_utils.lora.slots import adapter_shard_topology, megatron_shard_name
from miles.backends.training_utils.parallel import get_parallel_state


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _weight_shard_name() -> str:
    parallel_state = get_parallel_state()
    return megatron_shard_name(
        parallel_state.tp.rank, parallel_state.pp.rank, parallel_state.ep.rank, parallel_state.ep.size
    )


def _optim_shard_name() -> str:
    return f"optim_rank{_rank()}.pt"


def save_slot(model: Sequence[DDP], optimizer: MegatronOptimizer, slot: int, path: str) -> None:
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

    is_shard_writer, _ = adapter_shard_topology()
    final_dir = Path(path)
    tmp_dir = final_dir.parent / f"_tmp_{final_dir.name}"

    def make_tmp_dir():
        if _rank() == 0:
            tmp_dir.mkdir(parents=True, exist_ok=True)

    def write_shards():
        if is_shard_writer:
            with expose_adapter_slot(model, slot):
                shard = {
                    name: param.data.cpu()
                    for model_chunk in model
                    for name, param in model_chunk.named_parameters()
                    if ".adapter." in name
                }
            assert shard, f"slot {slot} exposed no adapter tensors"
            torch.save(shard, tmp_dir / _weight_shard_name())
        torch.save(_optimizer_slot_state(optimizer, slot), tmp_dir / _optim_shard_name())

    def publish_dir():
        # write-then-rename so readers never see a partial checkpoint; on overwrite
        # the old version survives (as _old_<name>) until the replacement is in place
        if _rank() == 0:
            if final_dir.exists():
                old_dir = final_dir.parent / f"_old_{final_dir.name}"
                if old_dir.exists():
                    shutil.rmtree(old_dir)
                os.replace(final_dir, old_dir)
                os.replace(tmp_dir, final_dir)
                shutil.rmtree(old_dir)
            else:
                os.replace(tmp_dir, final_dir)

    _run_checkpoint_phase(make_tmp_dir)
    _run_checkpoint_phase(write_shards)
    _run_checkpoint_phase(publish_dir)


def load_slot(model: Sequence[DDP], optimizer: MegatronOptimizer, slot: int, path: str, load_optimizer: bool) -> None:
    from megatron.bridge.peft.multi_lora_layers import load_adapter

    checkpoint_dir = Path(path)
    shards: dict = {}

    def read_shards():
        shards["weights"] = torch.load(checkpoint_dir / _weight_shard_name(), map_location="cpu", weights_only=True)
        if load_optimizer:
            optim_state = torch.load(checkpoint_dir / _optim_shard_name(), map_location="cpu", weights_only=True)
            _check_optimizer_slot_state(optimizer, slot, optim_state)
            shards["optim"] = optim_state

    def apply_shards():
        loaded = load_adapter(model, slot, shards["weights"])
        assert loaded > 0, f"loaded 0 adapter tensors from {checkpoint_dir / _weight_shard_name()}"
        optimizer.reload_model_params()
        if load_optimizer:
            _load_optimizer_slot_state(optimizer, slot, shards["optim"])
        # weights-only load keeps the fresh Adam state the slot init just zeroed

    # every rank reads and validates its shards before any rank touches the live slot,
    # so a missing or mismatched checkpoint cannot leave mixed or rank-divergent state
    _run_checkpoint_phase(read_shards)
    _run_checkpoint_phase(apply_shards)


def _run_checkpoint_phase(operation) -> None:
    """Run a checkpoint phase and propagate any rank's failure to every rank."""
    error = None
    try:
        operation()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    if not dist.is_initialized():
        if error is not None:
            raise RuntimeError(error)
        return
    errors = [None] * _world_size()
    dist.all_gather_object(errors, error)
    failed = [e for e in errors if e is not None]
    if failed:
        raise RuntimeError(f"slot checkpoint failed on {len(failed)} rank(s): {failed[0]}")


def _optimizer_slot_state(optimizer: MegatronOptimizer, slot: int) -> dict:
    children_states = []
    for child in _slot_children(optimizer, slot):
        inner = child.optimizer
        group_steps = []
        for group in inner.param_groups:
            step = group.get("step", 0)
            group_steps.append(step.cpu() if torch.is_tensor(step) else step)
        params = []
        masters = []
        for main_param in child.get_parameters():
            # a never-stepped slot has no per-param state yet
            state = inner.state[main_param] if main_param in inner.state else {}
            params.append({key: value.cpu() if torch.is_tensor(value) else value for key, value in state.items()})
            # rebuilding FP32 masters from BF16 would lose low bits on resume
            masters.append(main_param.data.cpu())
        children_states.append({"group_steps": group_steps, "params": params, "masters": masters})
    return {"world_size": _world_size(), "children": children_states}


def _check_optimizer_slot_state(optimizer: MegatronOptimizer, slot: int, saved: dict) -> None:
    """Validate the saved layout against the live slot before any state is touched."""
    assert saved["world_size"] == _world_size(), (
        f"optimizer state was saved with world_size={saved['world_size']}; "
        f"resume requires the same topology (got {_world_size()})"
    )
    children = _slot_children(optimizer, slot)
    assert len(children) == len(saved["children"]), "optimizer layout changed since save"
    for child, child_state in zip(children, saved["children"], strict=True):
        assert len(child_state["group_steps"]) == len(
            child.optimizer.param_groups
        ), "optimizer layout changed since save"
        params = child.get_parameters()
        assert len(child_state["params"]) == len(params), "optimizer layout changed since save"
        masters = child_state.get("masters")
        assert masters is None or len(masters) == len(params), "optimizer layout changed since save"


def _load_optimizer_slot_state(optimizer: MegatronOptimizer, slot: int, saved: dict) -> None:
    children = _slot_children(optimizer, slot)
    for child, child_state in zip(children, saved["children"], strict=True):
        inner = child.optimizer
        # FusedAdam clocks steps on the param group, not in per-param state
        for group, step in zip(inner.param_groups, child_state["group_steps"], strict=True):
            existing = group.get("step")
            if torch.is_tensor(existing):
                existing.copy_(torch.as_tensor(step))
            elif "step" in group or step:
                group["step"] = step
        # checkpoints written before masters were saved restore from BF16 as before
        masters = child_state.get("masters") or [None] * len(child_state["params"])
        for main_param, param_state, master in zip(
            child.get_parameters(), child_state["params"], masters, strict=True
        ):
            if master is not None:
                main_param.data.copy_(master.to(main_param.device))
            state = inner.state[main_param]
            for key, value in param_state.items():
                if not torch.is_tensor(value):
                    state[key] = value
                elif torch.is_tensor(state.get(key)):
                    state[key].copy_(value.to(state[key].device))
                else:
                    state[key] = value.to(main_param.device)
