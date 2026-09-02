import logging
import re
from dataclasses import replace as dataclass_replace
from pathlib import Path

import ray
import torch
import torch.distributed as dist

from miles.backends.megatron_utils.api_backends.multi_lora.checkpoint import (
    load_slot_state,
    named_state_dir,
    save_slot_state,
)
from miles.backends.megatron_utils.api_backends.multi_lora.executor import MultiLoraParameterExecutor
from miles.backends.megatron_utils.api_backends.multi_lora.optimizer import (
    reload_adapter_slot_model_params,
    zero_adapter_slot_grads,
)
from miles.backends.training_utils.operation_execution import run_optim_controls
from miles.ray.multi_lora.controller import get_multi_lora_controller
from miles.ray.multi_lora.residency import lease_from_metadata
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

_STATE_TAG = re.compile(r"[A-Za-z0-9._-]+")


def zero_optimizer_state_for_adapter(optimizer, model, slot: int) -> None:
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, _iter_multi_lora_modules

    target_main_params = set()
    for module in _iter_multi_lora_modules(model):
        if not isinstance(module, MultiLoRALinear):
            continue
        for param in module.adapters[slot].parameters():
            main = getattr(param, "main_param", None)
            target_main_params.add(id(main if main is not None else param))

    chained = getattr(optimizer, "chained_optimizers", [optimizer])
    for chained_optimizer in chained:
        inner = getattr(chained_optimizer, "optimizer", chained_optimizer)
        if inner is None:
            continue
        # TE/apex FusedAdam tracks the Adam step per param GROUP, not per param.
        for group in inner.param_groups:
            if group.get("miles_multi_lora_slot") == slot and "step" in group:
                if isinstance(group["step"], torch.Tensor):
                    group["step"].zero_()
                else:
                    group["step"] = 0
        for param, state in inner.state.items():
            if id(param) not in target_main_params:
                continue
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].zero_()
            if "step" in state:
                if isinstance(state["step"], torch.Tensor):
                    state["step"].zero_()
                else:
                    state["step"] = 0


def _install_adapter(adapter, args, model, optimizer) -> int | None:
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot

    log_prefix = f"[tinker] ({adapter.name})"
    try:
        restored_step = load_slot_state(args, model, optimizer, adapter)
    except ValueError as e:
        logger.warning(f"{log_prefix} sidecar state not restorable into slot {adapter.slot} ({e}); fresh init")
        restored_step = None
    if restored_step is not None:
        logger.info(f"{log_prefix} resumed slot {adapter.slot} from sidecar at step {restored_step}")
        return restored_step
    init_adapter_slot(model, adapter.slot, rank=adapter.config.rank, alpha=adapter.config.alpha)
    logger.info(f"{log_prefix} fresh init at slot {adapter.slot}")
    return None


def load_adapters(args, model, optimizer, adapters) -> int:
    from miles.backends.megatron_utils.initialize import is_first_replica_megatron_main_rank

    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    if not adapters:
        return 0
    installed_steps: dict[str, int | None] = {}
    for adapter in adapters:
        installed_steps[adapter.name] = _install_adapter(adapter, args, model, optimizer)
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    for adapter in adapters:
        if installed_steps[adapter.name] is None:
            reload_adapter_slot_model_params(optimizer, adapter.slot)
    if is_first_replica_megatron_main_rank():
        controller = get_multi_lora_controller()
        for name, step in installed_steps.items():
            if step:
                ray.get(controller.set_adapter_step.remote(name, step))
        ray.get(controller.mark_ready.remote(sorted(installed_steps)))
    return len(adapters)


def cleanup_adapters(args, model, optimizer, adapters) -> int:
    from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot

    from miles.backends.megatron_utils.initialize import is_first_replica_megatron_main_rank

    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    if not adapters:
        return 0
    for adapter in adapters:
        save_slot_state(args, model, optimizer, adapter, reason="final")
        clear_adapter_slot(model, adapter.slot)
        zero_optimizer_state_for_adapter(optimizer, model, adapter.slot)
        zero_adapter_slot_grads(model, adapter.slot)
        reload_adapter_slot_model_params(optimizer, adapter.slot)
        logger.info(f"[tinker] ({adapter.name}) slot {adapter.slot} retired and scrubbed")
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    if is_first_replica_megatron_main_rank():
        for adapter in adapters:
            ray.get(get_multi_lora_controller().free_slot.remote(adapter.name))
    return len(adapters)


def reconcile_adapters(args, model, optimizer, loaded_adapters: dict, pending_push: set, weights_backuper) -> None:
    from miles.backends.megatron_utils.initialize import is_first_replica_megatron_main_rank

    broadcast_buffer = [None]
    if is_first_replica_megatron_main_rank():
        controller = get_multi_lora_controller()
        ray.get(controller.retire_adapters.remote())
        # Queued registrations take freed slots so this reconcile loads them.
        ray.get(controller.bootstrap_pending.remote())
        snapshot = ray.get(controller.snapshot.remote())
        cleanup_steps = {name: ray.get(controller.adapter_step.remote(name)) for name in snapshot["cleanup"]}
        broadcast_buffer[0] = (snapshot, cleanup_steps)
    if dist.is_initialized():
        dist.broadcast_object_list(broadcast_buffer, src=0, group=get_gloo_group())
    snapshot, cleanup_steps = broadcast_buffer[0]
    should_be_loaded = {
        name: run
        for name, run in {**snapshot["pending"], **snapshot["ready"], **snapshot["retiring"]}.items()
        # Queued-but-unbound registrations have no residency to reconcile yet.
        if run.slot is not None
    }
    cleanup_names = set(snapshot["cleanup"])

    loaded_names = set(loaded_adapters)
    # Sorted so per-adapter collectives run in the same order on every rank;
    # set iteration order is process-specific.
    adapters_to_load = sorted(
        (adapter for name, adapter in should_be_loaded.items() if name not in loaded_names),
        key=lambda adapter: adapter.name,
    )
    adapters_to_clean_up = sorted(
        (loaded_adapters[n] for n in loaded_names if n in cleanup_names or n not in should_be_loaded),
        key=lambda adapter: adapter.name,
    )
    if adapters_to_load:
        load_adapters(args, model, optimizer, adapters_to_load)
        for adapter in adapters_to_load:
            loaded_adapters[adapter.name] = adapter
        weights_backuper.backup("actor")
    if adapters_to_clean_up:
        # The registry's step clock is authoritative for the final state; the
        # loaded views were captured at load time and lag it.
        refreshed = [
            dataclass_replace(adapter, step=cleanup_steps.get(adapter.name, adapter.step))
            for adapter in adapters_to_clean_up
        ]
        cleanup_adapters(args, model, optimizer, refreshed)
        for adapter in adapters_to_clean_up:
            loaded_adapters.pop(adapter.name, None)
            pending_push.discard(adapter.name)
        weights_backuper.backup("actor")

    # Deregistered before ever being loaded: nothing to save or clear.
    if is_first_replica_megatron_main_rank():
        for name in cleanup_names - loaded_names:
            ray.get(get_multi_lora_controller().free_slot.remote(name))


def execute_controls(
    args, model, optimizer, loaded_adapters, pending_push, weights_backuper, operations, lease_metadata
) -> dict:
    lease = lease_from_metadata(lease_metadata)
    executor = MultiLoraParameterExecutor(model=model, optimizer=optimizer, loaded_adapters=loaded_adapters)
    results = run_optim_controls(operations, lease, executor)

    def state_order(op: dict):
        binding = lease.binding_of(op["operation_id"])
        return (op["kind"], binding.training_slot if binding is not None else -1)

    for op in sorted(
        (op for op in operations if op["kind"] in ("save_weights_for_sampler", "save_state", "load_state")),
        key=state_order,
    ):
        results[op["operation_id"]] = _execute_state_op(
            op, lease, args, model, optimizer, loaded_adapters, pending_push
        )
        if results[op["operation_id"]].get("ok") and op["kind"] == "load_state":
            weights_backuper.backup("actor")

    for op in operations:
        if op["operation_id"] not in results:
            results[op["operation_id"]] = dict(
                ok=False, error=f"operation kind '{op['kind']}' has no executor", category="server"
            )
    return results


def _execute_state_op(op: dict, lease, args, model, optimizer, loaded_adapters, pending_push) -> dict:
    name, kind = op["name"], op["kind"]
    binding = lease.binding_of(op["operation_id"])
    if binding is None:
        return dict(
            ok=False, error=f"operation '{op['operation_id']}' has no binding in the batch lease", category="server"
        )
    bound_name, bound_registration_id = binding.registration_key
    if bound_name != name:
        return dict(
            ok=False,
            error=f"operation '{op['operation_id']}' names adapter '{name}' but its lease binding "
            f"names '{bound_name}'",
            category="server",
        )
    run = loaded_adapters.get(name)
    if run is None or run.registration_id != bound_registration_id or run.slot != binding.training_slot:
        return dict(
            ok=False, error=f"adapter '{name}' is not resident in slot {binding.training_slot}", category="server"
        )
    # The registry's clocks are authoritative; the loaded view can lag.
    run = dataclass_replace(run, step=op.get("step", run.step), version=op.get("serving_version", run.version))

    if kind == "save_weights_for_sampler":
        pending_push.add(name)
        return dict(ok=True, deferred="publish")

    payload = op.get("payload") or {}
    if kind == "save_state":
        tag = str(payload.get("tag") or f"step_{run.step}")
        if not _STATE_TAG.fullmatch(tag) or tag in (".", ".."):
            return dict(ok=False, error=f"invalid state tag '{tag}'", category="user")
        base = named_state_dir(run, tag)
        if base is None:
            return dict(ok=False, error=f"adapter '{name}' has no save dir", category="user")
        if (base / "manifest.pt").exists():
            return dict(ok=False, error=f"state '{tag}' already exists; states are immutable", category="user")
        save_slot_state(
            args, model, optimizer, run, reason=f"state:{tag}", base=base, ttl_seconds=payload.get("ttl_seconds")
        )
        return dict(ok=True, result=dict(path=str(base), step=run.step))

    assert kind == "load_state"
    path = payload.get("path")
    try:
        restored_step = load_slot_state(args, model, optimizer, run, base=Path(path))
    except ValueError as e:
        return dict(ok=False, error=str(e), category="user")
    if restored_step is None:
        return dict(ok=False, error=f"no loadable state at '{path}' for adapter '{name}'", category="user")
    pending_push.add(name)
    return dict(ok=True, deferred="publish", result=dict(step=restored_step, path=str(path)))


def validate_batch_lease(rollout_data, loaded_adapters: dict) -> None:
    lease = rollout_data.get("batch_execution_lease")
    if lease is None:
        raise RuntimeError("tinker batch carries no execution lease")
    for op_id, (name, registration_id, slot) in lease["bindings_by_operation"]:
        run = loaded_adapters.get(name)
        if run is None or run.registration_id != registration_id or run.slot != slot:
            raise RuntimeError(
                f"operation '{op_id}': lease binding ('{name}', {registration_id[:8]}, slot {slot}) "
                "does not match this rank's loaded adapters; refusing to mutate"
            )


def commit_batch(rollout_data, pending_push: set) -> None:
    from miles.backends.megatron_utils.initialize import is_first_replica_megatron_main_rank

    logprobs_by_op = _gather_logprobs(rollout_data)
    if is_first_replica_megatron_main_rank():
        try:
            registration_by_lane = rollout_data.get("registration_by_lane", {})
            # Forward batches accumulate nothing: no dirty streams.
            accumulated = (
                []
                if rollout_data.get("tinker_forward_only")
                else sorted({tuple(key) for key in registration_by_lane.values()})
            )
            operation_ids = [op_id for op_id in rollout_data.get("operation_by_lane", {}).values() if op_id]
            ray.get(get_multi_lora_controller().commit_tinker_batch.remote(accumulated, operation_ids, logprobs_by_op))
        finally:
            if (lease := rollout_data.get("batch_execution_lease")) is not None:
                ray.get(get_multi_lora_controller().release_batch_lease.remote(lease))


def _gather_logprobs(rollout_data) -> dict[str, list[list[float]]]:
    collector = rollout_data.get("tinker_logprob_collector") or {}
    if dist.is_initialized():
        shards = [None] * dist.get_world_size(get_gloo_group())
        dist.all_gather_object(shards, collector, group=get_gloo_group())
        merged: dict = {}
        for shard in shards:
            merged.update(shard or {})
    else:
        merged = dict(collector)

    op_by_lane = rollout_data.get("operation_by_lane", {})
    logprobs_by_op: dict[str, list[list[float]]] = {}
    for op_lane, op_id in op_by_lane.items():
        if op_id is None:
            continue
        # row -1 is DP padding: never part of the operation's result plane.
        rows = sorted((row, lp) for (lane, row), lp in merged.items() if lane == op_lane and row >= 0)
        logprobs_by_op[op_id] = [lp for _, lp in rows]
    return logprobs_by_op


def select_adapters_to_push(loaded_adapters: dict, pending_push: set, has_new_engines: bool) -> tuple[dict, list]:
    pending = pending_push & set(loaded_adapters)
    push_names = set(loaded_adapters) if has_new_engines else pending
    return {name: loaded_adapters[name] for name in sorted(push_names)}, sorted(pending)


def commit_weight_push(version_update_names: list, is_main_rank: bool) -> None:
    if version_update_names and is_main_rank:
        ray.get(get_multi_lora_controller().record_weight_update.remote(version_update_names))
