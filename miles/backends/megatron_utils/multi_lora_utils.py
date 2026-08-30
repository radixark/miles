import json
import logging
import os
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path

import ray
import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.ray.multi_lora.controller import get_multi_lora_controller
from miles.utils.adapter_config import AdapterRun
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

# Cached by adapter_shard_topology(); the topology is fixed for the run.
_shard_topology: tuple[bool, tuple[tuple[int, int, int], ...]] | None = None


def create_multi_lora_instance(args: Namespace):
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

    # exclude_modules was already folded into target_modules during arg validation.
    return MultiLoRA(
        target_modules=convert_target_modules_to_megatron(args.target_modules, lora_type=lora_cls),
        n_adapters=args.multi_lora_n_adapters,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=getattr(args, "lora_dropout", 0.0),
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )


def megatron_shard_name(tp_rank: int, pp_rank: int, ep_rank: int, ep_size: int) -> str:
    """Adapter shard name for one (tp, pp, ep) coordinate; EP ranks hold different local
    experts. The ep suffix is omitted at ep_size == 1 so legacy checkpoints stay loadable."""
    name = f"adapter_megatron_tp{tp_rank}_pp{pp_rank}"
    if ep_size > 1:
        name += f"_ep{ep_rank}"
    return name + ".pt"


def adapter_shard_topology() -> tuple[bool, tuple[tuple[int, int, int], ...]]:
    """Return ``(this_rank_writes_its_shard, realized (tp, pp, ep) coords)`` via one cached gloo all-gather."""
    global _shard_topology
    if _shard_topology is not None:
        return _shard_topology
    parallel_state = get_parallel_state()
    coords = (parallel_state.tp.rank, parallel_state.pp.rank, parallel_state.ep.rank)
    if not dist.is_initialized():
        _shard_topology = (True, (coords,))
        return _shard_topology

    current_rank = dist.get_rank()
    group = get_gloo_group()
    gathered: list[object] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, (coords, current_rank), group=group)
    is_writer = current_rank == min(rank for entry_coords, rank in gathered if entry_coords == coords)
    _shard_topology = (is_writer, tuple(sorted({entry_coords for entry_coords, _ in gathered})))
    return _shard_topology


def all_megatron_checkpoints_exist(step_dir: Path, shard_names) -> bool:
    return all((step_dir / name).exists() for name in shard_names)


def find_latest_checkpoint(ckpt_dir: Path) -> tuple[Path | None, int]:
    _, coords = adapter_shard_topology()
    if not ckpt_dir.exists():
        return None, 0

    parallel_state = get_parallel_state()
    ep_size = parallel_state.ep.size
    my_coords = (parallel_state.tp.rank, parallel_state.pp.rank, parallel_state.ep.rank)

    expected = {megatron_shard_name(*coord, ep_size) for coord in coords}
    my_shard = megatron_shard_name(*my_coords, ep_size)
    # Legacy pre-expert-adapter layout: no ep suffix; safe for all EP ranks to read (EP-replicated).
    legacy = {megatron_shard_name(tp, pp, 0, 1) for tp, pp, _ in coords}
    my_legacy = megatron_shard_name(my_coords[0], my_coords[1], 0, 1)

    def get_step(d):
        return int(d.name.split("_")[1])

    step_dirs = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=get_step,
        reverse=True,
    )
    for step_dir in step_dirs:
        step = get_step(step_dir)
        if all_megatron_checkpoints_exist(step_dir, expected):
            return step_dir / my_shard, step
        if ep_size > 1 and all_megatron_checkpoints_exist(step_dir, legacy):
            logger.info(f"[multilora] resuming from pre-expert-parallel shard layout in {step_dir}")
            return step_dir / my_legacy, step

    return None, 0


def zero_optimizer_state_for_adapter(optimizer, model, idx: int) -> None:
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, _iter_multi_lora_modules

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
        if inner is None:
            continue
        # TE/apex FusedAdam tracks the Adam step per param GROUP, not per param;
        # reset the retired slot's groups so the next tenant restarts bias correction.
        for group in inner.param_groups:
            if group.get("miles_multi_lora_slot") == idx and "step" in group:
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
            # Bias correction restarts for the slot's next tenant.
            if "step" in state:
                if isinstance(state["step"], torch.Tensor):
                    state["step"].zero_()
                else:
                    state["step"] = 0


def slice_lora_to_rank(hf_name: str, tensor: torch.Tensor, adapter_rank: int) -> torch.Tensor:
    """Trim a max-rank-padded LoRA tensor to ``adapter_rank`` on the rank axis, addressed
    from the end so packed grouped-expert exports are not sliced on the expert axis."""
    if "lora_A" in hf_name:
        rank_dim = tensor.ndim - 2
        if adapter_rank < tensor.shape[rank_dim]:
            remainder = tensor.narrow(rank_dim, adapter_rank, tensor.shape[rank_dim] - adapter_rank)
            assert remainder.abs().max() == 0, (
                f"lora_A padded dims are non-zero: {hf_name}, "
                f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
            )
            return tensor.narrow(rank_dim, 0, adapter_rank)
        return tensor
    if "lora_B" in hf_name:
        rank_dim = tensor.ndim - 1
        if adapter_rank < tensor.shape[rank_dim]:
            remainder = tensor.narrow(rank_dim, adapter_rank, tensor.shape[rank_dim] - adapter_rank)
            assert remainder.abs().max() == 0, (
                f"lora_B padded dims are non-zero: {hf_name}, "
                f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
            )
            return tensor.narrow(rank_dim, 0, adapter_rank)
        return tensor
    return tensor


def _neutral_state_key(name: str, slot: int) -> str:
    """Replace the slot index inside a param name so saved state can load into a different slot."""
    return name.replace(f".adapters.{slot}.", ".adapters.{slot}.".replace(str(slot), "{slot}"))


def slot_training_state_dict(optimizer, model, slot: int) -> dict:
    """This rank's owned share of one slot's training state (fp32 masters, Adam moments, step clocks)."""
    from miles.backends.megatron_utils.multi_lora_optimizer import _slot_children, adapter_slot_parameters

    name_by_id = {}
    model_chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            name_by_id[id(param)] = name

    state_by_main_id: dict[int, dict] = {}
    group_steps: list = []
    for child in _slot_children(optimizer, slot):
        inner = getattr(child, "optimizer", child)
        for param, state in inner.state.items():
            state_by_main_id[id(param)] = state
        for group in inner.param_groups:
            if "step" in group:
                step = group["step"]
                group_steps.append(int(step.item()) if isinstance(step, torch.Tensor) else int(step))

    masters: dict[str, torch.Tensor] = {}
    moments: dict[str, dict] = {}
    for param in adapter_slot_parameters(model, slot):
        main = getattr(param, "main_param", None)
        if main is None:
            continue
        key = _neutral_state_key(name_by_id[id(param)], slot)
        masters[key] = main.detach().cpu().clone()
        state = state_by_main_id.get(id(main), {})
        entry = {}
        for moment_key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if moment_key in state:
                entry[moment_key] = state[moment_key].detach().cpu().clone()
        if "step" in state:
            step = state["step"]
            entry["step"] = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
        moments[key] = entry
    return {"masters": masters, "moments": moments, "group_steps": group_steps}


def load_slot_training_state(optimizer, model, slot: int, ckpt_dir: Path) -> None:
    """Restore one slot's full training state from this rank's file and sync masters into the model weights."""
    from miles.backends.megatron_utils.multi_lora_optimizer import _slot_children, adapter_slot_parameters

    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    state_path = Path(ckpt_dir) / f"training_state_rank{rank}.pt"
    assert state_path.exists(), f"missing slot training state for rank {rank}: {state_path}"
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    assert (
        payload.get("world_size") == world
    ), f"slot training state was saved with world_size={payload.get('world_size')}, current world_size={world}"
    state = payload["state"]

    name_by_id = {}
    model_chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            name_by_id[id(param)] = name

    state_by_main_id: dict[int, dict] = {}
    children = _slot_children(optimizer, slot)
    for child in children:
        inner = getattr(child, "optimizer", child)
        for param, param_state in inner.state.items():
            state_by_main_id[id(param)] = param_state

    for param in adapter_slot_parameters(model, slot):
        main = getattr(param, "main_param", None)
        if main is None:
            continue
        key = _neutral_state_key(name_by_id[id(param)], slot)
        assert key in state["masters"], f"slot training state has no entry for {key}; topology mismatch?"
        main.data.copy_(state["masters"][key])
        entry = state["moments"].get(key, {})
        param_state = state_by_main_id.get(id(main), {})
        for moment_key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if moment_key in entry and moment_key in param_state:
                param_state[moment_key].copy_(entry[moment_key])
        if "step" in entry and "step" in param_state:
            step = param_state["step"]
            if isinstance(step, torch.Tensor):
                step.fill_(entry["step"])
            else:
                param_state["step"] = entry["step"]

    group_steps = list(state.get("group_steps", []))
    group_index = 0
    for child in children:
        inner = getattr(child, "optimizer", child)
        for group in inner.param_groups:
            if "step" in group and group_index < len(group_steps):
                if isinstance(group["step"], torch.Tensor):
                    group["step"].fill_(group_steps[group_index])
                else:
                    group["step"] = group_steps[group_index]
                group_index += 1

    # Masters are authoritative now: sync master -> model; reload_model_params would overwrite them (model -> master).
    for child in children:
        copy_fn = getattr(child, "_copy_main_params_to_model_params", None)
        assert copy_fn is not None, "full-state load requires the Float16 master-to-model sync"
        copy_fn()


def export_adapter_slot(args, model, slot: int, *, adapter_rank: int, bridge=None) -> dict[str, torch.Tensor]:
    """Gather one slot's adapter weights as HF-named CPU tensors sliced to the real rank; no push, no commit."""
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

    from miles.utils import megatron_bridge_utils

    if bridge is None:
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

    hf_state: dict[str, torch.Tensor] = {}
    with expose_adapter_slot(model, slot):
        with megatron_bridge_utils.patch_megatron_model(model):
            for hf_name, weight, _megatron_name in bridge.export_adapter_weights(model, cpu=True, show_progress=False):
                # Slice from the shared --lora-rank down to the adapter's real rank; clone() detaches from live storage.
                hf_state[hf_name] = slice_lora_to_rank(hf_name, weight, adapter_rank).clone()
    return hf_state


def save_multi_lora_checkpoints(
    args,
    model,
    adapter_steps: Mapping[str, int],
    adapters: Mapping[str, AdapterRun],
    optimizer=None,
    include_optimizer_state: bool = False,
):
    """Save per-adapter checkpoints in two formats per adapter.

    Layout (per adapter)::

        {adapter.save}/checkpoints/step_{iteration}/
        ├── adapter_megatron_tp{tp}_pp{pp}[_ep{ep}].pt  ← per-rank shard, fast resume
        ├── adapter_model.safetensors           ← gathered HF, inference / external
        └── adapter_config.json                 ← HF PEFT metadata (r, alpha, ...)
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot
    from safetensors.torch import save_file as save_safetensors

    from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_hf

    parallel_state = get_parallel_state()
    tp_rank = parallel_state.tp.rank
    pp_rank = parallel_state.pp.rank
    ep_rank = parallel_state.ep.rank
    ep_size = parallel_state.ep.size
    # Exactly one writer per (tp, pp, ep) shard; see adapter_shard_topology.
    is_shard_writer, _ = adapter_shard_topology()
    is_global_writer = is_shard_writer and tp_rank == 0 and pp_rank == 0 and ep_rank == 0

    target_modules_hf = (
        convert_target_modules_to_hf(list(args.target_modules))
        if args.target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

    for adapter_name, adapter in adapters.items():
        config = adapter.config
        log_prefix = f"[multilora] ({adapter_name})"
        iteration = adapter_steps[adapter_name]

        if config.save is None:
            logger.info(f"{log_prefix} skipping checkpoint (no save dir configured)")
            continue

        final_dir = config.save / "checkpoints" / f"step_{iteration}"
        tmp_dir = config.save / "checkpoints" / f"_tmp_step_{iteration}"
        write_training_state = include_optimizer_state and optimizer is not None
        if is_shard_writer or write_training_state:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        with expose_adapter_slot(model, adapter.slot):
            # Megatron checkpoints
            if is_shard_writer:
                shard: dict[str, torch.Tensor] = {
                    name: param.data.cpu()
                    for batch in model
                    for name, param in batch.named_parameters()
                    if ".adapter." in name
                }
                native_path = tmp_dir / megatron_shard_name(tp_rank, pp_rank, ep_rank, ep_size)
                torch.save(shard, native_path)
                logger.info(f"{log_prefix} saved Megatron shard " f"({len(shard)} tensors) to {native_path}")

        hf_state = export_adapter_slot(args, model, adapter.slot, adapter_rank=config.rank, bridge=bridge)

        if write_training_state:
            rank = dist.get_rank() if dist.is_initialized() else 0
            world = dist.get_world_size() if dist.is_initialized() else 1
            payload = {
                "format": 1,
                "world_size": world,
                "state": slot_training_state_dict(optimizer, model, adapter.slot),
            }
            torch.save(payload, tmp_dir / f"training_state_rank{rank}.pt")
            logger.info(f"{log_prefix} saved slot training state (rank {rank})")

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
            logger.info(f"{log_prefix} saved HF PEFT to {tmp_dir} " f"({len(hf_state)} tensors)")

        if dist.is_initialized():
            dist.barrier()

        # Write to a temp dir and move into place so readers never see a
        # partially written checkpoint.
        if is_global_writer:
            if final_dir.exists():
                import shutil

                shutil.rmtree(final_dir)
            os.replace(tmp_dir, final_dir)
            logger.info(f"{log_prefix} promoted checkpoint to {final_dir}")
        if dist.is_initialized():
            dist.barrier()


def _seeded_rng_scope(seed: int):
    """Context: fork global+CUDA RNG and the megatron CUDA-RNG tracker, all reseeded deterministically."""
    import contextlib

    @contextlib.contextmanager
    def scope():
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            tracker, saved_states = None, None
            try:
                from megatron.core.tensor_parallel import get_cuda_rng_tracker

                tracker = get_cuda_rng_tracker()
                saved_states = tracker.get_states()
                seeded_states = {}
                for index, name in enumerate(sorted(saved_states)):
                    torch.cuda.manual_seed(seed + index + 1)
                    seeded_states[name] = torch.cuda.get_rng_state()
                tracker.set_states(seeded_states)
                torch.cuda.manual_seed(seed)
            except Exception:
                tracker, saved_states = None, None
            try:
                yield
            finally:
                if tracker is not None and saved_states is not None:
                    tracker.set_states(saved_states)

    return scope()


def install_adapter_slot(
    model, slot: int, *, rank: int, alpha: float, seed: int | None = None, state_dict: dict | None = None
) -> int:
    """Policy-free slot install on this rank's model shard; returns the number of loaded tensors (0 = fresh init)."""
    from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot, init_adapter_slot, load_adapter

    loaded = 0
    if state_dict is not None:
        loaded = load_adapter(model, slot, state_dict)
        assert (
            loaded > 0
        ), f"loaded 0 tensors into slot {slot} (state_dict has {len(state_dict)} entries) — name mismatch?"
        init_adapter_slot(model, slot, rank=rank, alpha=alpha)
    elif seed is None:
        # Legacy fresh install: weights are whatever construction/release re-init left in the slot.
        init_adapter_slot(model, slot, rank=rank, alpha=alpha)
    else:
        # Deterministic fresh install: re-init the slot weights under the seed (bridge init draws from the
        # megatron CUDA-RNG tracker, so it must be reseeded too), identical on every rank for a given seed.
        with _seeded_rng_scope(seed):
            clear_adapter_slot(model, slot)
            init_adapter_slot(model, slot, rank=rank, alpha=alpha)
    return loaded


def release_adapter_slot(model, optimizer, slot: int) -> None:
    """Policy-free slot release: clear weights, optimizer state, grads, and scheduler so the next tenant starts clean."""
    from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot

    from miles.backends.megatron_utils.multi_lora_optimizer import zero_adapter_slot_grads
    from miles.backends.megatron_utils.multi_lora_scheduler import drop_slot_scheduler

    clear_adapter_slot(model, slot)
    zero_optimizer_state_for_adapter(optimizer, model, slot)
    zero_adapter_slot_grads(model, slot)
    drop_slot_scheduler(optimizer, slot)
    optimizer.reload_model_params()


def _register_adapter(adapter: AdapterRun, model) -> int:
    """E2E install wrapper: checkpoint auto-resume around the policy-free install; returns the resumed step (0 = fresh)."""
    name = adapter.name
    config = adapter.config
    slot = adapter.slot
    log_prefix = f"[multilora] ({name})"

    step = 0
    if config.save is not None:
        ckpt_root = config.save / "checkpoints"
        ckpt, step = find_latest_checkpoint(ckpt_root)
    else:
        ckpt = None

    state_dict = None
    if ckpt is None:
        logger.info(f"{log_prefix} no checkpoint, starting from random init")
        step = 0
    else:
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)

    loaded = install_adapter_slot(model, slot, rank=config.rank, alpha=config.alpha, state_dict=state_dict)
    if loaded:
        logger.info(f"{log_prefix} loaded from {ckpt} ({loaded} tensors)")
    logger.info(f"{log_prefix} installed at slot {slot}")
    return step


def _deregister_adapter(adapter: AdapterRun, args, model, optimizer) -> None:
    """E2E release wrapper: final-checkpoint policy around the policy-free release."""
    name = adapter.name
    slot = adapter.slot
    log_prefix = f"[multilora] ({name})"

    if args.save_interval is not None:
        # The controller still holds the step count until free_slot runs.
        step = ray.get(get_multi_lora_controller().adapter_step.remote(name))
        save_multi_lora_checkpoints(args, model, {name: step}, {name: adapter})
        logger.info(f"{log_prefix} saved final checkpoint at step {step}")
    else:
        logger.info(f"{log_prefix} save_interval unset; skipping final checkpoint")

    release_adapter_slot(model, optimizer, slot)
    logger.info(f"{log_prefix} released slot {slot}: weights, optimizer state, grads, and scheduler cleared")


def load_adapters(args, model, optimizer, adapters) -> int:
    """Load adapters into Megatron slots; resumes step counts from checkpoints."""
    from miles.backends.megatron_utils.initialize import is_first_replica_megatron_main_rank
    from miles.utils.distributed_utils import get_gloo_group

    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    if not adapters:
        return 0
    from miles.backends.megatron_utils.multi_lora_scheduler import install_slot_scheduler

    resume_steps: dict[str, int] = {}
    for adapter in adapters:
        resume_steps[adapter.name] = _register_adapter(adapter, model)
        # Per-adapter LR/WD schedule, positioned at the resumed step count.
        install_slot_scheduler(args, optimizer, adapter, resume_steps[adapter.name])
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    optimizer.reload_model_params()
    if is_first_replica_megatron_main_rank():
        for name, step in resume_steps.items():
            if step > 0:
                ray.get(get_multi_lora_controller().set_adapter_step.remote(name, step))
    return len(adapters)


def cleanup_adapters(args, model, optimizer, adapters) -> int:
    """Save final ckpt + clear Megatron slot, then free_slot on the controller."""
    from miles.backends.megatron_utils.initialize import is_first_replica_megatron_main_rank
    from miles.utils.distributed_utils import get_gloo_group

    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    if not adapters:
        return 0
    for adapter in adapters:
        _deregister_adapter(adapter, args, model, optimizer)
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    if is_first_replica_megatron_main_rank():
        for adapter in adapters:
            ray.get(get_multi_lora_controller().free_slot.remote(adapter.name))
    return len(adapters)


def step_stepped_adapter_slots(args, model, optimizer, rollout_data, rollout_id: int, step_id: int) -> float:
    """Optimizer-step the slots whose adapter batch completes with this train batch and advance
    their per-adapter LR/WD schedules. Returns the max grad norm across stepped slots (0.0 if none)."""
    from miles.backends.megatron_utils.multi_lora_optimizer import step_adapter_slots
    from miles.backends.megatron_utils.multi_lora_scheduler import step_slot_schedulers
    from miles.utils.tracking_utils.structured_log import log_structured

    # slot -> adapter_global_batch_size for adapter batches completing now; SCHEDULED_MEAN scales by 1/batch.
    step_batch_sizes = dict(rollout_data.get("step_adapter_batch_sizes", {}))
    grad_norms_by_slot = step_adapter_slots(
        optimizer,
        model,
        {slot: 1.0 / batch_size for slot, batch_size in step_batch_sizes.items()},
        {slot: args.clip_grad for slot in step_batch_sizes},
    )

    if lr_by_slot := step_slot_schedulers(optimizer, step_batch_sizes):
        log_structured(
            logger.info,
            op="adapter_lr",
            rollout=rollout_id,
            step=step_id,
            **{f"slot_{slot}": lr for slot, lr in lr_by_slot.items()},
        )
    return max((norm for norm in grad_norms_by_slot.values() if norm is not None), default=0.0)


def commit_trained_batch(rollout_data, rollout_id: int, pending_push: set) -> None:
    """A train call landed: schedule the stepped adapters' engine push and
    commit the batch on the controller (main rank only). The stepped set ships
    with the train data, identical on all ranks."""
    from miles.backends.megatron_utils.initialize import is_first_replica_megatron_main_rank

    pending_push.update(rollout_data.get("step_adapter_names", []))
    if is_first_replica_megatron_main_rank():
        ray.get(get_multi_lora_controller().mark_batch_trained.remote(rollout_id))


def save_due_adapter_checkpoints(args, model) -> bool:
    """Save per-adapter checkpoints for adapters at a save-interval multiple
    without a checkpoint on disk. Rank 0 picks and broadcasts, so the
    collective export lines up. Returns False when nothing is due."""
    from miles.backends.megatron_utils.initialize import is_first_replica_megatron_main_rank
    from miles.utils.distributed_utils import get_gloo_group

    due_buffer = [None]
    if is_first_replica_megatron_main_rank() and args.save_interval is not None:
        snapshot = ray.get(get_multi_lora_controller().snapshot.remote())
        adapters = {**snapshot["active"], **snapshot["retiring"]}
        due_buffer[0] = {
            name: adapter
            for name, adapter in adapters.items()
            if adapter.step > 0
            and adapter.step % args.save_interval == 0
            and adapter.config.save is not None
            and not (Path(adapter.config.save) / "checkpoints" / f"step_{adapter.step}").exists()
        }
    if dist.is_initialized():
        dist.broadcast_object_list(due_buffer, src=0, group=get_gloo_group())
    due_adapters = due_buffer[0]
    if not due_adapters:
        return False
    adapter_steps = {name: adapter.step for name, adapter in due_adapters.items()}
    save_multi_lora_checkpoints(args, model, adapter_steps, due_adapters)
    return True


def select_adapters_to_push(loaded_adapters: dict, pending_push: set, has_new_engines: bool) -> tuple[dict, list]:
    """Pick the stale adapters to push (all loaded adapters when engines are new). Returns
    (adapters to push keyed by name, names to version-bump — only those whose weights changed)."""
    pending = pending_push & set(loaded_adapters)
    push_names = set(loaded_adapters) if has_new_engines else pending
    return {name: loaded_adapters[name] for name in sorted(push_names)}, sorted(pending)


def commit_weight_push(version_update_names: list, is_main_rank: bool) -> None:
    """A weight push landed: bump the pushed adapters' slot versions on the
    controller (promotes PENDING adapters to ACTIVE)."""
    if version_update_names and is_main_rank:
        ray.get(get_multi_lora_controller().record_weight_update.remote(version_update_names))
