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

logger = logging.getLogger(__name__)


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

    return MultiLoRA(
        target_modules=convert_target_modules_to_megatron(args.target_modules, lora_type=lora_cls),
        n_adapters=args.multi_lora_n_adapters,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=getattr(args, "lora_dropout", 0.0),
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )


def all_megatron_checkpoints_exist(step_dir: Path, tp_size, pp_size) -> bool:
    return all(
        (step_dir / f"adapter_megatron_tp{tp}_pp{pp}.pt").exists() for tp in range(tp_size) for pp in range(pp_size)
    )


def find_latest_checkpoint(ckpt_dir: Path) -> tuple[Path | None, int]:
    if not ckpt_dir.exists():
        return None, 0

    parallel_state = get_parallel_state()
    tp_size = parallel_state.tp.size
    pp_size = parallel_state.pp.size
    tp_rank = parallel_state.tp.rank
    pp_rank = parallel_state.pp.rank

    def get_step(d):
        return int(d.name.split("_")[1])

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
    adapters: Mapping[str, AdapterRun],
):
    """Save per-adapter checkpoints in two formats per adapter.

    Layout (per adapter)::

        {adapter.save}/checkpoints/step_{iteration}/
        ├── adapter_megatron_tp{tp}_pp{pp}.pt   ← per-rank shard, fast resume
        ├── adapter_model.safetensors           ← gathered HF, inference / external
        └── adapter_config.json                 ← HF PEFT metadata (r, alpha, ...)
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot
    from safetensors.torch import save_file as save_safetensors

    from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_hf
    from miles.utils import megatron_bridge_utils

    parallel_state = get_parallel_state()
    tp_rank = parallel_state.tp.rank
    pp_rank = parallel_state.pp.rank
    # One writer per (tp, pp) shard: LoRA params are replicated across DP AND
    # CP, so gate on the combined dp×cp group. Gating on intra_dp alone left
    # every CP rank writing the same shard file and racing the os.replace.
    is_dp_cp_rank_0 = parallel_state.intra_dp_cp.rank == 0
    is_global_writer = is_dp_cp_rank_0 and tp_rank == 0 and pp_rank == 0

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
        if is_dp_cp_rank_0:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        with expose_adapter_slot(model, adapter.slot):
            # Megatron checkpoints
            if is_dp_cp_rank_0:
                shard: dict[str, torch.Tensor] = {
                    name: param.data.cpu()
                    for batch in model
                    for name, param in batch.named_parameters()
                    if ".adapter." in name
                }
                native_path = tmp_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
                torch.save(shard, native_path)
                logger.info(f"{log_prefix} saved Megatron shard " f"({len(shard)} tensors) to {native_path}")

            hf_state: dict[str, torch.Tensor] = {}
            with megatron_bridge_utils.patch_megatron_model(model):
                for hf_name, weight, _megatron_name in bridge.export_adapter_weights(
                    model,
                    cpu=True,
                    show_progress=False,
                ):
                    # Slice from the shared --lora-rank down to this adapter's real rank to
                    # match adapter_config's r; clone() since safetensors rejects aliased views.
                    hf_state[hf_name] = slice_lora_to_rank(hf_name, weight, config.rank).clone()

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


def _register_adapter(adapter: AdapterRun, model) -> int:
    """Install one adapter on this rank's local model shard. Returns the step
    of the checkpoint it resumed from (0 for a fresh adapter)."""
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot, load_adapter

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

    if ckpt is None:
        logger.info(f"{log_prefix} no checkpoint, starting from random init")
        step = 0
    else:
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        loaded = load_adapter(model, slot, state_dict)
        assert loaded > 0, (
            f"{log_prefix} loaded 0 tensors from {ckpt} "
            f"(state_dict has {len(state_dict)} entries) — name mismatch?"
        )
        logger.info(f"{log_prefix} loaded from {ckpt} ({loaded} tensors)")

    init_adapter_slot(model, slot, rank=config.rank, alpha=config.alpha)
    logger.info(f"{log_prefix} installed at slot {slot}")
    return step


def _deregister_adapter(adapter: AdapterRun, args, model, optimizer) -> None:
    """Model-side cleanup for one adapter."""
    from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot

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

    clear_adapter_slot(model, slot)
    logger.info(f"{log_prefix} cleared adapter slot {slot}")

    # Prevent future slot tenants from inheriting optimizer momentum or the
    # previous tenant's partially accumulated gradients.
    from miles.backends.megatron_utils.multi_lora_optimizer import zero_adapter_slot_grads

    from miles.backends.megatron_utils.multi_lora_scheduler import drop_slot_scheduler

    zero_optimizer_state_for_adapter(optimizer, model, slot)
    zero_adapter_slot_grads(model, slot)
    drop_slot_scheduler(optimizer, slot)
    optimizer.reload_model_params()
    logger.info(f"{log_prefix} cleared optimizer state and retained grads for slot {slot}")


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

    # slot -> adapter_global_batch_size for adapter batches completing now.
    step_batch_sizes = dict(rollout_data.get("step_adapter_batch_sizes", {}))
    grad_norms_by_slot = step_adapter_slots(
        optimizer,
        model,
        step_batch_sizes,
        clip_grad=args.clip_grad,
    )

    if lr_by_slot := step_slot_schedulers(optimizer, step_batch_sizes):
        log_structured(
            logger.info,
            op="adapter_lr",
            rollout=rollout_id,
            step=step_id,
            **{f"slot_{slot}": lr for slot, lr in lr_by_slot.items()},
        )
    return max(grad_norms_by_slot.values(), default=0.0)


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
