"""Dynamic test for the multi-LoRA online add/remove lifecycle.

Runs the standard train loop alongside a small scheduler task that fires
register/deregister events at predefined points. The trainer reacts via
its existing lifecycle hooks (``load_pending_adapters``, the idle gate,
``unload_drained_adapters``) — it has no knowledge of the schedule.

Schedule:
  1. idle 30s (no adapters)
  2. register dapo_math   -> wait 3 productive cycles
  3. register gsm8k       -> wait 3 productive cycles (both active)
  4. deregister dapo_math -> wait 3 productive cycles (gsm8k only)
  5. deregister gsm8k     -> idle 30s (no adapters)
  6. register gsm8k       -> wait 3 productive cycles
  7. register dapo_math   -> trainer runs to --num-rollout
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import ray

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.multi_lora_controller import create_multi_lora_controller
from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.adapter_config import AdapterState, ADAPTER_INACTIVE_STATES, ADAPTER_ROLLOUT_STATES
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)


@dataclass
class Step:
    name: str
    register: tuple[str, ...] = ()
    deregister: tuple[str, ...] = ()
    wait_cycles: int = 0
    wait_seconds: float = 0.0


SCHEDULE: tuple[Step, ...] = (
    Step("idle1",              wait_seconds=30.0),
    Step("load_dapo",          register=("dapo_math",), wait_cycles=2),
    Step("load_gsm8k",         register=("gsm8k",),     wait_cycles=2),
    Step("unload_dapo",        deregister=("dapo_math",), wait_cycles=2),
    Step("unload_gsm8k_idle",  deregister=("gsm8k",),   wait_seconds=30.0),
    Step("reload_gsm8k",       register=("gsm8k",),     wait_cycles=2),
    Step("reload_dapo_to_end", register=("dapo_math",)),
)


async def run_schedule(controller, multi_lora_dir: Path, shared_state: list[int]) -> None:
    """Drive register/deregister events. Talks only to the controller."""
    for step in SCHEDULE:
        logger.info(f"[schedule] >>> {step.name}")
        for name in step.register:
            await controller.register_adapter.remote(str(multi_lora_dir / name))
            logger.info(f"[schedule] registered {name}")
        for name in step.deregister:
            await controller.deregister_adapter.remote(name)
            logger.info(f"[schedule] deregistered {name}")

        # Sample the cycle baseline now so wait_cycles counts cycles completed
        # from this point (the productive cycle handling the deregister, if
        # any, is included).
        cycle_target = None
        if step.wait_cycles > 0:
            start = shared_state[0]
            cycle_target = start + step.wait_cycles

        if step.wait_seconds > 0:
            await asyncio.sleep(step.wait_seconds)
        if cycle_target is not None:
            while shared_state[0] < cycle_target:
                await asyncio.sleep(2.0)

        # Hard prereq for the next step: any name we just deregistered must
        # actually be gone from the controller before we move on. Otherwise
        # a follow-up register on the same name fails with "already
        # registered" (deregister flips state to DRAINING/DRAINED; only
        # unload_drained_adapters frees the slot and removes the entry).
        for name in step.deregister:
            while name in (await controller.adapter_configs.remote()):
                await asyncio.sleep(2.0)
            logger.info(f"[schedule] {name} removed from controller")

        logger.info(f"[schedule] <<< {step.name} done")
    logger.info("[schedule] all steps done; trainer continues to --num-rollout")


async def run_trainer(args, controller, rollout_manager, actor_model, num_rollout_per_epoch, shared_state: list[int]) -> None:
    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()

    # sync starting weights to sglang
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(action="compare")

    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

    async def offload_train():
        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def offload_rollout():
        # Offload if need to train or if need to update adapters
        if args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            await rollout_manager.offload.remote(tags=offload_tags)

    async def save(rollout_id):
        await actor_model.save_model(
            rollout_id,
            force_sync=rollout_id == args.num_rollout - 1,
        )
        if args.rollout_global_dataset:
            await rollout_manager.save.remote(rollout_id)

    rollout_id = args.start_rollout_id

    # Note: in colocated, rollout is inherently tied to train (1 rollout means 1 train) --
    # In async, we should have a run_rollout to gate the rollout.
    def should_run_train(adapter_configs):
        return any(config.state in ADAPTER_ROLLOUT_STATES for config in adapter_configs.values())

    def should_update_adapters(adapter_configs):
        return any(config.state in ADAPTER_INACTIVE_STATES for config in adapter_configs.values())

    # TODO: improve loop readability
    while True:
        adapter_configs = await controller.adapter_configs.remote()
        run_train = should_run_train(adapter_configs)
        update_adapters = should_update_adapters(adapter_configs)

        # Run training
        if run_train:
            rollout_data_ref = await rollout_manager.generate.remote(rollout_id)
            await offload_rollout()

            await actor_model.train(rollout_id, rollout_data_ref)
            await controller.report_training_completed.remote(rollout_id)

        # Load/unload adapteres
        if update_adapters:
            # Train already offloads the rollout
            if not run_train:
                await offload_rollout()
                await actor_model.onload()

            n_loaded = await actor_model.load_pending_adapters()
            n_unloaded = await actor_model.unload_drained_adapters(rollout_id)

        # Both cases need to push weights
        if run_train or update_adapters:
            # For run train, at the end, update rollout id and checkpoint if needed
            if run_train:
                if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
                    await save(rollout_id)
                rollout_id += 1
                shared_state[0] = rollout_id

            # Push the weights to sglang
            await offload_train()
            if args.offload_rollout:
                await rollout_manager.onload_weights.remote()
            await actor_model.update_weights()
            if args.offload_rollout:
                await rollout_manager.onload_kv.remote()
        else:
            print("Nothing to do: sleeping for 5s")
            await asyncio.sleep(5)

    await rollout_manager.dispose.remote()


async def main(args):
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    # No startup registration — the schedule task drives all events.
    controller = create_multi_lora_controller(args.multi_lora_n_adapters, args.lora_rank)
    args.data_source_path = "miles.rollout.multi_lora_data_source.MultiLoRADataSource"
    args.custom_generate_state_path = "miles.ray.multi_lora_controller.MultiLoRAGenerateState"

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, _ = await create_training_models(args, pgs, rollout_manager)

    shared_state = [args.start_rollout_id]

    await asyncio.gather(
        run_trainer(args, controller, rollout_manager, actor_model, num_rollout_per_epoch, shared_state),
        run_schedule(controller, Path(args.multi_lora_dir), shared_state),
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
