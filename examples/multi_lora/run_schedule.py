
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
import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import ray

from miles.ray.multi_lora_controller import get_multi_lora_controller

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

async def run_schedule(controller, multi_lora_dir: Path) -> None:
    """Drive register/deregister events. Talks only to the controller."""
    for step in SCHEDULE:
        print(f"[schedule] >>> {step.name}")
        for name in step.register:
            await controller.register_adapter.remote(name, str(multi_lora_dir / name / "adapter.yaml"))
            print(f"[schedule] registered {name}")
        for name in step.deregister:
            await controller.deregister_adapter.remote(name)
            print(f"[schedule] deregistered {name}")

        # Sample the cycle baseline now so wait_cycles counts cycles completed
        # from this point (the productive cycle handling the deregister, if
        # any, is included).
        cycle_target = None
        if step.wait_cycles > 0:
            track_name = step.register[0] if step.register else step.deregister[0]
            start = await controller.last_trained_rollout_id.remote()
            cycle_target = start + step.wait_cycles

        if step.wait_seconds > 0:
            await asyncio.sleep(step.wait_seconds)
        if cycle_target is not None:
            while True:
                last_train_step = await controller.last_trained_rollout_id.remote()
                if last_train_step >= cycle_target:
                    break
                await asyncio.sleep(2.0)

        # Hard prereq for the next step: any name we just deregistered must
        # actually be gone from the controller before we move on. Otherwise
        # a follow-up register on the same name fails with "already
        # registered" (deregister flips state to DRAINING/DRAINED; only
        # unload_drained_adapters frees the slot and removes the entry).
        for name in step.deregister:
            while name in (await controller.adapter_configs.remote()):
                await asyncio.sleep(2.0)
            print(f"[schedule] {name} removed from controller")

        print(f"[schedule] <<< {step.name} done")
    print("[schedule] all steps done; trainer continues to YAML num_row or num_epoch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-LoRA adapter schedule against a live trainer.")
    parser.add_argument("--multi-lora-dir", type=str, required=True,
                        help="Path to directory containing adapter subdirectories with adapter.yaml files")
    parser.add_argument("--ray-address", type=str, default="auto",
                        help="Ray cluster address (default: auto)")
    args = parser.parse_args()

    # Wait for Ray cluster to be available
    while True:
        try:
            ray.init(address=args.ray_address)
            break
        except Exception:
            print("Waiting for Ray cluster to start...")
            time.sleep(5)

    # Wait for the trainer to create the controller
    controller = None
    while controller is None:
        try:
            controller = get_multi_lora_controller()
        except ValueError:
            print("Waiting for trainer to create the controller...")
            time.sleep(5)

    asyncio.run(run_schedule(controller, Path(args.multi_lora_dir)))
