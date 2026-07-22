"""Eval against a dedicated eval fleet pinned to HF checkpoint snapshots.

The eval fleet never joins training weight updates; weights reach it only through
``update_weights_from_disk`` on a snapshot exported for a specific rollout_id.
"""

import copy
from argparse import Namespace

from miles.rollout.inference_rollout.inference_rollout_common import GenerateState

__all__ = ["retarget_args", "make_eval_args", "make_eval_generate_state", "EvalFleetSession"]


def retarget_args(args: Namespace, router_ip, router_port, num_gpus: int, num_gpus_per_engine: int) -> Namespace:
    """Shallow-copy ``args`` with the router address and GPU sizing swapped for eval.

    Generate functions read the router from ``args`` and ``GenerateState`` sizes its
    semaphore off the GPU counts, so a retargeted copy runs the standard eval path
    against a different set of engines unchanged.
    """
    eval_args = copy.copy(args)
    eval_args.sglang_router_ip = router_ip
    eval_args.sglang_router_port = router_port
    eval_args.rollout_num_gpus = num_gpus
    eval_args.rollout_num_gpus_per_engine = num_gpus_per_engine
    return eval_args


def make_eval_args(args: Namespace) -> Namespace:
    router_ip, router_port = args.sglang_model_routers["eval"]
    return retarget_args(args, router_ip, router_port, args.eval_num_gpus, args.eval_num_gpus_per_engine)


def make_eval_generate_state(args: Namespace) -> GenerateState:
    return GenerateState(make_eval_args(args))


class EvalFleetSession:
    """Eval-fleet ``GenerateState`` + prompt cache, built lazily on first use.

    Lazy because the eval router only exists once the servers are up, while rollout
    functions are constructed earlier.
    """

    def __init__(self, args: Namespace):
        self.args = args
        self._state: GenerateState | None = None
        self._prompt_dataset_cache: dict = {}

    async def run(self) -> dict:
        from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets

        if self._state is None:
            self._state = make_eval_generate_state(self.args)
        return await run_eval_datasets(self._state, self._prompt_dataset_cache)
