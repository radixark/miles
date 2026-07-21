"""Eval against a dedicated eval fleet pinned to checkpoint snapshots.

The eval fleet is a second model entry (``name="eval"``, ``update_weights=False``)
behind its own router; it never joins the training weight-update group and is never
paused or aborted by training. Weights reach it exclusively through
``update_weights_from_disk`` on an HF snapshot exported for a specific rollout_id,
so every eval point measures one well-defined weight version.

The helpers here retarget an args namespace at the eval fleet so the existing
generate machinery (which reads ``args.sglang_router_ip/port`` and sizes its
concurrency semaphore off ``args.rollout_num_gpus``) works against it unchanged.
"""

import copy
from argparse import Namespace

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnEvalOutput
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState

__all__ = ["retarget_args", "make_eval_args", "make_eval_generate_state", "CheckpointEvalRolloutFn"]


def retarget_args(args: Namespace, router_ip, router_port, num_gpus: int, num_gpus_per_engine: int) -> Namespace:
    """Shallow-copy ``args`` with the router address and GPU sizing swapped for eval.

    Generate functions read the router from ``args`` and ``GenerateState`` sizes its
    semaphore off the GPU counts, so a retargeted copy is all that is needed to run
    the standard eval path against a different set of engines. The original ``args``
    is not modified.
    """
    eval_args = copy.copy(args)
    eval_args.sglang_router_ip = router_ip
    eval_args.sglang_router_port = router_port
    eval_args.rollout_num_gpus = num_gpus
    eval_args.rollout_num_gpus_per_engine = num_gpus_per_engine
    return eval_args


def make_eval_args(args: Namespace) -> Namespace:
    """Retarget ``args`` at the in-job eval fleet's router (requires servers started)."""
    router_ip, router_port = args.sglang_model_routers["eval"]
    return retarget_args(args, router_ip, router_port, args.eval_num_gpus, args.eval_num_gpus_per_engine)


def make_eval_generate_state(args: Namespace) -> GenerateState:
    return GenerateState(make_eval_args(args))


class CheckpointEvalRolloutFn:
    """Eval-only rollout function running against the dedicated eval fleet.

    Addressable via ``--eval-function-path`` when the train rollout function should
    not serve eval itself.
    """

    def __init__(self, input: RolloutFnConstructorInput):
        self.args = input.args
        self._state: GenerateState | None = None
        self._prompt_dataset_cache = {}

    async def __call__(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets

        assert input.evaluation, "CheckpointEvalRolloutFn only serves eval"
        if self._state is None:
            self._state = make_eval_generate_state(self.args)
        results = await run_eval_datasets(self._state, self._prompt_dataset_cache)
        return RolloutFnEvalOutput(data=results)
