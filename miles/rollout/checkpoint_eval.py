"""Run eval on a backend of your own.

Subclass ``CheckpointEvalFn`` and point ``--eval-function-path`` at it. Every eval
point hands you a finished HF checkpoint directory and takes back results; what
happens in between is yours — an sglang server you launched, a remote scoring
service, a black box with an HTTP API.

The trainer exports the snapshot, dispatches it without blocking the training loop,
logs the result at the snapshot's step, and reclaims the directory afterwards. Raise
``EvalSkip(reason)`` to skip a point with attribution instead of counting as a crash.

Requires ``train_async.py`` and a snapshot source (``--eval-hf-dir`` or ``--save-hf``).
``examples/infra_features/fully_async/external_eval_fn.py`` is a working implementation.
"""

import abc
import copy
import inspect
import logging
from argparse import Namespace

from miles.rollout.base_types import BaseRolloutFn, RolloutFnEvalInput, RolloutFnEvalOutput, RolloutFnInput
from miles.utils.misc import load_function

__all__ = [
    "retarget_args",
    "EvalSkip",
    "CheckpointEvalFn",
    "is_checkpoint_eval_fn",
]

logger = logging.getLogger(__name__)


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


class EvalSkip(Exception):
    """Raise from a ``CheckpointEvalFn`` to skip this eval point with an attributable
    reason (logged as ``eval/skipped_{reason}``) instead of counting as a crash."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class CheckpointEvalFn(BaseRolloutFn, abc.ABC):
    """Contract for eval backends that consume HF checkpoint snapshots.

    ``__init__`` takes a ``RolloutFnConstructorInput`` and prepares the backend —
    launch a server, attach to one, open a client. ``evaluate_checkpoint`` then runs
    one eval point against the snapshot at ``checkpoint_dir``.
    """

    @abc.abstractmethod
    async def evaluate_checkpoint(self, checkpoint_dir: str, input: RolloutFnEvalInput) -> RolloutFnEvalOutput: ...

    async def __call__(self, input: RolloutFnInput) -> RolloutFnEvalOutput:
        assert input.evaluation, "CheckpointEvalFn only serves eval; keep the train fn on --rollout-function-path"
        assert input.hf_dir is not None, (
            "no snapshot was dispatched — checkpoint eval fns require train_async.py "
            "and a snapshot source (--eval-hf-dir or --save-hf)"
        )
        return await self.evaluate_checkpoint(input.hf_dir, input)

    def dispose(self) -> None:  # noqa: B027 — optional hook, deliberately a no-op default
        """Tear down anything launched in ``__init__``. Called by RolloutManager.dispose()."""


def is_checkpoint_eval_fn(eval_function_path: str | None) -> bool:
    """Whether ``--eval-function-path`` points at a black-box checkpoint backend."""
    eval_fn = load_function(eval_function_path)
    return inspect.isclass(eval_fn) and issubclass(eval_fn, CheckpointEvalFn)
