"""Eval fn backed by an external sglang server — an example of the
``eval_needs_snapshot`` contract.

Declaring ``eval_needs_snapshot = True`` makes the trainer export an HF snapshot per
eval point and hand its path to this fn via ``RolloutFnEvalInput.hf_dir``; the fn owns
delivering it to its backend. Here that is ``pin_and_verify`` against any
sglang-compatible HTTP server, then the standard miles eval datasets. Because the fn
runs inside the training job, all eval config (sampling, rm, datasets, lora, ...)
comes from the training args — nothing is hand-copied.

Usage (the server can be launched however you like, on any host)::

    python -m sglang.launch_server --model-path /models/Qwen3.5-4B --port 31000 ...

    MILES_EXTERNAL_EVAL_URL=http://eval-host:31000 <train launcher> \\
        --eval-function-path examples.fully_async.external_eval_fn.ExternalSglangEvalFn \\
        --eval-hf-dir /dev/shm/eval_snapshots --eval-dispatch async ...

``checkpoint_eval_service.py`` drives this same fn from a standalone process instead
(poll a snapshot dir, no training job needed).
"""

import os

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnEvalOutput
from miles.rollout.checkpoint_eval import HttpServerTarget, pin_and_verify, retarget_args
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets


class ExternalSglangEvalFn:
    """Pin an external sglang server to each eval snapshot and eval against it."""

    eval_needs_snapshot = True

    def __init__(self, input: RolloutFnConstructorInput):
        args = input.args
        url = getattr(args, "external_eval_url", None) or os.environ.get("MILES_EXTERNAL_EVAL_URL")
        if not url:
            raise ValueError(
                "ExternalSglangEvalFn needs the external server address: set MILES_EXTERNAL_EVAL_URL "
                "(or args.external_eval_url) to e.g. http://eval-host:31000"
            )
        ip, _, port = url.removeprefix("http://").rpartition(":")
        self._url = f"http://{ip}:{port}"
        # Sizes the client-side concurrency semaphore (#engines behind the URL).
        num_gpus = int(
            getattr(args, "external_eval_num_gpus", 0) or os.environ.get("MILES_EXTERNAL_EVAL_NUM_GPUS", "1")
        )
        per_engine = int(getattr(args, "external_eval_num_gpus_per_engine", 0) or num_gpus)
        self._eval_args = retarget_args(args, ip, int(port), num_gpus, per_engine)
        self._state: GenerateState | None = None
        self._cache: dict = {}

    async def __call__(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        assert input.evaluation, "ExternalSglangEvalFn only serves eval; keep the train fn on --rollout-function-path"
        assert input.hf_dir is not None, "eval_needs_snapshot fns are always dispatched with a snapshot"
        if not await pin_and_verify([HttpServerTarget(self._url)], input.hf_dir, input.weight_version):
            raise RuntimeError(f"weight_version pin failed for {input.hf_dir} (expected {input.weight_version})")
        if self._state is None:
            self._state = GenerateState(self._eval_args)
        results = await run_eval_datasets(self._state, self._cache)
        return RolloutFnEvalOutput(data=results)
