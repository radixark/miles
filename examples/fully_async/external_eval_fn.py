"""Self-contained external sglang eval backend — a reference ``CheckpointEvalFn``.

``__init__`` prepares the backend: launch our own sglang server on spare GPUs, or
attach to one already running anywhere. Each eval then pins the server to the
snapshot in ``checkpoint_dir`` and runs the standard miles eval datasets. Because
the fn runs inside the training job, all eval config (datasets, sampling, rm,
lora, ...) comes from the real training args — nothing is hand-copied.

Config (env vars, set them in the run launcher):

    MILES_EXTERNAL_EVAL_URL          attach to a running server, e.g. http://host:31000
    MILES_EXTERNAL_EVAL_GPUS         launch our own, e.g. "6,7" (tp = #gpus); the
                                     server runs on this node; ignored when URL is set
    MILES_EXTERNAL_EVAL_PORT         launch-mode port (default 31000)
    MILES_EXTERNAL_EVAL_SERVER_ARGS  extra raw sglang flags for launch mode, e.g.
                                     "--attention-backend fa3 --mem-fraction-static 0.9"

See ``run_qwen3_5_4b_fully_async_eval.py`` for a full launcher. A non-sglang black
box implements the same contract without any of this module: ``evaluate_checkpoint``
submits ``checkpoint_dir`` to the external service and maps its response into
``RolloutFnEvalOutput(data=...)``; raise ``EvalSkip(reason)`` for an attributable skip.
"""

import os
import shlex
import subprocess
import sys

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnEvalOutput
from miles.rollout.checkpoint_eval import CheckpointEvalFn, HttpServerTarget, pin_and_verify, retarget_args
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.utils.http_utils import wait_http_ok


class ExternalSglangEvalFn(CheckpointEvalFn):
    """Launch (or attach to) a standalone sglang server and eval snapshots on it."""

    def __init__(self, input: RolloutFnConstructorInput):
        args = input.args
        url = os.environ.get("MILES_EXTERNAL_EVAL_URL")
        gpus = os.environ.get("MILES_EXTERNAL_EVAL_GPUS")
        self._proc: subprocess.Popen | None = None
        num_gpus = 1
        if url is None:
            if gpus is None:
                raise ValueError(
                    "ExternalSglangEvalFn needs a backend: set MILES_EXTERNAL_EVAL_URL (attach) "
                    "or MILES_EXTERNAL_EVAL_GPUS (launch our own, e.g. '6,7')"
                )
            port = int(os.environ.get("MILES_EXTERNAL_EVAL_PORT", "31000"))
            num_gpus = len(gpus.split(","))
            cmd = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                args.eval_model_path or args.hf_checkpoint,
                "--tp",
                str(num_gpus),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--mem-fraction-static",
                "0.8",
                "--trust-remote-code",
                *shlex.split(os.environ.get("MILES_EXTERNAL_EVAL_SERVER_ARGS", "")),
            ]
            # Explicit GPU pinning: under Ray the manager process has no visible GPUs.
            self._proc = subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus})
            url = f"http://127.0.0.1:{port}"

        ip, _, srv_port = url.removeprefix("http://").rpartition(":")
        self._url = f"http://{ip}:{srv_port}"
        # Sizes the client-side concurrency semaphore (one engine behind the URL).
        self._eval_args = retarget_args(args, ip, int(srv_port), num_gpus, num_gpus)
        self._state: GenerateState | None = None
        self._cache: dict = {}
        self._ready = False  # health-wait deferred: server load must not block training start

    async def evaluate_checkpoint(self, checkpoint_dir: str, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        if not self._ready:
            await wait_http_ok(f"{self._url}/health_generate", timeout=1800.0)
            self._ready = True
        if not await pin_and_verify([HttpServerTarget(self._url)], checkpoint_dir, input.weight_version):
            raise RuntimeError(f"weight_version pin failed for {checkpoint_dir} (expected {input.weight_version})")
        if self._state is None:
            self._state = GenerateState(self._eval_args)
        return RolloutFnEvalOutput(data=await run_eval_datasets(self._state, self._cache))

    def dispose(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
