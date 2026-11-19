"""Launcher entrypoint for vLLM (including vllm-torchtpu) rollout servers.

When ``MILES_TPU_POP_WEBHOOK_ENV=1`` (sub-host TPU sharing on stock ``torch_tpu``), installs a
picklable ``multiprocessing.process.BaseProcess.__init__`` hook so spawned ``EngineCore`` and
``mp`` TPU workers pop pod-wide GKE webhook environment variables and bind per-worker
``TPU_VISIBLE_DEVICES`` / ``TPU_VISIBLE_CHIPS`` / MegaScale ports before ``torch_tpu`` initializes.
"""

from __future__ import annotations

import logging
import multiprocessing.process
import os
import shlex
import sys
from collections.abc import Callable
from typing import Any

from miles.utils.tpu_env import pop_gke_webhook_env_vars

logger = logging.getLogger(__name__)

VLLM_TPU_SYNC_WORKER_EXTENSION_CLS = "miles.backends.vllm_utils.tpu_sync_worker_extension.VllmTpuSyncWorkerExtension"


_orig_base_process_init: Callable[..., Any] | None = None


def _configure_subhost_worker_env(
    orig_configure: Callable[[int, int, int, int], None],
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
) -> None:
    """Wrap ``tpu_worker._configure_tpu_process_env`` with sub-host TPU isolation."""
    pop_gke_webhook_env_vars()
    offset = int(os.environ.get("DEBUG_TPU_LOCAL_RANK_OFFSET") or os.environ.get("TPU_LOCAL_RANK_OFFSET") or "0")
    # TPUWorker adds DEBUG_TPU_LOCAL_RANK_OFFSET before calling _configure_tpu_process_env;
    # handle callers that pass the raw worker-local rank directly as well.
    effective_local_rank = local_rank if local_rank >= offset else offset + local_rank
    effective_local_world = max(local_world_size, offset + world_size)
    orig_configure(rank, effective_local_rank, world_size, effective_local_world)

    slot_rank = effective_local_rank - offset
    # Platform.rollout_worker_env_vars owns the sub-host port layout and always exports the block base
    port_base = int(os.environ["MILES_TPU_SUBHOST_PORT_BASE"])
    tpu_log_dir = f"/tmp/tpu_logs/rank_{effective_local_rank}"
    os.makedirs(tpu_log_dir, exist_ok=True)

    os.environ["TPU_VISIBLE_DEVICES"] = str(effective_local_rank)
    os.environ["TPU_VISIBLE_CHIPS"] = str(effective_local_rank)
    os.environ["MEGASCALE_PORT"] = str(port_base + 2 * slot_rank)
    os.environ["MEGASCALE_DEBUG_PORT"] = str(port_base + 2 * slot_rank + 1)
    os.environ["TPU_LOG_DIR"] = tpu_log_dir


def patch_vllm_tpu_worker_env() -> bool:
    """Patch ``vllm_torchtpu.worker.tpu_worker._configure_tpu_process_env`` idempotently."""
    pop_gke_webhook_env_vars()
    try:
        # Deferred: vllm_torchtpu is an optional TPU-only dependency.
        from vllm_torchtpu.worker import tpu_worker
    except ImportError:
        logger.debug("vllm_torchtpu not importable; skipping _configure_tpu_process_env patch")
        return False

    orig = tpu_worker._configure_tpu_process_env
    if getattr(
        orig, "_miles_subhost_patched", False
    ):  # config-access-exempt: idempotency sentinel attached to patched function
        return True

    def _wrapped(rank: int, local_rank: int, world_size: int, local_world_size: int) -> None:
        _configure_subhost_worker_env(orig, rank, local_rank, world_size, local_world_size)

    _wrapped._miles_subhost_patched = True  # type: ignore[attr-defined]
    tpu_worker._configure_tpu_process_env = _wrapped
    return True


class PickleableProcessWrapper:
    """Picklable target wrapper propagating TPU sub-host env patches to spawned children."""

    def __init__(self, target: Callable[..., Any]) -> None:
        self._target = target

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Re-install hook so EngineCore's own spawned mp TPU workers receive the patch too.
        install_subhost_process_patch()
        patch_vllm_tpu_worker_env()
        return self._target(*args, **kwargs)


def install_subhost_process_patch() -> None:
    """Idempotently wrap ``multiprocessing.process.BaseProcess.__init__`` for sub-host TPU."""
    global _orig_base_process_init
    if _orig_base_process_init is not None:
        return
    orig_init = multiprocessing.process.BaseProcess.__init__
    _orig_base_process_init = orig_init

    def _patched_init(
        self: multiprocessing.process.BaseProcess,
        group: Any = None,
        target: Callable[..., Any] | None = None,
        name: str | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        *,
        daemon: bool | None = None,
    ) -> None:
        if target is not None and not isinstance(target, PickleableProcessWrapper):
            target = PickleableProcessWrapper(target)
        orig_init(
            self,
            group=group,
            target=target,
            name=name,
            args=args,
            kwargs={} if kwargs is None else kwargs,
            daemon=daemon,
        )

    multiprocessing.process.BaseProcess.__init__ = _patched_init  # type: ignore[method-assign]


def uninstall_subhost_process_patch() -> None:
    """Restore the original ``BaseProcess.__init__`` (used by unit tests)."""
    global _orig_base_process_init
    if _orig_base_process_init is None:
        return
    multiprocessing.process.BaseProcess.__init__ = _orig_base_process_init  # type: ignore[method-assign]
    _orig_base_process_init = None


def compute_vllm_launch_cmd(
    args: Any,
    *,
    interpreter_prefix: list[str],
    model_path: str,
    tp_size: int,
    host: str,
    port: int,
    random_seed: int | None = None,
) -> str:
    """Build the shell command that starts ``miles.backends.vllm_utils.launch_server``."""
    argv = [
        *interpreter_prefix,
        "-m",
        "miles.backends.vllm_utils.launch_server",
        model_path,
        "--host",
        host.strip("[]"),
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp_size),
        "--distributed-executor-backend",
        "mp",
        "--worker-extension-cls",
        VLLM_TPU_SYNC_WORKER_EXTENSION_CLS,
        "--trust-remote-code",
        "--dtype",
        str(args.vllm_dtype),
        "--gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        # vLLM defaults to raw_logprobs (before temperature); the trainer's logprobs and SGLang's are
        # temperature-scaled, so the importance ratios are only right with the processed ones
        "--logprobs-mode",
        "processed_logprobs",
        # vLLM 0.32 serves /inference/v1/generate, which vllm_single_turn posts to, only with scale-out on;
        # otherwise the route answers 403
        "--enable-scale-out",
    ]
    if random_seed is not None:
        argv.extend(["--seed", str(random_seed)])
    if args.vllm_max_model_len is not None:
        argv.extend(["--max-model-len", str(args.vllm_max_model_len)])
    if args.vllm_max_num_seqs is not None:
        argv.extend(["--max-num-seqs", str(args.vllm_max_num_seqs)])
    if args.vllm_enforce_eager:
        argv.append("--enforce-eager")
    if args.vllm_extra_args:
        argv.extend(shlex.split(args.vllm_extra_args))
    return shlex.join(argv)


def main(argv: list[str] | None = None) -> None:
    if os.environ.get("MILES_TPU_POP_WEBHOOK_ENV") == "1":
        pop_gke_webhook_env_vars()
        install_subhost_process_patch()
        patch_vllm_tpu_worker_env()

    cli_args = list(sys.argv[1:] if argv is None else argv)
    if cli_args and cli_args[0] == "serve":
        sys.argv = ["vllm", *cli_args]
    else:
        sys.argv = ["vllm", "serve", *cli_args]

    # Deferred: vllm is an optional rollout engine dependency.
    from vllm.entrypoints.cli.main import main as vllm_cli_main

    vllm_cli_main()


if __name__ == "__main__":
    main()
