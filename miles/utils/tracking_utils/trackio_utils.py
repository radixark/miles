"""
trackio tracking backend for miles.

trackio (https://github.com/gradio-app/trackio) is a lightweight, wandb-API
compatible, local-first experiment tracker from HuggingFace.

Key differences from wandb that shape this backend:
  - A run is identified by ``project + name`` (no opaque run id), so we propagate
    the resolved ``name`` to secondary ranks via ``args.trackio_run_name``.
  - ``trackio.init`` has no ``dir`` parameter -- the local SQLite location is
    controlled by the ``TRACKIO_DIR`` env var instead.
  - There is no ``define_metric``; we log the ``*/step`` counters as plain
    scalars and let the dashboard pick the x-axis (mirrors WandbBackend.log).
  - There is no native distributed "shared" run, and ``resume="allow"`` only
    finds a run by name once that run has logged a metric, so ranks that start
    together would each mint their own run id. The primary rank therefore
    publishes ``run.id`` as ``args.trackio_run_id`` and local-mode secondaries
    open ``trackio.run.Run`` with that id directly (one SQLite db under
    ``TRACKIO_DIR``, WAL mode, so concurrent same-host writers are fine;
    multi-node needs the dir on a shared filesystem). Remote targets
    (``server_url`` / ``space_id``) attach by name through ``trackio.init``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from miles.utils.tracking_utils.wandb_utils import _compute_config_for_logging

logger = logging.getLogger(__name__)


def _resolve_identity(args) -> tuple[str | None, str | None, str | None]:
    # trackio identifies a run by project + name; fall back to the wandb naming
    # so a single --wandb-* config drives both backends.
    project = args.trackio_project or args.wandb_project
    name = args.trackio_run_name or args.wandb_group
    group = args.trackio_group or args.wandb_group
    return project, name, group


def _remote_kwargs(args) -> dict[str, str]:
    # Only a remote backend (self-hosted server or HF Space) can safely coalesce
    # multi-process / multi-node logging into one run. Env vars act as fallback.
    kwargs: dict[str, str] = {}
    server_url = args.trackio_server_url or os.environ.get("TRACKIO_SERVER_URL")
    space_id = args.trackio_space_id or os.environ.get("TRACKIO_SPACE_ID")
    if server_url:
        kwargs["server_url"] = server_url
    if space_id:
        kwargs["space_id"] = space_id
    return kwargs


def _export_local_dir(args, remote: dict[str, str]) -> None:
    # trackio.init has no dir param; the local SQLite path is TRACKIO_DIR, and
    # trackio resolves it once at import time, so this must run before the import.
    # Skip it for a remote server/space (TRACKIO_DIR is ignored there, and creating
    # the dir on every node could raise a PermissionError/FileNotFoundError).
    if remote or not args.trackio_dir or os.environ.get("TRACKIO_DIR"):
        return
    os.makedirs(args.trackio_dir, exist_ok=True)
    os.environ["TRACKIO_DIR"] = args.trackio_dir
    logger.info("trackio local logs will be stored in: %s", args.trackio_dir)


# The Run this process logs to. Held here so log/finish do not depend on
# trackio's context variable, which only trackio.init sets.
_run = None
# Whether trackio.init created _run (then trackio.finish must close it, so the
# atexit hook trackio registers does not finish it a second time).
_run_from_init = False


def init_trackio(args, *, primary: bool = True, **kwargs) -> bool:
    """Initialise trackio for this process.

    Returns whether this process logs (False only when trackio is disabled; the
    backend stores this and no-ops log/finish when False).
    """
    global _run, _run_from_init
    if not args.use_trackio:
        args.trackio_run_name = None
        args.trackio_run_id = None
        return False

    project, name, group = _resolve_identity(args)
    remote = _remote_kwargs(args)
    _export_local_dir(args, remote)

    import trackio

    if primary:
        run = trackio.init(
            project=project,
            name=name,
            group=group,
            config=_compute_config_for_logging(args),
            resume="allow",
            **remote,
        )
        # Propagate the resolved run so secondary ranks attach to it (rides along
        # with args, like wandb_run_id). Capture the name trackio actually used,
        # in case it auto-generated one (name was None).
        args.trackio_run_name = getattr(run, "name", None) or name
        args.trackio_run_id = getattr(run, "id", None)
        _run, _run_from_init = run, True
        logger.info(
            "trackio run started: project=%s name=%s id=%s (%s)",
            project,
            args.trackio_run_name,
            args.trackio_run_id,
            "remote" if remote else "local",
        )
        return True

    # Secondary rank: attach to the primary's run so train/rollout/eval metrics
    # from the actor and rollout_manager processes land in one run.
    if remote:
        _run = trackio.init(
            project=project,
            name=args.trackio_run_name or name,
            group=group,
            resume="allow",
            **remote,
        )
        _run_from_init = True
        return True

    from trackio.run import Run

    _run = Run(
        url=None,
        project=project,
        client=None,
        name=args.trackio_run_name or name,
        run_id=args.trackio_run_id,
        group=group,
    )
    _run_from_init = False
    return True


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    # Mirror WandbBackend.log: do not pass step. trackio has no define_metric, and
    # the */step counters differ per metric family; they are logged as plain
    # scalars so the dashboard can be set to use any of them as the x-axis.
    _run.log(metrics)


def finish() -> None:
    global _run, _run_from_init
    if _run is None:
        return
    if _run_from_init:
        import trackio

        trackio.finish()
    else:
        _run.finish()
    _run, _run_from_init = None, False
