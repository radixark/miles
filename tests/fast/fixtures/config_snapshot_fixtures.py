from argparse import Namespace
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


def _capture_configuration(monkeypatch: pytest.MonkeyPatch, directory: Path) -> Namespace:
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(directory))
    monkeypatch.setattr(ConfigSnapshotDumper, "_state", None)
    return Namespace(
        ci_disable_config_snapshot=False,
        ci_test=True,
        config_snapshot_name="first",
        deploy_component="all",
        deploy_instance_id=None,
        run_uuid="snapshot-run",
        save_debug_event_data=None,
    )


def _stop_checkpoint_load(*args: object, **kwargs: object) -> None:
    raise RuntimeError("checkpoint SDK stopped before model access")


def _cpu_worker_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from miles.utils import logging_utils, object_store
    from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR, SUBPROCESS_INDEX_ENV_VAR

    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
    monkeypatch.setenv(SUBPROCESS_INDEX_ENV_VAR, "0")
    monkeypatch.setattr(logging_utils.logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(logging_utils, "_STRICT_ASYNC_WARNINGS_INSTALLED", True)
    monkeypatch.setattr(logging_utils, "_ENV_REPORTER", None)
    monkeypatch.setattr(logging_utils, "start_env_reporting", lambda args: None)
    monkeypatch.setattr(object_store, "_INSTANCE", None)


def _empty_rollout_data(**kwargs: object) -> tuple[dict[str, list[object]], AbstractContextManager[None]]:
    from contextlib import nullcontext

    return {"tokens": []}, nullcontext()


def _stop_training_loop(*args: object, **kwargs: object) -> None:
    raise RuntimeError("training SDK stopped before optimizer access")
