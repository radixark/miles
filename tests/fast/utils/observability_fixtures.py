import logging
import sys
import warnings
from collections.abc import Iterator

import pytest

from miles.utils import logging_utils
from miles.utils.args.configs.ci import CiConfig
from miles.utils.args.configs.debug import DebugConfig
from miles.utils.args.configs.prometheus import PrometheusConfig
from miles.utils.args.configs.wandb import WandbConfig
from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.event_logger import logger as event_logger_module


class _LoggingConfig(DebugConfig, CiConfig):
    pass


class _TrackingConfig(PrometheusConfig, WandbConfig):
    pass


@pytest.fixture
def isolated_observability_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(logging.root, "handlers", [])
    monkeypatch.setattr(logging.root, "level", logging.root.level)
    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)
    monkeypatch.setattr(logging_utils, "_STRICT_ASYNC_WARNINGS_INSTALLED", False)
    monkeypatch.setattr(logging_utils, "_ENV_REPORTER", None)
    monkeypatch.setattr(ConfigSnapshotDumper, "_state", None)
    monkeypatch.setattr(event_logger_module, "_event_logger", None)
    with warnings.catch_warnings():
        yield
