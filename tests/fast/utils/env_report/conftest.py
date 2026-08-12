import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from miles.utils.audit_utils.event_logger.logger import EventLogger, set_event_logger
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

SAMPLE_PIP_INSPECT = {
    "version": "1",
    "pip_version": "24.0",
    "installed": [
        {
            "metadata": {"name": "miles", "version": "0.2.1"},
            "direct_url": {
                "url": "file:///workspace/miles",
                "dir_info": {"editable": True},
            },
        },
        {
            "metadata": {"name": "sglang", "version": "0.4.0"},
            "direct_url": {
                "url": "file:///workspace/sglang",
                "dir_info": {"editable": True},
            },
        },
        {
            "metadata": {"name": "torch", "version": "2.5.0"},
        },
        {
            "metadata": {"name": "numpy", "version": "1.26.0"},
            "direct_url": {
                "url": "https://files.pythonhosted.org/numpy-1.26.0.tar.gz",
                "archive_info": {},
            },
        },
    ],
}


def make_args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(**{"env_report": "", **overrides})


def mock_pip_inspect() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["pip", "inspect"], returncode=0, stdout=json.dumps(SAMPLE_PIP_INSPECT), stderr=""
    )


@pytest.fixture()
def mocked_pip_inspect():
    with patch("miles.utils.env_report.collector.subprocess.run", return_value=mock_pip_inspect()):
        yield


@pytest.fixture()
def event_log_dir(tmp_path: Path) -> Path:
    set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main")))
    yield tmp_path
    set_event_logger(None)


@pytest.fixture()
def without_event_logger():
    set_event_logger(None)
    yield
    set_event_logger(None)
