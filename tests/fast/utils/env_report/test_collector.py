import json
import os
import subprocess
from unittest.mock import patch

from tests.fast.utils.env_report.conftest import SAMPLE_PIP_INSPECT

from miles.utils.env_report.collector import EditablePackageInfo, _is_editable, _parse_pip_entry, collect_pip_info


class TestParsePipEntry:
    def test_normal_package(self) -> None:
        entry = _parse_pip_entry({"metadata": {"name": "torch", "version": "2.5.0"}})
        assert entry == {"name": "torch", "version": "2.5.0"}

    def test_missing_metadata(self) -> None:
        entry = _parse_pip_entry({})
        assert entry == {"name": "", "version": ""}


class TestIsEditable:
    def test_editable_package(self) -> None:
        pkg = {"direct_url": {"url": "file:///workspace/miles", "dir_info": {"editable": True}}}
        assert _is_editable(pkg) is True

    def test_non_editable_package(self) -> None:
        assert _is_editable({"metadata": {"name": "torch"}}) is False

    def test_archive_url_not_editable(self) -> None:
        pkg = {"direct_url": {"url": "https://example.com/foo.tar.gz", "archive_info": {}}}
        assert _is_editable(pkg) is False


class TestCollectPipInfo:
    def test_parses_pip_inspect_output(self) -> None:
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(SAMPLE_PIP_INSPECT),
            stderr="",
        )
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=mock_result):
            editable, full_list = collect_pip_info()

        assert len(full_list) == 4
        assert full_list[0] == {"name": "miles", "version": "0.2.1"}
        assert full_list[2] == {"name": "torch", "version": "2.5.0"}

        assert len(editable) == 2
        assert editable[0] == EditablePackageInfo(
            name="miles",
            version="0.2.1",
            location="/workspace/miles",
        )
        assert editable[1] == EditablePackageInfo(
            name="sglang",
            version="0.4.0",
            location="/workspace/sglang",
        )

    def test_pip_inspect_failure_returns_empty(self) -> None:
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=1,
            stdout="",
            stderr="error",
        )
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=mock_result):
            editable, full_list = collect_pip_info()
        assert editable == []
        assert full_list == []

    def test_pip_inspect_exception_returns_empty(self) -> None:
        with patch("miles.utils.env_report.collector.subprocess.run", side_effect=OSError("no pip")):
            editable, full_list = collect_pip_info()
        assert editable == []
        assert full_list == []

    def test_pip_inspect_excludes_pythonpath_from_env(self) -> None:
        """PYTHONPATH must be excluded when running pip inspect, otherwise pip
        misses editable packages whose source is on the PYTHONPATH."""
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(SAMPLE_PIP_INSPECT),
            stderr="",
        )
        with patch.dict(os.environ, {"PYTHONPATH": "/workspace/Megatron-LM"}):
            with patch("miles.utils.env_report.collector.subprocess.run", return_value=mock_result) as mock_run:
                collect_pip_info()

        passed_env = mock_run.call_args.kwargs.get("env")
        assert passed_env is not None, "subprocess.run must be called with explicit env"
        assert "PYTHONPATH" not in passed_env
