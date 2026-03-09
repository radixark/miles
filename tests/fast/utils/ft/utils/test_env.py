"""Tests for miles.utils.ft.utils.env."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from miles.utils.ft.utils.env import (
    get_exception_inject_path,
    get_ft_id,
    get_k8s_node_name,
    get_k8s_pod_name,
    get_training_run_id,
)


class TestGetFtId:
    def test_returns_value_when_set(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_ID": "ft-123"}):
            assert get_ft_id() == "ft-123"

    def test_returns_empty_string_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_ft_id() == ""


class TestGetTrainingRunId:
    def test_returns_value_when_set(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_TRAINING_RUN_ID": "run-abc"}):
            assert get_training_run_id() == "run-abc"

    def test_returns_empty_string_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_training_run_id() == ""


class TestGetExceptionInjectPath:
    def test_returns_path_when_set(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_EXCEPTION_INJECT_PATH": "/tmp/inject.py"}):
            result = get_exception_inject_path()
            assert result == Path("/tmp/inject.py")
            assert isinstance(result, Path)

    def test_returns_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_exception_inject_path() is None

    def test_returns_none_when_empty_string(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_EXCEPTION_INJECT_PATH": ""}):
            assert get_exception_inject_path() is None


class TestGetK8sNodeName:
    def test_returns_value_when_set(self) -> None:
        with patch.dict("os.environ", {"K8S_NODE_NAME": "gke-node-01"}):
            assert get_k8s_node_name() == "gke-node-01"

    def test_returns_empty_string_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_k8s_node_name() == ""


class TestGetK8sPodName:
    def test_returns_value_when_set(self) -> None:
        with patch.dict("os.environ", {"K8S_POD_NAME": "trainer-pod-abc"}):
            assert get_k8s_pod_name() == "trainer-pod-abc"

    def test_returns_empty_string_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_k8s_pod_name() == ""
