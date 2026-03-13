"""Tests for miles.utils.ft.utils.env."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from miles.utils.ft.utils.env import (
    build_exception_inject_flag_path,
    get_exception_inject_dir,
    get_exception_inject_path,
    get_ft_id,
    get_run_id,
)


class TestGetFtId:
    def test_returns_value_when_set(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_ID": "ft-123"}):
            assert get_ft_id() == "ft-123"

    def test_returns_empty_string_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_ft_id() == ""


class TestGetRunId:
    def test_returns_value_when_set(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_RUN_ID": "run-abc"}):
            assert get_run_id() == "run-abc"

    def test_raises_when_unset(self) -> None:
        import pytest

        with patch.dict("os.environ", {}, clear=True), pytest.raises(RuntimeError, match="MILES_FT_RUN_ID"):
            get_run_id()

    def test_raises_when_empty(self) -> None:
        import pytest

        with patch.dict("os.environ", {"MILES_FT_RUN_ID": ""}), pytest.raises(RuntimeError, match="MILES_FT_RUN_ID"):
            get_run_id()


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


class TestGetExceptionInjectDir:
    def test_returns_path_when_set(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_EXCEPTION_INJECT_DIR": "/tmp/ft_exc"}):
            result = get_exception_inject_dir()
            assert result == Path("/tmp/ft_exc")
            assert isinstance(result, Path)

    def test_returns_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_exception_inject_dir() is None

    def test_returns_none_when_empty_string(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_EXCEPTION_INJECT_DIR": ""}):
            assert get_exception_inject_dir() is None


class TestBuildExceptionInjectFlagPath:
    """Each rank must get a distinct flag path so broadcast injection does
    not degenerate into single-consumer race."""

    def test_different_ranks_produce_different_paths(self) -> None:
        base = Path("/tmp/ft_exc")
        path_0 = build_exception_inject_flag_path(base, rank=0)
        path_1 = build_exception_inject_flag_path(base, rank=1)
        assert path_0 != path_1

    def test_path_contains_rank_identifier(self) -> None:
        path = build_exception_inject_flag_path(Path("/tmp/ft_exc"), rank=5)
        assert "rank-5" in str(path)

    def test_path_is_under_inject_dir(self) -> None:
        base = Path("/tmp/ft_exc")
        path = build_exception_inject_flag_path(base, rank=0)
        assert path.parent == base
