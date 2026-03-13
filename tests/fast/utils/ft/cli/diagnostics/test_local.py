"""Unit tests for cli/diagnostics/local.py (P0 item 6)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from miles.utils.ft.cli.diagnostics.local import _LOCAL_EXCLUDED, local


class TestLocalExcludedSet:
    def test_nccl_pairwise_excluded(self) -> None:
        assert "nccl_pairwise" in _LOCAL_EXCLUDED


class TestDiskMountsParsing:
    def test_single_mount(self) -> None:
        """disk_mounts="/" → [Path("/")]"""
        with patch("miles.utils.ft.cli.diagnostics.local.build_all_diagnostics") as mock_build, patch(
            "miles.utils.ft.cli.diagnostics.local.NodeDiagnosticDispatcher"
        ) as mock_dispatcher, patch("miles.utils.ft.cli.diagnostics.local.socket") as mock_socket, patch(
            "miles.utils.ft.cli.diagnostics.local.validate_check_names"
        ), patch(
            "miles.utils.ft.cli.diagnostics.local.print_results"
        ), patch(
            "miles.utils.ft.cli.diagnostics.local.exit_with_results"
        ):

            mock_socket.gethostname.return_value = "node-0"
            mock_build.return_value = {}
            dispatcher_instance = MagicMock()
            dispatcher_instance.available_types = []
            dispatcher_instance.run_selected = MagicMock()
            mock_dispatcher.return_value = dispatcher_instance

            with patch("miles.utils.ft.cli.diagnostics.local.asyncio") as mock_asyncio:
                mock_asyncio.run.return_value = []
                local(disk_mounts="/")

            call_kwargs = mock_build.call_args.kwargs
            assert call_kwargs["disk_mounts"] == [Path("/")]

    def test_comma_separated_mounts(self) -> None:
        """disk_mounts="/data,/scratch" → [Path("/data"), Path("/scratch")]"""
        with patch("miles.utils.ft.cli.diagnostics.local.build_all_diagnostics") as mock_build, patch(
            "miles.utils.ft.cli.diagnostics.local.NodeDiagnosticDispatcher"
        ) as mock_dispatcher, patch("miles.utils.ft.cli.diagnostics.local.socket") as mock_socket, patch(
            "miles.utils.ft.cli.diagnostics.local.validate_check_names"
        ), patch(
            "miles.utils.ft.cli.diagnostics.local.print_results"
        ), patch(
            "miles.utils.ft.cli.diagnostics.local.exit_with_results"
        ):

            mock_socket.gethostname.return_value = "node-0"
            mock_build.return_value = {}
            dispatcher_instance = MagicMock()
            dispatcher_instance.available_types = []
            dispatcher_instance.run_selected = MagicMock()
            mock_dispatcher.return_value = dispatcher_instance

            with patch("miles.utils.ft.cli.diagnostics.local.asyncio") as mock_asyncio:
                mock_asyncio.run.return_value = []
                local(disk_mounts="/data,/scratch")

            call_kwargs = mock_build.call_args.kwargs
            assert call_kwargs["disk_mounts"] == [Path("/data"), Path("/scratch")]


class TestDiskMountsEmptyStringFiltering:
    def test_trailing_comma_does_not_produce_empty_path(self) -> None:
        """Previously disk_mounts="/data," produced [Path("/data"), Path(".")],
        because "".split(",") yields "" which Path() converts to ".". Now
        empty segments are filtered out."""
        with patch("miles.utils.ft.cli.diagnostics.local.build_all_diagnostics") as mock_build, patch(
            "miles.utils.ft.cli.diagnostics.local.NodeDiagnosticDispatcher"
        ) as mock_dispatcher, patch("miles.utils.ft.cli.diagnostics.local.socket") as mock_socket, patch(
            "miles.utils.ft.cli.diagnostics.local.validate_check_names"
        ), patch(
            "miles.utils.ft.cli.diagnostics.local.print_results"
        ), patch(
            "miles.utils.ft.cli.diagnostics.local.exit_with_results"
        ):

            mock_socket.gethostname.return_value = "node-0"
            mock_build.return_value = {}
            dispatcher_instance = MagicMock()
            dispatcher_instance.available_types = []
            dispatcher_instance.run_selected = MagicMock()
            mock_dispatcher.return_value = dispatcher_instance

            with patch("miles.utils.ft.cli.diagnostics.local.asyncio") as mock_asyncio:
                mock_asyncio.run.return_value = []
                local(disk_mounts="/data,")

            call_kwargs = mock_build.call_args.kwargs
            assert call_kwargs["disk_mounts"] == [Path("/data")]

    def test_empty_string_produces_empty_list(self) -> None:
        with patch("miles.utils.ft.cli.diagnostics.local.build_all_diagnostics") as mock_build, patch(
            "miles.utils.ft.cli.diagnostics.local.NodeDiagnosticDispatcher"
        ) as mock_dispatcher, patch("miles.utils.ft.cli.diagnostics.local.socket") as mock_socket, patch(
            "miles.utils.ft.cli.diagnostics.local.validate_check_names"
        ), patch(
            "miles.utils.ft.cli.diagnostics.local.print_results"
        ), patch(
            "miles.utils.ft.cli.diagnostics.local.exit_with_results"
        ):

            mock_socket.gethostname.return_value = "node-0"
            mock_build.return_value = {}
            dispatcher_instance = MagicMock()
            dispatcher_instance.available_types = []
            dispatcher_instance.run_selected = MagicMock()
            mock_dispatcher.return_value = dispatcher_instance

            with patch("miles.utils.ft.cli.diagnostics.local.asyncio") as mock_asyncio:
                mock_asyncio.run.return_value = []
                local(disk_mounts="")

            call_kwargs = mock_build.call_args.kwargs
            assert call_kwargs["disk_mounts"] == []


class TestLocalDefaultsExcludeNcclPairwise:
    def test_nccl_pairwise_not_in_defaults(self) -> None:
        with patch("miles.utils.ft.cli.diagnostics.local.build_all_diagnostics") as mock_build, patch(
            "miles.utils.ft.cli.diagnostics.local.NodeDiagnosticDispatcher"
        ) as mock_dispatcher, patch("miles.utils.ft.cli.diagnostics.local.socket") as mock_socket, patch(
            "miles.utils.ft.cli.diagnostics.local.validate_check_names"
        ) as mock_validate, patch(
            "miles.utils.ft.cli.diagnostics.local.print_results"
        ), patch(
            "miles.utils.ft.cli.diagnostics.local.exit_with_results"
        ):

            mock_socket.gethostname.return_value = "node-0"
            mock_build.return_value = {}
            dispatcher_instance = MagicMock()
            dispatcher_instance.available_types = ["gpu_check", "nccl_pairwise", "disk_check"]
            mock_dispatcher.return_value = dispatcher_instance

            with patch("miles.utils.ft.cli.diagnostics.local.asyncio") as mock_asyncio:
                mock_asyncio.run.return_value = []
                local(checks=None)

            selected = mock_validate.call_args[0][0]
            assert "nccl_pairwise" not in selected
            assert "gpu_check" in selected
            assert "disk_check" in selected
