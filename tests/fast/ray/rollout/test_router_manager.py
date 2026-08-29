from __future__ import annotations

from unittest.mock import patch

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.router_manager import _resolve_session_server_ports, start_router, start_session_server


class TestStartRouter:
    def test_returns_existing_when_already_configured(self):
        """Happy path: ``sglang_router_ip`` and ``sglang_router_port`` are
        already set and ``force_new=False`` → skip subprocess launch entirely
        and return the existing tuple."""
        args = make_args(
            use_miles_router=False,
            sglang_router_ip="10.1.2.3",
            sglang_router_port=4567,
        )
        # No mocks needed — the function returns before touching anything.
        ip, port = start_router(args, force_new=False)
        assert (ip, port) == ("10.1.2.3", 4567)

    def test_pd_disagg_with_miles_router_asserts(self):
        args = make_args(use_miles_router=True, sglang_router_ip=None, sglang_router_port=None)
        with patch("miles.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), patch(
            "miles.ray.rollout.router_manager.find_available_port", return_value=20000
        ):
            with pytest.raises(AssertionError, match="miles router does not support PD"):
                start_router(args, has_pd_disaggregation=True, force_new=False)

    def test_port_conflict_raises_runtime_error(self):
        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=None)
        with patch("miles.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), patch(
            "miles.ray.rollout.router_manager.find_available_port", return_value=20000
        ), patch("miles.ray.rollout.router_manager.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="already in use"):
                start_router(args)


class TestStartSessionServer:
    def test_disabled_returns_silently(self):
        """Happy no-op: ``use_session_server=False`` → return without raising,
        without touching any other config."""
        args = make_args(use_session_server=False)
        start_session_server(args)

    def test_enabled_without_hf_checkpoint_raises(self):
        args = make_args(use_session_server=True, hf_checkpoint=None)
        with pytest.raises(ValueError, match="hf-checkpoint"):
            start_session_server(args)

    def test_enabled_port_conflict_raises_runtime_error(self):
        """When a configured ``session_server_port`` is already bound, fail
        loud rather than silently re-using the stale process."""
        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            sglang_router_ip="127.0.0.1",
            sglang_router_port=20000,
            session_server_ip="127.0.0.1",
            session_server_port=20001,
        )
        with patch("miles.ray.rollout.router_manager.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="already in use"):
                start_session_server(args)


class TestResolveSessionServerPorts:
    def test_none_auto_allocates_one_port(self):
        with patch("miles.ray.rollout.router_manager.find_available_port", return_value=20002), patch(
            "miles.ray.rollout.router_manager.is_port_available", return_value=True
        ):
            assert _resolve_session_server_ports(None, 1) == [20002]

    def test_none_auto_allocates_an_entire_available_range(self):
        with patch(
            "miles.ray.rollout.router_manager.find_available_port", side_effect=[6360, 6500]
        ) as find_port, patch(
            "miles.ray.rollout.router_manager.is_port_available", side_effect=lambda port: port != 6379
        ):
            ports = _resolve_session_server_ports(None, 32)

        assert ports == list(range(6500, 6532))
        assert find_port.call_count == 2

    def test_one_worker_uses_the_starting_port(self):
        assert _resolve_session_server_ports(30000, 1) == [30000]

    def test_workers_expand_from_the_starting_port(self):
        assert _resolve_session_server_ports(30000, 4) == [30000, 30001, 30002, 30003]

    @pytest.mark.parametrize("workers", [0, -1])
    def test_non_positive_workers_raise(self, workers):
        with pytest.raises(ValueError, match="at least 1"):
            _resolve_session_server_ports(30000, workers)
