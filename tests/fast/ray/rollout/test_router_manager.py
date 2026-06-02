from __future__ import annotations

from unittest.mock import patch

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.router_manager import start_router, start_session_server


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
        """When the configured ``session_server_port`` is already bound, fail
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
