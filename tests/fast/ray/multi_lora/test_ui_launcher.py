"""Tests for the standalone console launcher (ui/launcher.py): serves the
console, reports server-down status, and proxies /v1 with a clean 502."""

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]


def load_launcher():
    spec = importlib.util.spec_from_file_location("miles_ui_launcher", REPO_ROOT / "ui" / "launcher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client():
    launcher = load_launcher()
    # point the proxy at a port nothing listens on
    cfg = launcher.LauncherConfig(api_url="http://127.0.0.1:1", log_path="/tmp/nonexistent-miles-ui.log")
    with TestClient(launcher.create_app(cfg)) as c:
        yield c


def test_launcher_serves_console(client):
    r = client.get("/ui")
    assert r.status_code == 200
    assert "MILES MULTI-LORA" in r.text
    assert "server-panel" in r.text  # launch button UI present


def test_status_reports_server_down(client):
    body = client.get("/launcher/status").json()
    assert body["serverUp"] is False
    assert body["launching"] is False
    assert body["defaultSlots"] == 4


def test_proxy_returns_clean_502_when_server_down(client):
    r = client.get("/v1/info")
    assert r.status_code == 502
    err = r.json()["error"]
    assert err["status"] == "UNAVAILABLE" and "launch it first" in err["message"]
