from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from miles.utils.tracking_utils.base import TensorboardBackend, TrackingManager, WandbBackend


class _FakeTable:
    def __init__(self, columns, data):
        self.columns = columns
        self.data = data


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = ModuleType("wandb")
    fake.Table = _FakeTable
    fake.log_calls = []
    fake.log = lambda payload: fake.log_calls.append(payload)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_wandb_backend_logs_a_table(fake_wandb):
    WandbBackend().log_table("rollout/completions", ["prompt", "response"], [["p", "r"]])

    assert len(fake_wandb.log_calls) == 1
    table = fake_wandb.log_calls[0]["rollout/completions"]
    assert isinstance(table, _FakeTable)
    assert table.columns == ["prompt", "response"]
    assert table.data == [["p", "r"]]


def test_backends_without_tables_ignore_log_table():
    # The ABC default is a no-op, so a table never reaches a backend that
    # cannot render one.
    TensorboardBackend().log_table("rollout/completions", ["prompt"], [["p"]])


def test_manager_fans_out_to_enabled_backends(fake_wandb, monkeypatch):
    manager = TrackingManager({"wandb": (WandbBackend, "use_wandb")})
    monkeypatch.setattr(WandbBackend, "init", lambda self, args, primary=True, **kwargs: None)
    manager.init(SimpleNamespace(use_wandb=True), primary=True)

    manager.log_table("rollout/completions", ["prompt"], [["p"]])

    assert len(fake_wandb.log_calls) == 1
