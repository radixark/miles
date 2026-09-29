import os
from types import SimpleNamespace as NS

import pytest

from miles.backends.training_utils.weight_update import nccl


@pytest.fixture(autouse=True)
def clear_channels(monkeypatch):
    for key in nccl.CHANNEL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_options_configure_only_the_new_communicator(monkeypatch):
    options = NS(config=NS(min_ctas=-1, max_ctas=-1), is_high_priority_stream=False)
    monkeypatch.setattr(nccl.dist, "ProcessGroupNCCL", NS(Options=lambda: options), raising=False)
    before = dict(os.environ)
    result = nccl.weight_update_nccl_options(16)
    assert result is options
    assert result.config.min_ctas == result.config.max_ctas == 16
    assert not result.is_high_priority_stream
    assert dict(os.environ) == before


def test_unaffected_launch_does_not_require_nccl(monkeypatch):
    monkeypatch.delattr(nccl.dist, "ProcessGroupNCCL", raising=False)
    assert nccl.weight_update_nccl_options(None) is None


@pytest.mark.parametrize("key", nccl.CHANNEL_ENV_KEYS)
def test_all_channel_override_names_are_checked(monkeypatch, key):
    monkeypatch.setenv(key, "24")
    with pytest.raises(ValueError, match=f"{key}=24"):
        nccl.weight_update_nccl_options(8)
    monkeypatch.setenv(key, "8")
    nccl.validate_channel_env(os.environ, 8, role="trainer")


def test_missing_nccl_has_actionable_error(monkeypatch):
    monkeypatch.delattr(nccl.dist, "ProcessGroupNCCL", raising=False)
    with pytest.raises(RuntimeError, match="build with NCCL support"):
        nccl.weight_update_nccl_options(8)


@pytest.mark.parametrize("options", [NS(), NS(config=NS()), NS(config=NS(min_ctas=0))])
def test_missing_communicator_options_has_actionable_error(monkeypatch, options):
    monkeypatch.setattr(nccl.dist, "ProcessGroupNCCL", NS(Options=lambda: options), raising=False)
    with pytest.raises(RuntimeError, match="supported Miles CUDA image"):
        nccl.weight_update_nccl_options(8)
