"""Tests for AdapterConfig parsing, validation, and lifecycle state."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")


from pathlib import Path

import pytest
import yaml

from miles.utils.adapter_config import (
    ADAPTER_INACTIVE_STATES,
    ADAPTER_ROLLOUT_STATES,
    AdapterConfig,
    AdapterState,
    parse_adapter_yaml,
)


MINIMAL_YAML = {
    "rank": 16,
    "alpha": 32,
    "data": "/tmp/data.parquet",
    "label_key": "label",
    "rm_type": "math",
}


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


# ---------------------------------------------------------------------------
# parse_adapter_yaml
# ---------------------------------------------------------------------------


class TestParseAdapterYaml:
    def test_minimal_happy_path(self, tmp_path):
        path = write_yaml(tmp_path / "adapter.yaml", MINIMAL_YAML)
        cfg = parse_adapter_yaml(path)
        assert cfg.rank == 16
        assert cfg.alpha == 32
        assert cfg.data == "/tmp/data.parquet"
        assert cfg.label_key == "label"
        assert cfg.rm_type == "math"
        assert cfg.input_key == "text"
        assert cfg.metadata_key is None
        assert cfg.custom_rm_path is None
        assert cfg.num_epoch is None
        assert cfg.num_row is None
        assert cfg.slot == -1
        assert cfg.state == AdapterState.PENDING

    def test_optional_fields_passthrough(self, tmp_path):
        path = write_yaml(tmp_path / "adapter.yaml", {
            **MINIMAL_YAML,
            "input_key": "messages",
            "metadata_key": "meta",
            "num_epoch": 3,
            "num_row": 100,
        })
        cfg = parse_adapter_yaml(path)
        assert cfg.input_key == "messages"
        assert cfg.metadata_key == "meta"
        assert cfg.num_epoch == 3
        assert cfg.num_row == 100

    def test_dir_explicit(self, tmp_path):
        path = write_yaml(tmp_path / "adapter.yaml", {**MINIMAL_YAML, "dir": "/some/place"})
        cfg = parse_adapter_yaml(path)
        assert Path(cfg.dir) == Path("/some/place")

    @pytest.mark.parametrize("dir_value", [None, ""])
    def test_dir_falls_back_to_parent(self, tmp_path, dir_value):
        raw = MINIMAL_YAML if dir_value is None else {**MINIMAL_YAML, "dir": dir_value}
        path = write_yaml(tmp_path / "adapter.yaml", raw)
        cfg = parse_adapter_yaml(path)
        assert Path(cfg.dir) == tmp_path

    @pytest.mark.parametrize("missing", ["rank", "alpha", "data"])
    def test_missing_required_key_raises(self, tmp_path, missing):
        bad = {k: v for k, v in MINIMAL_YAML.items() if k != missing}
        path = write_yaml(tmp_path / "adapter.yaml", bad)
        with pytest.raises(KeyError):
            parse_adapter_yaml(path)


# ---------------------------------------------------------------------------
# AdapterConfig.__post_init__
# ---------------------------------------------------------------------------


class TestAdapterConfigValidation:
    def make(self, **overrides):
        defaults = dict(
            rank=8, alpha=16, data="/d", dir="/x",
            input_key="text", label_key="label",
            rm_type="math",
        )
        return AdapterConfig(**(defaults | overrides))

    def test_rm_type_only_ok(self):
        self.make(rm_type="math", custom_rm_path=None)

    def test_custom_rm_path_only_ok(self):
        self.make(rm_type=None, custom_rm_path="rm.py")

    def test_both_set_raises(self):
        with pytest.raises(ValueError, match="Only one of"):
            self.make(rm_type="math", custom_rm_path="rm.py")

    def test_neither_set_raises(self):
        with pytest.raises(ValueError, match="Only one of"):
            self.make(rm_type=None, custom_rm_path=None)


# ---------------------------------------------------------------------------
# AdapterState
# ---------------------------------------------------------------------------


class TestAdapterState:
    def test_lifecycle_strictly_increasing(self):
        """Controller relies on `state < new_state` for forward-only transitions."""
        order = [
            AdapterState.PENDING,
            AdapterState.ACTIVE,
            AdapterState.DRAINING_DATASOURCE,
            AdapterState.DRAINING_INFLIGHT,
            AdapterState.DRAINING_TRAINABLE,
            AdapterState.DRAINED,
        ]
        for prev, nxt in zip(order, order[1:]):
            assert prev < nxt

    def test_rollout_states(self):
        assert ADAPTER_ROLLOUT_STATES == {
            AdapterState.ACTIVE,
            AdapterState.DRAINING_DATASOURCE,
        }

    def test_inactive_states(self):
        assert ADAPTER_INACTIVE_STATES == {
            AdapterState.PENDING,
            AdapterState.DRAINED,
        }
