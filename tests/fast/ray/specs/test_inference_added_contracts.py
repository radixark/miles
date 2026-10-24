import sys
from argparse import Namespace
from pathlib import Path

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.ray.specs.inference import specs_inference_engine as _specs


@pytest.fixture(autouse=True)
def _interpreter_without_runner_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-m", "pytest"])


def _make_args(**overrides: object) -> Namespace:
    args = make_args(**overrides)
    args.sglang = SglangConfig.parse_args(args)
    return args


class TestGroupDpPortReservation:
    def test_group_dp_overrides_reserve_independent_master_port_ranges(self, tmp_path: Path) -> None:
        """Each engine group reserves ports for its resolved DP size rather than the global default."""
        config_path = tmp_path / "groups.yaml"
        config_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 2\n"
            "        overrides:\n"
            "          dp_size: 5\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 2\n"
        )
        args = _make_args(sglang_config=str(config_path), rollout_num_gpus=8, sglang_dp_size=3)

        specs = _specs(args)

        assert [
            next(port.num_consecutive for port in spec.port_infos if port.name == "dist_init") for spec in specs
        ] == [35, 33]
