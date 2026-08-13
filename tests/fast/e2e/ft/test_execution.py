import dataclasses
from pathlib import Path

from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_ft_args
from tests.e2e.ft.conftest_ft.modes import MODES


class TestGetCommonTrainArgs:
    def test_a_colocated_real_rollout_mode_emits_the_colocate_flag(self, tmp_path: Path) -> None:
        """A colocated mode must tell the trainer to share its gpus with the rollout engines."""
        args = get_common_train_args(MODES["kill_rollout__dp4__colocate"], dump_dir=str(tmp_path))

        assert "--colocate " in args

    def test_a_disaggregated_real_rollout_mode_omits_the_colocate_flag(self, tmp_path: Path) -> None:
        """Rollout engines on their own gpus must not be colocated with the trainer."""
        args = get_common_train_args(MODES["kill_train__dp2_cp2__moe_5layer"], dump_dir=str(tmp_path))

        assert "--rollout-num-gpus" in args
        assert "--colocate" not in args

    def test_a_debug_rollout_mode_omits_the_colocate_flag_even_when_the_mode_is_colocated(
        self, tmp_path: Path
    ) -> None:
        """Without real rollout engines there is nothing to colocate, whatever the mode declares."""
        mode = dataclasses.replace(
            MODES["kill_rollout__dp4__colocate"], rollout_num_engines=0, ft_components=("train",)
        )

        args = get_common_train_args(mode, dump_dir=str(tmp_path))

        assert mode.colocate is True
        assert "--debug-train-only" in args
        assert "--colocate" not in args


class TestGetFtArgs:
    def test_a_rollout_only_ft_mode_propagates_the_rollout_component_and_api_server_port(self) -> None:
        """Rollout-only fault tolerance must not silently enable trainer fault tolerance."""
        args = get_ft_args(MODES["kill_rollout__dp4__colocate"])

        assert args == "--use-fault-tolerance --ft-components rollout --api-server-port 0 "

    def test_a_trainer_mode_propagates_the_train_component(self) -> None:
        """The trainer-fault-tolerance modes keep sending the train component."""
        args = get_ft_args(MODES["kill_train__dp2_cp2__moe_5layer"])

        assert args == "--use-fault-tolerance --ft-components train --api-server-port 0 "
