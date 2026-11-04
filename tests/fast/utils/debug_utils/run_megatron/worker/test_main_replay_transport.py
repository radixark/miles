import pytest
from tests.fast.utils.debug_utils.run_megatron.worker.main_replay_fixtures import (
    MainReplayBoundary,
    main_replay_boundary,
)

from miles.utils.debug_utils.run_megatron.worker import main as worker
from miles.utils.replay_base import routing_replay_manager
from miles.utils.workers.serving.utils import override_argv

__all__ = ["main_replay_boundary"]


class TestMainReplayTransport:
    @pytest.mark.parametrize("sequence_parallel", [False, True])
    def test_main_applies_the_parsed_sequence_parallel_setting_to_actual_replay_tensor_slicing(
        self, main_replay_boundary: MainReplayBoundary, sequence_parallel: bool
    ) -> None:
        """Main forwards parsed sequence parallelism so rank one receives the appropriate saved token indices."""
        boundary = main_replay_boundary

        with override_argv(
            [
                "--script-hf-checkpoint",
                str(boundary.hf),
                "--script-token-ids-file",
                str(boundary.tokens),
                "--script-routing-replay-load-path",
                str(boundary.replay_directory),
            ]
        ):
            worker.main()

        assert boundary.replay.top_indices_list[0].tolist() == ([4, 5, 6, 7] if sequence_parallel else list(range(8)))
        assert boundary.replay.forward_index == 0
        assert boundary.replay.backward_index == 0
        assert routing_replay_manager.stage == "replay_forward"
        assert boundary.model.training
