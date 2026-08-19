import pytest
from tests.e2e.deploy.conftest_deploy.common.example_args import (
    assert_example_parallelism_matches,
    build_deterministic_test_args,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode


def _mode(parallel_args: str) -> FTTestMode:
    return FTTestMode(
        model_name="Qwen3-0.6B",
        model_hf_repo="Qwen/Qwen3-0.6B",
        megatron_model_type="qwen3-0.6B",
        num_cells=2,
        train_gpus_per_node=4,
        parallel_args=parallel_args,
    )


class TestAssertTheExampleBuildsTheParallelismOf:
    def test_an_example_building_the_parallelism_the_mode_declares_passes(self):
        """The happy path has to stay reachable, or the refusals below prove nothing."""
        assert_example_parallelism_matches(
            _mode("--context-parallel-size 2"), train_args="--lr 1e-6 --context-parallel-size 2 "
        )

    def test_an_example_building_another_degree_than_the_mode_declares_is_caught(self):
        """A mode describing a topology nobody launched is a comparison against a different run."""
        with pytest.raises(AssertionError, match="builds --context-parallel-size as"):
            assert_example_parallelism_matches(
                _mode("--context-parallel-size 2"), train_args="--lr 1e-6 --context-parallel-size 4 "
            )

    def test_an_example_that_never_builds_the_flag_at_all_is_caught(self):
        """The mode's whole point is the parallelism it names, and an example ignoring it runs another shape."""
        with pytest.raises(AssertionError, match="builds no --context-parallel-size at all"):
            assert_example_parallelism_matches(_mode("--context-parallel-size 2"), train_args="--lr 1e-6 ")

    def test_a_flag_the_mode_declares_without_a_value_is_checked_for_being_built_at_all(self):
        """--sequence-parallel takes no value, so the flag being there is the whole of what it declares."""
        assert_example_parallelism_matches(
            _mode("--sequence-parallel --context-parallel-size 2"),
            train_args="--sequence-parallel --lr 1e-6 --context-parallel-size 2 ",
        )

    def test_a_flag_the_mode_declares_without_a_value_and_the_example_omits_is_caught(self):
        """A run built without it parallelises differently, however little the flag itself says."""
        with pytest.raises(AssertionError, match="builds no --sequence-parallel at all"):
            assert_example_parallelism_matches(
                _mode("--sequence-parallel --context-parallel-size 2"),
                train_args="--lr 1e-6 --context-parallel-size 2 ",
            )


class TestBuildDeterministicTestArgs:
    def test_the_run_is_told_to_record_inference_engine_weight_checksums(self):
        """compare_deterministic_sides asserts on those events, so a run that never records them can only fail."""
        args = build_deterministic_test_args(
            mode=_mode("--context-parallel-size 2"), dump_dir="/dumps/run", enable_dumper=False
        )

        assert "--save-inference-engine-weight-checksum " in args

    def test_the_events_the_checksums_are_written_into_are_also_enabled(self):
        """The checksum emitter is a no-op unless the event logger is initialised from --save-debug-event-data."""
        args = build_deterministic_test_args(
            mode=_mode("--context-parallel-size 2"), dump_dir="/dumps/run", enable_dumper=False
        )

        assert "--save-debug-event-data /dumps/run/" in args
