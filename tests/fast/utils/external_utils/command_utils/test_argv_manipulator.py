import pytest

from miles.utils.external_utils.command_utils.common import ArgvManipulator


class TestGet:
    def test_reads_the_space_form(self):
        """This is how a launch script writes a flag, and the value is the next token."""
        assert ArgvManipulator.get(["--train-backend", "fsdp"], "--train-backend") == ["fsdp"]

    def test_reads_the_equals_form(self):
        """`--flag=value` is one token, and argparse takes it, so a reader that misses it reads a stale argv."""
        assert ArgvManipulator.get(["--train-backend=fsdp"], "--train-backend") == ["fsdp"]

    def test_reads_every_occurrence_in_order(self):
        """argparse takes the last of a repeated flag, so a caller has to be able to see that there were two."""
        argv = ["--train-backend", "megatron", "--train-backend=fsdp"]

        assert ArgvManipulator.get(argv, "--train-backend") == ["megatron", "fsdp"]

    def test_does_not_mistake_another_flags_value_for_the_flag(self):
        """A substring search finds the flag inside `--data-path=--train-backend fsdp` and reads a value nobody set."""
        assert ArgvManipulator.get(["--data-path=--train-backend fsdp"], "--train-backend") == []

    def test_refuses_a_trailing_flag_that_names_nothing(self):
        """argparse would fail on this argv inside the pod, where the failure is much harder to read."""
        with pytest.raises(AssertionError, match="last argument"):
            ArgvManipulator.get(["--rollout-num-gpus", "8", "--train-backend"], "--train-backend")


class TestIsDefined:
    @pytest.mark.parametrize("argv", [["--run-uuid", "abc"], ["--run-uuid=abc"], ["--run-uuid"]])
    def test_sees_a_flag_however_it_is_written(self, argv):
        """It answers whether the argv already speaks about the flag, not what it says about it."""
        assert ArgvManipulator.is_defined(argv, "--run-uuid")

    def test_does_not_see_a_flag_that_is_only_a_prefix_of_another(self):
        """`--run-uuid-file` is a different flag, and treating it as this one would skip setting this one."""
        assert not ArgvManipulator.is_defined(["--run-uuid-file", "/tmp/x"], "--run-uuid")


class TestSet:
    def test_appends_the_flag_an_argv_never_named(self):
        """A run that never asked for the flag still has to leave with the value the launcher computed."""
        rewritten = ArgvManipulator.set(["--rollout-num-gpus", "8"], "--run-uuid", "abc")

        assert rewritten == ["--rollout-num-gpus", "8", "--run-uuid", "abc"]

    def test_overwrites_a_value_the_argv_already_carried(self):
        """The launcher's value wins, and it has to win in place so the flag keeps its position."""
        argv = ["--run-uuid", "old", "--rollout-num-gpus", "8"]

        rewritten = ArgvManipulator.set(argv, "--run-uuid", "new")

        assert rewritten == ["--run-uuid", "new", "--rollout-num-gpus", "8"]

    def test_refuses_a_flag_the_argv_wrote_as_one_token(self):
        """`--run-uuid=old` carries no value token to overwrite, and a second one would be read instead of it."""
        with pytest.raises(AssertionError, match="no value token"):
            ArgvManipulator.set(["--run-uuid=old"], "--run-uuid", "new")

    def test_does_not_mutate_what_it_was_given(self):
        """The caller keeps its own argv, and a launcher builds several from one list."""
        argv = ["--rollout-num-gpus", "8"]

        ArgvManipulator.set(argv, "--run-uuid", "abc")

        assert argv == ["--rollout-num-gpus", "8"]
