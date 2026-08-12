import pytest

from miles.utils.external_utils.command_utils.common import ArgvManipulator


class TestValuesOf:
    def test_reads_the_space_form(self):
        """This is how a launch script writes a flag, and the value is the next token."""
        assert ArgvManipulator.values_of(["--train-backend", "fsdp"], "--train-backend") == ["fsdp"]

    def test_reads_the_equals_form(self):
        """`--flag=value` is one token, and argparse takes it, so a reader that misses it reads a stale argv."""
        assert ArgvManipulator.values_of(["--train-backend=fsdp"], "--train-backend") == ["fsdp"]

    def test_reads_every_occurrence_in_order(self):
        """argparse takes the last of a repeated flag, so a caller has to be able to see that there were two."""
        argv = ["--train-backend", "megatron", "--train-backend=fsdp"]

        assert ArgvManipulator.values_of(argv, "--train-backend") == ["megatron", "fsdp"]

    def test_does_not_mistake_another_flags_value_for_the_flag(self):
        """A substring search finds the flag inside `--data-path=--train-backend fsdp` and reads a value nobody set."""
        assert ArgvManipulator.values_of(["--data-path=--train-backend fsdp"], "--train-backend") == []

    def test_refuses_a_trailing_flag_that_names_nothing(self):
        """argparse would fail on this argv inside the pod, where the failure is much harder to read."""
        with pytest.raises(AssertionError, match="last argument"):
            ArgvManipulator.values_of(["--rollout-num-gpus", "8", "--train-backend"], "--train-backend")


class TestDeclares:
    @pytest.mark.parametrize("argv", [["--run-uuid", "abc"], ["--run-uuid=abc"], ["--run-uuid"]])
    def test_sees_a_flag_however_it_is_written(self, argv):
        """It answers whether the argv already speaks about the flag, not what it says about it."""
        assert ArgvManipulator.declares(argv, "--run-uuid")

    def test_does_not_see_a_flag_that_is_only_a_prefix_of_another(self):
        """`--run-uuid-file` is a different flag, and treating it as this one would skip setting this one."""
        assert not ArgvManipulator.declares(["--run-uuid-file", "/tmp/x"], "--run-uuid")


class TestWithFlag:
    def test_appends_a_flag_the_argv_does_not_have(self):
        """A pod that is not told a value takes the default, which is the whole reason the launcher sets it."""
        argv = ArgvManipulator.with_flag(["--rollout-num-gpus", "8"], "--cluster-backend", "kubernetes")

        assert argv == ["--rollout-num-gpus", "8", "--cluster-backend", "kubernetes"]

    @pytest.mark.parametrize("argv", [["--cluster-backend", "kubernetes"], ["--cluster-backend=kubernetes"]])
    def test_leaves_an_argv_that_already_says_it_alone(self, argv):
        """argparse would take the last of two, so appending a duplicate is a silent way to change the value."""
        assert ArgvManipulator.with_flag(argv, "--cluster-backend", "kubernetes") == argv

    def test_does_not_mutate_what_it_was_given(self):
        """The caller keeps its own argv, and a launcher builds several from one list."""
        argv = ["--rollout-num-gpus", "8"]

        ArgvManipulator.with_flag(argv, "--run-uuid", "abc")

        assert argv == ["--rollout-num-gpus", "8"]


class TestReplacingValue:
    def test_rewrites_the_value_in_place(self):
        """The rewritten argv is what the pods run, so the flag has to keep its position among the others."""
        argv = ["--mooncake-store-init-kwargs", "{}", "--rollout-num-gpus", "8"]

        rewritten = ArgvManipulator.replacing_value(argv, "--mooncake-store-init-kwargs", '{"a": 1}')

        assert rewritten == ["--mooncake-store-init-kwargs", '{"a": 1}', "--rollout-num-gpus", "8"]

    def test_refuses_an_argv_that_never_named_the_flag(self):
        """Replacing nothing would leave the pods with the launcher's own address and no sign of why."""
        with pytest.raises(AssertionError, match="not among the arguments"):
            ArgvManipulator.replacing_value(["--rollout-num-gpus", "8"], "--mooncake-store-init-kwargs", "{}")
