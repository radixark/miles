import sys

import pytest

from miles.utils.function_registry import function_registry
from miles.utils.workers.serving.utils import compute_serve_worker_spec, override_argv, split_worker_argv
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

SPECS_FN = "test:serving-utils-specs"
POOL_ID = "trainer"


def _base_spec_fields() -> dict[str, object]:
    return {
        "name": POOL_ID,
        "port_infos": [PortInfo(name="rpc", static_port=8000)],
        "env_var": lambda context: {},
        "scheduling": SchedulingSpec.single(num_gpus_per_worker=0),
    }


def _serve_spec() -> ServeWorkerSpec:
    return ServeWorkerSpec(
        **_base_spec_fields(),
        worker_class="test.worker",
        ctor_kwargs=lambda context: {},
    )


class TestSplitWorkerArgv:
    @pytest.mark.parametrize("argv", [["--"], ["--host", "127.0.0.1", "--"]])
    def test_trailing_separator_yields_empty_worker_argv(self, argv: list[str]) -> None:
        """A separator with nothing after it is accepted and leaves the worker argv empty."""
        own_argv, worker_argv = split_worker_argv(argv)

        assert own_argv == argv[:-1]
        assert worker_argv == []


class TestComputeServeWorkerSpec:
    @pytest.mark.parametrize(
        "specs, error_match",
        [
            ([_serve_spec(), _serve_spec()], "not one spec named"),
            (
                [CommandWorkerSpec(**_base_spec_fields(), launch_command=lambda context: "true")],
                "CommandWorkerSpec, which is not served",
            ),
        ],
    )
    def test_a_pool_must_match_exactly_one_serve_worker_spec(
        self, specs: list[ServeWorkerSpec | CommandWorkerSpec], error_match: str
    ) -> None:
        """A pool is rejected when its name is ambiguous or belongs to a non-served spec."""

        def compute_specs(worker_argv: list[str]) -> list[ServeWorkerSpec | CommandWorkerSpec]:
            return specs

        with function_registry.temporary(SPECS_FN, compute_specs):
            with pytest.raises(AssertionError, match=error_match):
                compute_serve_worker_spec(specs_fn=SPECS_FN, pool_id=POOL_ID, worker_argv=[])


class TestOverrideArgv:
    def test_override_argv_restores_the_original_after_an_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exceptional context exit restores the exact argv object that preceded the override."""
        original_argv = ["runner.py", "--original"]
        monkeypatch.setattr(sys, "argv", original_argv)

        with pytest.raises(RuntimeError, match="boom"):
            with override_argv(["--replacement"]):
                assert sys.argv == ["runner.py", "--replacement"]
                raise RuntimeError("boom")

        assert sys.argv is original_argv
