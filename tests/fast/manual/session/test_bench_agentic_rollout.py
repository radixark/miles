from tests.fast.import_isolation_utils import run_with_blocked_imports


def test_spawn_entrypoint_does_not_load_training_dependencies() -> None:
    """Spawned mock processes load the benchmark without training dependencies."""
    run_with_blocked_imports(
        script='import runpy; runpy.run_module("tests.manual.session.bench_agentic_rollout", run_name="__mp_main__")',
        blocked_prefixes=("miles", "torch", "transformers"),
    )
