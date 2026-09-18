"""Two-rank Gloo regressions for Hub preparation; no CUDA devices or Hub requests."""

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

_SCENARIOS = (
    "success",
    "leader_failure",
    "follower_failure",
    "missing_function",
    "different_revision",
    "different_build",
    "invalid_mapping",
    "different_mapping",
    "empty_mapping",
    "different_strict",
)


def _worker(rank: int, rendezvous: str, strict: bool) -> None:
    from tests.fast.backends.test_fsdp_hub_kernels import _all_hub_modules, _build_gdn_model, _make_args, _stub_kernels

    from miles.backends.fsdp_utils.kernels import hub
    from miles.backends.fsdp_utils.kernels.module_patches import apply_hub_module_kernels
    from miles.backends.fsdp_utils.kernels.presets import SLOT_CAUSAL_CONV1D, SLOT_GATED_DELTA_RULE
    from miles.utils.distributed_utils import init_gloo_group

    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    init_gloo_group()
    try:
        for scenario in _SCENARIOS:
            hub._RESOLVED.clear()
            hub._SLOT_FAILURES.clear()
            args = _make_args(kernel_strict=strict)
            modules = _all_hub_modules()
            with pytest.MonkeyPatch.context() as patch:
                _stub_kernels(patch, modules=modules)
                fake = sys.modules["kernels"]
                original_get = fake.get_kernel
                mapping = hub.load_module_kernels(args)
                error = None

                def get_kernel(repo_id, *, scenario=scenario, original_get=original_get, fake=fake, **kwargs):
                    if repo_id == "kernels-community/fla":
                        if (scenario == "leader_failure" and rank == 0) or (
                            scenario == "follower_failure" and rank == 1
                        ):
                            raise OSError("simulated rank-local import failure")
                    result = original_get(repo_id, **kwargs)
                    if repo_id == "kernels-community/fla" and rank == 1:
                        if scenario == "missing_function":
                            del result.chunk_gated_delta_rule
                        for loaded in fake.get_loaded_kernels():
                            if loaded.module is result:
                                if scenario == "different_revision":
                                    loaded.repo_info.revision = "another-commit"
                                if scenario == "different_build":
                                    loaded.metadata.id = "another-build"
                    return result

                fake.get_kernel = get_kernel
                if rank == 1:
                    if scenario == "invalid_mapping":

                        def invalid_mapping(args):
                            raise ValueError("simulated invalid slot declaration")

                        patch.setattr(hub, "load_module_kernels", invalid_mapping)
                    elif scenario in ("different_mapping", "empty_mapping"):
                        altered = (
                            {} if scenario == "empty_mapping" else {SLOT_CAUSAL_CONV1D: mapping[SLOT_CAUSAL_CONV1D]}
                        )
                        patch.setattr(hub, "load_module_kernels", lambda args, altered=altered: altered)
                    elif scenario == "different_strict":
                        args.kernel_strict = not strict

                try:
                    hub.prefetch_hub_module_kernels(args)
                except (ValueError, RuntimeError) as exc:
                    error = str(exc)

                configuration_failure = scenario in (
                    "invalid_mapping",
                    "different_mapping",
                    "empty_mapping",
                    "different_strict",
                )
                should_fail = configuration_failure or (strict and scenario != "success")
                assert (error is not None) == should_fail, (rank, scenario, strict, error)
                if not should_fail:
                    # Repeated binding exercises the policy and reference model path after consensus.
                    for _ in range(2):
                        model = _build_gdn_model()
                        assert apply_hub_module_kernels(model, args) == {"gated_deltanet": 2}
                        assert hub.resolve_slot(args, SLOT_CAUSAL_CONV1D) is not None
                        chosen = hub.resolve_slot(args, SLOT_GATED_DELTA_RULE)
                        assert (chosen is not None) == (scenario == "success")
                        if scenario == "success":
                            assert (
                                model.layers[0].chunk_gated_delta_rule
                                is modules["kernels-community/fla"].chunk_gated_delta_rule
                            )
                        else:
                            assert model.layers[0].chunk_gated_delta_rule.__name__ == "_torch_chunk_stand_in"
                errors = [None, None]
                dist.all_gather_object(errors, error)
                assert errors[0] == errors[1], (scenario, errors)
                if rank == 0:
                    print(f"PASS strict={strict} {scenario}", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strict", [False, True])
def test_hub_preparation_agrees_across_ranks(tmp_path: Path, strict: bool) -> None:
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, __file__, "--worker", str(tmp_path / "rendezvous"), str(int(strict))],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for scenario in _SCENARIOS:
        assert f"PASS strict={strict} {scenario}" in result.stdout


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        mp.spawn(_worker, args=(sys.argv[2], bool(int(sys.argv[3]))), nprocs=2, join=True)
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))
