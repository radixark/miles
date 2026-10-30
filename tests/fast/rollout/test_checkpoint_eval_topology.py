import pytest

from miles.rollout.checkpoint_eval import retarget_args
from miles.utils.args.component_rollout import InferenceRuntimeMutState
from miles.utils.args.custom_view import ImmutableNamespace


@pytest.mark.parametrize("engine_gpu_counts,expected", [([2, 4], (2, 6)), ([], (0, 0))])
def test_retargeting_uses_observed_heterogeneous_or_empty_fleets(
    engine_gpu_counts: list[int], expected: tuple[int, int]
) -> None:
    """Observed topology replaces configured counts in a detached evaluation argument view."""
    runtime = InferenceRuntimeMutState(engine_count=8, gpu_count=32)
    args = ImmutableNamespace.model_validate(
        {"sglang_router_ip": "10.0.0.1", "sglang_router_port": 30000, "inference_runtime_mut_state": runtime}
    )

    result = retarget_args(args=args, router_ip="10.0.0.9", router_port=39000, engine_gpu_counts=engine_gpu_counts)

    assert (result.inference_runtime_mut_state.engine_count, result.inference_runtime_mut_state.gpu_count) == expected
    assert (result.sglang_router_ip, result.sglang_router_port) == ("10.0.0.9", 39000)
    assert (args.inference_runtime_mut_state.engine_count, args.inference_runtime_mut_state.gpu_count) == (8, 32)
    assert (args.sglang_router_ip, args.sglang_router_port) == ("10.0.0.1", 30000)
