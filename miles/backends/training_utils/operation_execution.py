from dataclasses import dataclass
from typing import Protocol

from miles.utils.operation_contract import BatchExecutionLease, BindingT

# Adam defaults currently matching the Tinker protocol adapter's AdamParams.
ADAM_PARAM_DEFAULTS = dict(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0, grad_clip_norm=0.0)


def resolve_adam_params(adam_params: dict | None) -> dict:
    return {**ADAM_PARAM_DEFAULTS, **{k: v for k, v in (adam_params or {}).items() if v is not None}}


@dataclass(frozen=True)
class StepRequest:
    operation_id: str
    adam_params: dict


class ParameterExecutor(Protocol[BindingT]):
    def discard_many(self, lease: BatchExecutionLease[BindingT], operation_ids: list[str]) -> dict[str, dict]: ...

    def step_many(self, lease: BatchExecutionLease[BindingT], requests: list[StepRequest]) -> dict[str, dict]: ...


def run_optim_controls(
    operations: list[dict],
    lease: BatchExecutionLease[BindingT],
    executor: ParameterExecutor[BindingT],
) -> dict[str, dict]:
    all_optim = [op for op in operations if op["kind"] == "optim_step"]
    results: dict[str, dict] = {}

    poisoned = [op for op in all_optim if op.get("poison")]
    if poisoned:
        discard_outcomes = executor.discard_many(lease, [op["operation_id"] for op in poisoned])
        for op in poisoned:
            outcome = discard_outcomes.get(op["operation_id"])
            if outcome is None:
                results[op["operation_id"]] = dict(
                    ok=False,
                    error=f"executor returned no discard outcome for operation '{op['operation_id']}'",
                    category="server",
                )
                continue
            results[op["operation_id"]] = (
                dict(ok=False, error=op["poison"], category="user", gradient_window_consumed=True)
                if outcome.get("ok")
                else outcome
            )

    clean = [op for op in all_optim if not op.get("poison")]
    if clean:
        requests = [
            StepRequest(
                operation_id=op["operation_id"],
                adam_params=resolve_adam_params((op.get("payload") or {}).get("adam_params")),
            )
            for op in clean
        ]
        step_outcomes = executor.step_many(lease, requests)
        for op in clean:
            outcome = step_outcomes.get(op["operation_id"])
            if outcome is None:
                outcome = dict(
                    ok=False,
                    error=f"executor returned no step outcome for operation '{op['operation_id']}'",
                    category="server",
                )
            results[op["operation_id"]] = outcome
    return results


def reset_grad_metadata_keep_grads(model_chunks) -> None:
    for model_chunk in model_chunks:
        if getattr(model_chunk.config, "cuda_graph_impl", "none") != "transformer_engine":
            for param in model_chunk.params_with_grad:
                param.grad_added_to_main_grad = False
        for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
            bucket_group.reset()
