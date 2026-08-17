"""Parameterization-neutral tinker execution helpers
(codex-rollout-fullparameter-design-0810 §3.2/§3.5).

Everything here is tinker OPERATION semantics — the client owns the optimizer
boundary — with no Multi-LoRA in it: no AdapterRegistry, no SlotPool, no
AdapterRun, no slot numbers (the dependency rule of §3.7). The OPTIMIZER-
boundary Multi-LoRA pieces live behind the ``ParameterExecutor`` port
(miles/backends/megatron_utils/tinker_backend/executor.py); the trainer-side
DATA-batch path does not have an equivalent port yet — lease validation,
logprob gathering, and batch commit are Multi-LoRA-owned in
``megatron_utils/actor.py`` + ``tinker_backend/trainer.py``, so a future
full-parameter executor reuses the operation/result semantics but still needs
a small trainer-side data-hook extraction (external review 0811: narrow the
claim rather than pre-build the hook).
"""

from dataclasses import dataclass
from typing import Protocol

from miles.utils.tinker_backend import BatchExecutionLease, BindingT

# Tinker AdamParams defaults, per the SDK's AdamParams model.
ADAM_PARAM_DEFAULTS = dict(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0, grad_clip_norm=0.0)


def resolve_adam_params(adam_params: dict | None) -> dict:
    """One optim_step's effective AdamParams: the operation's own values over
    the SDK defaults (each optim_step carries its own AdamParams; no scheduler
    ever writes between operations). None means absent."""
    return {**ADAM_PARAM_DEFAULTS, **{k: v for k, v in (adam_params or {}).items() if v is not None}}


@dataclass(frozen=True)
class StepRequest:
    """One optim_step for the executor: operation_id + resolved AdamParams and
    NOTHING else — a request can never smuggle a second binding; the executor
    resolves bindings exclusively from the batch lease."""

    operation_id: str
    adam_params: dict


class ParameterExecutor(Protocol[BindingT]):
    """Batch-shaped physical execution port: distributed ranks must run
    controls in one deterministic order, so the executor receives whole
    batches, resolves each operation's binding from the validated opaque
    lease, and keys every outcome by operation ID (two operations on one
    physical target can never collide). Storage/publish verbs (save_state,
    load_state, save_weights_for_sampler) stay target-specific — they are
    deliberately NOT forced into this interface."""

    def discard_many(self, lease: BatchExecutionLease[BindingT], operation_ids: list[str]) -> dict[str, dict]: ...

    def step_many(self, lease: BatchExecutionLease[BindingT], requests: list[StepRequest]) -> dict[str, dict]: ...


def run_optim_controls(
    operations: list[dict],
    lease: BatchExecutionLease[BindingT],
    executor: ParameterExecutor[BindingT],
) -> dict[str, dict]:
    """Generic coordinator for the tinker optimizer boundary (§3.5):

    - reads the poison the ledger already derived onto each claim (the ledger
      stays the only poison authority);
    - routes poisoned steps to the executor's discard — they still EXECUTE
      (every rank must clear the window) but terminal-fail as user errors
      carrying the poison evidence;
    - resolves per-call AdamParams defaults into StepRequests;
    - hands the validated opaque lease to the executor and normalizes its
      results into operation-ID-keyed outcomes.

    Clean optim_steps (no prior F/B in the window) execute exactly like any
    other — no dirty prerequisite exists or may be added. Claim order and
    compatibility policy are untouched: this only partitions and formats.

    Every outcome answers two independent questions: did the OPERATION succeed
    (``ok``), and were the window's physical gradients consumed
    (``gradient_window_consumed`` — a step, a discard, or a veto that zeroed
    them). A missing executor outcome fails CLOSED as a server error with the
    consumed bit unset: claiming a phantom discard/step here is exactly the
    partial-gradient leak the window invariant forbids."""
    all_optim = [op for op in operations if op["kind"] == "optim_step"]
    results: dict[str, dict] = {}

    poisoned = [op for op in all_optim if op.get("poison")]
    if poisoned:
        discard_outcomes = executor.discard_many(lease, [op["operation_id"] for op in poisoned])
        for op in poisoned:
            outcome = discard_outcomes.get(op["operation_id"])
            if outcome is None:
                # Fail closed: without an explicit discard outcome nothing
                # says the gradients were cleared, so this must not read as
                # the user-poison terminal (which delimits the window).
                results[op["operation_id"]] = dict(
                    ok=False,
                    error=f"executor returned no discard outcome for operation '{op['operation_id']}'",
                    category="server",
                )
                continue
            # A successful discard is the POLICY failure (user, poison
            # evidence attached, window consumed); an executor-side refusal
            # wins as-is (and carries no consumed bit).
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
    """Reset DDP grad bookkeeping WITHOUT zeroing buffers, so cross-call
    gradient accumulation survives (replaces ``zero_grad_buffer`` under
    explicit-step semantics). Selects no slot — this is how ANY tinker
    parameterization retains its gradient sum between train calls."""
    for model_chunk in model_chunks:
        if getattr(model_chunk.config, "cuda_graph_impl", "none") != "transformer_engine":
            for param in model_chunk.params_with_grad:
                param.grad_added_to_main_grad = False
        for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
            bucket_group.reset()
