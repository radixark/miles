"""Execute protocol-neutral operations against one whole-model optimizer.
Each dispatch lease contains exactly one operation for the target."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from miles.backends.training_utils.operation_execution import StepRequest, resolve_adam_params
from miles.utils.operation_contract import BatchExecutionLease


@dataclass(frozen=True)
class FullParameterBinding:
    """Immutable binding for one executor-owned whole-model target.
    ``target_id`` identifies the deployment rather than an adapter slot."""

    target_id: str


def _server_error(message: str, *, consumed: bool = False) -> dict:
    outcome = dict(ok=False, error=message, category="server")
    if consumed:
        outcome["gradient_window_consumed"] = True
    return outcome


@dataclass
class FullParameterExecutor:
    """Execute one operation at a time against a stock Megatron optimizer.
    Validate the request and singleton lease before mutating model state."""

    model_chunks: Sequence[Any]
    optimizer: Any
    binding: FullParameterBinding

    def discard_many(
        self,
        lease: BatchExecutionLease[FullParameterBinding],
        operation_ids: list[str],
    ) -> dict[str, dict]:
        if not operation_ids:
            return {}
        refusal = self._validate_singleton_lease(lease, operation_ids)
        if refusal is not None:
            return {operation_id: _server_error(refusal) for operation_id in operation_ids}

        runtime_error = self._validate_clear_runtime()
        if runtime_error is not None:
            return {operation_ids[0]: _server_error(runtime_error)}

        self._clear_gradient_window()
        return {operation_ids[0]: dict(ok=True, gradient_window_consumed=True)}

    def step_many(
        self,
        lease: BatchExecutionLease[FullParameterBinding],
        requests: list[StepRequest],
    ) -> dict[str, dict]:
        if not requests:
            return {}
        operation_ids = [request.operation_id for request in requests]
        refusal = self._validate_singleton_lease(lease, operation_ids)
        if refusal is not None:
            return {operation_id: _server_error(refusal) for operation_id in operation_ids}

        request = requests[0]
        runtime_error = self._validate_step_runtime()
        if runtime_error is not None:
            return {request.operation_id: _server_error(runtime_error)}

        try:
            adam = resolve_adam_params(request.adam_params)
        except Exception as exc:
            return {request.operation_id: _server_error(f"invalid Adam parameters: {exc}")}

        update_successful = False
        grad_norm: float | None = None
        nonfinite_veto = False
        primary_error: BaseException | None = None
        primary_traceback = None
        finalization_errors: list[str] = []
        config = self.optimizer.config
        previous_clip = config.clip_grad
        try:
            self._apply_adam_to_param_groups(adam)
            # MCore measures grad_norm only in its clip branch; infinity keeps
            # ``0 = no clipping`` while still requesting the measurement.
            config.clip_grad = adam["grad_clip_norm"] if adam["grad_clip_norm"] > 0.0 else float("inf")
            nonfinite_veto = self._has_nonfinite_gradient_norm()
            if not nonfinite_veto:
                raw_outcome = self.optimizer.step()
                if not isinstance(raw_outcome, tuple) or len(raw_outcome) != 3:
                    raise RuntimeError(
                        "stock optimizer.step() did not return (update_successful, grad_norm, num_zeros)"
                    )
                update_successful, raw_grad_norm, _ = raw_outcome
                update_successful = bool(update_successful)
                if update_successful and raw_grad_norm is None:
                    raise RuntimeError("stock optimizer.step() did not report a gradient norm")
                if raw_grad_norm is not None:
                    grad_norm = float(raw_grad_norm)
        except BaseException as exc:
            # Execution failures may follow a partial physical update, so keep
            # them fatal rather than returning a recoverable operation result.
            primary_error = exc
            primary_traceback = exc.__traceback__
        finally:
            try:
                config.clip_grad = previous_clip
            except Exception as exc:
                finalization_errors.append(f"failed to restore optimizer clip_grad: {exc}")
            try:
                self._clear_gradient_window()
            except Exception as exc:
                finalization_errors.append(str(exc))

        if primary_error is not None:
            if finalization_errors:
                raise RuntimeError(
                    f"full-parameter optimizer execution failed ({primary_error}); "
                    f"finalization also failed: {'; '.join(finalization_errors)}"
                ) from primary_error
            raise primary_error.with_traceback(primary_traceback)
        if finalization_errors:
            raise RuntimeError("; ".join(finalization_errors))
        if nonfinite_veto:
            return {
                request.operation_id: _server_error(
                    "non-finite gradient norm; step vetoed and gradients cleared",
                    consumed=True,
                )
            }
        if not update_successful:
            return {
                request.operation_id: _server_error(
                    "stock optimizer vetoed the step; gradients cleared",
                    consumed=True,
                )
            }
        return {
            request.operation_id: dict(
                ok=True,
                gradient_window_consumed=True,
                result=dict(grad_norm=grad_norm, learning_rate=adam["learning_rate"]),
            )
        }

    def _validate_singleton_lease(
        self,
        lease: BatchExecutionLease[FullParameterBinding],
        operation_ids: list[str],
    ) -> str | None:
        if len(operation_ids) != 1:
            return (
                f"full-parameter execution requires exactly one operation per dispatch; received {len(operation_ids)}"
            )
        try:
            bindings = lease.bindings_by_operation
            if len(bindings) != 1:
                return f"full-parameter execution requires a singleton whole-model lease; received {len(bindings)} bindings"
            leased_operation_id, leased_binding = bindings[0]
            operation_id = operation_ids[0]
            if leased_operation_id != operation_id:
                return f"operation '{operation_id}' is not the singleton operation in dispatch '{lease.dispatch_id}'"
            if leased_binding != self.binding:
                return f"operation '{operation_id}' is not bound to this executor's whole-model target"
        except Exception as exc:
            return f"invalid full-parameter batch lease: {exc}"
        return None

    def _validate_clear_runtime(self) -> str | None:
        if not isinstance(self.model_chunks, Sequence):
            return "full-parameter executor model must be a sequence of model chunks"
        if not self.model_chunks:
            return "full-parameter executor requires at least one model chunk"
        for index, model_chunk in enumerate(self.model_chunks):
            if not callable(getattr(model_chunk, "zero_grad_buffer", None)):
                return f"model chunk {index} does not provide zero_grad_buffer()"
            if not callable(getattr(model_chunk, "parameters", None)):
                return f"model chunk {index} does not provide parameters()"
        if not callable(getattr(self.optimizer, "zero_grad", None)):
            return "stock optimizer does not provide zero_grad()"
        return None

    def _validate_step_runtime(self) -> str | None:
        clear_error = self._validate_clear_runtime()
        if clear_error is not None:
            return clear_error
        if not callable(getattr(self.optimizer, "step", None)):
            return "stock optimizer does not provide step()"
        config = getattr(self.optimizer, "config", None)
        if config is None or not hasattr(config, "clip_grad"):
            return "stock optimizer config does not provide clip_grad"
        if str(getattr(config, "optimizer", "")).lower() != "adam":
            return "full-parameter explicit operations require an Adam optimizer"
        try:
            param_groups = self.optimizer.param_groups
        except Exception as exc:
            return f"stock optimizer param_groups are unavailable: {exc}"
        if not isinstance(param_groups, Sequence) or not param_groups:
            return "stock optimizer must expose at least one parameter group"
        if any(not isinstance(group, dict) for group in param_groups):
            return "stock optimizer parameter groups must be dictionaries"
        return None

    def _apply_adam_to_param_groups(self, adam: dict[str, float]) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = adam["learning_rate"]
            group["betas"] = (adam["beta1"], adam["beta2"])
            group["eps"] = adam["eps"]
            group["weight_decay"] = adam["weight_decay"]

    def _has_nonfinite_gradient_norm(self) -> bool:
        """Veto NaN/Inf before the stock BF16 optimizer mutates parameters.
        Scan all visible gradients and reduce their squared norm across ranks."""

        gradients: list[torch.Tensor] = []
        seen: set[int] = set()

        def append_gradient(candidate: Any) -> None:
            if candidate is None:
                return
            candidate = getattr(candidate, "_local_tensor", candidate)
            if not isinstance(candidate, torch.Tensor) or id(candidate) in seen:
                return
            seen.add(id(candidate))
            gradients.append(candidate.coalesce().values() if candidate.is_sparse else candidate)

        for model_chunk in self.model_chunks:
            for parameter in model_chunk.parameters():
                append_gradient(getattr(parameter, "main_grad", None))
                append_gradient(getattr(parameter, "grad", None))
                append_gradient(getattr(parameter, "decoupled_grad", None))
        for group in self.optimizer.param_groups:
            for parameter in group.get("params", ()):
                append_gradient(getattr(parameter, "main_grad", None))
                append_gradient(getattr(parameter, "grad", None))
                append_gradient(getattr(parameter, "decoupled_grad", None))

        if gradients:
            reduction_device = gradients[0].device
        elif dist.is_initialized() and dist.get_backend() == "nccl":
            reduction_device = torch.device("cuda", torch.cuda.current_device())
        else:
            reduction_device = torch.device("cpu")

        squared_norm = torch.zeros(1, dtype=torch.float32, device=reduction_device)
        for gradient in gradients:
            local_norm = torch.linalg.vector_norm(gradient.detach().float())
            squared_norm.add_(local_norm.to(reduction_device).square())
        if dist.is_initialized():
            dist.all_reduce(squared_norm, op=dist.ReduceOp.SUM)
        return not bool(torch.isfinite(squared_norm).item())

    def _clear_gradient_window(self) -> None:
        errors: list[str] = []
        for index, model_chunk in enumerate(self.model_chunks):
            try:
                model_chunk.zero_grad_buffer()
            except Exception as exc:
                errors.append(f"model chunk {index} zero_grad_buffer() failed: {exc}")
        try:
            self.optimizer.zero_grad()
        except Exception as exc:
            errors.append(f"optimizer zero_grad() failed: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))
