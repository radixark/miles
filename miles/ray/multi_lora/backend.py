import logging
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from miles.ray.multi_lora.config import AdapterRunConfig
from miles.ray.multi_lora.gradient_windows import GradientWindowTracker
from miles.ray.multi_lora.identity import rid_prefix, serving_lora_name
from miles.ray.multi_lora.inference_admin import RouterInferenceAdmin
from miles.ray.multi_lora.operations import OperationLedger
from miles.ray.multi_lora.registry import AdapterRegistry, AdapterState
from miles.ray.multi_lora.residency import FixedSlotResidency, ResidentBinding, lease_from_metadata, lease_to_metadata
from miles.utils.operation_contract import BatchExecutionLease

logger = logging.getLogger(__name__)

SUPPORTED_LOSS_FNS = ("cross_entropy", "importance_sampling", "ppo")
_ADAM_FIELDS = ("learning_rate", "beta1", "beta2", "eps", "weight_decay", "grad_clip_norm")
_SAMPLE_TENSOR_FIELDS = ("loss_mask", "loss_weights", "advantages", "rollout_log_probs")
_LOSS_REQUIRED_CHANNELS = {
    "cross_entropy": ("loss_weights",),
    "importance_sampling": ("rollout_log_probs", "advantages"),
    "ppo": ("rollout_log_probs", "advantages"),
}


class MultiLoraOperationBackend:
    """Multi-LoRA implementation selected by ``--multi-lora-backend-path``."""

    def __init__(self, args: Any, router_url: str) -> None:
        self.args = args
        self.registry = AdapterRegistry(args.multi_lora_n_adapters)
        self.operations = OperationLedger(
            gap_timeout=getattr(args, "tinker_operation_gap_timeout", 600.0),
            claimed_ttl=getattr(args, "tinker_operation_claimed_ttl", 1800.0),
        )
        # Ledger lifetime rides the registry's completed ring: ring eviction purges the tenant's ledger state.
        self.registry.on_completed_evicted = self.operations.drop_tenant
        self.gradient_windows = GradientWindowTracker()
        self.residency = FixedSlotResidency(self.registry)
        self.router_url = router_url.rstrip("/")
        self.inference_admin = RouterInferenceAdmin(self.router_url)
        self.trainer_ready = False

    def mark_trainer_ready(self) -> None:
        self.trainer_ready = True

    async def init(self) -> None:
        await self.inference_admin.init()

    async def close(self) -> None:
        await self.inference_admin.close()

    # ---------------- registration ----------------

    async def validate_adapter(self, name: str, config: Any) -> None:
        """Override to reject registrations (raise ValueError)."""

    def resolve_adapter_config(self, name: str, config: Any) -> Any:
        """Resolve client fields against deployment defaults. The public
        surface takes rank/save/num_step/metadata only; alpha is server-set."""
        if config is None or not isinstance(config, AdapterRunConfig):
            return config
        rank = config.rank if config.rank is not None else getattr(self.args, "lora_rank", 1)
        if type(rank) is not int or rank <= 0:
            raise ValueError(f"Adapter '{name}' rank must be a positive integer")
        if rank > getattr(self.args, "lora_rank", rank):
            raise ValueError(f"Adapter '{name}' rank {rank} exceeds the deployment maximum {self.args.lora_rank}")
        if config.alpha is not None:
            raise ValueError(f"Adapter '{name}' must not set alpha; it is deployment-configured (--lora-alpha)")
        alpha = getattr(self.args, "lora_alpha", None) or rank
        if config.num_step is not None and (type(config.num_step) is not int or config.num_step <= 0):
            raise ValueError(f"Adapter '{name}' num_step must be a positive integer")
        save = Path(config.save) if config.save is not None else None
        if save is None:
            if getattr(self.args, "save", None) is None:
                raise ValueError(f"Adapter '{name}' has no save dir: set 'save' in the config or pass --save")
            save = Path(self.args.save) / "adapters" / name
        return replace(config, rank=rank, alpha=alpha, save=save)

    async def register(self, name: str, config: Any) -> dict:
        config = self.resolve_adapter_config(name, config)
        await self.validate_adapter(name, config)
        result = self.registry.register(name, config)
        self.gradient_windows.open(self.registry.records[name].tenant)
        logger.info(f"[tinker] adapter '{name}' registered (slot {result['slot']})")
        return result

    async def deregister(self, name: str, expected_registration_id: str | None = None) -> None:
        if expected_registration_id is not None:
            record = self.registry.find(name)
            if record is None or record.registration_id != expected_registration_id:
                return  # the handle's registration is already gone; never touch a successor
        self.registry.deregister(name)

    async def retire_adapters(self) -> list[str]:
        names = self.registry.retire_adapters()
        for name in names:
            record = self.registry.records.get(name)
            if record is not None:
                # Fence before the engine abort: no operation of the dead
                # registration may be claimed once retirement is underway.
                self.operations.fence(name, record.registration_id)
                await self.abort_adapter_requests(name, record.registration_id)
        return names

    async def free_slot(self, name: str) -> int:
        """Free the adapter's slot after one final abort round: requests can
        survive the retire abort and must not leak to the slot's next tenant."""
        record = self.registry.records.get(name)
        if record is not None and record.state is AdapterState.CLEANUP:
            await self.abort_adapter_requests(name, record.registration_id)
        slot = self.registry.free_slot(name)
        if record is not None and slot != -1:
            self.gradient_windows.close(record.tenant)
        return slot

    # ---------------- training-stream clocks ----------------

    def set_adapter_step(self, name: str, step: int) -> None:
        record = self.registry.find(name)
        if record is None:
            return
        self.gradient_windows.restore_step(record.tenant, step)
        self.registry.set_step(name, step)

    def adapter_step(self, name: str) -> int:
        record = self.registry.find(name)
        return self.gradient_windows.step_of(record.tenant) if record is not None else 0

    # ---------------- operation preflight (compatibility matrix) ----------------

    def enqueue_operation(
        self,
        name: str,
        operation_id: str,
        ordinal: int,
        kind: str,
        payload: dict | None = None,
        expected_registration_id: str | None = None,
    ) -> dict:
        record = self.registry.find(name)
        if record is None or record.state not in (AdapterState.PENDING, AdapterState.READY):
            raise ValueError(f"Adapter '{name}' is not accepting operations (not registered or retiring)")
        self._check_expected_registration(name, record, expected_registration_id)
        payload = payload or {}
        self._preflight(name, kind, payload)
        return self.operations.enqueue(operation_id, name, record.registration_id, ordinal, kind, payload)

    @staticmethod
    def _check_expected_registration(name: str, record: Any, expected_registration_id: str | None) -> None:
        if expected_registration_id is not None and record.registration_id != expected_registration_id:
            raise ValueError(
                f"Adapter '{name}' registration {expected_registration_id[:8]} was retired and the name "
                f"re-registered ({record.registration_id[:8]}); operations from the stale handle are fenced"
            )

    def reject_operation(
        self,
        name: str,
        operation_id: str,
        ordinal: int,
        kind: str,
        payload: dict | None,
        error: str,
        expected_registration_id: str | None = None,
    ) -> dict:
        """Record a boundary-rejected submission as terminal FAILED(user) at
        its ordinal (see ``OperationLedger.record_rejected``): a frontend that
        refuses a request AFTER the client spent the ordinal must still keep
        the registration's arrival sequence gap-free. Like ``enqueue_operation``,
        a pinned ``expected_registration_id`` fences stale handles — a rejection
        must never consume an ordinal slot of a same-name successor."""
        record = self.registry.find(name)
        if record is None or record.state not in (AdapterState.PENDING, AdapterState.READY):
            raise ValueError(f"Adapter '{name}' is not accepting operations (not registered or retiring)")
        self._check_expected_registration(name, record, expected_registration_id)
        return self.operations.record_rejected(
            operation_id, name, record.registration_id, ordinal, kind, payload or {}, error
        )

    def _preflight(self, name: str, kind: str, payload: dict) -> None:
        if kind in ("forward_backward", "forward"):
            samples = payload.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"{kind} payload needs a non-empty 'samples' list")
            required_channels: tuple[str, ...] = ()
            if kind == "forward_backward":
                loss = payload.get("loss") or {}
                loss_fn = loss.get("loss_fn", "cross_entropy")
                if loss_fn not in SUPPORTED_LOSS_FNS:
                    raise ValueError(
                        f"loss_fn '{loss_fn}' is not supported in v1; supported: {', '.join(SUPPORTED_LOSS_FNS)}"
                    )
                required_channels = _LOSS_REQUIRED_CHANNELS[loss_fn]
            for i, sample in enumerate(samples):
                self._preflight_sample(name, kind, i, sample, required_channels)
        elif kind == "optim_step":
            self._preflight_adam_params(payload.get("adam_params") or {})
        elif kind == "save_state":
            tag = payload.get("tag")
            if tag is not None:
                if not isinstance(tag, str):
                    raise ValueError("save_state 'tag' must be a string")
                # Containment: the tag is a single directory name under the
                # adapter's states/ dir — '.'/'..' would escape it.
                if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", tag) or tag in (".", ".."):
                    raise ValueError(
                        f"save_state tag '{tag}' is invalid: 1-128 chars of [A-Za-z0-9._-], not '.' or '..'"
                    )
        elif kind == "load_state":
            if not isinstance(payload.get("path"), str) or not payload["path"]:
                raise ValueError("load_state needs a 'path'")
        elif kind == "save_weights_for_sampler":
            pass
        else:
            raise ValueError(f"unknown operation kind '{kind}'")

    def _preflight_sample(
        self, name: str, kind: str, index: int, sample: Any, required_channels: tuple[str, ...] = ()
    ) -> None:
        where = f"{kind} sample[{index}]"
        if not isinstance(sample, dict):
            raise ValueError(f"{where} must be an object")
        for banned in ("multimodal_inputs", "multimodal_train_inputs"):
            if sample.get(banned):
                raise ValueError(f"{where}: multimodal inputs are not supported in v1 (text-only)")
        tokens = sample.get("tokens")
        response_length = sample.get("response_length")
        if not isinstance(tokens, list) or not tokens or not all(isinstance(t, int) for t in tokens):
            raise ValueError(f"{where}: 'tokens' must be a non-empty list of ints (1-D; no top-K targets in v1)")
        # Strictly below len(tokens): targets are shifted, so the first response
        # token's logprob conditions on at least one preceding token.
        if not isinstance(response_length, int) or not (0 < response_length < len(tokens)):
            raise ValueError(f"{where}: 'response_length' must be an int in (0, len(tokens)) — shifted targets")
        for field_name in required_channels:
            if sample.get(field_name) is None:
                raise ValueError(f"{where}: per-token '{field_name}' is required by this operation's loss_fn")
        for field_name in _SAMPLE_TENSOR_FIELDS:
            value = sample.get(field_name)
            if value is None:
                continue
            if not isinstance(value, list) or len(value) != response_length:
                raise ValueError(f"{where}: '{field_name}' must be a flat list of length response_length (1-D only)")
            if any(isinstance(v, (list, dict)) for v in value):
                raise ValueError(f"{where}: '{field_name}' must be 1-D; nested targets are not supported in v1")

    def _preflight_adam_params(self, adam: dict) -> None:
        for field_name, value in adam.items():
            if field_name not in _ADAM_FIELDS:
                raise ValueError(f"unknown adam_params field '{field_name}'")
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"adam_params.{field_name} must be a finite number")
        for field_name in ("learning_rate", "weight_decay", "grad_clip_norm"):
            if (value := adam.get(field_name)) is not None and value < 0:
                raise ValueError(f"adam_params.{field_name} must be >= 0")
        for field_name in ("beta1", "beta2"):
            if (value := adam.get(field_name)) is not None and not (0 <= value < 1):
                raise ValueError(f"adam_params.{field_name} must be in [0, 1)")
        if (value := adam.get("eps")) is not None and value <= 0:
            raise ValueError("adam_params.eps must be > 0")

    # ---------------- data-operation claims ----------------

    def claim_data_operation(self, name: str, registration_id: str) -> dict | None:
        binding = self.residency.binding_for((name, registration_id))
        if binding is None:
            return None
        operation = self.operations.claim_data_operation(name, registration_id)
        if operation is None:
            return None
        operation["binding"] = binding
        return operation

    def acquire_batch_lease(self, bindings_by_operation: list) -> BatchExecutionLease[ResidentBinding]:
        return self.residency.acquire_batch(
            tuple((operation_id, binding) for operation_id, binding in bindings_by_operation)
        )

    def release_batch_lease(self, lease_metadata: dict) -> None:
        """Completion-boundary lifecycle hook; no-op under fixed residency."""
        self.residency.release_batch(lease_from_metadata(lease_metadata))

    # ---------------- control-operation claims ----------------

    EXECUTABLE_CONTROL_KINDS = ("optim_step", "save_weights_for_sampler", "save_state", "load_state")
    DIRTY_GATED_KINDS = ("save_state", "load_state")

    def sweep_operation_timeouts(self) -> None:
        # Both liveness backstops ride the same heartbeat: QUEUED gap holes and orphaned CLAIMED heads.
        self.operations.sweep_gap_timeouts()
        for view in self.operations.claimed_timeouts():
            error = (
                f"claimed-operation timeout: {view['kind']} '{view['operation_id']}' held CLAIMED for "
                f"{view['claimed_age']:.0f}s (TTL {self.operations.claimed_ttl:.0f}s) without a terminal "
                "outcome; its executor dispatch is presumed lost — resubmit the operation"
            )
            logger.warning(f"[tinker] {error}")
            self.fail_tinker_batch([view["operation_id"]], error)

    def claim_ready_control_operations(self) -> dict:
        self.sweep_operation_timeouts()
        ready: list[dict] = []
        bindings: list[tuple[str, ResidentBinding]] = []
        for name, registration_id in self.operations.claimable_control_tenants():
            record = self.registry.find(name)
            if record is None or record.registration_id != registration_id:
                continue
            binding = self.residency.binding_for((name, registration_id))
            if binding is None:
                continue
            operation = self.operations.claim_control_operation(
                name, registration_id, kinds=self.EXECUTABLE_CONTROL_KINDS
            )
            if operation is None:
                continue
            if operation["kind"] == "optim_step":
                blocker = self.operations.poisoned_window_blocker(name, registration_id, operation["ordinal"])
                if blocker is not None:
                    operation["poison"] = (
                        f"a forward_backward in this gradient window failed ({blocker}); the window's "
                        "accumulated gradients were discarded — resubmit the batch and optim_step again"
                    )
            if operation["kind"] in self.DIRTY_GATED_KINDS and self.gradient_windows.is_dirty(record.tenant):
                self.operations.fail(
                    operation["operation_id"],
                    f"adapter '{name}' holds unstepped gradients; optim_step (or deregister) before "
                    f"{operation['kind']}",
                    "user",
                )
                continue
            operation["step"] = self.gradient_windows.step_of(record.tenant)
            operation["serving_version"] = record.serving_version
            ready.append(operation)
            bindings.append((operation["operation_id"], binding))
        if not ready:
            return {"operations": [], "lease": None}
        lease = self.residency.acquire_batch(tuple(bindings))
        return {"operations": ready, "lease": lease_to_metadata(lease)}

    def complete_control_operations(self, results: dict[str, dict]) -> None:
        for operation_id, outcome in results.items():
            operation = self.operations.get(operation_id)
            # Only still-CLAIMED operations complete: a swept/fenced (already terminal) one keeps its outcome.
            if operation is None or operation["state"] != "CLAIMED":
                continue
            if outcome.get("ok"):
                result = outcome.get("result")
                if operation["kind"] == "save_weights_for_sampler":
                    # Completing after the push landed (the publish barrier):
                    # stamp the authoritative post-push serving identity.
                    record = self.registry.find(operation["name"])
                    result = {
                        **(result or {}),
                        "serving_version": record.serving_version if record else None,
                        "serving_name": serving_lora_name(operation["name"], operation["registration_id"]),
                    }
                self.operations.complete(operation_id, result)
                key = (operation["name"], operation["registration_id"])
                if operation["kind"] == "optim_step":
                    self.operations.mark_window_consumed(operation_id)
                    step = self.gradient_windows.commit_step(key)
                    self.registry.on_step_committed(operation["name"], operation["registration_id"], step)
                elif operation["kind"] == "load_state":
                    step = int((outcome.get("result") or {}).get("step", 0))
                    self.gradient_windows.restore_step(key, step)
                    self.registry.set_step(operation["name"], step)
            else:
                self.operations.fail(
                    operation_id, outcome.get("error", "control operation failed"), outcome.get("category", "server")
                )
                if operation["kind"] == "optim_step" and outcome.get("gradient_window_consumed"):
                    self.operations.mark_window_consumed(operation_id)
                    self.gradient_windows.clear_after_executed_optim((operation["name"], operation["registration_id"]))
                    self.registry.clear_dirty(operation["name"])

    def commit_tinker_batch(
        self,
        accumulated: list[tuple[str, str]],
        operation_ids: list[str],
        logprobs_by_op: dict[str, list] | None = None,
    ) -> None:
        for name, registration_id in accumulated:
            record = self.registry.find(name)
            if record is None or record.registration_id != registration_id:
                continue
            self.gradient_windows.mark_forward_backward_succeeded(record.tenant)
            # Multi-LoRA mirror: pin the accumulating slot's state immovable.
            self.registry.mark_accumulated([name])
        logprobs_by_op = logprobs_by_op or {}
        for operation_id in operation_ids:
            operation = self.operations.get(operation_id)
            if operation is not None and operation["state"] == "CLAIMED":
                logprobs = logprobs_by_op.get(operation_id)
                result = {"logprobs": logprobs}
                if operation["kind"] == "forward_backward" and logprobs is not None:
                    result["metrics"] = operation_result_metrics(self.operations.payload(operation_id), logprobs)
                self.operations.complete(operation_id, result)

    def fail_tinker_batch(self, operation_ids: list[str], error: str, lease_metadata: dict | None = None) -> None:
        try:
            for operation_id in operation_ids:
                operation = self.operations.get(operation_id)
                if operation is not None and operation["state"] == "CLAIMED":
                    self.operations.fail(operation_id, error, "server")
        finally:
            if lease_metadata is not None:
                self.residency.release_batch(lease_from_metadata(lease_metadata))

    # ---------------- engine-facing ----------------

    async def abort_adapter_requests(self, adapter_name: str, registration_id: str) -> None:
        await self.inference_admin.abort_registration(rid_prefix(adapter_name, registration_id))

    # ---------------- frontend facade ----------------
    # The HTTP frontend sees projections and verbs only — never the registry,
    # the ledger, or the router URL (codex-rollout-fullparameter-design-0810
    # §4.2; §3.7 dependency rule: frontend -> backend facade + sampling
    # transport). A future lifecycle strategy replaces what sits behind these
    # without forking the frontend.

    def registration_view(self, name: str) -> dict | None:
        """Projection of the name's CURRENT registration: identity, lifecycle
        state, resolved rank, bound-ness, and serving version."""
        record = self.registry.find(name)
        if record is None:
            return None
        return dict(
            name=record.name,
            registration_id=record.registration_id,
            state=record.state.value,
            rank=getattr(record.config, "rank", None),
            bound=record.slot is not None,
            serving_version=record.serving_version,
        )

    def operation_view(self, operation_id: str) -> dict | None:
        self.sweep_operation_timeouts()
        view = self.operations.get(operation_id)
        if view is not None and view["state"] == "QUEUED":
            for stall in self.operations.gap_stalls():
                if (stall["name"], stall["registration_id"]) == (view["name"], view["registration_id"]):
                    view["waiting_on_ordinal"] = stall["missing_ordinal"]
                    view["gap_stalled_for"] = stall["stalled_for"]
        return view

    def ack_operation(self, operation_id: str) -> None:
        self.operations.ack(operation_id)

    def sampling_endpoint(self) -> str:
        """Base URL sampling requests go to: the SGLang router today, the
        InferenceController-provided endpoint after PR #1842."""
        return self.router_url

    # ---------------- info ----------------

    def service_info(self) -> dict:
        self.sweep_operation_timeouts()
        args = self.args
        return dict(
            base_model=getattr(args, "hf_checkpoint", None),
            lora_rank_max=getattr(args, "lora_rank", None),
            n_adapters=getattr(args, "multi_lora_n_adapters", None),
            occupied_slots=self.registry.slot_pool.occupied_slot_ids(),
            ready_adapters=sorted(self.registry.in_state(AdapterState.READY)),
            supported_loss_fns=list(SUPPORTED_LOSS_FNS),
            operation_gap_timeout=self.operations.gap_timeout,
            operation_claimed_ttl=self.operations.claimed_ttl,
            gap_stalls=self.operations.gap_stalls(),
        )


def operation_result_metrics(payload: dict, logprobs: list[list[float]]) -> dict[str, float]:
    spec = payload.get("loss") or {}
    loss_fn = spec.get("loss_fn", "cross_entropy")
    config = spec.get("loss_fn_config") or {}
    total = 0.0
    weighted_tokens = 0.0
    loss_weight_sum = 0.0
    for sample, sample_logprobs in zip(payload.get("samples") or [], logprobs, strict=False):
        mask = sample.get("loss_mask") or [1.0] * len(sample_logprobs)
        weighted_tokens += sum(1.0 for m in mask if m)
        if loss_fn == "cross_entropy":
            weights = sample.get("loss_weights") or []
            total += sum(-lp * w * m for lp, w, m in zip(sample_logprobs, weights, mask, strict=False))
            loss_weight_sum += sum(w * m for w, m in zip(weights, mask, strict=False))
        else:
            old = sample.get("rollout_log_probs") or []
            advantages = sample.get("advantages") or []
            for lp, old_lp, advantage, m in zip(sample_logprobs, old, advantages, mask, strict=False):
                ratio = math.exp(min(lp - old_lp, 80.0))
                surrogate = ratio * advantage
                if loss_fn == "ppo":
                    low = config.get("clip_low_threshold", 0.8)
                    high = config.get("clip_high_threshold", 1.2)
                    surrogate = min(surrogate, min(max(ratio, low), high) * advantage)
                total += -surrogate * m
    metrics = {"loss:sum": total, "unmasked_tokens:sum": weighted_tokens}
    if loss_fn == "cross_entropy":
        metrics["loss_weight:sum"] = loss_weight_sum
    return metrics
