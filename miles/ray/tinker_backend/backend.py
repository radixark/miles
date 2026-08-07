"""Tinker backend control plane: registry + operation ledger + engine-facing
aborts, shared by the controller Ray actor and the HTTP server. Every client
input is validated here, at the boundary — an unsupported loss, shape, or
payload must never reach the shared GPU driver."""

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from miles.ray.tinker_backend.config import AdapterRunConfig
from miles.ray.tinker_backend.operations import OperationLedger
from miles.ray.tinker_backend.registry import AdapterRegistry, AdapterState
from miles.utils.http_utils import router_worker_base_urls
from miles.utils.tinker_backend import rid_prefix

logger = logging.getLogger(__name__)

# v1 compatibility matrix (README table mirrors this): anything outside is a
# typed user error at enqueue time, never a GPU-side crash.
SUPPORTED_LOSS_FNS = ("cross_entropy", "importance_sampling", "ppo")
_ADAM_FIELDS = ("learning_rate", "beta1", "beta2", "eps", "weight_decay", "grad_clip_norm")
_SAMPLE_TENSOR_FIELDS = ("loss_mask", "loss_weights", "advantages", "rollout_log_probs")


class TinkerBackend:
    """Subclass via --multi-lora-backend-path."""

    def __init__(self, args: Any, router_url: str) -> None:
        self.args = args
        self.registry = AdapterRegistry(args.multi_lora_n_adapters)
        self.operations = OperationLedger()
        self.router_url = router_url.rstrip("/")
        self.client: httpx.AsyncClient | None = None

    async def init(self) -> None:
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

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
        logger.info(f"[tinker] adapter '{name}' registered (slot {result['slot']})")
        return result

    async def deregister(self, name: str) -> None:
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
        return self.registry.free_slot(name)

    # ---------------- operation preflight (compatibility matrix) ----------------

    def enqueue_operation(
        self, name: str, operation_id: str, ordinal: int, kind: str, payload: dict | None = None
    ) -> dict:
        """Enqueue one client operation against the name's CURRENT
        registration, after full boundary validation."""
        record = self.registry.find(name)
        if record is None or record.state not in (AdapterState.PENDING, AdapterState.READY):
            raise ValueError(f"Adapter '{name}' is not accepting operations (not registered or retiring)")
        payload = payload or {}
        self._preflight(name, kind, payload)
        return self.operations.enqueue(operation_id, name, record.registration_id, ordinal, kind, payload)

    def _preflight(self, name: str, kind: str, payload: dict) -> None:
        if kind in ("forward_backward", "forward"):
            samples = payload.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"{kind} payload needs a non-empty 'samples' list")
            for i, sample in enumerate(samples):
                self._preflight_sample(name, kind, i, sample)
            if kind == "forward_backward":
                loss = payload.get("loss") or {}
                loss_fn = loss.get("loss_fn", "cross_entropy")
                if loss_fn not in SUPPORTED_LOSS_FNS:
                    raise ValueError(
                        f"loss_fn '{loss_fn}' is not supported in v1; supported: {', '.join(SUPPORTED_LOSS_FNS)}"
                    )
        elif kind == "optim_step":
            adam = payload.get("adam_params") or {}
            for field_name, value in adam.items():
                if field_name not in _ADAM_FIELDS:
                    raise ValueError(f"unknown adam_params field '{field_name}'")
                if not isinstance(value, (int, float)):
                    raise ValueError(f"adam_params.{field_name} must be a number")
        elif kind == "save_state":
            tag = payload.get("tag")
            if tag is not None and not isinstance(tag, str):
                raise ValueError("save_state 'tag' must be a string")
        elif kind == "load_state":
            if not isinstance(payload.get("path"), str) or not payload["path"]:
                raise ValueError("load_state needs a 'path'")
        elif kind == "save_weights_for_sampler":
            pass
        else:
            raise ValueError(f"unknown operation kind '{kind}'")

    def _preflight_sample(self, name: str, kind: str, index: int, sample: Any) -> None:
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
        if not isinstance(response_length, int) or not (0 < response_length <= len(tokens)):
            raise ValueError(f"{where}: 'response_length' must be an int in (0, len(tokens)]")
        for field_name in _SAMPLE_TENSOR_FIELDS:
            value = sample.get(field_name)
            if value is None:
                continue
            if not isinstance(value, list) or len(value) != response_length:
                raise ValueError(f"{where}: '{field_name}' must be a flat list of length response_length (1-D only)")
            if any(isinstance(v, (list, dict)) for v in value):
                raise ValueError(f"{where}: '{field_name}' must be 1-D; nested targets are not supported in v1")

    # ---------------- control-operation claims ----------------

    EXECUTABLE_CONTROL_KINDS = ("optim_step", "save_weights_for_sampler", "save_state", "load_state")
    # Moving state under unstepped gradients would silently drop them (no
    # checkpoint carries grads): the client must step or deregister first.
    DIRTY_GATED_KINDS = ("save_state", "load_state")

    def claim_ready_control_operations(self) -> list[dict]:
        """Claim every registration whose next open operation is an executable
        control kind on a slot-resident READY adapter. The claimed view
        carries the registry's authoritative clocks."""
        ready = []
        for name, registration_id in self.operations.claimable_control_tenants():
            record = self.registry.find(name)
            if (
                record is None
                or record.registration_id != registration_id
                or record.state is not AdapterState.READY
                or record.slot is None
            ):
                continue
            operation = self.operations.claim_control_operation(
                name, registration_id, kinds=self.EXECUTABLE_CONTROL_KINDS
            )
            if operation is None:
                continue
            if operation["kind"] in self.DIRTY_GATED_KINDS and self.registry.is_dirty(name):
                self.operations.fail(
                    operation["operation_id"],
                    f"adapter '{name}' holds unstepped gradients; optim_step (or deregister) before "
                    f"{operation['kind']}",
                    "user",
                )
                continue
            operation["slot"] = record.slot
            operation["step"] = record.step
            operation["serving_version"] = record.serving_version
            ready.append(operation)
        return ready

    def complete_control_operations(self, results: dict[str, dict]) -> None:
        """Book the trainer's control-phase outcomes: an optim_step success
        advances the step clock and either outcome releases the dirty pin (a
        veto zeroes the gradients on every rank); a load_state success
        repositions the step clock."""
        for operation_id, outcome in results.items():
            operation = self.operations.get(operation_id)
            if operation is None:
                continue
            if outcome.get("ok"):
                self.operations.complete(operation_id, outcome.get("result"))
                if operation["kind"] == "optim_step":
                    self.registry.commit_tinker_step(operation["name"])
                elif operation["kind"] == "load_state":
                    self.registry.set_step(operation["name"], int((outcome.get("result") or {}).get("step", 0)))
            else:
                self.operations.fail(
                    operation_id, outcome.get("error", "control operation failed"), outcome.get("category", "server")
                )
                if operation["kind"] == "optim_step":
                    self.registry.clear_dirty(operation["name"])

    def commit_tinker_batch(
        self, accumulated: list[str], operation_ids: list[str], logprobs_by_op: dict[str, list] | None = None
    ) -> None:
        """A data selection landed: forward_backward adapters now hold
        unstepped gradients (pin them); every listed operation completes with
        its per-datum target logprobs in the operation's row order."""
        self.registry.mark_accumulated(accumulated)
        logprobs_by_op = logprobs_by_op or {}
        for operation_id in operation_ids:
            operation = self.operations.get(operation_id)
            if operation is not None and operation["state"] == "CLAIMED":
                self.operations.complete(operation_id, {"logprobs": logprobs_by_op.get(operation_id)})

    # ---------------- engine-facing ----------------

    async def worker_urls(self) -> list[str]:
        assert self.client is not None
        for endpoint, extract in (
            ("/list_workers", lambda body: body["urls"]),
            ("/workers", lambda body: [worker["url"] for worker in body["workers"]]),
        ):
            try:
                resp = await self.client.get(f"{self.router_url}{endpoint}")
                if resp.status_code == 200:
                    return router_worker_base_urls(extract(resp.json()))
            except Exception:
                continue
        return []

    async def abort_adapter_requests(self, adapter_name: str, registration_id: str) -> None:
        # Registration-scoped: a retiring tenant's abort must never match a
        # same-name successor's in-flight requests (rid carries the registration).
        prefix = rid_prefix(adapter_name, registration_id)
        urls = await self.worker_urls()
        if not urls:
            logger.warning(f"[tinker] abort for '{adapter_name}': no workers discovered at {self.router_url}")
            return
        results = await asyncio.gather(
            *(self.client.post(f"{url}/abort_request", json={"rid": prefix, "prefix": True}) for url in urls),
            return_exceptions=True,
        )
        if failures := sum(isinstance(r, Exception) for r in results):
            logger.warning(f"[tinker] abort for '{adapter_name}': {failures}/{len(results)} posts failed")

    # ---------------- info ----------------

    def service_info(self) -> dict:
        """Deployment facts a tinker frontend needs for get_server_capabilities
        and weights_info: one base model per deployment, the rank ceiling,
        slot occupancy, and the v1 loss allowlist."""
        args = self.args
        return dict(
            base_model=getattr(args, "hf_checkpoint", None),
            lora_rank_max=getattr(args, "lora_rank", None),
            n_adapters=getattr(args, "multi_lora_n_adapters", None),
            occupied_slots=self.registry.slot_pool.occupied_slot_ids(),
            ready_adapters=sorted(self.registry.in_state(AdapterState.READY)),
            supported_loss_fns=list(SUPPORTED_LOSS_FNS),
        )
