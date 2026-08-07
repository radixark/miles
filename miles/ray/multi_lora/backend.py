"""Multi-LoRA backend: the registry plus engine-facing aborts, shared by the
controller Ray actor and the HTTP server. Subclass via
``--multi-lora-backend-path``."""

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from miles.ray.multi_lora.operations import OperationLedger
from miles.ray.multi_lora.registry import AdapterRegistry, AdapterState
from miles.utils.adapter_config import AdapterRunConfig
from miles.utils.http_utils import router_worker_base_urls
from miles.utils.multi_lora import rid_prefix

logger = logging.getLogger(__name__)


class MultiLoRABackend:
    """Registry + engine-facing aborts, shared by the Ray actor and HTTP server.
    Subclass via --multi-lora-backend-path."""

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

    async def validate_adapter(self, name: str, config: Any) -> None:
        """Override to reject adapter registrations (raise ValueError)."""

    def resolve_adapter_config(self, name: str, config: Any) -> Any:
        """Resolve optional adapter-local values against process-wide defaults
        and validate the batch shape against the trainer's DP layout.

        All batch-shape constraints are enforced here, at registration, so a
        bad config fails immediately instead of crashing an arbitrary later
        train batch.
        """
        if config is None or not isinstance(config, AdapterRunConfig):
            return config

        if config.input_mode not in ("multi-lora", "thinker"):
            raise ValueError(
                f"Adapter '{name}' input_mode must be 'multi-lora' or 'thinker', got '{config.input_mode}'"
            )
        if config.input_mode == "thinker":
            return self._resolve_thinker_config(name, config)
        if config.data is None:
            raise ValueError(f"Adapter '{name}' needs a dataset path: set 'data' (or use input_mode: thinker)")

        rank = config.rank if config.rank is not None else getattr(self.args, "lora_rank", 1)
        alpha = config.alpha if config.alpha is not None else getattr(self.args, "lora_alpha", rank)
        rollout_batch_size = (
            config.rollout_batch_size
            if config.rollout_batch_size is not None
            else getattr(self.args, "rollout_batch_size", None)
        )
        n_samples_per_prompt = (
            config.n_samples_per_prompt
            if config.n_samples_per_prompt is not None
            else getattr(self.args, "n_samples_per_prompt", 1)
        )

        if type(rank) is not int or rank <= 0:
            raise ValueError(f"Adapter '{name}' rank must be a positive integer")
        if rank > getattr(self.args, "lora_rank", rank):
            raise ValueError(f"Adapter '{name}' rank {rank} exceeds the allocated maximum rank {self.args.lora_rank}")
        if alpha is None or alpha <= 0:
            raise ValueError(f"Adapter '{name}' must have a positive alpha")
        if type(rollout_batch_size) is not int or rollout_batch_size <= 0:
            raise ValueError(f"Adapter '{name}' rollout_batch_size must be a positive integer (prompt groups)")
        if type(n_samples_per_prompt) is not int or n_samples_per_prompt <= 0:
            raise ValueError(f"Adapter '{name}' n_samples_per_prompt must be a positive integer")
        if config.num_step is not None and (type(config.num_step) is not int or config.num_step <= 0):
            raise ValueError(f"Adapter '{name}' num_step must be a positive integer")
        if config.num_epoch is not None and (type(config.num_epoch) is not int or config.num_epoch <= 0):
            raise ValueError(f"Adapter '{name}' num_epoch must be a positive integer")
        if config.num_step is not None and config.num_epoch is not None:
            logger.warning(f"Adapter '{name}' sets both num_step and num_epoch; num_step takes precedence")

        # A bad data path or unresolvable reward config does not fail at this
        # API otherwise: the data path kills the shared rollout producer thread
        # and an empty reward config burns every generated sample, either way
        # stalling ALL adapters behind a misleading empty-batch timeout.
        if not Path(config.data).expanduser().exists():
            raise ValueError(
                f"Adapter '{name}' data path '{config.data}' does not exist "
                "(checked from the controller process, which runs on the head node with the rollout data source)"
            )
        if (
            config.custom_rm_path is None
            and not (config.rm_type or "").strip()
            and getattr(self.args, "custom_rm_path", None) is None
            and not (getattr(self.args, "rm_type", None) or "").strip()
        ):
            raise ValueError(
                f"Adapter '{name}' has no reward config: set rm_type or custom_rm_path in the adapter "
                "config, or launch with --rm-type / --custom-rm-path"
            )

        adapter_global_batch_size = rollout_batch_size * n_samples_per_prompt
        if (max_batch := getattr(self.args, "multi_lora_max_adapter_global_batch_size", None)) is not None:
            if adapter_global_batch_size > max_batch:
                raise ValueError(
                    f"Adapter '{name}' consumes {adapter_global_batch_size} samples per step "
                    f"(rollout_batch_size {rollout_batch_size} x n_samples_per_prompt {n_samples_per_prompt}), "
                    f"exceeding --multi-lora-max-adapter-global-batch-size {max_batch}"
                )
        save = self._resolve_save(name, config)

        return replace(
            config,
            rank=rank,
            alpha=alpha,
            rollout_batch_size=rollout_batch_size,
            n_samples_per_prompt=n_samples_per_prompt,
            save=save,
        )

    def _resolve_thinker_config(self, name: str, config: AdapterRunConfig) -> AdapterRunConfig:
        """Thinker adapters have no dataset, reward, or server-side batch
        shape: clients push batches through the operation queue and end the
        run by explicit deregistration (num_step remains an optional bound)."""
        rank = config.rank if config.rank is not None else getattr(self.args, "lora_rank", 1)
        alpha = config.alpha if config.alpha is not None else getattr(self.args, "lora_alpha", rank)
        if type(rank) is not int or rank <= 0:
            raise ValueError(f"Adapter '{name}' rank must be a positive integer")
        if rank > getattr(self.args, "lora_rank", rank):
            raise ValueError(f"Adapter '{name}' rank {rank} exceeds the allocated maximum rank {self.args.lora_rank}")
        if alpha is None or alpha <= 0:
            raise ValueError(f"Adapter '{name}' must have a positive alpha")
        if config.data is not None:
            raise ValueError(f"Thinker adapter '{name}' must not set 'data'; batches arrive via operations")
        if config.rm_type is not None or config.custom_rm_path is not None:
            raise ValueError(f"Thinker adapter '{name}' must not set a reward; losses come per operation")
        if config.rollout_function_path is not None:
            raise ValueError(
                f"Thinker adapter '{name}' must not set rollout_function_path; the queue child is built in"
            )
        if config.num_epoch is not None:
            raise ValueError(f"Thinker adapter '{name}' must not set num_epoch (there is no dataset); use num_step")
        if config.num_step is not None and (type(config.num_step) is not int or config.num_step <= 0):
            raise ValueError(f"Adapter '{name}' num_step must be a positive integer")
        return replace(config, rank=rank, alpha=alpha, save=self._resolve_save(name, config))

    def _resolve_save(self, name: str, config: AdapterRunConfig) -> Path:
        if config.save is not None:
            return Path(config.save)
        if getattr(self.args, "save", None) is None:
            raise ValueError(f"Adapter '{name}' has no save dir: set 'save' in the adapter config or pass --save")
        return Path(self.args.save) / "adapters" / name

    async def register(self, name: str, config: Any) -> dict:
        config = self.resolve_adapter_config(name, config)
        await self.validate_adapter(name, config)
        result = self.registry.register(name, config)
        resolved = getattr(config, "save", None)
        if resolved is not None:
            logger.info(f"Adapter '{name}' registered (slot {result['slot']}), checkpoints -> {resolved}")
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

    # ---------------- thinker operations ----------------

    def enqueue_operation(
        self, name: str, operation_id: str, ordinal: int, kind: str, payload: dict | None = None
    ) -> dict:
        """Enqueue one client operation against the name's CURRENT registration
        (clients address adapters by name; the resolved registration_id rides
        the operation, so a re-registered name can never execute stale work)."""
        record = self.registry.find(name)
        if record is None or record.state not in (AdapterState.PENDING, AdapterState.ACTIVE):
            raise ValueError(f"Adapter '{name}' is not accepting operations (not registered or retiring)")
        if getattr(record.config, "input_mode", "multi-lora") != "thinker":
            raise ValueError(f"Adapter '{name}' is a dataset run; operations apply to input_mode: thinker only")
        return self.operations.enqueue(operation_id, name, record.registration_id, ordinal, kind, payload)

    # Control kinds the driver's control phase can execute today; the rest
    # (save_state / load_state / publish_snapshot) keep their queue turn until
    # their executors land.
    EXECUTABLE_CONTROL_KINDS = ("optim_step", "publish_snapshot", "save_state", "load_state")
    # Moving state under unstepped gradients would silently drop them (no
    # sidecar carries grads): the client must step or deregister first.
    DIRTY_GATED_KINDS = ("save_state", "load_state")

    def claim_ready_control_operations(self) -> list[dict]:
        """Claim every registration whose next open operation is an executable
        control kind on a slot-resident ACTIVE adapter. Claims are strictly
        serialized per registration, so an optim_step is claimable only after
        its preceding forward_backward batches reached terminal states. The
        claimed view carries the registry's authoritative clocks."""
        ready = []
        for name, registration_id in self.operations.claimable_control_tenants():
            record = self.registry.find(name)
            if (
                record is None
                or record.registration_id != registration_id
                or record.state is not AdapterState.ACTIVE
                or record.slot is None
            ):
                continue
            operation = self.operations.claim_control_operation(
                name, registration_id, kinds=self.EXECUTABLE_CONTROL_KINDS
            )
            if operation is None:
                continue
            if operation["kind"] in self.DIRTY_GATED_KINDS and self._slot_dirty(record):
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

    def _slot_dirty(self, record) -> bool:
        entry = self.registry.slot_pool.entry_of(record.tenant)
        return entry is not None and "dirty-grads" in entry.pins

    def complete_control_operations(self, results: dict[str, dict]) -> None:
        """Book the trainer's control-phase outcomes: an optim_step success
        advances the adapter's step clock and either outcome releases the
        dirty-gradient pin (a veto zeroes the accumulated gradients on every
        rank); a load_state success repositions the step clock."""
        for operation_id, outcome in results.items():
            operation = self.operations.get(operation_id)
            if operation is None:
                continue
            if outcome.get("ok"):
                self.operations.complete(operation_id, outcome.get("result"))
                if operation["kind"] == "optim_step":
                    self.registry.commit_thinker_step(operation["name"])
                elif operation["kind"] == "load_state":
                    self.registry.set_step(operation["name"], int((outcome.get("result") or {}).get("step", 0)))
            else:
                self.operations.fail(
                    operation_id, outcome.get("error", "control operation failed"), outcome.get("category", "server")
                )
                if operation["kind"] == "optim_step":
                    self.registry.clear_dirty(operation["name"])

    def commit_thinker_batch(
        self, names: list[str], operation_ids: list[str], logprobs_by_op: dict[str, list] | None = None
    ) -> None:
        """A thinker train call landed: its slots now hold unstepped
        gradients (pin them) and its forward_backward operations complete with
        their per-datum target logprobs (row order = the operation's datum
        order), which the frontend maps into ForwardBackwardOutput."""
        self.registry.mark_accumulated(names)
        logprobs_by_op = logprobs_by_op or {}
        for operation_id in operation_ids:
            operation = self.operations.get(operation_id)
            if operation is not None and operation["state"] == "CLAIMED":
                self.operations.complete(operation_id, {"logprobs": logprobs_by_op.get(operation_id)})

    async def free_slot(self, name: str) -> int:
        """Free the adapter's slot after one final abort round: requests can survive the
        ``retire_adapters`` abort (e.g. multi-turn groups), and must not leak to the slot's next tenant."""
        record = self.registry.records.get(name)
        if record is not None and record.state is AdapterState.CLEANUP:
            await self.abort_adapter_requests(name, record.registration_id)
        return self.registry.free_slot(name)

    def service_info(self) -> dict:
        """Deployment facts a thinker frontend needs for get_server_capabilities
        and weights_info: one base model per deployment, the rank ceiling, and
        slot occupancy."""
        args = self.args
        return dict(
            base_model=getattr(args, "hf_checkpoint", None),
            lora_rank_max=getattr(args, "lora_rank", None),
            n_adapters=getattr(args, "multi_lora_n_adapters", None),
            occupied_slots=sorted(e.slot for e in self.registry.slot_pool.entries if e.tenant is not None),
            active_adapters=sorted(self.registry.in_state(AdapterState.ACTIVE)),
        )

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
            logger.warning(f"Abort for adapter '{adapter_name}': no workers discovered at {self.router_url}")
            return
        results = await asyncio.gather(
            *(self.client.post(f"{url}/abort_request", json={"rid": prefix, "prefix": True}) for url in urls),
            return_exceptions=True,
        )
        if failures := sum(isinstance(r, Exception) for r in results):
            logger.warning(f"Abort for adapter '{adapter_name}': {failures}/{len(results)} posts failed")
