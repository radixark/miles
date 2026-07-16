"""Multi-LoRA adapter registry, backend, and control-plane HTTP server."""

import asyncio
import logging
import re
import uuid
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from miles.utils.adapter_config import AdapterRun, AdapterRunConfig, parse_adapter_run_yaml

logger = logging.getLogger(__name__)

__all__ = [
    "AdapterRegistry",
    "AdapterState",
    "MultiLoRABackend",
    "MultiLoRAHTTPServer",
    "RID_SEPARATOR",
    "is_multi_lora_enabled",
    "make_rid",
    "parse_adapter",
    "slot_lora_name",
]


# Must not appear in adapter names so rid prefix aborts can't cross adapters.
RID_SEPARATOR = "::"

VALID_ADAPTER_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def is_multi_lora_enabled(args: Any) -> bool:
    return getattr(args, "multi_lora", False)


def make_rid(adapter_name: str) -> str:
    return f"{adapter_name}{RID_SEPARATOR}{uuid.uuid4().hex}"


def parse_adapter(rid: str) -> str:
    return rid.rsplit(RID_SEPARATOR, 1)[0]


def slot_lora_name(slot: int) -> str:
    """Engine-side LoRA adapter name for a controller slot. Weight pushes and
    every inference request (rollout and prefill scoring) must agree on this."""
    return f"__miles_slot_{slot}"


def min_groups_per_dp_split(n_samples_per_prompt: int, dp_size: int) -> int:
    """Minimum prompt-group count that splits cleanly across data-parallel
    ranks.

    Train batches only pop groups in multiples of this value, so each popped
    slice has a sample count divisible by ``dp_size`` with no trimming.

    Requires ``n_samples_per_prompt`` and ``dp_size`` to divide each other
    (one must be a multiple of the other).
    """
    larger = max(dp_size, n_samples_per_prompt)
    smaller = min(dp_size, n_samples_per_prompt)
    if larger % smaller == 0:
        return larger // n_samples_per_prompt
    raise ValueError(
        f"n_samples_per_prompt={n_samples_per_prompt} must be a divisor or a multiple of "
        f"the data-parallel size {dp_size} so whole prompt groups can split evenly across ranks"
    )


class AdapterState(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    RETIRING = "RETIRING"
    CLEANUP = "CLEANUP"
    COMPLETED = "COMPLETED"


# States that hold a slot.
LIVE_STATES = (
    AdapterState.PENDING,
    AdapterState.ACTIVE,
    AdapterState.RETIRING,
    AdapterState.CLEANUP,
)


@dataclass
class AdapterRecord:
    name: str
    slot: int
    config: Any
    step: int = 0
    # Baseline step for relative num_step stopping (supports checkpoint resume).
    start_step: int = 0
    # Committed prompt groups accumulated toward the current optimizer step.
    # Only advanced by mark_batch_trained (after a successful train call).
    accumulated_groups: int = 0
    state: AdapterState = AdapterState.PENDING


MAX_BATCH_RECORDS = 16
MAX_COMPLETED_RECORDS = 1024


class AdapterRegistry:
    """One record per name; ``slot_versions`` never reset, so (slot, version)
    never recurs across slot reuse."""

    def __init__(self, max_adapters: int) -> None:
        self.max_adapters = max_adapters
        self.free_slots: set[int] = set(range(max_adapters))
        self.slot_versions: list[int] = [0] * max_adapters
        self.records: dict[str, AdapterRecord] = {}
        self.batch_records: dict[int, dict] = {}

    def in_state(self, *states: AdapterState) -> dict[str, AdapterRecord]:
        return {name: r for name, r in self.records.items() if r.state in states}

    def find(self, name: str) -> AdapterRecord | None:
        record = self.records.get(name)
        return record if record is not None and record.state in LIVE_STATES else None

    def is_active(self, name: str) -> bool:
        record = self.records.get(name)
        return record is not None and record.state in (AdapterState.ACTIVE, AdapterState.RETIRING)

    def register(self, name: str, config: Any) -> dict:
        if not VALID_ADAPTER_NAME.match(name) or name in (".", ".."):
            raise ValueError(f"Adapter name '{name}' is invalid: use only letters, digits, '.', '_' and '-'")
        if (existing := self.records.get(name)) is not None:
            if existing.state in (AdapterState.PENDING, AdapterState.ACTIVE):
                raise ValueError(f"Adapter '{name}' already registered")
            if existing.state in (AdapterState.RETIRING, AdapterState.CLEANUP):
                raise ValueError(f"Adapter '{name}' is still cleaning up; retry shortly")
        if (save_dir := getattr(config, "save", None)) is not None:
            for record in self.in_state(*LIVE_STATES).values():
                other_save = getattr(record.config, "save", None)
                if other_save is not None and Path(other_save).resolve() == Path(save_dir).resolve():
                    raise ValueError(
                        f"Adapter '{name}' save dir '{save_dir}' is already used by adapter '{record.name}'"
                    )
        if not self.free_slots:
            raise RuntimeError(f"No free adapter slots (max {self.max_adapters})")
        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.records.pop(name, None)
        self.records[name] = AdapterRecord(name=name, slot=slot, config=config)
        return {"name": name, "slot": slot}

    def deregister(self, name: str) -> None:
        record = self.records.get(name)
        if record is not None and record.state in (AdapterState.PENDING, AdapterState.ACTIVE):
            record.state = AdapterState.RETIRING

    def retire_adapters(self) -> list[str]:
        retired = sorted(self.in_state(AdapterState.RETIRING))
        for name in retired:
            self.records[name].state = AdapterState.CLEANUP
        return retired

    def free_slot(self, name: str) -> int:
        record = self.records.get(name)
        if record is None or record.state is not AdapterState.CLEANUP:
            return -1
        self.free_slots.add(record.slot)
        record.state = AdapterState.COMPLETED
        self.records[name] = self.records.pop(name)
        completed = self.in_state(AdapterState.COMPLETED)
        for oldest in list(completed)[: len(completed) - MAX_COMPLETED_RECORDS]:
            self.records.pop(oldest)
        return record.slot

    def adapter_state(self, name: str) -> AdapterState | None:
        record = self.records.get(name)
        if record is None:
            return None
        if record.state is AdapterState.COMPLETED:
            self.records[name] = self.records.pop(name)
        return record.state

    def record_weight_update(self, names: list[str]) -> None:
        """A weight push landed: bump slot versions, promote PENDING to ACTIVE."""
        for name in names:
            record = self.find(name)
            if record is None:
                continue
            self.slot_versions[record.slot] += 1
            if record.state is AdapterState.PENDING:
                record.state = AdapterState.ACTIVE

    def record_batch_adapters(self, rollout_id: int, groups: dict[str, int], step_names: list[str]) -> None:
        """Register what a train batch contains before it trains.

        ``groups`` maps adapter name -> prompt groups riding in this batch;
        ``step_names`` lists adapters whose adapter batch completes with
        this batch (decided by the collection loop, which caps per-adapter
        contributions at the adapter's remaining groups).
        """
        unknown = set(step_names) - set(groups)
        assert not unknown, f"step adapters {sorted(unknown)} not present in batch groups"
        self.batch_records[rollout_id] = {"groups": dict(groups), "step_names": list(step_names)}
        while len(self.batch_records) > MAX_BATCH_RECORDS:
            self.batch_records.pop(next(iter(self.batch_records)))

    def mark_batch_trained(self, rollout_id: int) -> list[str]:
        """A train call over this batch succeeded: bank each adapter's groups, fire steps.

        This is the only place accumulation/step state advances, so a failed or
        retried train call leaves the registry untouched. Returns the adapters
        that stepped.
        """
        record_entry = self.batch_records.pop(rollout_id, None)
        if record_entry is None:
            return []
        stepped = []
        reached_num_step = []
        for name, n_groups in record_entry["groups"].items():
            record = self.records.get(name)
            if record is None or record.state not in (
                AdapterState.ACTIVE,
                AdapterState.RETIRING,
                AdapterState.CLEANUP,
            ):
                continue
            record.accumulated_groups += n_groups
            if name in record_entry["step_names"]:
                target = record.config.rollout_batch_size
                if record.accumulated_groups != target:
                    logger.warning(
                        f"Adapter '{name}' stepped with accumulated_groups={record.accumulated_groups} "
                        f"!= rollout_batch_size={target}; adapter batch accounting drifted"
                    )
                record.step += 1
                record.accumulated_groups = 0
                stepped.append(name)
                if (
                    getattr(record.config, "num_step", None) is not None
                    and record.state is AdapterState.ACTIVE
                    and (record.step - record.start_step) >= record.config.num_step
                ):
                    reached_num_step.append(name)
        for name in reached_num_step:
            logger.info(
                f"Adapter '{name}' reached num_step={self.records[name].config.num_step} "
                f"(start_step={self.records[name].start_step}, step={self.records[name].step}), deregistering"
            )
            self.deregister(name)
        return stepped

    def set_step(self, name: str, step: int) -> None:
        if (record := self.find(name)) is not None:
            record.step = step
            record.start_step = step

    def step_count(self, name: str) -> int:
        record = self.find(name)
        return record.step if record is not None else 0

    def view(self, record: AdapterRecord) -> AdapterRun:
        return AdapterRun(
            name=record.name,
            config=record.config,
            slot=record.slot,
            version=self.slot_versions[record.slot],
            step=record.step,
            accumulated_groups=record.accumulated_groups,
        )

    def active_adapters(self) -> dict[str, AdapterRun]:
        """Sampleable view: RETIRING keeps serving until retired."""
        return {
            name: self.view(record)
            for name, record in self.in_state(AdapterState.ACTIVE, AdapterState.RETIRING).items()
        }

    def snapshot(self) -> dict:
        def views(state: AdapterState) -> dict[str, AdapterRun]:
            return {name: self.view(record) for name, record in self.in_state(state).items()}

        return {
            "pending": views(AdapterState.PENDING),
            "active": views(AdapterState.ACTIVE),
            "retiring": views(AdapterState.RETIRING),
            "cleanup": list(self.in_state(AdapterState.CLEANUP)),
            "completed": list(self.in_state(AdapterState.COMPLETED)),
        }


class MultiLoRABackend:
    """Registry + engine-facing aborts, shared by the Ray actor and HTTP server.
    Subclass via --multi-lora-backend-path."""

    def __init__(self, args: Any, router_url: str) -> None:
        self.args = args
        self.registry = AdapterRegistry(args.multi_lora_n_adapters)
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
        if config.num_row is not None and (type(config.num_row) is not int or config.num_row <= 0):
            raise ValueError(f"Adapter '{name}' num_row must be a positive integer")
        if config.num_step is not None and config.num_row is not None:
            logger.warning(
                f"Adapter '{name}' sets both num_step and num_row; num_step takes precedence and num_row is ignored"
            )
        elif config.num_step is None and config.num_row is not None:
            logger.warning(f"Adapter '{name}' uses deprecated num_row={config.num_row}; prefer num_step")
        adapter_global_batch_size = rollout_batch_size * n_samples_per_prompt
        if (max_batch := getattr(self.args, "multi_lora_max_adapter_global_batch_size", None)) is not None:
            if adapter_global_batch_size > max_batch:
                raise ValueError(
                    f"Adapter '{name}' consumes {adapter_global_batch_size} samples per step "
                    f"(rollout_batch_size {rollout_batch_size} x n_samples_per_prompt {n_samples_per_prompt}), "
                    f"exceeding --multi-lora-max-adapter-global-batch-size {max_batch}"
                )
        if (dp_size := getattr(self.args, "multi_lora_dp_size", None)) is not None:
            try:
                group_multiple = min_groups_per_dp_split(n_samples_per_prompt, dp_size)
            except ValueError as e:
                raise ValueError(f"Adapter '{name}': {e}") from None
            if rollout_batch_size % group_multiple != 0:
                raise ValueError(
                    f"Adapter '{name}' rollout_batch_size {rollout_batch_size} must be a multiple of "
                    f"its min_groups_per_dp_split ({group_multiple} at dp_size={dp_size}), so the "
                    f"adapter batch can complete from evenly-splitting takes"
                )

        save = Path(config.save) if config.save is not None else None
        if save is None:
            if getattr(self.args, "save", None) is None:
                raise ValueError(f"Adapter '{name}' has no save dir: set 'save' in the adapter config or pass --save")
            save = Path(self.args.save) / "adapters" / name

        return replace(
            config,
            rank=rank,
            alpha=alpha,
            rollout_batch_size=rollout_batch_size,
            n_samples_per_prompt=n_samples_per_prompt,
            save=save,
        )

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
            await self.abort_adapter_requests(name)
        return names

    async def free_slot(self, name: str) -> int:
        """Free the adapter's slot, after one final abort round.

        The abort in ``retire_adapters`` fires once at the RETIRING->CLEANUP
        flip, but requests can survive it: a multi-turn group between turns
        submits its next turn only after that round, and a request still inside
        the engine's tokenizer adapter batch can be missed by the scheduler-side
        matching. Aborting again here — right before the slot becomes reusable —
        closes those escapes, so a later tenant of the slot cannot serve a
        retired adapter's orphaned requests.
        """
        record = self.registry.records.get(name)
        if record is not None and record.state is AdapterState.CLEANUP:
            await self.abort_adapter_requests(name)
        return self.registry.free_slot(name)

    async def worker_urls(self) -> list[str]:
        assert self.client is not None
        for endpoint, extract in (
            ("/list_workers", lambda body: body["urls"]),
            ("/workers", lambda body: [worker["url"] for worker in body["workers"]]),
        ):
            try:
                resp = await self.client.get(f"{self.router_url}{endpoint}")
                if resp.status_code == 200:
                    return extract(resp.json())
            except Exception:
                continue
        return []

    async def abort_adapter_requests(self, adapter_name: str) -> None:
        prefix = f"{adapter_name}{RID_SEPARATOR}"
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


class RegisterAdapterRequest(BaseModel):
    """Exactly one of ``config`` (inline) or ``yaml_path`` must be set."""

    name: str
    config: AdapterRunConfig | None = None
    yaml_path: str | None = None


_NAMES_QUERY = Query(default_factory=list)


class MultiLoRAHTTPServer:
    """Control-plane API over a MultiLoRABackend. Subclass via
    --multi-lora-http-server-path (add_routes / create_app)."""

    def __init__(self, backend, host="127.0.0.1", api_port=0):
        self.backend = backend
        self.host = host
        self.api_port = api_port
        self.api_server: uvicorn.Server | None = None
        self.api_task: asyncio.Task | None = None

    @property
    def actual_api_port(self) -> int:
        if self.api_server is not None and self.api_server.started:
            return self.api_server.servers[0].sockets[0].getsockname()[1]
        return self.api_port

    def create_app(self) -> FastAPI:
        app = FastAPI(title="Miles Multi-LoRA Controller")

        @app.exception_handler(ValueError)
        async def value_error_handler(request: Request, exc: ValueError):
            return JSONResponse({"detail": str(exc)}, status_code=400)

        @app.exception_handler(RuntimeError)
        async def runtime_error_handler(request: Request, exc: RuntimeError):
            status = 409 if "No free adapter slots" in str(exc) else 500
            return JSONResponse({"detail": str(exc)}, status_code=status)

        return app

    def add_routes(self, app: FastAPI) -> None:
        app.get("/health")(self.health)
        app.get("/adapter_runs")(self.list_adapters)
        app.get("/adapter_runs/state")(self.adapter_states)  # before /adapter_runs/{name}
        app.get("/adapter_runs/{name}")(self.get_adapter)
        app.post("/adapter_runs")(self.register_adapter)
        app.delete("/adapter_runs/{name}")(self.deregister_adapter)

    async def start(self) -> None:
        app = self.create_app()
        self.add_routes(app)
        config = uvicorn.Config(app, host=self.host, port=self.api_port, log_level="warning", access_log=False)
        self.api_server = uvicorn.Server(config)
        self.api_task = asyncio.create_task(self.api_server.serve())
        while not self.api_server.started:
            if self.api_task.done():
                self.api_task.result()
                raise RuntimeError("uvicorn exited before startup completed")
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        if self.api_server is not None:
            self.api_server.should_exit = True
            await self.api_task
        self.api_server = self.api_task = None

    async def health(self) -> dict:
        return {"status": "healthy"}

    def adapter_statuses(self) -> list[dict]:
        registry = self.backend.registry
        statuses = []
        for record in registry.records.values():
            flat = asdict(registry.view(record))
            flat |= flat.pop("config")
            flat["save"] = str(flat["save"])
            flat["state"] = record.state
            if record.state is AdapterState.COMPLETED:
                flat["version"] = None
            statuses.append(flat)
        return statuses

    async def list_adapters(self) -> dict:
        return {"adapters": self.adapter_statuses()}

    async def adapter_states(self, names: list[str] = _NAMES_QUERY) -> dict:
        return {"states": {name: self.backend.registry.adapter_state(name) for name in names}}

    async def get_adapter(self, name: str) -> dict:
        for status in self.adapter_statuses():
            if status["name"] == name:
                return status
        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not registered")

    async def register_adapter(self, request: RegisterAdapterRequest) -> dict:
        if (request.config is None) == (request.yaml_path is None):
            raise HTTPException(status_code=400, detail="Exactly one of 'config' or 'yaml_path' must be set")
        if request.yaml_path is not None:
            config = parse_adapter_run_yaml(Path(request.yaml_path))
        else:
            config = request.config
        return await self.backend.register(request.name, config)

    async def deregister_adapter(self, name: str) -> dict:
        state = self.backend.registry.adapter_state(name)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Adapter '{name}' not registered")
        await self.backend.deregister(name)
        return {"status": "ok", "name": name}
