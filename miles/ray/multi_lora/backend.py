"""Multi-LoRA backend: the registry plus engine-facing aborts, shared by the
controller Ray actor and the HTTP server. Subclass via
``--multi-lora-backend-path``."""

import asyncio
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from miles.ray.multi_lora.registry import AdapterRegistry, AdapterState
from miles.utils.adapter_config import AdapterRunConfig
from miles.utils.http_utils import router_worker_base_urls
from miles.utils.multi_lora import RID_SEPARATOR, min_groups_per_dp_split

logger = logging.getLogger(__name__)


class UsageJournal:
    """Append-only JSONL token-usage ledger under ``{--save}/multi_lora_controller/``.

    The registry's in-memory meters are volatile (a controller restart loses
    them, and a same-name re-registration evicts the COMPLETED record); the
    journal is the authoritative record an external billing backend can
    consume. Events are timestamped at append time; replay on startup
    restores terminal-registration usage."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def append(self, event: dict) -> None:
        record = {"ts": time.time(), **event}
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def read_events(self) -> list[dict]:
        if not self.path.exists():
            return []
        events = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping corrupt usage journal line in {self.path}")
        return events

    def final_usage(self, registration_id: str) -> dict | None:
        """The frozen counters of a finalized registration whose in-memory
        meter was evicted. Rare read path (only past MAX_USAGE_RECORDS), so a
        full journal scan is acceptable; the newest 'final' event wins."""
        result = None
        for event in self.read_events():
            if event.get("kind") == "final" and event.get("registration_id") == registration_id:
                result = event.get("usage")
        return result

    def close(self) -> None:
        self._file.close()


class MultiLoRABackend:
    """Registry + engine-facing aborts, shared by the Ray actor and HTTP server.
    Subclass via --multi-lora-backend-path."""

    def __init__(self, args: Any, router_url: str) -> None:
        self.args = args
        self.registry = AdapterRegistry(args.multi_lora_n_adapters)
        self.router_url = router_url.rstrip("/")
        self.client: httpx.AsyncClient | None = None
        self.usage_journal: UsageJournal | None = None

    async def init(self) -> None:
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        if getattr(self.args, "save", None):
            self.usage_journal = UsageJournal(Path(self.args.save) / "multi_lora_controller" / "usage.jsonl")
            self.registry.replay_usage_journal(self.usage_journal.read_events())
            self.registry.usage_journal = self.usage_journal.append
            self.registry.usage_fallback = self.usage_journal.final_usage
        else:
            logger.warning("No --save dir: token-usage journal disabled; usage counters are in-memory only")

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None
        if self.usage_journal is not None:
            self.usage_journal.close()
            self.usage_journal = None

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
        """Free the adapter's slot after one final abort round: requests can survive the
        ``retire_adapters`` abort (e.g. multi-turn groups), and must not leak to the slot's next tenant."""
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
                    return router_worker_base_urls(extract(resp.json()))
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
