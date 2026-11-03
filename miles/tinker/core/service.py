"""Sessions, models, futures, and ordered trainer dispatch.

The trainer lock serializes trainer calls across dispatch, model creation, and lease expiry."""

import asyncio
import logging
import os
import secrets
import time
import uuid
from datetime import UTC, datetime
from typing import NamedTuple

from miles.tinker.core.checkpoint_expiry import CheckpointExpiry

from miles.tinker.core.future import RequestFuture, RequestFutureStore
from miles.tinker.core.input_validation import (
    validate_batch_payload,
    validate_checkpoint_segment,
    validate_model_config,
    validate_sample_payload,
    validate_seq_id,
    validate_ttl_seconds,
)
from miles.tinker.core.model_queue import ModelRequestQueue
from miles.tinker.core.scheduler import BarrierUnit, BatchUnit, RequestScheduler
from miles.tinker.core.types import (
    Command,
    CommandOp,
    GatewayConfig,
    ModelRecord,
    NotFoundError,
    OwnershipError,
    SamplingSessionRecord,
    SessionRecord,
    UserInputError,
)
from miles.tinker.core.utils import (
    build_parametrization_fields,
    parse_tinker_path,
    read_checkpoint_metadata,
    resolve_checkpoint_dir,
    resolve_sampler_checkpoint,
    tenant_digest,
)

logger = logging.getLogger(__name__)

# how long the Tinker SDK keeps retrying a finish (InternalClientHolder.execute_with_retries)
_SDK_FINISH_RETRY_S = 300.0


class _EndedSession(NamedTuple):
    tenant: str
    lease_expired: bool
    ended_at: float


class _Download(NamedTuple):
    tenant: str
    tinker_path: str
    # a re-save under the same name after a delete is another checkpoint, which this URL must not serve
    commit_id: str
    expires_at: float


# as official Tinker: a download URL lives an hour, and never past its checkpoint's expiry
_DOWNLOAD_URL_TTL_S = 3600
_EXPIRY_SWEEP_S = 60


def _is_same_base_model(a: str, b: str) -> bool:
    """By name: equal, or the same last path component when one side is a local path (e.g. --hf-checkpoint)."""

    def is_path(name: str) -> bool:
        return os.path.isabs(name) or os.path.exists(name)

    def last_component(name: str) -> str:
        return os.path.basename(os.path.normpath(name))

    return a == b or ((is_path(a) or is_path(b)) and last_component(a) == last_component(b))


class TinkerService:
    def __init__(self, backend, config: GatewayConfig) -> None:
        self.backend = backend
        self.config = config
        self.futures = RequestFutureStore()
        self.scheduler = RequestScheduler(config.batch_token_budget)
        self.models: dict[str, ModelRecord] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self.sampling_sessions: dict[str, SamplingSessionRecord] = {}
        self.free_slots = set(range(config.n_slots))
        self._wake = asyncio.Event()
        self._trainer_lock = asyncio.Lock()
        self._sample_tasks: dict[str, tuple] = {}  # request_id -> (task, session_id)
        self._create_tasks: set = set()
        self._cleanup_tasks: set = set()
        self._arrival_counter = 0
        self._batch_counter = 0
        self._background_error: BaseException | None = None
        # why each model closed, so later requests get the reason instead of "unknown model"
        self._close_reasons: dict[str, str] = {}
        # ended sessions, kept a lease long and at least the SDK's retry window so a retried finish is answered
        self._ended_sessions: dict[str, _EndedSession] = {}
        self.expiry = CheckpointExpiry(config.checkpoint_root)
        # issued download tokens; unguessable, held only in memory, so a restart invalidates every URL
        self._downloads: dict[str, _Download] = {}

    async def run(self) -> None:
        await self._expire_checkpoints()
        sweep_task = asyncio.create_task(self._run_lease_sweeper())
        sweep_task.add_done_callback(self._observe_background_task)
        expiry_task = asyncio.create_task(self._run_expiry_sweeper())
        expiry_task.add_done_callback(self._observe_background_task)
        try:
            while True:
                if self._background_error is not None:
                    raise self._background_error
                # unit selection shares the critical section with execution, so
                # lease expiry cannot reclaim a model queue between the two
                async with self._trainer_lock:
                    rejections = self.scheduler.ready_rejections()
                    if rejections:
                        for model_queue, pending in rejections:
                            await self._finish_request(
                                model_queue,
                                pending,
                                {"error": pending.command.validation_error, "error_category": "user"},
                            )
                        continue
                    unit = self.scheduler.schedule_next()
                    if unit is not None:
                        if isinstance(unit, BatchUnit):
                            await self._run_batch(unit)
                        else:
                            await self._run_barrier(unit)
                        if await self.backend.trainer_dead():
                            raise RuntimeError("the trainer workers died; exiting so clients get refused connections")
                        continue
                await self._wake.wait()
                self._wake.clear()
        finally:
            tasks = [sweep_task, expiry_task, *self._create_tasks, *(task for task, _ in self._sample_tasks.values())]
            for task in tasks:
                task.cancel()
            # Cleanup failures must not replace the error that stopped the dispatcher.
            await asyncio.gather(*tasks, return_exceptions=True)

    def _observe_background_task(self, task: asyncio.Task) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            if self._background_error is None:
                self._background_error = error
            self._wake.set()

    def create_session(self, tenant: str) -> str:
        session_id = f"session-{uuid.uuid4().hex}"
        self.sessions[session_id] = SessionRecord(tenant=tenant, last_heartbeat=time.monotonic())
        return session_id

    def _session_for(self, tenant: str, session_id: str) -> SessionRecord:
        session = self.sessions.get(session_id)
        if session is None or session.tenant != tenant:
            raise UserInputError(f"unknown session {session_id!r}; create a session first")
        return session

    def heartbeat(self, tenant: str, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if session is None or session.tenant != tenant:
            return False
        session.last_heartbeat = time.monotonic()
        return True

    def create_model(self, tenant: str, payload: dict) -> tuple[str, str]:
        """Two-phase like every command: allocate now, initialize the slot behind the future."""
        session = self._session_for(tenant, payload["session_id"])
        model_seq_id = validate_seq_id(payload["model_seq_id"], "model_seq_id", minimum=0)
        if (previous := session.models_by_seq.get(model_seq_id)) is not None:
            request_id, model_id = previous
            request_id = self.futures.request_id_for_retry(request_id, model_id, tenant)
            session.models_by_seq[model_seq_id] = (request_id, model_id)
            return request_id, model_id
        base_model = payload["base_model"]
        if base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        lora_config = payload.get("lora_config") or {}
        validate_model_config(lora_config, self.config)
        rank = lora_config.get("rank", 32)
        alpha = self.config.lora_alpha if self.config.lora_alpha is not None else float(2 * rank)
        if not self.free_slots:
            raise UserInputError(f"no free adapter slots (capacity {self.config.n_slots})")
        slot = min(self.free_slots)
        self.free_slots.remove(slot)

        model_id = f"model-{uuid.uuid4().hex[:12]}"
        future = self.futures.create(model_id, tenant)
        record = ModelRecord(
            model_id=model_id,
            session_id=payload["session_id"],
            tenant=tenant,
            slot=slot,
            base_model=base_model,
            lora_rank=rank,
            lora_alpha=alpha,
            create_request_id=future.request_id,
        )
        self.models[model_id] = record
        self.scheduler.add_model_queue(ModelRequestQueue(model_id, tenant, slot))
        task = asyncio.create_task(self._run_create_model(record))
        self._create_tasks.add(task)
        task.add_done_callback(self._create_tasks.discard)
        task.add_done_callback(self._observe_background_task)
        session.models_by_seq[model_seq_id] = (future.request_id, model_id)
        return future.request_id, model_id

    async def _run_create_model(self, record: ModelRecord) -> None:
        async with self._trainer_lock:
            if self.models.get(record.model_id) is not record:
                return
            failure = await self.backend.load_slot(record.slot, record.lora_rank, record.lora_alpha)
            record.slot_initialized = True
            if failure is not None:
                await self._close_model(record.model_id, failure["error"], "server")
                return
            self.futures.resolve(record.create_request_id, {"op": "create_model", "model_id": record.model_id})

    def get_model(self, tenant: str, model_id: str) -> ModelRecord:
        record = self.models.get(model_id)
        if record is None:
            reason = self._close_reasons.get(model_id)
            if reason is not None:
                raise UserInputError(f"model {model_id!r} was unloaded: {reason}")
            raise UserInputError(f"unknown model {model_id!r}")
        if record.tenant != tenant:
            raise OwnershipError(f"model {model_id} does not belong to this tenant")
        return record

    async def _close_model(self, model_id: str, error: str, category: str) -> None:
        """Free a model's slot and fail its pending requests; requires the trainer lock. Idempotent."""
        record = self.models.pop(model_id, None)
        if record is None:
            return
        self._close_reasons[model_id] = error
        while len(self._close_reasons) > 4 * self.config.n_slots:
            self._close_reasons.pop(next(iter(self._close_reasons)))
        self.scheduler.remove_model_queue(model_id)
        for request_id in [record.create_request_id, *record.request_id_by_seq.values()]:
            if self.futures.get(request_id, record.tenant) is not None:
                self.futures.fail(request_id, error, category)
        if await self.backend.trainer_dead():
            return
        if record.slot_initialized:
            failure = await self.backend.unload_slot(record.slot)
            if failure is not None:
                logger.error("slot %s remains unavailable after unload failed: %s", record.slot, failure["error"])
                return
        self.free_slots.add(record.slot)

    def submit(self, tenant: str, op: str, payload: dict) -> str:
        """Admit a decoded command; content errors settle its future as a user failure."""
        try:
            op = CommandOp(op)
        except ValueError:
            raise UserInputError(f"unknown command op {op!r}") from None
        model_id = payload["model_id"]
        record = self.get_model(tenant, model_id)
        seq_id = validate_seq_id(payload["seq_id"], "seq_id")
        model_queue = self.scheduler.model_queue(model_id)

        # retries must not accumulate gradients twice
        if seq_id in record.request_id_by_seq:
            request_id = self.futures.request_id_for_retry(record.request_id_by_seq[seq_id], model_id, tenant)
            record.request_id_by_seq[seq_id] = request_id
            return request_id

        future = self.futures.create(model_id, tenant)
        record.request_id_by_seq[seq_id] = future.request_id
        self._arrival_counter += 1
        validation_error = payload.get("validation_error")
        if validation_error is None:
            try:
                validate_batch_payload(op, payload, self.config)
            except UserInputError as error:
                validation_error = str(error)
        model_queue.submit(
            Command(
                model_id=model_id,
                seq_id=seq_id,
                op=op,
                payload={key: value for key, value in payload.items() if key != "validation_error"},
                request_id=future.request_id,
                arrival=self._arrival_counter,
                validation_error=validation_error,
            )
        )
        self._wake.set()
        return future.request_id

    def retrieve_future(self, tenant: str, request_id: str) -> RequestFuture | None:
        return self.futures.get(request_id, tenant)

    async def _run_batch(self, batch: BatchUnit) -> None:
        # slot-contiguous order; outputs come back aligned to it
        refs = sorted(batch.datums, key=lambda ref: ref.model_queue.slot)
        slot_datums = [(ref.model_queue.slot, ref.datum) for ref in refs]
        self._batch_counter += 1
        forward = (
            self.backend.forward_backward if batch.op == CommandOp.FORWARD_BACKWARD else self.backend.forward_only
        )
        try:
            outputs = await forward(self._batch_counter, slot_datums, batch.loss_fn, batch.loss_fn_config)
        except UserInputError as error:
            await self._fail_batch(batch, str(error), "user")
            return
        if isinstance(outputs, dict) and "error" in outputs:
            await self._fail_batch(batch, outputs["error"], "server")
            return

        assert len(outputs) == len(refs), f"unit returned {len(outputs)} outputs for {len(refs)} datums"
        for ref, output in zip(refs, outputs, strict=True):
            request = ref.request
            if request.record_output(ref.local_index, output):
                await self._finish_request(
                    ref.model_queue, request, {"op": request.command.op, "outputs": request.outputs}
                )

    async def _fail_batch(self, batch: BatchUnit, error: str, category: str) -> None:
        requests = {ref.request.command.request_id: (ref.model_queue, ref.request) for ref in batch.datums}
        for model_queue, pending in requests.values():
            await self._finish_request(model_queue, pending, {"error": error, "error_category": category})

    async def _run_barrier(self, barrier: BarrierUnit) -> None:
        try:
            outcomes = await self._dispatch_barrier_op(barrier)
        except (UserInputError, OwnershipError) as error:
            outcomes = [{"error": str(error), "error_category": "user"} for _ in barrier.entries]
        for (model_queue, pending), outcome in zip(barrier.entries, outcomes, strict=True):
            await self._finish_request(model_queue, pending, outcome)

    async def _dispatch_barrier_op(self, barrier: BarrierUnit) -> list[dict]:
        """Return one result or error per entry; only the caller settles futures and retires models."""
        if barrier.op == CommandOp.OPTIM_STEP:
            return await self._step_optimizers(barrier.entries)
        ((model_queue, pending),) = barrier.entries  # every other barrier is single-entry
        record = self.models[model_queue.model_id]
        payload = pending.command.payload
        if barrier.op == CommandOp.SAVE_STATE:
            return [await self._save_state(record, payload)]
        if barrier.op == CommandOp.LOAD_STATE:
            return [await self._load_state(record, payload)]
        if barrier.op == CommandOp.SAVE_WEIGHTS_FOR_SAMPLER:
            return [await self._save_weights_for_sampler(record, payload)]
        if barrier.op == CommandOp.SAVE_WEIGHTS_EXTERNAL:
            return [await self._save_weights_external(record, payload)]
        raise UserInputError(f"unknown barrier op {barrier.op!r}")

    async def _step_optimizers(self, entries: list) -> list[dict]:
        adam_params_by_slot = {
            model_queue.slot: pending.command.payload["adam_params"] for model_queue, pending in entries
        }
        slot_outcomes = await self.backend.optim_step(adam_params_by_slot)
        outcomes = []
        for model_queue, _ in entries:
            outcome = slot_outcomes[model_queue.slot]
            if "error" in outcome:
                outcomes.append(outcome)
            else:
                outcomes.append({"op": "optim_step", "metrics": {key: float(value) for key, value in outcome.items()}})
        return outcomes

    async def _finish_request(self, model_queue, pending, outcome: dict) -> None:
        if "error" in outcome:
            category = outcome.get("error_category", "server")
            if pending.command.op.requires_model_close_on_failure():
                await self._close_model(
                    model_queue.model_id,
                    f"model training failed ({outcome['error']}); create a new model and restore from a checkpoint",
                    category,
                )
                return
            self.futures.fail(pending.command.request_id, outcome["error"], category)
        else:
            self.futures.resolve(pending.command.request_id, outcome)
        model_queue.finish(pending)

    async def _save_state(self, record: ModelRecord, payload: dict) -> dict:
        """Save parameters and optimizer state; call after optim_step to persist accumulated training work."""
        name = payload["name"] or f"checkpoint-{payload['seq_id']:06d}"
        validate_checkpoint_segment(name)
        checkpoint_dir = resolve_checkpoint_dir(self.config.checkpoint_root, record.model_id, "weights", name)
        if not payload["overwrite"] and os.path.lexists(checkpoint_dir):
            raise UserInputError(f"checkpoint {name!r} already exists; pass overwrite=True to replace it")
        root = os.path.dirname(checkpoint_dir)
        written = await self.backend.save_slot(
            slot=record.slot,
            root=root,
            name=name,
            tenant_digest=tenant_digest(record.tenant),
            parametrization_fields=build_parametrization_fields(record, self.config),
        )
        if isinstance(written, dict):
            return written
        tinker_path = f"tinker://{record.model_id}/weights/{name}"
        if payload["overwrite"]:
            # a crash between the commit and the new TTL must not leave the new commit the old one's expiry
            self.expiry.drop(tinker_path)
        self.backend.commit_adapter(root, name, written, payload["overwrite"])
        self._delete_retired_outside_the_lock(root)
        self.expiry.set(tinker_path, payload["ttl_seconds"], time.time())
        return {"op": "save_state", "path": tinker_path}

    def _delete_retired_outside_the_lock(self, root: str) -> None:
        task = asyncio.create_task(asyncio.to_thread(self.backend.delete_retired_adapters, root))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def _load_state(self, record: ModelRecord, payload: dict) -> dict:
        if payload["path"].startswith("file://"):
            return await self._load_imported_adapter(record, payload)
        source_id, kind, name = parse_tinker_path(payload["path"])
        if kind != "weights":
            raise UserInputError(f"cannot load {kind} into a training model; use a save_state checkpoint")
        checkpoint_dir = resolve_checkpoint_dir(self.config.checkpoint_root, source_id, kind, name)
        source_tenant = payload.get("weights_access_token")
        if source_tenant is None:
            source_tenant = record.tenant
        committed = self.backend.resolve_adapter(os.path.dirname(checkpoint_dir), name, source_tenant, source_id)
        if committed is None:
            # saved before commits: torch_dist with META.json, which the trainer's legacy loader reads
            source, ckpt_path = (
                read_checkpoint_metadata(checkpoint_dir, source_tenant, payload["path"]),
                checkpoint_dir,
            )
        else:
            source, ckpt_path = committed["parametrization"], committed["path"]
        # the trainer checks the rest of the parametrization against the model
        for key, expected in (("base_model", record.base_model), ("lora_rank", record.lora_rank)):
            if source[key] != expected:
                raise UserInputError(
                    f"checkpoint {payload['path']!r} has {key}={source[key]!r}; this model has {expected!r}"
                )
        fields = build_parametrization_fields(record, self.config) | {"lora_alpha": source["lora_alpha"]}
        failure = await self.backend.load_slot(
            record.slot,
            record.lora_rank,
            source["lora_alpha"],
            ckpt_path=os.path.realpath(ckpt_path),
            load_optimizer=payload["optimizer"],
            parametrization_fields=fields,
        )
        if failure is not None:
            return failure
        # the tenant record owns alpha; the SDK cannot set it, so a loaded adapter keeps its own
        record.lora_alpha = source["lora_alpha"]
        return {"op": "load_state"}

    async def _save_weights_for_sampler(self, record: ModelRecord, payload: dict) -> dict:
        version = payload.get("sampler_path")
        if version is None:
            version = str(record.next_sampler_version)
            record.next_sampler_version += 1
        else:
            validate_checkpoint_segment(version)
        path = resolve_checkpoint_dir(self.config.checkpoint_root, record.model_id, "sampler_weights", version)
        if os.path.lexists(path):
            raise UserInputError(f"sampler weights {version!r} already exist; save under a new name")
        root = os.path.dirname(path)
        # the tenant's key stays on the gateway; the trainer ranks see only its digest
        written = await self.backend.export_slot(
            slot=record.slot,
            root=root,
            name=version,
            tenant_digest=tenant_digest(record.tenant),
            parametrization_fields=build_parametrization_fields(record, self.config),
        )
        if isinstance(written, dict):
            return written
        self.backend.commit_adapter(root, version, written, False)
        result = {
            "op": "save_weights_for_sampler",
            "path": f"tinker://{record.model_id}/sampler_weights/{version}",
        }
        self.expiry.set(result["path"], payload["ttl_seconds"], time.time())
        if payload.get("sampler_path") is None:
            # unnamed saves return a sampling session bound to the new version
            result["sampling_session_id"] = self._new_sampling_session(
                record.tenant, record.session_id, result["path"]
            )
        return result

    async def _load_imported_adapter(self, record: ModelRecord, payload: dict) -> dict:
        """An adapter from under --tinker-import-root; it has no Adam state this gateway could resume."""
        if payload["optimizer"]:
            raise UserInputError(f"{payload['path']} loads weights only; load it without optimizer state")
        path, source = self._read_imported_adapter(payload["path"])
        if source["lora_alpha"] is None:
            raise UserInputError(f"{payload['path']} records no LoRA alpha, which a Tinker adapter must carry")
        if source["lora_rank"] != record.lora_rank:
            raise UserInputError(
                f"{payload['path']} has lora_rank={source['lora_rank']}; this model has {record.lora_rank}"
            )
        fields = build_parametrization_fields(record, self.config) | {"lora_alpha": source["lora_alpha"]}
        failure = await self.backend.load_slot(
            record.slot,
            record.lora_rank,
            source["lora_alpha"],
            ckpt_path=path,
            load_optimizer=False,
            parametrization_fields=fields,
        )
        if failure is not None:
            return failure
        record.lora_alpha = source["lora_alpha"]
        return {"op": "load_state"}

    def _read_imported_adapter(self, uri: str) -> tuple[str, dict]:
        path = self._imported_path(uri)
        source = self.backend.read_imported_adapter(path)
        # a source that records no base (official Tinker exports write null) is taken as trained on this one
        if source["base_model"] is not None and not _is_same_base_model(source["base_model"], self.config.base_model):
            raise UserInputError(
                f"{uri} was trained on {source['base_model']!r}; this gateway serves {self.config.base_model!r}"
            )
        return path, source

    def _imported_path(self, uri: str) -> str:
        if self.config.import_root is None:
            raise UserInputError("this gateway imports no file:// adapters; start it with --tinker-import-root")
        root = os.path.realpath(self.config.import_root)
        path = os.path.realpath(uri.removeprefix("file://"))
        if not path.startswith(root + os.sep):
            raise UserInputError(f"{uri} is outside the import root {self.config.import_root}")
        return path

    async def _save_weights_external(self, record: ModelRecord, payload: dict) -> dict:
        """The official Tinker export of the adapter, for download (get_external_weights_urls)."""
        name = payload["name"]
        validate_checkpoint_segment(name)
        path = resolve_checkpoint_dir(self.config.checkpoint_root, record.model_id, "external_weights", name)
        if os.path.lexists(path):
            raise UserInputError(f"external weights {name!r} already exist; save under a new name")
        root = os.path.dirname(path)
        written = await self.backend.export_external(
            slot=record.slot,
            root=root,
            name=name,
            tenant_digest=tenant_digest(record.tenant),
            parametrization_fields=build_parametrization_fields(record, self.config),
            age_recipients=payload["age_recipients"],
        )
        if isinstance(written, dict):
            return written
        self.backend.commit_adapter(root, name, written, False)
        tinker_path = f"tinker://{record.model_id}/external_weights/{name}"
        self.expiry.set(tinker_path, payload["ttl_seconds"], time.time())
        files = self.backend.resolve_checkpoint_files(root, name, record.tenant)
        size = sum(os.path.getsize(os.path.join(files["path"], f)) for f in files["files"])
        return {"op": "save_weights_external", "path": tinker_path, "size_bytes": size}

    def _owned_checkpoint(self, tenant: str, model_id: str, checkpoint_id: str) -> tuple[str, str, str, dict]:
        """(tinker path, its adapter root, name, its files) of a committed checkpoint ``tenant`` owns."""
        tinker_path = f"tinker://{model_id}/{checkpoint_id}"
        _, kind, name = parse_tinker_path(tinker_path)
        root = os.path.dirname(resolve_checkpoint_dir(self.config.checkpoint_root, model_id, kind, name))
        files = self.backend.resolve_checkpoint_files(root, name, tenant)
        if files is None:
            raise NotFoundError(f"checkpoint {tinker_path} does not exist")
        return tinker_path, root, name, files

    def external_weights_urls(self, tenant: str, model_id: str, checkpoint_id: str, base_url: str) -> dict:
        """Download URLs that carry no credential, bound to this tenant, the checkpoint and an expiry."""
        tinker_path, _, _, files = self._owned_checkpoint(tenant, model_id, checkpoint_id)
        if not checkpoint_id.startswith("external_weights/"):
            raise UserInputError(f"{tinker_path} is not an external_weights checkpoint")
        now = time.time()
        expires_at = now + _DOWNLOAD_URL_TTL_S
        if (checkpoint_expires_at := self.expiry.expires_at(tinker_path)) is not None:
            # expired, and only not yet swept
            if checkpoint_expires_at <= now:
                raise NotFoundError(f"checkpoint {tinker_path} has expired")
            expires_at = min(expires_at, checkpoint_expires_at)
        self._downloads = {t: d for t, d in self._downloads.items() if d.expires_at > now}
        token = secrets.token_urlsafe(32)
        self._downloads[token] = _Download(tenant, tinker_path, files["commit_id"], expires_at)
        return {
            "urls": {f: f"{base_url}api/v1/external_weights/{token}/{f}" for f in files["files"]},
            "expires": datetime.fromtimestamp(expires_at, UTC).isoformat(),
        }

    def download_path(self, token: str, filename: str) -> str:
        """The file a download URL names; gone once the URL expires or the checkpoint is deleted."""
        download = self._downloads.get(token)
        if download is None or download.expires_at <= time.time():
            raise NotFoundError("unknown or expired download URL")
        model_id, kind, name = parse_tinker_path(download.tinker_path)
        try:
            _, _, _, files = self._owned_checkpoint(download.tenant, model_id, f"{kind}/{name}")
        except NotFoundError:
            raise NotFoundError(f"{download.tinker_path} was deleted") from None
        if files["commit_id"] != download.commit_id:
            raise NotFoundError(f"{download.tinker_path} was deleted and saved again; sign a new URL")
        if filename not in files["files"]:
            raise NotFoundError(f"{download.tinker_path} has no file {filename!r}")
        return os.path.join(files["path"], filename)

    async def set_checkpoint_ttl(self, tenant: str, model_id: str, checkpoint_id: str, ttl_seconds) -> None:
        validate_ttl_seconds(ttl_seconds)
        async with self._trainer_lock:
            tinker_path, _, _, _ = self._owned_checkpoint(tenant, model_id, checkpoint_id)
            self.expiry.set(tinker_path, ttl_seconds, time.time())

    async def delete_checkpoint(self, tenant: str, model_id: str, checkpoint_id: str) -> None:
        async with self._trainer_lock:
            tinker_path, root, name, _ = self._owned_checkpoint(tenant, model_id, checkpoint_id)
            self._delete_checkpoint(tinker_path, root, name, tenant)

    def _delete_checkpoint(self, tinker_path: str, root: str, name: str, tenant: str | None) -> None:
        """Requires the trainer lock; ``tenant`` None skips the ownership check (expiry)."""
        self.backend.delete_adapter(root, name, tenant)
        self.expiry.drop(tinker_path)
        self._delete_retired_outside_the_lock(root)
        logger.info(f"deleted checkpoint {tinker_path}")

    async def _run_expiry_sweeper(self) -> None:
        while True:
            await asyncio.sleep(_EXPIRY_SWEEP_S)
            await self._expire_checkpoints()

    async def _expire_checkpoints(self) -> None:
        async with self._trainer_lock:
            for tinker_path in self.expiry.expired(time.time()):
                model_id, kind, name = parse_tinker_path(tinker_path)
                root = os.path.dirname(resolve_checkpoint_dir(self.config.checkpoint_root, model_id, kind, name))
                if self.backend.resolve_checkpoint_files(root, name, None) is None:
                    # deleted by hand, or never committed: nothing to remove
                    self.expiry.drop(tinker_path)
                else:
                    self._delete_checkpoint(tinker_path, root, name, None)

    def weights_info(self, tenant: str, tinker_path: str) -> dict:
        """What the SDK needs to rebuild a training client from a checkpoint."""
        if tinker_path.startswith("file://"):
            _, source = self._read_imported_adapter(tinker_path)
            return {
                "base_model": self.config.base_model,
                "is_lora": True,
                "lora_rank": source["lora_rank"],
                "train_attn": self.config.trains_attn,
                "train_mlp": self.config.trains_mlp,
                "train_unembed": self.config.trains_unembed,
            }
        model_id, kind, name = parse_tinker_path(tinker_path)
        if kind == "external_weights":
            raise UserInputError(f"{tinker_path} is a download; create a training client from a weights checkpoint")
        checkpoint_dir = resolve_checkpoint_dir(self.config.checkpoint_root, model_id, kind, name)
        if os.path.islink(checkpoint_dir):
            parametrization = self.backend.resolve_adapter(os.path.dirname(checkpoint_dir), name, tenant, model_id)[
                "parametrization"
            ]
            # an adapter dir records targets, not group flags; it was written under this gateway's layout
            return {
                "base_model": parametrization["base_model"],
                "is_lora": True,
                "lora_rank": parametrization["lora_rank"],
                "train_attn": self.config.trains_attn,
                "train_mlp": self.config.trains_mlp,
                "train_unembed": self.config.trains_unembed,
            }
        meta = read_checkpoint_metadata(checkpoint_dir, tenant, tinker_path)
        return {
            "base_model": meta["base_model"],
            "is_lora": True,
            "lora_rank": meta["lora_rank"],
            "train_attn": meta["train_attn"],
            "train_mlp": meta["train_mlp"],
            "train_unembed": meta["train_unembed"],
        }

    def create_sampling_session(self, tenant: str, payload: dict) -> str:
        session = self._session_for(tenant, payload["session_id"])
        base_model = payload.get("base_model")
        if base_model is not None and base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        seq_id = validate_seq_id(payload["sampling_session_seq_id"], "sampling_session_seq_id", minimum=0)
        if (previous := session.sampling_sessions_by_seq.get(seq_id)) is not None:
            return previous
        sampling_session_id = self._new_sampling_session(tenant, payload["session_id"], payload.get("model_path"))
        session.sampling_sessions_by_seq[seq_id] = sampling_session_id
        return sampling_session_id

    def _new_sampling_session(self, tenant: str, session_id: str, model_path: str | None) -> str:
        if model_path is not None:
            resolve_sampler_checkpoint(self.config, tenant, model_path, self.backend.resolve_adapter)
        sampling_session_id = f"sampling-{uuid.uuid4().hex}"
        self.sampling_sessions[sampling_session_id] = SamplingSessionRecord(
            tenant=tenant, model_path=model_path, session_id=session_id
        )
        return sampling_session_id

    def get_sampler(self, tenant: str, sampling_session_id: str) -> dict:
        session = self.sampling_sessions.get(sampling_session_id)
        if session is None:
            raise UserInputError(f"unknown sampling session {sampling_session_id!r}")
        if session.tenant != tenant:
            raise OwnershipError("sampling session does not belong to this tenant")
        return {
            "sampler_id": sampling_session_id,
            "base_model": self.config.base_model,
            "model_path": session.model_path,
        }

    def submit_sample(self, tenant: str, payload: dict) -> tuple[str, list[str]]:
        base_model = payload.get("base_model")
        if base_model is not None and base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        model_path = payload.get("model_path")
        sampling_session = None
        if payload.get("sampling_session_id"):
            sampling_session_id = payload["sampling_session_id"]
            sampling_session = self.sampling_sessions.get(sampling_session_id)
            if sampling_session is None:
                raise UserInputError(
                    f"unknown sampling session {sampling_session_id!r}; create a sampling session first"
                )
            if sampling_session.tenant != tenant:
                raise OwnershipError("sampling session does not belong to this tenant")
            model_path = model_path or sampling_session.model_path
            seq_id = validate_seq_id(payload["seq_id"], "seq_id", minimum=0)
            if (previous := sampling_session.samples_by_seq.get(seq_id)) is not None:
                request_id, sequence_ids = previous
                request_id = self.futures.request_id_for_retry(request_id, model_path or "base", tenant)
                sampling_session.samples_by_seq[seq_id] = (request_id, sequence_ids)
                return request_id, sequence_ids
        validate_sample_payload(payload, self.config)
        params = payload["sampling_params"]
        # the shared sampling/training window takes precedence over the requested generation maximum
        payload = payload | {
            "sampling_params": params
            | {
                "max_tokens": min(
                    params["max_tokens"], self.config.max_tokens_per_sample - len(payload["prompt_tokens"])
                )
            }
        }
        lora_name, lora_path = (
            resolve_sampler_checkpoint(self.config, tenant, model_path, self.backend.resolve_adapter)
            if model_path
            else (None, None)
        )
        future = self.futures.create(model_path or "base", tenant)
        sequence_ids = [f"seq-{uuid.uuid4().hex}" for _ in range(payload.get("num_samples", 1))]
        task = asyncio.create_task(self._run_sample(future.request_id, sequence_ids, payload, lora_name, lora_path))
        session_id = sampling_session.session_id if sampling_session is not None else None
        self._sample_tasks[future.request_id] = (task, session_id)
        task.add_done_callback(lambda _t, rid=future.request_id: self._sample_tasks.pop(rid, None))
        task.add_done_callback(self._observe_background_task)
        if sampling_session is not None:
            sampling_session.samples_by_seq[seq_id] = (future.request_id, sequence_ids)
        return future.request_id, sequence_ids

    async def _run_sample(
        self,
        request_id: str,
        sequence_ids: list[str],
        payload: dict,
        lora_name: str | None,
        lora_path: str | None = None,
    ) -> None:
        try:
            result = await self.backend.sample(payload, lora_name, lora_path)
        except asyncio.CancelledError:
            self.futures.fail(request_id, "cancelled", "user")
        except UserInputError as error:
            self.futures.fail(request_id, str(error), "user")
        else:
            if "error" in result:
                self.futures.fail(request_id, result["error"], "server")
            else:
                for sequence_id, sequence in zip(sequence_ids, result["sequences"], strict=True):
                    sequence["sequence_id"] = sequence_id
                self.futures.resolve(request_id, {"op": "sample", **result})

    def cancel(self, tenant: str, request_id: str) -> None:
        """Cancel an in-flight sampling future; training commands have no
        cancel in the protocol and are ignored."""
        if self.futures.get(request_id, tenant) is None:
            return
        entry = self._sample_tasks.get(request_id)
        if entry is not None:
            entry[0].cancel()

    async def _run_lease_sweeper(self) -> None:
        """Reclaim from stale sessions: cancel sampling, unload models, free
        slots. Training state dies with the lease; only checkpoints survive."""
        while True:
            await asyncio.sleep(30)
            await self._expire_sessions()

    async def _expire_sessions(self) -> None:
        async with self._trainer_lock:
            now = time.monotonic()
            expired_sessions = {
                session_id
                for session_id, session in self.sessions.items()
                if now - session.last_heartbeat >= self.config.lease_timeout_s
            }
            await self._end_sessions(expired_sessions, "lease expired", lease_expired=True)
            # a client silent for longer than a lease would have lost the session anyway
            retention_s = max(self.config.lease_timeout_s, _SDK_FINISH_RETRY_S)
            for session_id, ended in list(self._ended_sessions.items()):
                if now - ended.ended_at >= retention_s:
                    del self._ended_sessions[session_id]

    async def finish_session(self, tenant: str, session_id: str, reason: str, detail: str | None) -> bool:
        """SDK close(): reclaim the session now instead of at lease expiry. False if its lease already expired.

        The first finish wins; a repeat (an SDK retry) is accepted without effect.
        """
        async with self._trainer_lock:
            ended = self._ended_sessions.get(session_id)
            if ended is not None and ended.tenant == tenant:
                return not ended.lease_expired
            self._session_for(tenant, session_id)
            logger.info(f"session {session_id} finished ({reason}): {detail}")
            await self._end_sessions({session_id}, f"session finished ({reason})", lease_expired=False)
            return True

    async def _end_sessions(self, session_ids: set[str], error: str, *, lease_expired: bool) -> None:
        """Drop sessions with their sampling sessions and models; requires the trainer lock."""
        now = time.monotonic()
        for session_id in session_ids:
            self._ended_sessions[session_id] = _EndedSession(self.sessions.pop(session_id).tenant, lease_expired, now)
        for sampling_session_id, record in list(self.sampling_sessions.items()):
            if record.session_id in session_ids:
                del self.sampling_sessions[sampling_session_id]

        for request_id, (task, session_id) in list(self._sample_tasks.items()):
            if session_id in session_ids:
                logger.warning(f"{error} for session of sample {request_id}; cancelling")
                self.futures.fail(request_id, error, "user")
                task.cancel()
        for model_id, record in list(self.models.items()):
            if record.session_id in session_ids:
                logger.warning(f"{error} for {model_id}; freeing slot {record.slot}")
                await self._close_model(model_id, error, "user")
