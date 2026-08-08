"""Public /v1 resource surface for the multi-LoRA controller.

Declarative resources over the existing control plane: datasets, evaluators,
post-training jobs (kind=RFT), and models (adapters). The gateway is a pure
driver over the backend's register/deregister verbs and read-only registry
views — it never reaches into ``AdapterRegistry`` internals or the
trainer/rollout-plane RPCs, so a future Tinker-style primitives API can mount
as a sibling driver over the same core.

Design notes (multi_lora_api_design.md):
- Flat ``/v1/`` routes, camelCase JSON, AIP verbs (``:cancel``/``:download``);
  resource ids share the registry charset so ``jobId`` IS the adapter name.
- Slot indices never appear in responses (slot *counts* only, in /v1/info).
- Token usage is exposed per job/model in the Tinker rate-class shape;
  counting only — no rates, no cost fields (billing is an external backend).
- The legacy ``/adapter_runs`` ops plane stays mounted; ops deletions are
  reported back via :meth:`V1Gateway.note_ops_delete` so job state renders
  CANCELLED with ``stopReason: OPS_CANCELLED``; registrations made behind the
  gateway's back appear as shadow ``kind: EXTERNAL`` jobs.
"""

import asyncio
import functools
import hashlib
import importlib
import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from miles.ray.multi_lora.registry import (
    LIVE_STATES,
    VALID_ADAPTER_NAME,
    AdapterState,
    SlotsFullError,
)
from miles.utils.adapter_config import AdapterRunConfig

logger = logging.getLogger(__name__)

# ``base`` is the router's alias for the frozen base model; the rest guard
# against unroutable/ambiguous resource names.
RESERVED_IDS = frozenset({"base", "default", "models", "datasets", "evaluators", "postTrainingJobs"})

_JOB_STATE_MAP = {
    AdapterState.PENDING: "PENDING",
    AdapterState.ACTIVE: "RUNNING",
    AdapterState.RETIRING: "STOPPING",
    AdapterState.CLEANUP: "STOPPING",
}


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


class GatewayError(Exception):
    """Typed /v1 error carrying the Google-style envelope fields.

    Deliberately not a ValueError/RuntimeError subclass: those have app-level
    legacy handlers with different response shapes."""

    def __init__(
        self,
        code: int,
        status: str,
        reason: str,
        message: str,
        *,
        retryable: bool = False,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.reason = reason
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after

    def to_response(self) -> JSONResponse:
        detail: dict[str, Any] = {"reason": self.reason}
        if self.retryable:
            detail["retryable"] = True
        body = {
            "error": {
                "code": self.code,
                "status": self.status,
                "requestId": new_request_id(),
                "message": self.message,
                "details": [detail],
            }
        }
        headers = {}
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        return JSONResponse(body, status_code=self.code, headers=headers)


def _validate_resource_id(kind: str, resource_id: str) -> None:
    if not VALID_ADAPTER_NAME.match(resource_id) or resource_id in (".", ".."):
        raise GatewayError(
            400,
            "INVALID_ARGUMENT",
            "INVALID_ARGUMENT",
            f"{kind} id '{resource_id}' is invalid: use only letters, digits, '.', '_' and '-'",
        )
    if resource_id in RESERVED_IDS:
        raise GatewayError(400, "INVALID_ARGUMENT", "RESERVED_NAME", f"{kind} id '{resource_id}' is reserved")


def _bare_id(name: str, collection: str) -> str:
    """Accept both canonical ('datasets/x') and bare ('x') resource names."""
    prefix = f"{collection}/"
    return name[len(prefix) :] if name.startswith(prefix) else name


def usage_to_wire(usage: dict[str, Any], uid: str) -> dict[str, Any]:
    """Registry usage counters -> the public §4.4 shape. Counting only; the
    totals must never be rate-multiplied (classes differ in unit cost)."""
    rollout = {
        "prefillTokens": usage.get("prefill_tokens", 0),
        "cachedPrefillTokens": usage.get("cached_prefill_tokens", 0),
        "sampleTokens": usage.get("sample_tokens", 0),
        "scoringPrefillTokens": usage.get("scoring_prefill_tokens", 0),
        "detail": {
            "sampleTokensTrained": usage.get("sample_tokens_trained", 0),
            "sampleTokensDroppedStale": usage.get("sample_tokens_dropped_stale", 0),
            "sampleTokensDroppedFilter": usage.get("sample_tokens_dropped_filter", 0),
            "sampleTokensAborted": usage.get("sample_tokens_aborted", 0),
            "sampleTokensDroppedRetired": usage.get("sample_tokens_dropped_retired", 0),
        },
    }
    training = {
        "trainTokens": usage.get("train_tokens", 0),
        "trainForwardTokens": usage.get("train_forward_tokens", 0),
        "optimizerSteps": usage.get("optimizer_steps", 0),
    }
    # External OpenAI-passthrough metering lands here in P2; zeros until then.
    inference = {"prefillTokens": 0, "cachedPrefillTokens": 0, "sampleTokens": 0}
    rollout_total = (
        rollout["prefillTokens"]
        + rollout["cachedPrefillTokens"]
        + rollout["sampleTokens"]
        + rollout["scoringPrefillTokens"]
    )
    training_total = training["trainTokens"] + training["trainForwardTokens"]
    inference_total = inference["prefillTokens"] + inference["cachedPrefillTokens"] + inference["sampleTokens"]
    return {
        "meterVersion": usage.get("meter_version", 1),
        "uid": uid,
        "asOf": iso_now(),
        "rollout": rollout,
        "training": training,
        "inference": inference,
        "totals": {
            "rolloutTokens": rollout_total,
            "trainingTokens": training_total,
            "inferenceTokens": inference_total,
            "computedTokens": rollout_total + training_total + inference_total,
        },
    }


@dataclass
class DatasetRecord:
    dataset_id: str
    uid: str
    display_name: str
    cluster_path: str
    fmt: str
    input_key: str
    label_key: str | None
    metadata_key: str | None
    create_time: str

    def to_json(self) -> dict:
        return {
            "name": f"datasets/{self.dataset_id}",
            "uid": self.uid,
            "displayName": self.display_name,
            "state": "READY",
            "source": {"clusterPath": self.cluster_path},
            "format": self.fmt,
            "schema": {"inputKey": self.input_key, "labelKey": self.label_key, "metadataKey": self.metadata_key},
            "exampleCount": None,
            "createTime": self.create_time,
        }


@dataclass
class EvaluatorRecord:
    evaluator_id: str
    uid: str
    display_name: str
    kind: str  # BUILTIN | PYTHON_PATH
    rm_type: str | None
    entrypoint: str | None
    create_time: str

    def to_json(self) -> dict:
        body: dict[str, Any] = {
            "name": f"evaluators/{self.evaluator_id}",
            "uid": self.uid,
            "displayName": self.display_name,
            "kind": self.kind,
            "state": "READY",
            "createTime": self.create_time,
        }
        if self.kind == "BUILTIN":
            body["builtin"] = {"rmType": self.rm_type}
        else:
            body["pythonPath"] = {"entrypoint": self.entrypoint}
        return body


@dataclass
class JobRecord:
    job_id: str
    uid: str  # registration_id
    kind: str
    dataset: str  # canonical name
    dataset_uid: str
    evaluator: str | None
    output_model: str
    training_config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    create_time: str = ""
    stop_reason: str | None = None
    version_at_registration: int = 0
    payload_hash: str = ""


@dataclass
class ModelRecord:
    model_id: str
    job_id: str
    uid: str
    save_path: str
    lora_rank: int | None
    lora_alpha: int | None
    create_time: str


class V1Gateway:
    """The /v1 job layer. State: three resource tables plus a model index;
    job/model *status* is always derived live from the registry."""

    def __init__(self, backend) -> None:
        self.backend = backend
        self.datasets: dict[str, DatasetRecord] = {}
        self.evaluators: dict[str, EvaluatorRecord] = {}
        self.jobs: dict[str, JobRecord] = {}
        self.models: dict[str, ModelRecord] = {}
        # jobId -> stored create response, replayed for byte-identical retries.
        self._create_responses: dict[str, dict] = {}
        # jobIds with a create in flight (an async validator can suspend
        # between the table check and the insert).
        self._creating: set[str] = set()

    # ------------------------------------------------------------------ app

    def add_routes(self, app: FastAPI) -> None:
        app.add_exception_handler(GatewayError, self._gateway_error_handler)
        wrap = self._enveloped
        app.get("/v1/info")(wrap(self.get_info))
        app.post("/v1/datasets")(wrap(self.create_dataset))
        app.get("/v1/datasets")(wrap(self.list_datasets))
        app.get("/v1/datasets/{dataset_id}")(wrap(self.get_dataset))
        app.delete("/v1/datasets/{dataset_id}")(wrap(self.delete_dataset))
        app.post("/v1/evaluators")(wrap(self.create_evaluator))
        app.get("/v1/evaluators")(wrap(self.list_evaluators))
        app.get("/v1/evaluators/{evaluator_id}")(wrap(self.get_evaluator))
        app.delete("/v1/evaluators/{evaluator_id}")(wrap(self.delete_evaluator))
        app.post("/v1/postTrainingJobs")(wrap(self.create_job))
        app.get("/v1/postTrainingJobs")(wrap(self.list_jobs))
        app.get("/v1/postTrainingJobs:batchGetState")(wrap(self.batch_get_state))
        app.post("/v1/postTrainingJobs/{job_id}:cancel")(wrap(self.cancel_job))
        app.get("/v1/postTrainingJobs/{job_id}")(wrap(self.get_job))
        app.get("/v1/postTrainingJobs/{job_id}/usage")(wrap(self.get_job_usage))
        app.delete("/v1/postTrainingJobs/{job_id}")(wrap(self.delete_job))
        app.get("/v1/models")(wrap(self.list_models))
        app.get("/v1/models/{model_id}:download")(wrap(self.download_model))
        app.get("/v1/models/{model_id}")(wrap(self.get_model))
        app.get("/v1/models/{model_id}/usage")(wrap(self.get_model_usage))
        app.delete("/v1/models/{model_id}")(wrap(self.delete_model))
        app.get("/v1/usage")(wrap(self.list_usage))
        app.get("/ui")(self.serve_console)  # not wrapped: returns a file, not a JSON body

    @staticmethod
    async def _gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        return exc.to_response()

    async def serve_console(self) -> FileResponse:
        """Single-file console UI over the /v1 API (self-contained; no build
        step, no external assets — works over a bare SSH port-forward). Lives
        at the repo root (<repo>/ui/console.html), outside the python package,
        so it can be edited/replaced without touching the package."""
        console = Path(__file__).resolve().parents[3] / "ui" / "console.html"
        if not console.exists():
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"console not found at {console}")
        # no-cache: the console is edited live; a reload must always fetch it fresh
        return FileResponse(console, media_type="text/html", headers={"Cache-Control": "no-cache"})

    @staticmethod
    def _enveloped(handler):
        """Every /v1 error leaves through the §3.6 envelope: stray ValueErrors
        would otherwise hit the app-level legacy handlers ({"detail": ...},
        no requestId) registered for the /adapter_runs ops plane."""

        @functools.wraps(handler)
        async def wrapped(*args, **kwargs):
            try:
                return await handler(*args, **kwargs)
            except GatewayError:
                raise
            except json.JSONDecodeError as e:
                raise GatewayError(400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", f"invalid JSON body: {e}") from None
            except (ValueError, TypeError) as e:
                raise GatewayError(400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", str(e)) from None
            except Exception as e:
                logger.exception("unhandled error in /v1 gateway handler")
                raise GatewayError(500, "INTERNAL", "INTERNAL", str(e)) from None

        return wrapped

    # ----------------------------------------------------------------- info

    async def get_info(self) -> dict:
        args = self.backend.args
        registry = self.backend.registry
        target_modules = getattr(args, "target_modules", None)
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        return {
            "requestId": new_request_id(),
            "baseModel": getattr(args, "hf_checkpoint", None),
            "kind": "RFT",
            "maxLoraRank": getattr(args, "lora_rank", None),
            "targetModules": target_modules,
            "slots": {"total": registry.max_adapters, "free": len(registry.free_slots)},
            "limits": {
                "maxAdapterGlobalBatchSize": getattr(args, "multi_lora_max_adapter_global_batch_size", None),
                "dpSize": getattr(args, "multi_lora_dp_size", None),
                "maxWeightStaleness": getattr(args, "max_weight_staleness", None),
            },
        }

    # ------------------------------------------------------------- datasets

    async def create_dataset(self, request: Request) -> dict:
        payload = await request.json()
        dataset_id = payload.get("datasetId")
        if not dataset_id:
            raise GatewayError(400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", "datasetId is required")
        _validate_resource_id("dataset", dataset_id)
        if dataset_id in self.datasets:
            raise GatewayError(409, "ALREADY_EXISTS", "DATASET_EXISTS", f"Dataset '{dataset_id}' already exists")
        source = payload.get("source") or {}
        cluster_path = source.get("clusterPath")
        if not cluster_path:
            raise GatewayError(
                400,
                "INVALID_ARGUMENT",
                "INVALID_ARGUMENT",
                "source.clusterPath is required in v0 (the upload flow is not built yet)",
            )
        path = Path(cluster_path).expanduser()
        if not await asyncio.to_thread(path.exists):  # possibly slow network fs; keep the actor loop free
            raise GatewayError(
                400,
                "INVALID_ARGUMENT",
                "INVALID_ARGUMENT",
                f"source.clusterPath '{cluster_path}' does not exist "
                "(checked from the controller process on the head node)",
            )
        schema = payload.get("schema") or {}
        record = DatasetRecord(
            dataset_id=dataset_id,
            uid=uuid.uuid4().hex,
            display_name=payload.get("displayName", ""),
            cluster_path=str(path),
            fmt=payload.get("format") or (path.suffix.lstrip(".") or "unknown"),
            input_key=schema.get("inputKey", "text"),
            label_key=schema.get("labelKey"),
            metadata_key=schema.get("metadataKey"),
            create_time=iso_now(),
        )
        self.datasets[dataset_id] = record
        return {"requestId": new_request_id(), **record.to_json()}

    async def list_datasets(self) -> dict:
        return {
            "requestId": new_request_id(),
            "datasets": [record.to_json() for record in self.datasets.values()],
        }

    async def get_dataset(self, dataset_id: str) -> dict:
        record = self.datasets.get(dataset_id)
        if record is None:
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Dataset '{dataset_id}' not found")
        return {"requestId": new_request_id(), **record.to_json()}

    async def delete_dataset(self, dataset_id: str) -> dict:
        record = self.datasets.get(dataset_id)
        if record is None:
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Dataset '{dataset_id}' not found")
        for job in self.jobs.values():
            if job.dataset == f"datasets/{dataset_id}" and self._job_is_live(job):
                raise GatewayError(
                    409,
                    "FAILED_PRECONDITION",
                    "RESOURCE_IN_USE",
                    f"Dataset '{dataset_id}' is referenced by non-terminal job '{job.job_id}'",
                )
        self.datasets.pop(dataset_id)
        return {"requestId": new_request_id(), "deleted": True, "name": f"datasets/{dataset_id}"}

    # ----------------------------------------------------------- evaluators

    async def create_evaluator(self, request: Request) -> dict:
        payload = await request.json()
        evaluator_id = payload.get("evaluatorId")
        if not evaluator_id:
            raise GatewayError(400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", "evaluatorId is required")
        _validate_resource_id("evaluator", evaluator_id)
        if evaluator_id in self.evaluators:
            raise GatewayError(
                409, "ALREADY_EXISTS", "EVALUATOR_EXISTS", f"Evaluator '{evaluator_id}' already exists"
            )
        kind = payload.get("kind")
        if kind == "BUILTIN":
            rm_type = (payload.get("builtin") or {}).get("rmType")
            if not (rm_type or "").strip():
                raise GatewayError(400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", "builtin.rmType is required")
            entrypoint = None
        elif kind == "PYTHON_PATH":
            rm_type = None
            entrypoint = (payload.get("pythonPath") or {}).get("entrypoint")
            if not (entrypoint or "").strip():
                raise GatewayError(400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", "pythonPath.entrypoint is required")
            await asyncio.to_thread(self._check_importable, entrypoint)  # imports can block on fs
        else:
            raise GatewayError(
                400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", "kind must be 'BUILTIN' or 'PYTHON_PATH'"
            )
        record = EvaluatorRecord(
            evaluator_id=evaluator_id,
            uid=uuid.uuid4().hex,
            display_name=payload.get("displayName", ""),
            kind=kind,
            rm_type=rm_type,
            entrypoint=entrypoint,
            create_time=iso_now(),
        )
        self.evaluators[evaluator_id] = record
        return {"requestId": new_request_id(), **record.to_json()}

    @staticmethod
    def _check_importable(entrypoint: str) -> None:
        """Fail evaluator creation, not a later rollout, on an unresolvable
        reward path. importlib directly: miles.utils.misc pulls in ray."""
        module_path, _, attr = entrypoint.rpartition(".")
        if not module_path:
            raise GatewayError(
                400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", f"entrypoint '{entrypoint}' is not a dotted path"
            )
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise GatewayError(
                400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", f"entrypoint module '{module_path}' is not importable: {e}"
            ) from None
        if not hasattr(module, attr):
            raise GatewayError(
                400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", f"module '{module_path}' has no attribute '{attr}'"
            )

    async def list_evaluators(self) -> dict:
        return {
            "requestId": new_request_id(),
            "evaluators": [record.to_json() for record in self.evaluators.values()],
        }

    async def get_evaluator(self, evaluator_id: str) -> dict:
        record = self.evaluators.get(evaluator_id)
        if record is None:
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Evaluator '{evaluator_id}' not found")
        return {"requestId": new_request_id(), **record.to_json()}

    async def delete_evaluator(self, evaluator_id: str) -> dict:
        record = self.evaluators.get(evaluator_id)
        if record is None:
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Evaluator '{evaluator_id}' not found")
        for job in self.jobs.values():
            if job.evaluator == f"evaluators/{evaluator_id}" and self._job_is_live(job):
                raise GatewayError(
                    409,
                    "FAILED_PRECONDITION",
                    "RESOURCE_IN_USE",
                    f"Evaluator '{evaluator_id}' is referenced by non-terminal job '{job.job_id}'",
                )
        self.evaluators.pop(evaluator_id)
        return {"requestId": new_request_id(), "deleted": True, "name": f"evaluators/{evaluator_id}"}

    # ----------------------------------------------------------------- jobs

    def build_adapter_config(
        self, payload: dict, dataset: DatasetRecord, evaluator: EvaluatorRecord | None, output_model: str
    ) -> AdapterRunConfig:
        """The single JSON -> AdapterRunConfig translation seam. Keep every
        field mapping here so the later AdapterSpec/RunSpec split (design §6
        rule 5) stays a mechanical refactor."""
        args = self.backend.args
        training = payload.get("trainingConfig") or {}
        overrides = payload.get("datasetOverrides") or {}
        save = str(Path(args.save) / "adapters" / output_model) if getattr(args, "save", None) else None
        rm_type = custom_rm_path = None
        if evaluator is not None:
            if evaluator.kind == "BUILTIN":
                rm_type = evaluator.rm_type
            else:
                custom_rm_path = evaluator.entrypoint
        return AdapterRunConfig(
            data=dataset.cluster_path,
            rank=training.get("loraRank"),
            alpha=training.get("loraAlpha"),
            rollout_batch_size=training.get("batchSizePrompts"),
            n_samples_per_prompt=training.get("rolloutsPerPrompt"),
            num_step=training.get("maxSteps"),
            num_epoch=training.get("epochs"),
            save=save,
            input_key=overrides.get("inputKey", dataset.input_key),
            label_key=overrides.get("labelKey", dataset.label_key),
            metadata_key=overrides.get("metadataKey", dataset.metadata_key),
            rm_type=rm_type,
            custom_rm_path=custom_rm_path,
            metadata=payload.get("metadata") or {},
        )

    def _check_cluster_default(self, field_name: str, provided: Any, arg_name: str) -> None:
        """Per-job values are v-next; accept them only when they equal the
        cluster default (keeps clients that always send them working), 400 on a
        real conflict."""
        if provided is None:
            return
        cluster = getattr(self.backend.args, arg_name, None)
        if cluster is None or float(provided) != float(cluster):
            raise GatewayError(
                400,
                "INVALID_ARGUMENT",
                "PER_JOB_VALUE_UNSUPPORTED",
                f"{field_name}={provided} differs from the cluster default ({arg_name}={cluster}); "
                "per-job values are not supported yet — omit the field or match the cluster default",
            )

    async def create_job(self, request: Request) -> dict:
        payload = await request.json()
        job_id = payload.get("jobId") or f"job-{uuid.uuid4().hex[:8]}"
        _validate_resource_id("job", job_id)
        payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

        if (existing := self.jobs.get(job_id)) is not None:
            if existing.payload_hash == payload_hash and job_id in self._create_responses:
                return self._create_responses[job_id]  # idempotent replay
            raise GatewayError(
                409,
                "ALREADY_EXISTS",
                "JOB_EXISTS",
                f"Job '{job_id}' already exists with a different payload; delete it or pick a new jobId",
            )

        if (kind := payload.get("kind", "RFT")) != "RFT":
            raise GatewayError(400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", f"kind must be 'RFT', got '{kind}'")
        if payload.get("warmStartFrom") is not None:
            raise GatewayError(
                400, "INVALID_ARGUMENT", "NOT_IMPLEMENTED", "warmStartFrom is not supported yet (planned, P3)"
            )

        dataset_name = payload.get("dataset")
        if not dataset_name:
            raise GatewayError(400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", "dataset is required")
        dataset = self.datasets.get(_bare_id(dataset_name, "datasets"))
        if dataset is None:
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Dataset '{dataset_name}' not found")

        evaluator = None
        if (evaluator_name := payload.get("evaluator")) is not None:
            evaluator = self.evaluators.get(_bare_id(evaluator_name, "evaluators"))
            if evaluator is None:
                raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Evaluator '{evaluator_name}' not found")

        training = payload.get("trainingConfig") or {}
        self._check_cluster_default("trainingConfig.learningRate", training.get("learningRate"), "lr")
        inference_params = payload.get("inferenceParameters") or {}
        self._check_cluster_default(
            "inferenceParameters.temperature", inference_params.get("temperature"), "rollout_temperature"
        )
        self._check_cluster_default("inferenceParameters.topP", inference_params.get("topP"), "rollout_top_p")
        self._check_cluster_default(
            "inferenceParameters.maxOutputTokens", inference_params.get("maxOutputTokens"), "rollout_max_response_len"
        )

        output_model = payload.get("outputModel") or job_id
        _validate_resource_id("model", output_model)
        if output_model in self.models:
            raise GatewayError(
                409,
                "ALREADY_EXISTS",
                "MODEL_EXISTS",
                f"Model '{output_model}' already exists; outputModel reuse is not supported — "
                "pick a new outputModel or DELETE /v1/models/{id} first",
            )

        config = self.build_adapter_config(payload, dataset, evaluator, output_model)
        # A leftover checkpoint dir would make the trainer silently warm-start
        # this "new" job from another registration's weights (and resume its
        # step count); require an explicit clean-up instead.
        if config.save is not None:
            checkpoint_dir = Path(config.save) / "checkpoints"
            if await asyncio.to_thread(lambda: checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir())):
                raise GatewayError(
                    409,
                    "FAILED_PRECONDITION",
                    "STALE_CHECKPOINTS",
                    f"'{checkpoint_dir}' already contains checkpoints from a previous run; "
                    "DELETE /v1/models/{id}?force=true or pick a new outputModel",
                )

        if job_id in self._creating:
            raise GatewayError(
                409, "ABORTED", "CREATE_IN_FLIGHT", f"Job '{job_id}' create is in flight; retry", retryable=True
            )
        self._creating.add(job_id)
        try:
            result = await self._register(job_id, config)
        finally:
            self._creating.discard(job_id)

        registry = self.backend.registry
        record = registry.records[job_id]
        job = JobRecord(
            job_id=job_id,
            uid=record.registration_id,
            kind="RFT",
            dataset=f"datasets/{dataset.dataset_id}",
            dataset_uid=dataset.uid,
            evaluator=f"evaluators/{evaluator.evaluator_id}" if evaluator is not None else None,
            output_model=output_model,
            training_config={
                "loraRank": record.config.rank,
                "loraAlpha": record.config.alpha,
                "batchSizePrompts": record.config.rollout_batch_size,
                "rolloutsPerPrompt": record.config.n_samples_per_prompt,
                "maxSteps": record.config.num_step,
                "epochs": record.config.num_epoch,
            },
            metadata=payload.get("metadata") or {},
            create_time=iso_now(),
            # Baseline read in-process at register time: a follow-up GET would
            # race the driver's reconcile+push and be off by one.
            version_at_registration=registry.slot_versions[result["slot"]],
            payload_hash=payload_hash,
        )
        self.jobs[job_id] = job
        self.models[output_model] = ModelRecord(
            model_id=output_model,
            job_id=job_id,
            uid=job.uid,
            save_path=str(record.config.save),
            lora_rank=record.config.rank,
            lora_alpha=record.config.alpha,
            create_time=job.create_time,
        )
        response = self.job_to_json(job)
        self._create_responses[job_id] = response
        return response

    async def _register(self, name: str, config: AdapterRunConfig) -> dict:
        try:
            return await self.backend.register(name, config)
        except SlotsFullError as e:
            raise GatewayError(
                429, "RESOURCE_EXHAUSTED", "SLOT_CAPACITY", str(e), retryable=True, retry_after=5
            ) from None
        except ValueError as e:
            message = str(e)
            if "still cleaning up" in message:
                raise GatewayError(
                    409, "ABORTED", "NAME_CLEANING_UP", message, retryable=True, retry_after=5
                ) from None
            if "already registered" in message:
                raise GatewayError(409, "ALREADY_EXISTS", "JOB_EXISTS", message) from None
            if "save dir" in message and "already used" in message:
                raise GatewayError(409, "ALREADY_EXISTS", "MODEL_EXISTS", message) from None
            raise GatewayError(400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", message) from None
        except RuntimeError as e:
            if "No free adapter slots" in str(e):
                raise GatewayError(
                    429, "RESOURCE_EXHAUSTED", "SLOT_CAPACITY", str(e), retryable=True, retry_after=5
                ) from None
            raise

    def _registry_record_for(self, job_id: str, uid: str):
        record = self.backend.registry.records.get(job_id)
        if record is not None and record.registration_id == uid:
            return record
        return None

    def _job_is_live(self, job: JobRecord) -> bool:
        record = self._registry_record_for(job.job_id, job.uid)
        return record is not None and record.state in LIVE_STATES

    def _job_state(self, job: JobRecord) -> tuple[str, str | None]:
        record = self._registry_record_for(job.job_id, job.uid)
        if record is not None and record.state in LIVE_STATES:
            return _JOB_STATE_MAP[record.state], job.stop_reason
        stop_reason = job.stop_reason or "MAX_STEPS_REACHED"
        state = "CANCELLED" if stop_reason in ("USER_CANCELLED", "OPS_CANCELLED") else "COMPLETED"
        return state, stop_reason

    def job_to_json(self, job: JobRecord) -> dict:
        registry = self.backend.registry
        state, stop_reason = self._job_state(job)
        record = self._registry_record_for(job.job_id, job.uid)
        # maxSteps may only be resolved from epochs after the data source
        # learns the dataset length; prefer the live registry value.
        max_steps = job.training_config.get("maxSteps")
        if record is not None and getattr(record.config, "num_step", None) is not None:
            max_steps = record.config.num_step
        progress: dict[str, Any] = {"maxSteps": max_steps}
        if record is not None:
            progress |= {
                "completedSteps": record.step - record.start_step,
                "accumulatedPrompts": record.accumulated_groups,
                # Slot versions keep advancing for the slot's NEXT tenant, so a
                # terminal job must not read them (version leakage across reuse).
                "policyVersion": (
                    registry.slot_versions[record.slot] - job.version_at_registration
                    if record.state in LIVE_STATES
                    else None
                ),
            }
        else:
            progress |= {"completedSteps": None, "accumulatedPrompts": None, "policyVersion": None}
        return {
            "requestId": new_request_id(),
            "name": f"postTrainingJobs/{job.job_id}",
            "uid": job.uid,
            "kind": job.kind,
            "state": state,
            "stopReason": stop_reason,
            "createTime": job.create_time,
            "dataset": job.dataset,
            "datasetUid": job.dataset_uid,
            "evaluator": job.evaluator,
            "outputModel": f"models/{job.output_model}",
            "warmStartFrom": None,
            "trainingConfig": job.training_config,
            "metadata": job.metadata,
            "jobProgress": progress,
            "usage": usage_to_wire(registry.usage_dict(job.uid), job.uid),
        }

    def _shadow_job_json(self, record) -> dict:
        """A registration made through the legacy ops plane: visible here so
        no slot consumer or meter is ever invisible to /v1 clients. Shape is
        identical to a regular job (null-filled) so typed clients can parse
        one list."""
        state = _JOB_STATE_MAP.get(record.state, "COMPLETED")
        return {
            "name": f"postTrainingJobs/{record.name}",
            "uid": record.registration_id,
            "kind": "EXTERNAL",
            "state": state,
            "stopReason": None,
            "createTime": None,
            "dataset": None,
            "datasetUid": None,
            "evaluator": None,
            "outputModel": None,
            "warmStartFrom": None,
            "trainingConfig": None,
            "metadata": {},
            "jobProgress": {
                "completedSteps": record.step - record.start_step,
                "maxSteps": getattr(record.config, "num_step", None),
                "accumulatedPrompts": record.accumulated_groups,
                "policyVersion": None,
            },
            "usage": usage_to_wire(self.backend.registry.usage_dict(record.registration_id), record.registration_id),
        }

    async def list_jobs(self, pageSize: int | None = None, pageToken: str | None = None, filter: str | None = None) -> dict:
        if pageToken:
            raise GatewayError(
                400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", "pagination is not supported in v0 (pageToken must be empty)"
            )
        if filter:
            # Silently returning an unfiltered list would misinform the client.
            raise GatewayError(400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", "filtering is not supported in v0")
        jobs = [self.job_to_json(job) for job in self.jobs.values()]
        known_uids = {job.uid for job in self.jobs.values()}
        for record in self.backend.registry.records.values():
            if record.registration_id not in known_uids:
                jobs.append(self._shadow_job_json(record))
        # v0 returns the full list; nextPageToken omitted so pagination loops terminate.
        return {"requestId": new_request_id(), "postTrainingJobs": jobs, "totalSize": len(jobs)}

    def _newer_registration(self, job_id: str):
        """The registry record for this name when it belongs to a NEWER
        registration than the gateway's job (e.g. an ops-plane re-register).
        Coexistence rule: reads by id resolve to the latest registration;
        the older one stays addressable by uid via /v1/usage."""
        job = self.jobs.get(job_id)
        record = self.backend.registry.records.get(job_id)
        if job is not None and record is not None and record.registration_id != job.uid:
            return record
        return None

    async def batch_get_state(self, request: Request) -> dict:
        names = request.query_params.getlist("names")
        states: dict[str, str | None] = {}
        for name in names:
            job = self.jobs.get(name)
            newer = self._newer_registration(name)
            if newer is not None:
                states[name] = _JOB_STATE_MAP.get(newer.state, "COMPLETED")
            elif job is not None:
                states[name], _ = self._job_state(job)
            elif (record := self.backend.registry.records.get(name)) is not None:
                states[name] = _JOB_STATE_MAP.get(record.state, "COMPLETED")
            else:
                states[name] = None
        return {"requestId": new_request_id(), "states": states}

    async def get_job(self, job_id: str) -> dict:
        if (newer := self._newer_registration(job_id)) is not None:
            return {"requestId": new_request_id(), **self._shadow_job_json(newer)}
        job = self.jobs.get(job_id)
        if job is None:
            if (record := self.backend.registry.records.get(job_id)) is not None:
                return {"requestId": new_request_id(), **self._shadow_job_json(record)}
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Job '{job_id}' not found")
        return self.job_to_json(job)

    async def get_job_usage(self, job_id: str) -> dict:
        registry = self.backend.registry
        if (newer := self._newer_registration(job_id)) is not None:
            uid = newer.registration_id
        elif (job := self.jobs.get(job_id)) is not None:
            uid = job.uid
        elif (record := registry.records.get(job_id)) is not None:
            uid = record.registration_id
        else:
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Job '{job_id}' not found")
        return {
            "requestId": new_request_id(),
            "resource": f"postTrainingJobs/{job_id}",
            "usage": usage_to_wire(registry.usage_dict(uid), uid),
        }

    async def cancel_job(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Job '{job_id}' not found")
        record = self._registry_record_for(job_id, job.uid)
        if record is not None and record.state in (AdapterState.PENDING, AdapterState.ACTIVE):
            # Only a PENDING/ACTIVE record can still be cancelled by us; a
            # RETIRING/CLEANUP record's stop cause was already decided (e.g.
            # max-steps auto-deregister) and must not be relabelled.
            if job.stop_reason is None:
                job.stop_reason = "USER_CANCELLED"
            await self.backend.deregister(job_id)
        # Idempotent: cancelling a stopping/terminal job returns it unchanged.
        return self.job_to_json(job)

    async def delete_job(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Job '{job_id}' not found")
        if self._job_is_live(job):
            raise GatewayError(
                409, "FAILED_PRECONDITION", "JOB_LIVE", f"Job '{job_id}' is not terminal; cancel it first"
            )
        self.jobs.pop(job_id)
        self._create_responses.pop(job_id, None)
        # The usage ledger is never deleted with the job: the journal and
        # GET /v1/usage keep answering for this uid.
        return {"requestId": new_request_id(), "deleted": True, "name": f"postTrainingJobs/{job_id}"}

    def note_ops_delete(self, name: str) -> None:
        job = self.jobs.get(name)
        if job is None or job.stop_reason is not None:
            return
        record = self._registry_record_for(name, job.uid)
        # Same rule as cancel_job: RETIRING/CLEANUP already has a decided stop
        # cause (a redundant ops delete must not relabel a max-steps finish).
        if record is not None and record.state in (AdapterState.PENDING, AdapterState.ACTIVE):
            job.stop_reason = "OPS_CANCELLED"

    # --------------------------------------------------------------- models

    def _model_checkpoints(self, model: ModelRecord) -> list[dict]:
        checkpoint_dir = Path(model.save_path) / "checkpoints"
        if not checkpoint_dir.is_dir():
            return []
        checkpoints = []
        for entry in checkpoint_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("step_"):
                try:
                    step = int(entry.name.removeprefix("step_"))
                except ValueError:
                    continue
                # A dir without the exported adapter (crash mid-save; the
                # writer publishes atomically via tmp-dir + os.replace) is not
                # a checkpoint: it must not flip the model READY or download
                # as an empty file set.
                if not (entry / "adapter_model.safetensors").exists():
                    continue
                checkpoints.append(
                    {
                        "checkpointId": entry.name,
                        "step": step,
                        "createTime": datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                    }
                )
        return sorted(checkpoints, key=lambda c: c["step"])

    def _model_json(self, model: ModelRecord) -> dict:
        registry = self.backend.registry
        record = self.backend.registry.records.get(model.job_id)
        live = record is not None and record.registration_id == model.uid and record.state in LIVE_STATES
        checkpoints = self._model_checkpoints(model)
        if live:
            state = "TRAINING"
        else:
            state = "READY" if checkpoints else "INCOMPLETE"
        servable = (
            record is not None
            and record.registration_id == model.uid
            and record.state in (AdapterState.ACTIVE, AdapterState.RETIRING)
        )
        args = self.backend.args
        target_modules = getattr(args, "target_modules", None)
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        return {
            "requestId": new_request_id(),
            "name": f"models/{model.model_id}",
            "uid": model.uid,
            "kind": "LORA_ADAPTER",
            "baseModel": getattr(args, "hf_checkpoint", None),
            "state": state,
            "servable": servable,
            "loraRank": model.lora_rank,
            "loraAlpha": model.lora_alpha,
            "targetModules": target_modules,
            "job": f"postTrainingJobs/{model.job_id}",
            "checkpoints": checkpoints,
            "usage": usage_to_wire(registry.usage_dict(model.uid), model.uid),
        }

    async def list_models(self) -> dict:
        return {"requestId": new_request_id(), "models": [self._model_json(m) for m in self.models.values()]}

    async def get_model(self, model_id: str) -> dict:
        model = self.models.get(model_id)
        if model is None:
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Model '{model_id}' not found")
        return self._model_json(model)

    async def get_model_usage(self, model_id: str) -> dict:
        model = self.models.get(model_id)
        if model is None:
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Model '{model_id}' not found")
        return {
            "requestId": new_request_id(),
            "resource": f"models/{model_id}",
            "usage": usage_to_wire(self.backend.registry.usage_dict(model.uid), model.uid),
        }

    async def download_model(self, model_id: str, checkpoint: str | None = None) -> dict:
        model = self.models.get(model_id)
        if model is None:
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Model '{model_id}' not found")
        checkpoints = self._model_checkpoints(model)
        if not checkpoints:
            raise GatewayError(
                404, "NOT_FOUND", "CHECKPOINT_NOT_FOUND", f"Model '{model_id}' has no checkpoints on disk"
            )
        if checkpoint is None:
            selected = checkpoints[-1]
        else:
            matches = [c for c in checkpoints if c["checkpointId"] == checkpoint]
            if not matches:
                raise GatewayError(
                    404, "NOT_FOUND", "CHECKPOINT_NOT_FOUND", f"Checkpoint '{checkpoint}' not found for '{model_id}'"
                )
            selected = matches[0]
        checkpoint_dir = Path(model.save_path) / "checkpoints" / selected["checkpointId"]
        files = {}
        for filename in ("adapter_model.safetensors", "adapter_config.json"):
            file_path = checkpoint_dir / filename
            if file_path.exists():
                # downloadUrl stays null in v0; the shape is upload-flow
                # symmetric so signed URLs can land without a shape change.
                files[filename] = {"clusterPath": str(file_path), "downloadUrl": None}
        return {
            "requestId": new_request_id(),
            "checkpointId": selected["checkpointId"],
            "files": files,
        }

    async def delete_model(self, model_id: str, force: bool = False) -> dict:
        model = self.models.get(model_id)
        if model is None:
            raise GatewayError(404, "NOT_FOUND", "NOT_FOUND", f"Model '{model_id}' not found")
        job = self.jobs.get(model.job_id)
        if job is not None and job.uid == model.uid and self._job_is_live(job):
            raise GatewayError(
                409, "FAILED_PRECONDITION", "JOB_LIVE", f"Model '{model_id}' is produced by live job '{model.job_id}'"
            )
        self.models.pop(model_id)
        if force:
            checkpoint_dir = Path(model.save_path) / "checkpoints"
            if checkpoint_dir.is_dir():
                # Off-loop: a large checkpoint tree would otherwise stall every
                # training-plane RPC sharing this actor's event loop.
                await asyncio.to_thread(shutil.rmtree, checkpoint_dir, ignore_errors=True)
        return {"requestId": new_request_id(), "deleted": True, "name": f"models/{model_id}"}

    # ---------------------------------------------------------------- usage

    async def list_usage(self, uid: str | None = None) -> dict:
        jobs_by_uid = {job.uid: job.job_id for job in self.jobs.values()}
        entries = []
        for entry in self.backend.registry.usage_entries(uid):
            job_id = jobs_by_uid.get(entry["registration_id"])
            entries.append(
                {
                    "job": f"postTrainingJobs/{job_id}" if job_id else None,
                    "name": entry["name"],
                    "uid": entry["registration_id"],
                    "finalized": entry["finalized"],
                    "usage": usage_to_wire(entry["usage"], entry["registration_id"]),
                }
            )
        return {"requestId": new_request_id(), "entries": entries}
