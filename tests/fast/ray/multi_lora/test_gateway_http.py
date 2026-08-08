"""HTTP tests for the /v1 gateway (declarative resource surface) with a
mock router (no Ray, no SGLang)."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

from miles.ray.multi_lora.backend import MultiLoRABackend
from miles.ray.multi_lora.http_server import MultiLoRAHTTPServer
from miles.utils.token_usage import ROLLOUT_FIELDS

DATA_FILE = __file__


class GatewayHarness:
    def __init__(self, session: aiohttp.ClientSession, backend: MultiLoRABackend, srv: MultiLoRAHTTPServer):
        self.session = session
        self.backend = backend
        self.srv = srv
        self.aborts: list[dict] = []

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.srv.actual_api_port}"

    async def post(self, path: str, payload: dict) -> tuple[int, dict]:
        async with self.session.post(f"{self.api_base}{path}", json=payload) as resp:
            return resp.status, await resp.json()

    async def get(self, path: str) -> tuple[int, dict, dict]:
        async with self.session.get(f"{self.api_base}{path}") as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, await resp.json(), headers

    async def delete(self, path: str) -> tuple[int, dict]:
        async with self.session.delete(f"{self.api_base}{path}") as resp:
            return resp.status, await resp.json()

    async def create_dataset(self, dataset_id: str = "gsm8k") -> dict:
        status, body = await self.post(
            "/v1/datasets",
            {"datasetId": dataset_id, "source": {"clusterPath": DATA_FILE}, "schema": {"inputKey": "messages"}},
        )
        assert status == 200, body
        return body

    async def create_evaluator(self, evaluator_id: str = "math-verify") -> dict:
        status, body = await self.post(
            "/v1/evaluators", {"evaluatorId": evaluator_id, "kind": "BUILTIN", "builtin": {"rmType": "math"}}
        )
        assert status == 200, body
        return body

    def job_payload(self, job_id: str, **overrides) -> dict:
        payload = {
            "jobId": job_id,
            "dataset": "datasets/gsm8k",
            "evaluator": "evaluators/math-verify",
            "trainingConfig": {"loraRank": 16, "maxSteps": 400},
        }
        payload.update(overrides)
        return payload

    async def create_job(self, job_id: str, **overrides) -> tuple[int, dict]:
        return await self.post("/v1/postTrainingJobs", self.job_payload(job_id, **overrides))

    def promote(self, name: str) -> None:
        """Simulate the trainer's first weight push (PENDING -> ACTIVE)."""
        self.backend.registry.record_weight_update([name])

    async def finish(self, name: str) -> None:
        """Drive the registry through the full retire path."""
        await self.backend.retire_adapters()
        await self.backend.free_slot(name)


@asynccontextmanager
async def running_gateway(tmp_path: Path):
    router_url = ""
    harness: GatewayHarness | None = None

    async def router_handler(request):
        if request.path == "/list_workers":
            return web.json_response({"urls": [router_url]})
        if request.path == "/abort_request":
            harness.aborts.append(json.loads(await request.read()))
            return web.json_response({})
        return web.json_response({}, status=404)

    app = web.Application()
    app.router.add_resource("/{tail:.*}").add_route("*", router_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    router_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    backend = MultiLoRABackend(
        SimpleNamespace(
            multi_lora_n_adapters=4,
            save=str(tmp_path),
            lora_rank=32,
            lora_alpha=32,
            rollout_batch_size=16,
            n_samples_per_prompt=4,
            multi_lora_dp_size=2,
            multi_lora_max_adapter_global_batch_size=256,
            lr=1e-6,
            rollout_temperature=0.8,
            hf_checkpoint="zai-org/GLM-5.2",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            max_weight_staleness=2,
        ),
        router_url,
    )
    srv = MultiLoRAHTTPServer(backend)
    await backend.init()
    await srv.start()
    try:
        async with aiohttp.ClientSession() as session:
            harness = GatewayHarness(session, backend, srv)
            yield harness
    finally:
        await srv.stop()
        await backend.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_info_reports_capacity_not_slot_indices(tmp_path):
    async with running_gateway(tmp_path) as gw:
        status, body, _ = await gw.get("/v1/info")
        assert status == 200
        assert body["baseModel"] == "zai-org/GLM-5.2"
        assert body["slots"] == {"total": 4, "free": 4}
        assert body["maxLoraRank"] == 32
        assert body["limits"]["dpSize"] == 2


@pytest.mark.asyncio
async def test_dataset_lifecycle_and_validation(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        status, body, _ = await gw.get("/v1/datasets/gsm8k")
        assert status == 200 and body["state"] == "READY"
        assert body["schema"]["inputKey"] == "messages"

        # duplicate -> 409; bad path -> 400; reserved id -> 400
        status, body = await gw.post("/v1/datasets", {"datasetId": "gsm8k", "source": {"clusterPath": DATA_FILE}})
        assert status == 409 and body["error"]["details"][0]["reason"] == "DATASET_EXISTS"
        status, body = await gw.post("/v1/datasets", {"datasetId": "x", "source": {"clusterPath": "/nope/missing"}})
        assert status == 400
        status, body = await gw.post("/v1/datasets", {"datasetId": "base", "source": {"clusterPath": DATA_FILE}})
        assert status == 400 and body["error"]["details"][0]["reason"] == "RESERVED_NAME"

        status, _ = await gw.delete("/v1/datasets/gsm8k")
        assert status == 200
        status, _, _ = await gw.get("/v1/datasets/gsm8k")
        assert status == 404


@pytest.mark.asyncio
async def test_evaluator_python_path_import_check(tmp_path):
    async with running_gateway(tmp_path) as gw:
        status, body = await gw.post(
            "/v1/evaluators", {"evaluatorId": "ok", "kind": "PYTHON_PATH", "pythonPath": {"entrypoint": "json.dumps"}}
        )
        assert status == 200, body
        status, body = await gw.post(
            "/v1/evaluators",
            {"evaluatorId": "bad", "kind": "PYTHON_PATH", "pythonPath": {"entrypoint": "no.such.module.fn"}},
        )
        assert status == 400
        status, body = await gw.post("/v1/evaluators", {"evaluatorId": "weird", "kind": "OTHER"})
        assert status == 400


@pytest.mark.asyncio
async def test_job_create_shape_and_state_flow(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        status, job = await gw.create_job("tutor-a")
        assert status == 200, job
        assert job["name"] == "postTrainingJobs/tutor-a"
        assert job["state"] == "PENDING"
        assert job["outputModel"] == "models/tutor-a"
        assert "slot" not in json.dumps(job)  # slot indices never leak
        assert job["trainingConfig"]["loraRank"] == 16
        assert job["trainingConfig"]["batchSizePrompts"] == 16  # cluster default
        assert job["usage"]["totals"] == {
            "rolloutTokens": 0,
            "trainingTokens": 0,
            "inferenceTokens": 0,
            "computedTokens": 0,
        }
        assert job["jobProgress"]["policyVersion"] == 0

        gw.promote("tutor-a")
        status, job, _ = await gw.get("/v1/postTrainingJobs/tutor-a")
        assert job["state"] == "RUNNING"
        assert job["jobProgress"]["policyVersion"] == 1

        # model is TRAINING + servable while the job serves
        status, model, _ = await gw.get("/v1/models/tutor-a")
        assert model["state"] == "TRAINING" and model["servable"] is True

        # cancel -> STOPPING, then the trainer's retire path -> CANCELLED
        status, body = await gw.post("/v1/postTrainingJobs/tutor-a:cancel", {})
        assert status == 200 and body["state"] == "STOPPING" and body["stopReason"] == "USER_CANCELLED"
        await gw.finish("tutor-a")
        status, job, _ = await gw.get("/v1/postTrainingJobs/tutor-a")
        assert job["state"] == "CANCELLED" and job["stopReason"] == "USER_CANCELLED"

        # no checkpoints were written -> INCOMPLETE, download 404s
        status, model, _ = await gw.get("/v1/models/tutor-a")
        assert model["state"] == "INCOMPLETE" and model["servable"] is False
        status, body, _ = await gw.get("/v1/models/tutor-a:download")
        assert status == 404 and body["error"]["details"][0]["reason"] == "CHECKPOINT_NOT_FOUND"


@pytest.mark.asyncio
async def test_job_create_is_idempotent_for_identical_payloads(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        status, first = await gw.create_job("tutor-a")
        assert status == 200
        status, replay = await gw.create_job("tutor-a")
        assert status == 200 and replay["uid"] == first["uid"]  # replayed, not re-registered
        status, body = await gw.create_job("tutor-a", trainingConfig={"loraRank": 8})
        assert status == 409 and body["error"]["details"][0]["reason"] == "JOB_EXISTS"


@pytest.mark.asyncio
async def test_output_model_collision_and_per_job_values(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        await gw.create_job("tutor-a", outputModel="shared-model")
        status, body = await gw.create_job("tutor-b", outputModel="shared-model")
        assert status == 409 and body["error"]["details"][0]["reason"] == "MODEL_EXISTS"

        # per-job values: accepted at cluster default, 400 on a real conflict
        status, _ = await gw.create_job("tutor-c", trainingConfig={"loraRank": 16, "learningRate": 1e-6})
        assert status == 200
        status, body = await gw.create_job("tutor-d", trainingConfig={"loraRank": 16, "learningRate": 5e-5})
        assert status == 400 and body["error"]["details"][0]["reason"] == "PER_JOB_VALUE_UNSUPPORTED"


@pytest.mark.asyncio
async def test_slots_full_maps_to_429_with_retry_after(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        for i in range(4):
            status, _ = await gw.create_job(f"job-{i}")
            assert status == 200
        async with gw.session.post(
            f"{gw.api_base}/v1/postTrainingJobs", json=gw.job_payload("job-overflow")
        ) as resp:
            body = await resp.json()
            assert resp.status == 429
            assert body["error"]["status"] == "RESOURCE_EXHAUSTED"
            assert body["error"]["details"][0] == {"reason": "SLOT_CAPACITY", "retryable": True}
            assert resp.headers.get("Retry-After") == "5"


@pytest.mark.asyncio
async def test_ops_delete_renders_ops_cancelled_and_shadow_jobs_appear(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        await gw.create_job("tutor-a")
        gw.promote("tutor-a")

        # ops plane deletes behind the gateway's back
        status, _ = await gw.delete("/adapter_runs/tutor-a")
        assert status == 200
        await gw.finish("tutor-a")
        status, job, _ = await gw.get("/v1/postTrainingJobs/tutor-a")
        assert job["state"] == "CANCELLED" and job["stopReason"] == "OPS_CANCELLED"

        # ops plane registers behind the gateway's back -> shadow EXTERNAL job
        status, _ = await gw.post(
            "/adapter_runs", {"name": "ops-only", "config": {"data": DATA_FILE, "rm_type": "math"}}
        )
        assert status == 200
        _, listing, _ = await gw.get("/v1/postTrainingJobs")
        by_name = {j["name"]: j for j in listing["postTrainingJobs"]}
        assert by_name["postTrainingJobs/ops-only"]["kind"] == "EXTERNAL"

        _, states, _ = await gw.get("/v1/postTrainingJobs:batchGetState?names=ops-only&names=tutor-a&names=nope")
        assert states["states"]["ops-only"] == "PENDING"
        assert states["states"]["tutor-a"] == "CANCELLED"
        assert states["states"]["nope"] is None


@pytest.mark.asyncio
async def test_usage_flows_to_job_and_survives_job_delete(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        _, job = await gw.create_job("tutor-a")
        uid = job["uid"]
        gw.promote("tutor-a")
        registry = gw.backend.registry

        counters = {key: 0 for key in ROLLOUT_FIELDS}
        counters |= {"prefill_tokens": 700, "cached_prefill_tokens": 300, "sample_tokens": 500}
        registry.credit_rollout_usage("inc1", [{"name": "tutor-a", "registration_id": uid, "counters": counters}])
        registry.record_batch_adapters(
            1, {"tutor-a": 16}, step_names=["tutor-a"], token_sums={"tutor-a": {"train_tokens": 1200}}
        )
        registry.mark_batch_trained(1)

        _, body, _ = await gw.get("/v1/postTrainingJobs/tutor-a/usage")
        usage = body["usage"]
        assert usage["rollout"]["prefillTokens"] == 700
        assert usage["rollout"]["cachedPrefillTokens"] == 300
        assert usage["training"]["trainTokens"] == 1200
        assert usage["training"]["optimizerSteps"] == 1
        assert usage["totals"] == {
            "rolloutTokens": 1500,
            "trainingTokens": 1200,
            "inferenceTokens": 0,
            "computedTokens": 2700,
        }

        # terminal + deleted job: the ledger keeps answering by uid
        await gw.post("/v1/postTrainingJobs/tutor-a:cancel", {})
        await gw.finish("tutor-a")
        status, _ = await gw.delete("/v1/postTrainingJobs/tutor-a")
        assert status == 200
        _, body, _ = await gw.get(f"/v1/usage?uid={uid}")
        [entry] = body["entries"]
        assert entry["finalized"] is True
        assert entry["usage"]["totals"]["computedTokens"] == 2700

        # the usage journal on disk retains the events for the billing backend
        journal = Path(gw.backend.args.save) / "multi_lora_controller" / "usage.jsonl"
        kinds = [json.loads(line)["kind"] for line in journal.read_text().splitlines()]
        assert "rollout_snapshot" in kinds and "train_commit" in kinds and "final" in kinds


@pytest.mark.asyncio
async def test_model_checkpoints_and_download(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        await gw.create_job("tutor-a")
        gw.promote("tutor-a")

        checkpoint_dir = tmp_path / "adapters" / "tutor-a" / "checkpoints" / "step_100"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"fake")
        (checkpoint_dir / "adapter_config.json").write_text("{}")

        _, model, _ = await gw.get("/v1/models/tutor-a")
        assert [c["checkpointId"] for c in model["checkpoints"]] == ["step_100"]

        _, body, _ = await gw.get("/v1/models/tutor-a:download")
        assert body["checkpointId"] == "step_100"
        assert body["files"]["adapter_model.safetensors"]["clusterPath"].endswith(
            "adapters/tutor-a/checkpoints/step_100/adapter_model.safetensors"
        )
        assert body["files"]["adapter_model.safetensors"]["downloadUrl"] is None

        # live job blocks model deletion
        status, body = await gw.delete("/v1/models/tutor-a")
        assert status == 409

        await gw.post("/v1/postTrainingJobs/tutor-a:cancel", {})
        await gw.finish("tutor-a")
        _, model, _ = await gw.get("/v1/models/tutor-a")
        assert model["state"] == "READY" and model["servable"] is False
        status, _ = await gw.delete("/v1/models/tutor-a?force=true")
        assert status == 200
        assert not checkpoint_dir.exists()


@pytest.mark.asyncio
async def test_pagination_and_warm_start_are_rejected_in_v0(tmp_path):
    async with running_gateway(tmp_path) as gw:
        status, body, _ = await gw.get("/v1/postTrainingJobs?pageToken=abc")
        assert status == 400
        await gw.create_dataset()
        await gw.create_evaluator()
        status, body = await gw.create_job("tutor-a", warmStartFrom={"model": "models/x"})
        assert status == 400 and body["error"]["details"][0]["reason"] == "NOT_IMPLEMENTED"


@pytest.mark.asyncio
async def test_error_envelope_covers_malformed_json_and_bad_types(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        async with gw.session.post(
            f"{gw.api_base}/v1/postTrainingJobs", data=b"{not json", headers={"Content-Type": "application/json"}
        ) as resp:
            body = await resp.json()
            assert resp.status == 400
            assert body["error"]["status"] == "INVALID_ARGUMENT" and "requestId" in body["error"]
        status, body = await gw.create_job("tutor-x", trainingConfig={"learningRate": "fast"})
        assert status == 400 and "requestId" in body["error"]  # enveloped, not legacy {"detail"}


@pytest.mark.asyncio
async def test_get_by_id_resolves_to_latest_registration(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        _, job = await gw.create_job("tutor-a")
        old_uid = job["uid"]
        gw.promote("tutor-a")
        await gw.post("/v1/postTrainingJobs/tutor-a:cancel", {})
        await gw.finish("tutor-a")

        # ops plane re-registers the same name: reads by id follow the LATEST registration
        status, _ = await gw.post(
            "/adapter_runs", {"name": "tutor-a", "config": {"data": DATA_FILE, "rm_type": "math"}}
        )
        assert status == 200
        _, body, _ = await gw.get("/v1/postTrainingJobs/tutor-a")
        assert body["kind"] == "EXTERNAL" and body["uid"] != old_uid and body["state"] == "PENDING"
        _, states, _ = await gw.get("/v1/postTrainingJobs:batchGetState?names=tutor-a")
        assert states["states"]["tutor-a"] == "PENDING"
        # the old registration stays addressable by uid
        _, usage, _ = await gw.get(f"/v1/usage?uid={old_uid}")
        assert len(usage["entries"]) == 1 and usage["entries"][0]["finalized"] is True


@pytest.mark.asyncio
async def test_policy_version_freezes_on_terminal_jobs(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        await gw.create_job("tutor-a")
        gw.promote("tutor-a")
        await gw.post("/v1/postTrainingJobs/tutor-a:cancel", {})
        await gw.finish("tutor-a")

        # slot 0 is reused by another job whose pushes bump the slot version
        await gw.create_job("tutor-b")
        gw.promote("tutor-b")
        gw.promote("tutor-b")
        _, old_job, _ = await gw.get("/v1/postTrainingJobs/tutor-a")
        assert old_job["jobProgress"]["policyVersion"] is None  # not the next tenant's 2..3


@pytest.mark.asyncio
async def test_redundant_cancel_keeps_max_steps_completion(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        await gw.create_job("tutor-a", trainingConfig={"maxSteps": 1, "batchSizePrompts": 4})
        gw.promote("tutor-a")
        registry = gw.backend.registry
        registry.record_batch_adapters(1, {"tutor-a": 4}, step_names=["tutor-a"])
        registry.mark_batch_trained(1)  # reaches maxSteps -> auto-deregister (RETIRING)

        # a redundant cancel/ops-delete must not relabel the finished run
        await gw.post("/v1/postTrainingJobs/tutor-a:cancel", {})
        await gw.delete("/adapter_runs/tutor-a")
        await gw.finish("tutor-a")
        _, job, _ = await gw.get("/v1/postTrainingJobs/tutor-a")
        assert job["state"] == "COMPLETED" and job["stopReason"] == "MAX_STEPS_REACHED"


@pytest.mark.asyncio
async def test_stale_checkpoints_block_output_model_reuse(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        await gw.create_job("tutor-a", outputModel="m1")
        gw.promote("tutor-a")
        checkpoint_dir = tmp_path / "adapters" / "m1" / "checkpoints" / "step_5"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"fake")
        await gw.post("/v1/postTrainingJobs/tutor-a:cancel", {})
        await gw.finish("tutor-a")

        # non-force delete keeps checkpoints; reusing the outputModel would
        # silently warm-start from them -> rejected until cleaned up
        status, _ = await gw.delete("/v1/models/m1")
        assert status == 200
        status, body = await gw.create_job("tutor-b", outputModel="m1")
        assert status == 409 and body["error"]["details"][0]["reason"] == "STALE_CHECKPOINTS"


@pytest.mark.asyncio
async def test_incomplete_checkpoint_dir_is_not_a_checkpoint(tmp_path):
    async with running_gateway(tmp_path) as gw:
        await gw.create_dataset()
        await gw.create_evaluator()
        await gw.create_job("tutor-a")
        gw.promote("tutor-a")
        # crash-mid-save artifact: a step dir without the exported adapter
        (tmp_path / "adapters" / "tutor-a" / "checkpoints" / "step_1").mkdir(parents=True)
        await gw.post("/v1/postTrainingJobs/tutor-a:cancel", {})
        await gw.finish("tutor-a")
        _, model, _ = await gw.get("/v1/models/tutor-a")
        assert model["state"] == "INCOMPLETE" and model["checkpoints"] == []
        status, _, _ = await gw.get("/v1/models/tutor-a:download")
        assert status == 404


@pytest.mark.asyncio
async def test_console_ui_is_served(tmp_path):
    async with running_gateway(tmp_path) as gw:
        async with gw.session.get(f"{gw.api_base}/ui") as resp:
            text = await resp.text()
            assert resp.status == 200
            assert "text/html" in resp.headers["content-type"]
            assert "MILES MULTI-LORA" in text and "/v1/postTrainingJobs" in text
