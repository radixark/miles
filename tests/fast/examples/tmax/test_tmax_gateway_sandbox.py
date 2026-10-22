"""The TMax gateway sandbox backend against a fake LiteRegistry gateway.

The fake is a real HTTP server speaking literegistry 1.0.63's /affinity/* contract, driven through literegistry's own
client, with a tiny in-container model: the launcher exec records the command, wait execs answer with its scripted
result, uploads decode their base64 stdin. That is enough to check the launch-then-wait protocol, chunked uploads, the
episode logic shared with the podman backend, retries, and how a lost container ends the episode.
"""

import asyncio
import base64
import io
import sys
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("literegistry_podman_client")
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

TMAX_DIR = Path(__file__).resolve().parents[4] / "examples" / "tmax"
sys.path.insert(0, str(TMAX_DIR))

import tmax_agent_function  # noqa: E402
import tmax_gateway  # noqa: E402
import tmax_sandbox  # noqa: E402

from miles.rollout.agentic import rollout_limits  # noqa: E402

SANDBOX_LOST_PIDS = {
    "detail": {
        "code": "sandbox_lost",
        "reason": "pids_limit",
        "container_id": "container0001",
        "recoverable": False,
        "limit": 2048,
        "observed": 4538,
        "unit": "tasks",
        "enforcement": "userspace_watchdog",
    }
}
SANDBOX_STOPPED = {"detail": {"code": "sandbox_lost", "reason": "container_stopped", "recoverable": False}}
OWNER_LOST = {
    "error": "strict affinity server is no longer registered",
    "code": "affinity_owner_lost",
    "recoverable": False,
}


class FakeGateway:
    def __init__(self):
        self.containers: dict[str, dict] = {}
        self.handshakes = 0
        self.closed: list[str] = []
        self.commands: list[str] = []  # agent-visible commands, in launch order
        self.results: dict[str, tuple[int, str, str]] = {}  # command substring -> (exit, stdout, stderr)
        self.pending_rounds = 0  # wait execs that answer PENDING before DONE
        self.markerless_rounds = 0  # wait execs that answer with neither marker
        self.stopped_on_probe = False  # the `true` probe finds the container stopped
        self.fail_next: list[tuple[int, object]] = []  # (status, JSON body) for the next requests
        self.lost_on_first_exec: set[str] = set()  # containers the fleet loses before their first exec answers
        self.requests: list[str] = []  # every request's endpoint, failed ones included

    async def handle(self, request: web.Request) -> web.Response:
        payload = await request.json()
        self.requests.append(request.match_info["op"])
        if self.fail_next:
            status, body = self.fail_next.pop(0)
            return web.json_response(body, status=status)
        op = request.match_info["op"]
        if op == "handshake":
            self.handshakes += 1
            cid = f"container{self.handshakes:04d}"
            self.containers[cid] = {"files": {}, "jobs": {}}
            return web.json_response(
                {"affinity_id": cid, "container_id": cid, "instance_id": "replica-1", "image": payload["image"]}
            )
        if op == "close":
            self.closed.append(payload["affinity_id"])
            self.containers.pop(payload["affinity_id"], None)
            return web.json_response({"affinity_id": payload["affinity_id"], "removed": True})
        container = self.containers.get(payload["affinity_id"])
        if container is None:
            return web.json_response({"detail": "strict affinity binding was not found or has expired"}, status=404)
        if self.stopped_on_probe and payload["command"] == "true":
            return web.json_response(SANDBOX_STOPPED, status=410)
        if payload["affinity_id"] in self.lost_on_first_exec:
            self.lost_on_first_exec.discard(payload["affinity_id"])
            self.containers.pop(payload["affinity_id"])
            return web.json_response(SANDBOX_STOPPED, status=410)
        exit_code, stdout, stderr = self._exec(container, payload["command"], payload["stdin"])
        return web.json_response(
            {
                "container_id": payload["affinity_id"],
                "affinity_id": payload["affinity_id"],
                "stdout": stdout,
                "stderr": stderr,
                "success": exit_code == 0,
                "exit_code": exit_code,
                "execution_time": 0.01,
                "timed_out": False,
            }
        )

    def _exec(self, container: dict, command: str, stdin: str) -> tuple[int, str, str]:
        if "LAUNCHED" in command:  # the detached launcher; stdin is the command
            job = command.split("if mkdir ")[1].split(" ")[0]
            if job not in container["jobs"]:
                self.commands.append(stdin)
                container["jobs"][job] = self._result_for(stdin)
            return 0, "LAUNCHED\n", ""
        if "until [ -f" in command:  # a wait exec
            if self.markerless_rounds:
                self.markerless_rounds -= 1
                return 137, "", ""
            if self.pending_rounds:
                self.pending_rounds -= 1
                return 0, f"{tmax_gateway._PENDING_MARKER}\n", ""
            job = command.split("until [ -f ")[1].split("/rc")[0]
            exit_code, out, err = container["jobs"][job]
            return 0, f"{tmax_gateway._DONE_MARKER}\nRC={exit_code}\n{out}", err
        if command.startswith("base64 -d | tar -xf - -C /"):
            self._extract(container, base64.b64decode(stdin))
        elif command.startswith("base64 -d > "):  # an upload chunk
            container["files"][command.split(" > ")[1]] = base64.b64decode(stdin)
        elif command.startswith("cat ") and "part.*" in command:  # chunk assembly
            stage = command.split("cat ")[1].split("/part.*")[0]
            parts = sorted(k for k in container["files"] if k.startswith(f"{stage}/part."))
            self._extract(container, b"".join(container["files"].pop(k) for k in parts))
        return 0, "", ""  # also mkdir, rm -rf, true

    def _result_for(self, command: str) -> tuple[int, str, str]:
        for key, result in self.results.items():
            if key in command:
                return result
        return (0, "", "")

    @staticmethod
    def _extract(container: dict, tar_bytes: bytes) -> None:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
            for member in tar.getmembers():
                if member.isfile():
                    container["files"]["/" + member.name] = tar.extractfile(member).read()


@pytest.fixture
def gateway(monkeypatch, tmp_path):
    fake = FakeGateway()
    monkeypatch.setenv("TMAX_SANDBOX_BACKEND", "gateway")
    monkeypatch.setenv("TMAX_GATEWAY_KEEPALIVE_S", "0")
    monkeypatch.setenv("MILES_ROLLOUT_LIMITS_SCOPE", f"test-{tmp_path.name}")
    monkeypatch.setattr(tmax_gateway, "_RETRY_BASE_S", 0.0)
    monkeypatch.setattr(tmax_gateway, "_START_RETRY_BASE_S", 0.0)
    monkeypatch.setattr(tmax_gateway, "_START_RETRY_JITTER_S", 0.0)
    yield fake
    tmax_sandbox._settings.cache_clear()


def run(fake: FakeGateway, body, monkeypatch):
    """Serve ``fake`` on localhost, point the backend at it, and run ``body()``."""

    async def main():
        app = web.Application()
        app.router.add_post("/affinity/{op}", fake.handle)
        server = TestServer(app)
        await server.start_server()
        monkeypatch.setenv("TMAX_GATEWAY_URL", str(server.make_url("")).rstrip("/"))
        tmax_sandbox._settings.cache_clear()
        try:
            return await body()
        finally:
            await server.close()

    return asyncio.run(main())


@pytest.fixture
def task_dir(tmp_path):
    (tmp_path / "environment" / "seeds").mkdir(parents=True)
    (tmp_path / "environment" / "seeds" / "data.txt").write_text("seed")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test.sh").write_text("#!/bin/bash\necho ok > /logs/verifier/reward.txt\n")
    return str(tmp_path)


def test_episode_runs_commands_and_tests(gateway, task_dir, monkeypatch):
    gateway.results = {
        "wrapper.sh ls": (0, "a.txt\n", ""),
        "wrapper.sh false": (1, "", "boom"),
        "COMPLETE_TASK": (0, "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", ""),
        "cat /logs/verifier/reward.txt": (0, "1\n", ""),
    }
    gateway.pending_rounds = 2  # the first command takes two wait rounds

    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            ls = await box.bash("ls")
            fail = await box.bash("false")
            submitted = await box.bash("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
            reward = await box.run_tests()
            files = dict(gateway.containers["container0001"]["files"])
        return ls, fail, submitted, reward, files

    ls, fail, submitted, reward, files = run(gateway, episode, monkeypatch)
    assert ls == ("a.txt\n\n\n(exit_code=0)", False)
    assert fail == ("boom\n\n(exit_code=1)", False)
    assert submitted[1] is True
    assert reward == 1.0
    # Seeds, the persistent-shell wrapper and the withheld tests all reached the container.
    assert files["/workspace/data.txt"] == b"seed"
    assert tmax_sandbox._BASH_WRAPPER_PATH in files and "/tests/test.sh" in files
    # Agent commands run through the persistent-shell wrapper; the container is closed at the end.
    assert any(c.startswith(f"bash {tmax_sandbox._BASH_WRAPPER_PATH} ls") for c in gateway.commands)
    assert gateway.closed == ["container0001"]


def test_large_upload_is_chunked(gateway, task_dir, monkeypatch):
    monkeypatch.setattr(tmax_gateway, "_UPLOAD_CHUNK_BYTES", 1024)
    (Path(task_dir) / "environment" / "seeds" / "big.bin").write_bytes(bytes(range(256)) * 64)

    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir):
            return dict(gateway.containers["container0001"]["files"])

    files = run(gateway, episode, monkeypatch)
    assert files["/workspace/big.bin"] == bytes(range(256)) * 64
    assert not any("/part." in k for k in files)  # staging chunks consumed


def test_timeout_reports_like_podman(gateway, task_dir, monkeypatch):
    gateway.results = {"wrapper.sh 'sleep 999'": (124, "", "")}

    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            return await box.bash("sleep 999")

    observation, submitted = run(gateway, episode, monkeypatch)
    assert "Command timed out after 120s." in observation and observation.endswith("(exit_code=124)")
    assert not submitted


@pytest.mark.parametrize("status, body", [(410, SANDBOX_LOST_PIDS), (410, OWNER_LOST)])
def test_lost_container_ends_the_episode(gateway, task_dir, monkeypatch, caplog, status, body):
    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            gateway.fail_next = [(status, body)]
            await box.bash(":(){ :|:& };:")

    with pytest.raises(tmax_sandbox.SandboxLostError) as lost, caplog.at_level("WARNING", tmax_gateway.logger.name):
        run(gateway, episode, monkeypatch)
    assert lost.value.reason == body.get("detail", body).get("reason", body.get("code"))
    assert gateway.handshakes == 1  # never continued in a replacement container
    assert any(f"gateway container lost ({status})" in r.getMessage() for r in caplog.records)  # with its full detail


def test_expired_binding_ends_the_episode(gateway, task_dir, monkeypatch):
    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            gateway.containers.clear()  # every request for the binding now 404s
            await box.bash("echo hi")

    with pytest.raises(tmax_sandbox.SandboxLostError) as lost:
        run(gateway, episode, monkeypatch)
    assert lost.value.reason == "binding_lost"
    assert gateway.handshakes == 1


@pytest.mark.parametrize("stopped", [False, True])
def test_markerless_waits_probe_the_container(gateway, task_dir, monkeypatch, stopped):
    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            gateway.markerless_rounds = tmax_gateway._MAX_MARKERLESS_WAITS
            gateway.stopped_on_probe = stopped
            await box.bash("echo hi")

    with pytest.raises((tmax_sandbox.SandboxInfraError, tmax_sandbox.SandboxLostError)) as error:
        run(gateway, episode, monkeypatch)
    # A stopped container ends the episode; a live one that keeps answering oddly fails the step.
    assert type(error.value) is (tmax_sandbox.SandboxLostError if stopped else tmax_sandbox.SandboxInfraError)


def test_transient_errors_are_retried(gateway, task_dir, monkeypatch, caplog):
    unavailable = {"error": "strict affinity owner is temporarily unavailable", "code": "affinity_owner_unavailable"}

    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            gateway.fail_next = [(503, unavailable), (502, {"code": "affinity_upstream_disconnected"})]
            return await box.bash("echo hi")

    with caplog.at_level("WARNING", logger=tmax_gateway.logger.name):
        observation, _ = run(gateway, episode, monkeypatch)
    assert observation.endswith("(exit_code=0)")
    # Each retry logs the gateway's own code and how long the request took, not only the status.
    retries = [r.getMessage() for r in caplog.records if "gateway request failed" in r.getMessage()]
    assert len(retries) == 2
    assert "(503) after" in retries[0] and '"code":"affinity_owner_unavailable"' in retries[0]
    assert "(502) after" in retries[1] and "affinity_upstream_disconnected" in retries[1]


ADMISSION_TIMEOUT = {"code": "affinity_admission_timeout", "execution_outcome": "unknown"}


def test_a_timed_out_handshake_is_resent(gateway, task_dir, monkeypatch):
    gateway.fail_next = [(504, ADMISSION_TIMEOUT)]

    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            return await box.bash("echo hi")

    observation, _ = run(gateway, episode, monkeypatch)
    assert observation.endswith("(exit_code=0)")
    assert gateway.requests[:2] == ["handshake", "handshake"] and gateway.handshakes == 1


def test_handshakes_give_up_after_their_attempts(gateway, task_dir, monkeypatch):
    """A start sends at most _HANDSHAKE_ATTEMPTS handshakes: failing ones do not restart the start loop."""
    gateway.fail_next = [(504, ADMISSION_TIMEOUT)] * tmax_gateway._HANDSHAKE_ATTEMPTS

    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir):
            pass

    with pytest.raises(tmax_sandbox.SandboxInfraError):
        run(gateway, episode, monkeypatch)
    assert gateway.requests == ["handshake"] * tmax_gateway._HANDSHAKE_ATTEMPTS


def test_a_container_lost_while_starting_is_replaced(gateway, task_dir, monkeypatch):
    gateway.lost_on_first_exec = {"container0001"}

    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            return await box.bash("echo hi")

    observation, _ = run(gateway, episode, monkeypatch)
    assert observation.endswith("(exit_code=0)")
    # The fleet already removed the lost container, so only its replacement is closed.
    assert gateway.handshakes == 2 and gateway.closed == ["container0002"]


def test_abort_closes_live_containers(gateway, task_dir, monkeypatch):
    async def episode():
        box = tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir)
        await box.__aenter__()
        closed = await tmax_gateway.close_live_sandboxes()
        await box.__aexit__(None, None, None)
        return closed

    assert run(gateway, episode, monkeypatch) == 1
    assert "container0001" in gateway.closed


def test_job_exit_closes_every_rollout_process_containers(gateway, task_dir, monkeypatch, tmp_path):
    """close-all, which the head runs when the job stops, reaches the containers of every rollout process on the host."""
    monkeypatch.setattr(tmax_gateway.tempfile, "tempdir", str(tmp_path))  # only this test's live-container records

    async def two_rollout_processes():
        first = tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir)
        await first.__aenter__()
        monkeypatch.setenv(rollout_limits.SCOPE_ENV_VAR, "another-rollout-process")
        second = tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir)
        await second.__aenter__()
        closed = await tmax_gateway.close_all_live_sandboxes(timeout_s=10)
        for box in (first, second):  # the processes' own HTTP pools, which a killed process never closes
            await box._client.aclose()
        return closed

    assert run(gateway, two_rollout_processes, monkeypatch) == 2
    assert sorted(gateway.closed) == ["container0001", "container0002"] and not gateway.containers
    assert not [p for d in tmp_path.glob("miles-tmax-gateway-live-*") for p in d.iterdir()], "records removed"


def test_job_exit_with_nothing_live_closes_nothing(gateway, task_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(tmax_gateway.tempfile, "tempdir", str(tmp_path))
    assert run(gateway, lambda: tmax_gateway.close_all_live_sandboxes(timeout_s=10), monkeypatch) == 0
    assert gateway.closed == []


def test_missing_gateway_url_is_a_configuration_error(gateway, task_dir, monkeypatch):
    monkeypatch.setenv("TMAX_GATEWAY_URL", "")
    tmax_sandbox._settings.cache_clear()
    with pytest.raises(ValueError):
        tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir)


@pytest.mark.parametrize(
    "reason, reward",
    [
        ("unknown", None),
        ("affinity_owner_lost", None),
        ("binding_lost", None),
        ("memory_limit", 0.0),
        ("pids_limit", 0.0),
        ("container_stopped", 0.0),
    ],
)
def test_fleet_losses_drop_the_group_policy_losses_score_zero(reason, reward):
    assert tmax_agent_function._lost_sandbox_reward(reason) == reward
