"""The TMax gateway sandbox backend against a fake LiteRegistry gateway.

The fake implements the gateway's /affinity/* contract and a tiny in-container model: the launcher exec records the
command, wait execs answer with its scripted result, uploads decode their base64 stdin. That is enough to check the
launch-then-wait protocol, chunked uploads, the episode logic shared with the podman backend, and how lost or killed
containers end up.
"""

import asyncio
import base64
import io
import json
import sys
import tarfile
from pathlib import Path

import httpx
import pytest

TMAX_DIR = Path(__file__).resolve().parents[4] / "examples" / "tmax"
sys.path.insert(0, str(TMAX_DIR))

import tmax_gateway  # noqa: E402
import tmax_sandbox  # noqa: E402

GATEWAY = "http://gateway.test:8080"


class FakeGateway:
    def __init__(self):
        self.containers: dict[str, dict] = {}
        self.handshakes = 0
        self.closed: list[str] = []
        self.commands: list[str] = []  # agent-visible commands, in launch order
        self.results: dict[str, tuple[int, str, str]] = {}  # command substring -> (exit, stdout, stderr)
        self.pending_rounds = 0  # wait execs that answer PENDING before DONE
        self.fail_next: list[tuple[int, str]] = []  # (status, body) for the next posts
        self.lose_after_launches: int | None = None  # 404 every exec after this many launches

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if self.fail_next:
            status, body = self.fail_next.pop(0)
            return httpx.Response(status, text=body)
        path = request.url.path
        if path == "/affinity/handshake":
            self.handshakes += 1
            cid = f"container{self.handshakes:04d}"
            self.containers[cid] = {"files": {}, "jobs": {}}
            return httpx.Response(
                200, json={"affinity_id": cid, "instance_id": "replica-1", "image": payload["image"]}
            )
        if path == "/affinity/close":
            self.closed.append(payload["affinity_id"])
            self.containers.pop(payload["affinity_id"], None)
            return httpx.Response(200, json={"removed": True})
        assert path == "/affinity/podman"
        container = self.containers.get(payload["affinity_id"])
        if container is None:
            return httpx.Response(404, text="affinity binding not found")
        return httpx.Response(200, json=self._exec(container, payload["command"], payload["stdin"]))

    def _exec(self, container: dict, command: str, stdin: str) -> dict:
        ok = {"stdout": "", "stderr": "", "exit_code": 0}
        if "LAUNCHED" in command:  # the detached launcher; stdin is the command
            if self.lose_after_launches is not None and len(self.commands) >= self.lose_after_launches:
                self.containers.clear()
                return {"stdout": "", "stderr": "", "exit_code": 1}
            job = command.split("if mkdir ")[1].split(" ")[0]
            if job not in container["jobs"]:
                self.commands.append(stdin)
                container["jobs"][job] = self._result_for(stdin)
            return {**ok, "stdout": "LAUNCHED\n"}
        if "until [ -f" in command:  # a wait exec
            if self.pending_rounds:
                self.pending_rounds -= 1
                return {**ok, "stdout": f"{tmax_gateway._PENDING_MARKER}\n"}
            job = command.split("until [ -f ")[1].split("/rc")[0]
            exit_code, out, err = container["jobs"][job]
            return {"stdout": f"{tmax_gateway._DONE_MARKER}\nRC={exit_code}\n{out}", "stderr": err, "exit_code": 0}
        if command.startswith("base64 -d | tar -xf - -C /"):
            self._extract(container, base64.b64decode(stdin))
            return ok
        if " > " in command and command.startswith("base64 -d > "):  # an upload chunk
            container["files"][command.split(" > ")[1]] = base64.b64decode(stdin)
            return ok
        if command.startswith("cat ") and "part.*" in command:  # chunk assembly
            stage = command.split("cat ")[1].split("/part.*")[0]
            parts = sorted(k for k in container["files"] if k.startswith(f"{stage}/part."))
            self._extract(container, b"".join(container["files"].pop(k) for k in parts))
            return ok
        return ok  # mkdir, rm -rf, true

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
    monkeypatch.setenv("TMAX_GATEWAY_URL", GATEWAY)
    monkeypatch.setenv("TMAX_GATEWAY_KEEPALIVE_S", "0")
    monkeypatch.setenv("MILES_ROLLOUT_LIMITS_SCOPE", f"test-{tmp_path.name}")
    monkeypatch.setattr(tmax_gateway, "_RETRY_BASE_S", 0.0)
    monkeypatch.setattr(tmax_gateway, "_START_RETRY_BASE_S", 0.0)
    monkeypatch.setattr(tmax_gateway, "_START_RETRY_JITTER_S", 0.0)
    tmax_sandbox._settings.cache_clear()
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        tmax_gateway.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(fake.handler), **kwargs),
    )
    yield fake
    tmax_sandbox._settings.cache_clear()


@pytest.fixture
def task_dir(tmp_path):
    (tmp_path / "environment" / "seeds").mkdir(parents=True)
    (tmp_path / "environment" / "seeds" / "data.txt").write_text("seed")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test.sh").write_text("#!/bin/bash\necho ok > /logs/verifier/reward.txt\n")
    return str(tmp_path)


def run(coro):
    return asyncio.run(coro)


def test_episode_runs_commands_and_tests(gateway, task_dir):
    gateway.results = {
        "wrapper.sh ls": (0, "a.txt\n", ""),
        "wrapper.sh false": (1, "", "boom"),
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

    gateway.results["COMPLETE_TASK"] = (0, "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", "")
    ls, fail, submitted, reward, files = run(episode())
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

    files = run(episode())
    assert files["/workspace/big.bin"] == bytes(range(256)) * 64
    assert not any(".part." in k or "/part." in k for k in files)  # staging chunks consumed


def test_timeout_reports_like_podman(gateway, task_dir):
    gateway.results = {"wrapper.sh 'sleep 999'": (124, "", "")}

    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            return await box.bash("sleep 999")

    observation, submitted = run(episode())
    assert "Command timed out after 120s." in observation and observation.endswith("(exit_code=124)")
    assert not submitted


def test_lost_binding_is_rehandshaken_once(gateway, task_dir):
    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            await box.bash("echo one")
            gateway.containers.clear()  # the replica died: every request for the binding now 404s
            second = await box.bash("echo two")
            gateway.containers.clear()
            with pytest.raises(tmax_sandbox.SandboxInfraError):
                await box.bash("echo three")
            return second

    observation, _ = run(episode())
    assert observation.endswith("(exit_code=0)")
    assert gateway.handshakes == 2


def test_container_removed_by_replica_ends_the_episode(gateway, task_dir):
    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            gateway.fail_next = [(404, '{"code": "container_not_found"}')]
            await box.bash("python big.py")

    with pytest.raises(tmax_sandbox.SandboxLostError):
        run(episode())


def test_watchdog_removal_ends_the_episode(gateway, task_dir):
    body = (
        '{"detail":{"code":"sandbox_lost","reason":"pids_limit","recoverable":false,'
        '"limit":2048,"observed":4538,"unit":"tasks","enforcement":"userspace_watchdog"}}'
    )

    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            gateway.fail_next = [(410, body)]
            await box.bash(":(){ :|:& };:")

    with pytest.raises(tmax_sandbox.SandboxLostError):
        run(episode())
    assert gateway.handshakes == 1  # not re-handshaken into a fresh container


def test_start_retries_then_gives_up(gateway, task_dir):
    gateway.fail_next = [(503, "no replicas")] * (tmax_gateway._HANDSHAKE_ATTEMPTS * tmax_sandbox._START_ATTEMPTS)

    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir):
            pass

    with pytest.raises(tmax_sandbox.SandboxInfraError):
        run(episode())
    assert gateway.handshakes == 0


def test_transient_errors_are_retried(gateway, task_dir):
    async def episode():
        async with tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir) as box:
            gateway.fail_next = [(502, "bad gateway"), (503, "busy")]
            return await box.bash("echo hi")

    observation, _ = run(episode())
    assert observation.endswith("(exit_code=0)")


def test_abort_closes_live_containers(gateway, task_dir):
    async def episode():
        box = tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir)
        await box.__aenter__()
        closed = await tmax_gateway.close_live_sandboxes()
        await box.__aexit__(None, None, None)
        return closed

    assert run(episode()) == 1
    assert "container0001" in gateway.closed


def test_missing_gateway_url_is_a_configuration_error(gateway, task_dir, monkeypatch):
    monkeypatch.setenv("TMAX_GATEWAY_URL", "")
    tmax_sandbox._settings.cache_clear()
    with pytest.raises(ValueError):
        tmax_gateway.TmaxGatewaySandbox(image="img:1", task_dir=task_dir)
