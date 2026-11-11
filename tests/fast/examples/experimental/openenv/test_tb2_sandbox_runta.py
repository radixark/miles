"""Tests for the Runta sandbox half: checkpoint naming and reuse, the in-runtime
image build, the per-episode create and its cleanup, token supply, the
published-port URL and its launch preflight, and the orphan-TTL keepalive
lifecycle.

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q
"""

import sys
import threading
import time
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

import tb2_sandbox_runta as sandbox


def _patch_recipe(
    monkeypatch,
    base="ghcr.io/laude/task:1.0",
    commands=("apt-get install -y curl",),
    resources=(1, 2048, 10240),
):
    monkeypatch.setattr(sandbox, "resolve_docker_image", lambda task_dir, override: base)
    monkeypatch.setattr(sandbox, "server_layer_commands", lambda task_dir: list(commands))
    monkeypatch.setattr(sandbox, "task_env_resources", lambda task_dir: resources)


# --- the fake SDK ------------------------------------------------------------


class _Result:
    def __init__(self, exit_code=0, stdout_text="", stderr_text=""):
        self.exit_code = exit_code
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text


class _FakeRuntime:
    """Stands in for runta.Runtime: records what was done to it."""

    def __init__(self, client, name, kwargs):
        self.client = client
        self.name = name
        self.kwargs = kwargs
        self.id = f"rt-{len(client.runtimes.created)}"
        self.execs = []
        self.files_written = {}
        self.deleted = False
        self.rearmed = []
        self.files = types.SimpleNamespace(write=self.files_written.__setitem__)
        self.checkpoints = types.SimpleNamespace(create=self._checkpoint)

    def wait_until_ready(self, timeout):
        if self.client.never_ready:
            raise TimeoutError("runtime not ready")

    def exec(self, command, timeout=None):
        self.execs.append((command, timeout))
        return self.client.exec_result

    def _checkpoint(self, name=None, timeout=None):
        checkpoint = types.SimpleNamespace(id=f"ckpt-{name}", display_name=name, state="ready")
        self.client.stored.append(checkpoint)
        return checkpoint

    def update_timeout_policies(self, **policies):
        self.rearmed.append(policies)

    def delete(self):
        self.deleted = True


class _FakeClient:
    def __init__(self):
        self.stored = []  # checkpoints the account holds
        self.deleted_checkpoints = []
        self.waited_for = []
        self.never_ready = False
        self.exec_result = _Result()
        self.runtimes = types.SimpleNamespace(created=[], create=self._create)
        self.checkpoints = types.SimpleNamespace(
            list=lambda: list(self.stored),
            delete=self._delete_checkpoint,
            wait_until_ready=self._wait_checkpoint,
        )

    def _create(self, name, **kwargs):
        runtime = _FakeRuntime(self, name, kwargs)
        self.runtimes.created.append(runtime)
        return runtime

    def _delete_checkpoint(self, checkpoint_id):
        self.deleted_checkpoints.append(checkpoint_id)
        self.stored = [c for c in self.stored if c.id != checkpoint_id]

    def _wait_checkpoint(self, checkpoint_id, timeout):
        self.waited_for.append(checkpoint_id)
        return next(c for c in self.stored if c.id == checkpoint_id)


@pytest.fixture
def fake_runta(monkeypatch):
    """A fake `runta` module plus the client the module under test will be handed."""
    mod = types.ModuleType("runta")
    mod.IngressSpec = lambda protocol, runtime_port: (protocol, runtime_port)
    monkeypatch.setitem(sys.modules, "runta", mod)
    client = _FakeClient()
    monkeypatch.setattr(sandbox, "make_client", lambda: client)

    @contextmanager
    def no_lock(name):
        yield

    monkeypatch.setattr(sandbox, "rollout_lock", no_lock)
    return client


def _checkpoint(name, state="ready"):
    return types.SimpleNamespace(id=f"ckpt-{state}-{name}", display_name=name, state=state)


# --- checkpoint naming --------------------------------------------------------


def test_checkpoint_name_is_deterministic_and_sanitized(monkeypatch):
    _patch_recipe(monkeypatch)
    n1 = sandbox.checkpoint_name(Path("/tasks/Regex_Chess"))
    n2 = sandbox.checkpoint_name(Path("/tasks/Regex_Chess"))
    assert n1 == n2
    assert n1.startswith("tb2-regex-chess-")  # lowercased, non [a-z0-9-] mapped to '-'


def test_checkpoint_name_tracks_recipe_content(monkeypatch):
    """The digest must cover base image AND build commands: editing either
    re-bakes instead of silently restoring a stale image."""
    _patch_recipe(monkeypatch)
    original = sandbox.checkpoint_name(Path("/tasks/t"))
    _patch_recipe(monkeypatch, commands=("apt-get install -y curl", "extra step"))
    assert sandbox.checkpoint_name(Path("/tasks/t")) != original
    _patch_recipe(monkeypatch, base="ghcr.io/laude/task:2.0")
    assert sandbox.checkpoint_name(Path("/tasks/t")) != original


def test_checkpoint_name_tracks_runtime_size(monkeypatch):
    """A checkpoint is a snapshot of a sized machine and episodes restore it at
    that size, so a task.toml resource bump must re-bake."""
    _patch_recipe(monkeypatch)
    small = sandbox.checkpoint_name(Path("/tasks/t"))
    _patch_recipe(monkeypatch, resources=(4, 8192, 10240))
    assert sandbox.checkpoint_name(Path("/tasks/t")) != small


def test_checkpoint_name_tracks_the_build_user(monkeypatch):
    """The user is part of the baked image, so changing it must re-bake rather
    than restore an image built by whoever built it first."""
    _patch_recipe(monkeypatch)
    before = sandbox.checkpoint_name(Path("/tasks/t"))
    monkeypatch.setattr(sandbox, "_BUILD_USER", "someone-else")
    assert sandbox.checkpoint_name(Path("/tasks/t")) != before


# --- the recipe as a Dockerfile -------------------------------------------------


def test_dockerfile_layers_the_recipe_on_the_official_image_as_root(monkeypatch):
    """USER comes before the first RUN: the recipe's layers apt-install and
    write under /opt, and the user set here is also who the env server — and so
    the agent — runs as."""
    _patch_recipe(monkeypatch, commands=("apt-get install -y curl", "mkdir -p /opt/envserver"))
    assert sandbox.dockerfile(Path("/tasks/t")).splitlines() == [
        "FROM ghcr.io/laude/task:1.0",
        "USER root",
        "RUN apt-get install -y curl",
        "RUN mkdir -p /opt/envserver",
    ]


def test_runtime_is_sized_above_the_task_for_the_docker_daemon(tmp_path: Path):
    """The daemon runs in the runtime, outside the task container; without the
    headroom it would eat into the memory task.toml promises the task."""
    (tmp_path / "task.toml").write_text("[environment]\ncpus = 4\nmemory_mb = 8192\n")
    resources = sandbox.task_runtime_resources(tmp_path)
    assert resources == {"vcpus": 4, "memory_mib": 8192 + sandbox._DOCKER_OVERHEAD_MIB}


def test_runtime_resources_floors(tmp_path: Path):
    (tmp_path / "task.toml").write_text("[environment]\ncpus = 0\nmemory_mb = 128\n")
    resources = sandbox.task_runtime_resources(tmp_path)
    assert resources == {"vcpus": 1, "memory_mib": 2048 + sandbox._DOCKER_OVERHEAD_MIB}


# --- ensure_task_checkpoint -----------------------------------------------------


def test_ensure_checkpoint_short_circuits_on_a_ready_checkpoint(monkeypatch, fake_runta):
    _patch_recipe(monkeypatch)
    ready = _checkpoint(sandbox.checkpoint_name(Path("/tasks/t")))
    fake_runta.stored = [_checkpoint("tb2-other-task-0000000000"), ready]

    assert sandbox.ensure_task_checkpoint(Path("/tasks/t")) == ready.id
    assert fake_runta.runtimes.created == []  # no builder when the checkpoint exists


def test_ensure_checkpoint_builds_the_image_in_a_builder_and_snapshots_it(monkeypatch, fake_runta):
    _patch_recipe(monkeypatch, resources=(3, 4096, 10240))
    task_dir = Path("/tasks/t")
    name = sandbox.checkpoint_name(task_dir)

    checkpoint_id = sandbox.ensure_task_checkpoint(task_dir, build_timeout_s=1234.0)

    (builder,) = fake_runta.runtimes.created
    assert builder.name.startswith("miles-tb2-build-t-")
    assert builder.kwargs["vcpus"] == 3
    assert builder.kwargs["memory_mib"] == 4096 + sandbox._DOCKER_OVERHEAD_MIB
    assert builder.files_written == {"/opt/tb2-build/Dockerfile": sandbox.dockerfile(task_dir)}
    ((command, timeout),) = builder.execs
    assert command.startswith("docker build -t tb2-env:latest /opt/tb2-build")
    assert timeout == 1234.0
    assert [c.display_name for c in fake_runta.stored] == [name]
    assert checkpoint_id == fake_runta.stored[0].id
    assert builder.deleted  # the checkpoint is what is kept, never the builder


def test_builder_carries_a_ttl_that_outlasts_the_build(monkeypatch, fake_runta):
    """A builder has no keepalive, so the backstop is the only thing that
    reclaims one whose caller died — and it must not fire mid-build."""
    _patch_recipe(monkeypatch)
    sandbox.ensure_task_checkpoint(Path("/tasks/t"), build_timeout_s=1800.0)
    (builder,) = fake_runta.runtimes.created
    assert builder.kwargs["auto_shutdown_after_secs"] > 1800
    assert builder.kwargs["auto_delete_after_secs"] > 0
    assert builder.kwargs["wait"] is False  # the handle must exist before anything can fail


def test_failed_build_names_the_output_and_still_deletes_the_builder(monkeypatch, fake_runta):
    _patch_recipe(monkeypatch)
    fake_runta.exec_result = _Result(exit_code=1, stdout_text="E: Unable to locate package curl")
    with pytest.raises(RuntimeError, match="docker build failed.*Unable to locate package"):
        sandbox.ensure_task_checkpoint(Path("/tasks/t"))
    (builder,) = fake_runta.runtimes.created
    assert builder.deleted
    assert fake_runta.stored == []  # a broken image is never snapshotted


def test_builder_that_never_becomes_ready_is_deleted(monkeypatch, fake_runta):
    """A create that waits and times out would raise without returning the
    handle; creating with wait=False is what leaves something to delete."""
    _patch_recipe(monkeypatch)
    fake_runta.never_ready = True
    with pytest.raises(TimeoutError):
        sandbox.ensure_task_checkpoint(Path("/tasks/t"))
    (builder,) = fake_runta.runtimes.created
    assert builder.deleted
    assert builder.execs == []


def test_ensure_checkpoint_force_replaces_the_existing_one(monkeypatch, fake_runta):
    _patch_recipe(monkeypatch)
    stale = _checkpoint(sandbox.checkpoint_name(Path("/tasks/t")))
    fake_runta.stored = [stale]

    rebuilt = sandbox.ensure_task_checkpoint(Path("/tasks/t"), force=True)

    assert fake_runta.deleted_checkpoints == [stale.id]
    assert rebuilt != stale.id
    assert len(fake_runta.runtimes.created) == 1


def test_ensure_checkpoint_waits_for_one_another_rollout_is_creating(monkeypatch, fake_runta):
    """The build lock is per rollout; a checkpoint a different rollout is midway
    through creating is waited for, not built a second time."""
    _patch_recipe(monkeypatch)
    creating = _checkpoint(sandbox.checkpoint_name(Path("/tasks/t")), state="creating")
    fake_runta.stored = [creating]

    assert sandbox.ensure_task_checkpoint(Path("/tasks/t")) == creating.id
    assert fake_runta.waited_for == [creating.id]
    assert fake_runta.runtimes.created == []


def test_ensure_checkpoint_clears_a_failed_one_and_rebuilds(monkeypatch, fake_runta):
    _patch_recipe(monkeypatch)
    failed = _checkpoint(sandbox.checkpoint_name(Path("/tasks/t")), state="failed")
    fake_runta.stored = [failed]

    rebuilt = sandbox.ensure_task_checkpoint(Path("/tasks/t"))

    assert fake_runta.deleted_checkpoints == [failed.id]
    assert rebuilt != failed.id
    assert len(fake_runta.runtimes.created) == 1


def test_checkpoint_state_is_read_from_the_sdk_enum(monkeypatch, fake_runta):
    """The SDK reports state as a str enum, not a bare string; the real one must
    select the same branches the fakes above do."""
    monkeypatch.delitem(sys.modules, "runta")  # the fixture's fake: this test needs the real enum
    runta = pytest.importorskip("runta")
    _patch_recipe(monkeypatch)
    name = sandbox.checkpoint_name(Path("/tasks/t"))
    ready = types.SimpleNamespace(id="ckpt-enum", display_name=name, state=runta.CheckpointState.READY)
    failed = types.SimpleNamespace(id="ckpt-bad", display_name=name, state=runta.CheckpointState.FAILED)
    fake_runta.stored = [failed, ready]

    assert sandbox.ensure_task_checkpoint(Path("/tasks/t")) == "ckpt-enum"
    assert fake_runta.deleted_checkpoints == ["ckpt-bad"]


# --- create_task_sandbox --------------------------------------------------------


def _patch_endpoint(monkeypatch, endpoint="https://api.runta.com"):
    """What the SDK resolves, without the SDK: the fixture's fake `runta` has no
    config resolution to ask."""
    monkeypatch.setattr(sandbox, "api_endpoint", lambda: endpoint)
    monkeypatch.delenv("RUNTA_INGRESS_BASE_URL", raising=False)


def _patch_create(monkeypatch):
    monkeypatch.setattr(sandbox, "ensure_task_checkpoint", lambda task_dir: "ckpt-abc")
    monkeypatch.setattr(sandbox, "server_cmd", lambda *a, **k: "TB2_X=1 serve --port 8000")
    monkeypatch.setattr(sandbox, "wait_server_ready", lambda url, timeout_s: None)
    monkeypatch.setattr(sandbox, "_start_keepalive", lambda runtime, task_id: None)
    _patch_endpoint(monkeypatch)
    monkeypatch.delenv("OPENENV_RUN_ID", raising=False)


def test_create_restores_the_checkpoint_with_the_port_published_and_a_ttl(monkeypatch, fake_runta):
    _patch_recipe(monkeypatch, resources=(2, 4096, 10240))
    _patch_create(monkeypatch)

    runtime, url = sandbox.create_task_sandbox(Path("/tasks/t"))

    assert fake_runta.runtimes.created == [runtime]
    assert runtime.name.startswith("miles-tb2-run-t-")
    assert runtime.kwargs["checkpoint_id"] == "ckpt-abc"
    # A restore comes back at the checkpoint's size; the SDK drops a size passed
    # with checkpoint_id, so passing one would only claim a control that is not there.
    assert runtime.kwargs.keys().isdisjoint({"vcpus", "memory_mib", "memory_max_mib", "resources"})
    assert runtime.kwargs["ingress_specs"] == [("https", 8000)]
    # the dead-man's switch: armed at create, with the shut-down remains reclaimed after
    assert runtime.kwargs["auto_shutdown_after_secs"] == sandbox._SANDBOX_TTL_S
    assert runtime.kwargs["auto_delete_after_secs"] > 0
    assert url == f"https://8000-{runtime.id}.runta.dev"
    assert not runtime.deleted


def test_create_runs_the_env_server_as_the_images_one_container(monkeypatch, fake_runta):
    """Detached under the Docker daemon — a process backgrounded from the exec
    with `&` dies when the exec returns — with the server's port published to
    the runtime, where the ingress serves it."""
    _patch_recipe(monkeypatch)
    _patch_create(monkeypatch)

    runtime, _url = sandbox.create_task_sandbox(Path("/tasks/t"))

    ((command, _timeout),) = runtime.execs
    assert command == (
        "docker run --detach --name tb2-env --publish 8000:8000 --entrypoint bash "
        "tb2-env:latest -c 'TB2_X=1 serve --port 8000'"
    )


def test_create_deletes_the_runtime_on_partial_failure(monkeypatch, fake_runta):
    """A runtime whose server never becomes ready must not leak until TTL."""
    _patch_recipe(monkeypatch)
    _patch_create(monkeypatch)

    def never_ready(url, timeout_s):
        raise TimeoutError("server not ready")

    monkeypatch.setattr(sandbox, "wait_server_ready", never_ready)
    with pytest.raises(TimeoutError):
        sandbox.create_task_sandbox(Path("/tasks/t"), ready_timeout_s=0.01)
    (runtime,) = fake_runta.runtimes.created
    assert runtime.deleted


def test_create_deletes_the_runtime_when_the_server_does_not_start(monkeypatch, fake_runta):
    _patch_recipe(monkeypatch)
    _patch_create(monkeypatch)
    fake_runta.exec_result = _Result(exit_code=125, stderr_text="docker: no such image")
    with pytest.raises(RuntimeError, match="starting the env server failed.*no such image"):
        sandbox.create_task_sandbox(Path("/tasks/t"))
    (runtime,) = fake_runta.runtimes.created
    assert runtime.deleted


def test_create_deletes_a_runtime_that_never_becomes_ready(monkeypatch, fake_runta):
    _patch_recipe(monkeypatch)
    _patch_create(monkeypatch)
    fake_runta.never_ready = True
    with pytest.raises(TimeoutError):
        sandbox.create_task_sandbox(Path("/tasks/t"))
    (runtime,) = fake_runta.runtimes.created
    assert runtime.deleted


def test_create_refuses_an_underivable_endpoint_before_anything_is_created(monkeypatch, fake_runta):
    """Where published ports are served is known before the first API call; an
    endpoint that does not say must not cost a checkpoint build and a runtime."""
    _patch_recipe(monkeypatch)
    _patch_create(monkeypatch)
    _patch_endpoint(monkeypatch, "http://127.0.0.1:8080")
    ensured = []
    monkeypatch.setattr(sandbox, "ensure_task_checkpoint", lambda task_dir: ensured.append(task_dir) or "ckpt-abc")

    with pytest.raises(ValueError, match="RUNTA_INGRESS_BASE_URL"):
        sandbox.create_task_sandbox(Path("/tasks/t"))

    assert ensured == []
    assert fake_runta.runtimes.created == []


def test_runtime_names_carry_the_run_id_when_set(monkeypatch):
    """Runta has no labels, so the name is the only attribution a sweep can key on."""
    monkeypatch.setenv("OPENENV_RUN_ID", "Run_42")
    name = sandbox._runtime_name("run", Path("/tasks/Regex_Chess"))
    assert name.startswith("miles-tb2-run-run-42-regex-chess-")
    monkeypatch.delenv("OPENENV_RUN_ID")
    assert sandbox._runtime_name("run", Path("/tasks/Regex_Chess")).startswith("miles-tb2-run-regex-chess-")


def test_runtime_names_are_unique_per_create():
    names = {sandbox._runtime_name("run", Path("/tasks/t")) for _ in range(20)}
    assert len(names) == 20


# --- the client ------------------------------------------------------------------


def test_make_client_is_built_once_with_the_resolved_token(monkeypatch):
    built = []
    mod = types.ModuleType("runta")
    mod.Runta = lambda **kwargs: built.append(kwargs) or object()
    monkeypatch.setitem(sys.modules, "runta", mod)
    monkeypatch.setattr(sandbox, "_client", None)
    monkeypatch.setattr(sandbox, "resolve_token", lambda: "rt_from_file")

    first = sandbox.make_client()
    assert sandbox.make_client() is first
    # the endpoint is not passed: RUNTA_ENDPOINT is the SDK's own to read
    assert built == [{"token": "rt_from_file"}]


# --- token supply ----------------------------------------------------------------


def test_resolve_token_env_value_wins(monkeypatch, tmp_path: Path):
    token_file = tmp_path / "token"
    token_file.write_text("rt_from_file\n")
    monkeypatch.setenv("RUNTA_TOKEN", "rt_from_env")
    monkeypatch.setenv("RUNTA_TOKEN_FILE", str(token_file))
    assert sandbox.resolve_token() == "rt_from_env"


def test_resolve_token_falls_back_to_file(monkeypatch, tmp_path: Path):
    # The launcher forwards only this path (never the value, which would be
    # echoed into driver logs via ray runtime_env); workers must read it here.
    token_file = tmp_path / "token"
    token_file.write_text("rt_from_file\n")
    monkeypatch.delenv("RUNTA_TOKEN", raising=False)
    monkeypatch.setenv("RUNTA_TOKEN_FILE", str(token_file))
    assert sandbox.resolve_token() == "rt_from_file"  # whitespace stripped


def test_resolve_token_default_path_under_home(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("RUNTA_TOKEN", raising=False)
    monkeypatch.delenv("RUNTA_TOKEN_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".config" / "runta"
    cfg.mkdir(parents=True)
    (cfg / "token").write_text("rt_default\n")
    assert sandbox.resolve_token() == "rt_default"


def test_resolve_token_errors_when_absent(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("RUNTA_TOKEN", raising=False)
    monkeypatch.setenv("RUNTA_TOKEN_FILE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="missing or empty"):
        sandbox.resolve_token()


def test_default_token_file_matches_the_credential_registry():
    """The launcher preflights the registry's path and the worker reads this
    module's; they must be the same file."""
    from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS

    assert sandbox._DEFAULT_TOKEN_FILE == PROVIDER_CREDENTIALS["runta"]["default_path"]


# --- base_url ---------------------------------------------------------------


_RUNTIME = types.SimpleNamespace(id="01a1c7e7-07b3")


def test_base_url_defaults_to_runta_cloud(monkeypatch):
    _patch_endpoint(monkeypatch)
    assert sandbox.base_url(_RUNTIME) == "https://8000-01a1c7e7-07b3.runta.dev"


def test_base_url_follows_an_api_dot_domain_endpoint(monkeypatch):
    _patch_endpoint(monkeypatch, "https://API.Example.Test/")
    assert sandbox.base_url(_RUNTIME) == "https://8000-01a1c7e7-07b3.example.test"


def test_base_url_knows_both_hosted_deployments(monkeypatch):
    """Runta's hosted ingress domains are not their API domains, so neither
    follows from the api.<domain> rule: api.runta.me would probe runta.me."""
    _patch_endpoint(monkeypatch, "https://api.runta.com")
    assert sandbox.base_url(_RUNTIME) == "https://8000-01a1c7e7-07b3.runta.dev"
    _patch_endpoint(monkeypatch, "https://api.runta.me")
    assert sandbox.base_url(_RUNTIME) == "https://8000-01a1c7e7-07b3.runta.sh"


def test_base_url_ingress_override_wins(monkeypatch):
    # A self-hosted deployment may serve published ports over plain HTTP, on a
    # port, somewhere its API endpoint says nothing about.
    _patch_endpoint(monkeypatch, "http://127.0.0.1:8080")
    monkeypatch.setenv("RUNTA_INGRESS_BASE_URL", "http://ingress.internal:9000")
    assert sandbox.base_url(_RUNTIME) == "http://8000-01a1c7e7-07b3.ingress.internal:9000"
    # ... and over an endpoint that does imply a domain
    _patch_endpoint(monkeypatch, "https://api.runta.me")
    monkeypatch.setenv("RUNTA_INGRESS_BASE_URL", "http://ingress.internal:9000")
    assert sandbox.base_url(_RUNTIME) == "http://8000-01a1c7e7-07b3.ingress.internal:9000"


def test_base_url_refuses_to_guess_for_an_underivable_endpoint(monkeypatch):
    _patch_endpoint(monkeypatch, "http://127.0.0.1:8080")
    with pytest.raises(ValueError, match="RUNTA_INGRESS_BASE_URL"):
        sandbox.base_url(_RUNTIME)


def test_base_url_follows_the_endpoint_the_sdk_resolves(monkeypatch, tmp_path: Path):
    """The client is built with no endpoint, so the SDK picks it: RUNTA_ENDPOINT,
    else its config file, else Runta Cloud. The published-port URL must follow
    that same answer — read off the variable alone, a config-file endpoint puts
    the runtimes on one deployment and the health probe on another."""
    runta = pytest.importorskip("runta")
    config = tmp_path / "config.toml"
    config.write_text('endpoint = "https://api.selfhosted.test"\n')
    monkeypatch.setenv("RUNTA_CONFIG", str(config))
    monkeypatch.delenv("RUNTA_ENDPOINT", raising=False)
    monkeypatch.delenv("RUNTA_INGRESS_BASE_URL", raising=False)

    def client_endpoint():
        with runta.Runta(token="rt_test") as client:  # as make_client builds it; no request is made
            return client._http.endpoint

    assert sandbox.api_endpoint() == client_endpoint() == "https://api.selfhosted.test"
    assert sandbox.base_url(_RUNTIME) == "https://8000-01a1c7e7-07b3.selfhosted.test"

    monkeypatch.setenv("RUNTA_ENDPOINT", "https://api.runta.me")  # the variable outranks the file
    assert sandbox.api_endpoint() == client_endpoint() == "https://api.runta.me"
    assert sandbox.base_url(_RUNTIME) == "https://8000-01a1c7e7-07b3.runta.sh"

    monkeypatch.delenv("RUNTA_ENDPOINT")
    monkeypatch.setenv("RUNTA_CONFIG", str(tmp_path / "absent.toml"))
    assert sandbox.api_endpoint() == client_endpoint() == "https://api.runta.com"
    assert sandbox.base_url(_RUNTIME) == "https://8000-01a1c7e7-07b3.runta.dev"


# --- launch preflight --------------------------------------------------------


def test_preflight_fails_the_launch_for_an_underivable_endpoint(monkeypatch):
    """The launcher's check: no SDK call, no client, nothing created — just the
    error an operator would otherwise meet once per episode."""
    _patch_endpoint(monkeypatch, "http://127.0.0.1:8080")
    monkeypatch.setattr(sandbox, "make_client", lambda: pytest.fail("preflight must not need a client"))
    with pytest.raises(ValueError, match=r"127\.0\.0\.1:8080.*RUNTA_INGRESS_BASE_URL"):
        sandbox.preflight()


def test_preflight_passes_and_echoes_where_ports_are_served(monkeypatch, capsys):
    _patch_endpoint(monkeypatch, "https://api.runta.me")
    sandbox.preflight()
    assert "<port>-<runtime id>.runta.sh" in capsys.readouterr().out

    _patch_endpoint(monkeypatch, "http://127.0.0.1:8080")  # underivable, but named outright
    monkeypatch.setenv("RUNTA_INGRESS_BASE_URL", "http://ingress.internal:9000")
    sandbox.preflight()
    assert "http://<port>-<runtime id>.ingress.internal:9000" in capsys.readouterr().out


# --- TTL / keepalive ---------------------------------------------------------


def test_sandbox_ttl_armed_by_default():
    # The dead-man's-switch contract: runtimes are created with a bounded
    # lifetime, or a hard-killed caller's orphans run forever.
    assert sandbox._SANDBOX_TTL_S > 0
    assert sandbox._KEEPALIVE_INTERVAL_S < sandbox._SANDBOX_TTL_S  # beats within the window


def test_keepalive_beats_then_exits_on_persistent_failure(monkeypatch):
    monkeypatch.setattr(sandbox, "_KEEPALIVE_INTERVAL_S", 0.02)

    class Stub:
        def __init__(self):
            self.beats = []
            self.dead = False

        def update_timeout_policies(self, **policies):
            if self.dead:
                raise RuntimeError("runtime deleted")
            self.beats.append(policies)

    stub = Stub()
    sandbox._start_keepalive(stub, "regex-chess")
    deadline = time.time() + 2.0
    while len(stub.beats) < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert len(stub.beats) >= 3  # beats while the runtime is alive
    # each beat saves the session limit again, which is what restarts its clock
    assert stub.beats[0] == {"auto_shutdown_after_secs": sandbox._SANDBOX_TTL_S}

    stub.dead = True  # episode over, runtime deleted -> thread must exit
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not any("keepalive" in t.name for t in threading.enumerate()):
            break
        time.sleep(0.01)
    assert not any("keepalive" in t.name for t in threading.enumerate())
