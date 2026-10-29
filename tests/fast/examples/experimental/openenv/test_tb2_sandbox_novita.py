"""Tests for the Novita sandbox half: template naming, the pinned start
command, key supply, the public-host URL, and the orphan-TTL keepalive lifecycle.

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q
"""

import concurrent.futures
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import tb2_sandbox_novita as sandbox


# --- template naming -----------------------------------------------------------


def _patch_recipe(
    monkeypatch,
    base="ghcr.io/laude/task:1.0",
    commands=("apt-get install -y curl",),
    resources=None,
):
    resources = resources or {"cpu_count": 1, "memory_mb": 2048}
    monkeypatch.setattr(sandbox, "resolve_docker_image", lambda task_dir, override: base)
    monkeypatch.setattr(sandbox, "server_layer_commands", lambda task_dir: list(commands))
    monkeypatch.setattr(sandbox, "task_build_resources", lambda task_dir: dict(resources))


def test_template_name_is_deterministic_and_sanitized(monkeypatch):
    _patch_recipe(monkeypatch)
    n1 = sandbox.template_name(Path("/tasks/Regex_Chess"))
    n2 = sandbox.template_name(Path("/tasks/Regex_Chess"))
    assert n1 == n2
    assert n1.startswith("tb2-regex-chess-")  # lowercased, non [a-z0-9-] mapped to '-'


def test_template_name_tracks_recipe_content(monkeypatch):
    """The digest must cover base image AND build commands: editing either
    re-bakes instead of silently serving a stale template."""
    _patch_recipe(monkeypatch)
    original = sandbox.template_name(Path("/tasks/t"))
    _patch_recipe(monkeypatch, commands=("apt-get install -y curl", "extra step"))
    assert sandbox.template_name(Path("/tasks/t")) != original
    _patch_recipe(monkeypatch, base="ghcr.io/laude/task:2.0")
    assert sandbox.template_name(Path("/tasks/t")) != original


def test_template_name_tracks_build_resources(monkeypatch):
    """Novita sizes sandboxes at template-build time, so a task.toml resource
    bump must re-bake instead of starting under-provisioned sandboxes."""
    _patch_recipe(monkeypatch)
    small = sandbox.template_name(Path("/tasks/t"))
    _patch_recipe(monkeypatch, resources={"cpu_count": 4, "memory_mb": 8192})
    assert sandbox.template_name(Path("/tasks/t")) != small


def test_template_name_tracks_the_build_user_and_start_command(monkeypatch):
    """Both are part of the baked artifact, so changing either must re-bake
    rather than serve a template built the old way."""
    _patch_recipe(monkeypatch)
    before = sandbox.template_name(Path("/tasks/t"))
    monkeypatch.setattr(sandbox, "_BUILD_USER", "someone-else")
    assert sandbox.template_name(Path("/tasks/t")) != before
    monkeypatch.setattr(sandbox, "_BUILD_USER", "root")
    monkeypatch.setattr(sandbox, "_KEEPALIVE_CMD", "sleep 1d")
    assert sandbox.template_name(Path("/tasks/t")) != before


# --- fake novita SDK ---------------------------------------------------------


class _FakeReadyCmd:
    def __init__(self, cmd):
        self._cmd = cmd

    def get_cmd(self):
        return self._cmd


class _FakeFinal:
    """What set_start_cmd returns: a buildable template that takes no more steps."""

    def __init__(self, builder, start_cmd, ready_cmd):
        self.builder = builder
        self.start_cmd = start_cmd
        self.ready_cmd = ready_cmd


class _FakeTemplateCls:
    """Stands in for novita_sandbox.Template: records build calls, controls exists()."""

    exists_value = False
    builds = None

    def __init__(self, **kwargs):
        self.commands = []
        self.steps = []  # ordered build steps, to prove set_user precedes them

    def from_image(self, base, **kwargs):
        self.base = base
        return self

    def set_user(self, user):
        self.user = user
        self.steps.append(("set_user", user))
        return self

    def run_cmd(self, cmd, user=None):
        self.commands.append(cmd)
        self.steps.append(("run_cmd", cmd))
        return self

    def set_start_cmd(self, start_cmd, ready_cmd):
        self.steps.append(("set_start_cmd", start_cmd))
        return _FakeFinal(self, start_cmd, ready_cmd)

    @staticmethod
    def exists(name, **kwargs):
        _FakeTemplateCls.exists_queries.append(name)
        return _FakeTemplateCls.exists_value

    @staticmethod
    def build(template, name, **kwargs):
        _FakeTemplateCls.builds.append((template, name, kwargs))


class _FakeSandbox:
    def __init__(self, host="8000-sbx123.us-phx-1.sandbox.novita.ai"):
        self.killed = False
        self.ran = []
        self.timeouts = []
        self._host = host
        self.commands = types.SimpleNamespace(run=lambda *a, **k: self.ran.append((a, k)))

    def get_host(self, port):
        return self._host

    def set_timeout(self, timeout, **opts):
        self.timeouts.append(timeout)

    def kill(self, **kwargs):
        self.killed = True
        return True


@pytest.fixture
def fake_novita(monkeypatch):
    _FakeTemplateCls.exists_value = False
    _FakeTemplateCls.exists_queries = []
    _FakeTemplateCls.builds = []
    mod = types.ModuleType("novita_sandbox")
    mod.Template = _FakeTemplateCls
    mod.wait_for_timeout = lambda ms: _FakeReadyCmd(f"sleep {ms // 1000}")
    monkeypatch.setitem(sys.modules, "novita_sandbox", mod)
    monkeypatch.setattr(sandbox, "resolve_api_key", lambda: "nvt_" + "0" * 32)
    return mod


# --- template build ------------------------------------------------------------


def test_ensure_template_short_circuits_on_existing_name(monkeypatch, fake_novita):
    _patch_recipe(monkeypatch)
    _FakeTemplateCls.exists_value = True
    name = sandbox.ensure_task_template(Path("/tasks/t"))
    assert name == sandbox.template_name(Path("/tasks/t"))
    assert _FakeTemplateCls.exists_queries == [name]
    assert _FakeTemplateCls.builds == []  # no rebuild when the name exists


def test_ensure_template_builds_with_recipe_resources_and_key(monkeypatch, fake_novita):
    _patch_recipe(monkeypatch, resources={"cpu_count": 3, "memory_mb": 4096})
    name = sandbox.ensure_task_template(Path("/tasks/t"))
    ((final, built_name, kwargs),) = _FakeTemplateCls.builds
    assert built_name == name
    assert final.builder.base == "ghcr.io/laude/task:1.0"
    assert final.builder.commands == ["apt-get install -y curl"]
    assert kwargs["cpu_count"] == 3 and kwargs["memory_mb"] == 4096
    assert kwargs["skip_cache"] is False
    assert kwargs["api_key"].startswith("nvt_")


def test_ensure_template_builds_as_root_before_the_first_command(monkeypatch, fake_novita):
    """Novita's parser defaults subsequent instructions to a non-root user,
    which fails every layer of the recipe (apt-get exits 100, /opt is not
    writable). The user must be set BEFORE the first command, since set_user
    only affects what follows it."""
    _patch_recipe(monkeypatch, commands=("apt-get install -y curl", "mkdir -p /opt/envserver"))
    sandbox.ensure_task_template(Path("/tasks/t"))
    ((final, _, _),) = _FakeTemplateCls.builds
    assert final.builder.user == "root"
    assert final.builder.steps[0] == ("set_user", "root")


def test_ensure_template_pins_a_keepalive_start_command_last(monkeypatch, fake_novita):
    """The SDK inherits the task image's own CMD as the template start command;
    pinning a keepalive is what keeps port 8000 free for the env server the
    create exec's. It has to come last: set_start_cmd closes the builder."""
    _patch_recipe(monkeypatch, commands=("apt-get install -y curl", "mkdir -p /opt/envserver"))
    sandbox.ensure_task_template(Path("/tasks/t"))
    ((final, _, _),) = _FakeTemplateCls.builds
    assert final.start_cmd == "sleep infinity"
    assert final.ready_cmd.get_cmd() == "sleep 5"
    assert final.builder.steps[-1] == ("set_start_cmd", "sleep infinity")
    assert final.builder.steps[-2] == ("run_cmd", "mkdir -p /opt/envserver")


def test_ensure_template_force_rebuilds_skipping_cache(monkeypatch, fake_novita):
    _patch_recipe(monkeypatch)
    _FakeTemplateCls.exists_value = True  # force must rebuild anyway
    sandbox.ensure_task_template(Path("/tasks/t"), force=True)
    ((_, _, kwargs),) = _FakeTemplateCls.builds
    assert kwargs["skip_cache"] is True


def test_ensure_template_enforces_build_wall_clock(monkeypatch, fake_novita):
    """build_timeout_s bounds the whole blocking build, not one HTTP request."""
    _patch_recipe(monkeypatch)

    def hanging_build(template, name, **kwargs):
        time.sleep(2)

    monkeypatch.setattr(_FakeTemplateCls, "build", staticmethod(hanging_build))
    with pytest.raises(concurrent.futures.TimeoutError):
        sandbox.ensure_task_template(Path("/tasks/t"), build_timeout_s=0.05)


def test_task_build_resources_from_task_toml(tmp_path: Path):
    (tmp_path / "task.toml").write_text("[environment]\ncpus = 4\nmemory_mb = 8192\n")
    assert sandbox.task_build_resources(tmp_path) == {"cpu_count": 4, "memory_mb": 8192}


def test_task_build_resources_floors(tmp_path: Path):
    (tmp_path / "task.toml").write_text("[environment]\ncpus = 0\nmemory_mb = 128\n")
    assert sandbox.task_build_resources(tmp_path) == {"cpu_count": 1, "memory_mb": 2048}


# --- create path -----------------------------------------------------------------


def _wire_create(monkeypatch, fake_novita, created):
    fake_novita.Sandbox = type("S", (), {"create": staticmethod(lambda **k: created)})
    fake_novita.Sandbox.created_kwargs = None

    def create(**k):
        fake_novita.Sandbox.created_kwargs = k
        return created

    fake_novita.Sandbox.create = staticmethod(create)
    monkeypatch.setattr(sandbox, "ensure_task_template", lambda task_dir: "tb2-t-abc")
    monkeypatch.setattr(sandbox, "sandbox_labels", lambda task_dir: {"openenv-tbench2-task": task_dir.name})
    monkeypatch.setattr(sandbox, "server_cmd", lambda timeout, default_task_id="": f"serve {default_task_id}")


def test_create_passes_the_lifetime_and_ownership_contract(monkeypatch, fake_novita):
    created = _FakeSandbox()
    _wire_create(monkeypatch, fake_novita, created)
    monkeypatch.setattr(sandbox, "wait_server_ready", lambda url, timeout_s: None)
    monkeypatch.setattr(sandbox, "_start_keepalive", lambda sb, task_id: None)
    monkeypatch.setattr(sandbox, "_SANDBOX_TTL_S", 1200)

    sb, url = sandbox.create_task_sandbox(Path("/tasks/regex-chess"))

    assert sb is created
    assert url == "https://8000-sbx123.us-phx-1.sandbox.novita.ai"
    kwargs = fake_novita.Sandbox.created_kwargs
    assert kwargs["template"] == "tb2-t-abc"
    assert kwargs["timeout"] == 1200
    assert kwargs["metadata"] == {"openenv-tbench2-task": "regex-chess"}
    assert kwargs["api_key"].startswith("nvt_")
    assert not created.killed


def test_create_runs_the_env_server_as_root_in_the_background(monkeypatch, fake_novita):
    """The env server executes the agent's commands, so the user it runs as IS
    the task environment's user; and the SDK's per-command deadline must not
    apply to a server that runs for the whole episode."""
    created = _FakeSandbox()
    _wire_create(monkeypatch, fake_novita, created)
    monkeypatch.setattr(sandbox, "wait_server_ready", lambda url, timeout_s: None)
    monkeypatch.setattr(sandbox, "_start_keepalive", lambda sb, task_id: None)

    sandbox.create_task_sandbox(Path("/tasks/regex-chess"))

    ((args, kwargs),) = created.ran
    assert "serve regex-chess" in args[0]
    assert kwargs["user"] == "root"
    assert kwargs["background"] is True
    assert kwargs["timeout"] == 0


def test_create_task_sandbox_kills_on_partial_failure(monkeypatch, fake_novita):
    """A sandbox whose server never becomes ready must not leak until TTL."""
    created = _FakeSandbox()
    _wire_create(monkeypatch, fake_novita, created)

    def never_ready(url, timeout_s):
        raise TimeoutError("server not ready")

    monkeypatch.setattr(sandbox, "wait_server_ready", never_ready)
    with pytest.raises(TimeoutError):
        sandbox.create_task_sandbox(Path("/tasks/t"), ready_timeout_s=0.01)
    assert created.killed


def test_create_refuses_a_loopback_host(monkeypatch, fake_novita):
    """NOVITA_DEBUG makes get_host return localhost: a URL that silently targets
    the caller's machine. Refuse it, and do not leak the sandbox."""
    created = _FakeSandbox(host="localhost:8000")
    _wire_create(monkeypatch, fake_novita, created)
    with pytest.raises(RuntimeError, match="NOVITA_DEBUG"):
        sandbox.create_task_sandbox(Path("/tasks/t"))
    assert created.killed


def test_kill_sandbox_resupplies_the_key(monkeypatch, fake_novita):
    created = _FakeSandbox()
    seen = {}

    def kill(**opts):
        seen.update(opts)
        created.killed = True

    created.kill = kill
    sandbox.kill_sandbox(created)
    assert created.killed and seen["api_key"].startswith("nvt_")


# --- base_url --------------------------------------------------------------------


class _HostStub:
    def __init__(self, host):
        self._host = host

    def get_host(self, port):
        return f"{port}-{self._host}" if "localhost" not in self._host else self._host


def test_base_url_is_https_on_the_per_port_host():
    assert sandbox.base_url(_HostStub("sbx123.us-phx-1.sandbox.novita.ai")) == (
        "https://8000-sbx123.us-phx-1.sandbox.novita.ai"
    )


def test_base_url_refuses_loopback_and_schemed_hosts():
    with pytest.raises(RuntimeError, match="NOVITA_DEBUG"):
        sandbox.base_url(_HostStub("localhost:8000"))
    with pytest.raises(RuntimeError):
        sandbox.base_url(_HostStub("https://evil.example"))


# --- key supply -----------------------------------------------------------------


def test_resolve_api_key_env_value_wins(monkeypatch, tmp_path: Path):
    key_file = tmp_path / "api_key"
    key_file.write_text("nvt_from_file\n")
    monkeypatch.setenv("NOVITA_API_KEY", "nvt_from_env")
    monkeypatch.setenv("NOVITA_API_KEY_FILE", str(key_file))
    assert sandbox.resolve_api_key() == "nvt_from_env"


def test_resolve_api_key_falls_back_to_file(monkeypatch, tmp_path: Path):
    # The launcher forwards only this path (never the value, which would be
    # echoed into driver logs via ray runtime_env); workers must read it here.
    key_file = tmp_path / "api_key"
    key_file.write_text("nvt_from_file\n")
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    monkeypatch.setenv("NOVITA_API_KEY_FILE", str(key_file))
    assert sandbox.resolve_api_key() == "nvt_from_file"  # whitespace stripped


def test_resolve_api_key_default_path_under_home(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    monkeypatch.delenv("NOVITA_API_KEY_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".config" / "novita"
    cfg.mkdir(parents=True)
    (cfg / "api_key").write_text("nvt_default\n")
    assert sandbox.resolve_api_key() == "nvt_default"


def test_resolve_api_key_errors_when_absent(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    monkeypatch.setenv("NOVITA_API_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="missing or empty"):
        sandbox.resolve_api_key()


# --- TTL / keepalive --------------------------------------------------------------


def test_sandbox_ttl_armed_by_default():
    # The dead-man's-switch contract: sandboxes are created with a bounded
    # lifetime, or a hard-killed caller's orphans run forever. The TTL also has
    # to fit under Novita's 1 h set_timeout cap for accounts without the
    # long_running marker, or every re-arm would be refused.
    assert 0 < sandbox._SANDBOX_TTL_S <= 3600
    assert sandbox._KEEPALIVE_INTERVAL_S < sandbox._SANDBOX_TTL_S  # beats within the window


def test_keepalive_beats_then_exits_on_persistent_failure(monkeypatch):
    monkeypatch.setattr(sandbox, "_KEEPALIVE_INTERVAL_S", 0.02)
    monkeypatch.setattr(sandbox, "_connection_opts", lambda: {})

    class Stub:
        def __init__(self):
            self.beats = 0
            self.dead = False

        def set_timeout(self, timeout, **opts):
            if self.dead:
                raise RuntimeError("sandbox killed")
            self.beats += 1

    stub = Stub()
    sandbox._start_keepalive(stub, "regex-chess")
    deadline = time.time() + 2.0
    while stub.beats < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert stub.beats >= 3  # beats while the sandbox is alive

    stub.dead = True  # episode over, sandbox killed -> thread must exit
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not any("keepalive" in t.name for t in threading.enumerate()):
            break
        time.sleep(0.01)
    assert not any("keepalive" in t.name for t in threading.enumerate())
