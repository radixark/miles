"""Tests for the E2B sandbox half: template aliasing, key supply, and the
orphan-TTL keepalive lifecycle.

Not collected by the repo-level pytest run (testpaths = ./tests); run manually
when touching the recipe:

    pytest examples/experimental/openenv/tests/ -q
"""

import concurrent.futures
import sys
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tb2_sandbox_e2b as sandbox  # noqa: E402


# --- template aliasing -------------------------------------------------------


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


def test_template_alias_is_deterministic_and_sanitized(monkeypatch):
    _patch_recipe(monkeypatch)
    a1 = sandbox.template_alias(Path("/tasks/Regex_Chess"))
    a2 = sandbox.template_alias(Path("/tasks/Regex_Chess"))
    assert a1 == a2
    assert a1.startswith("tb2-regex-chess-")  # lowercased, non [a-z0-9-] mapped to '-'


def test_template_alias_tracks_recipe_content(monkeypatch):
    """The digest must cover base image AND build commands: editing either
    re-bakes instead of silently serving a stale template."""
    _patch_recipe(monkeypatch)
    original = sandbox.template_alias(Path("/tasks/t"))
    _patch_recipe(monkeypatch, commands=("apt-get install -y curl", "extra step"))
    assert sandbox.template_alias(Path("/tasks/t")) != original
    _patch_recipe(monkeypatch, base="ghcr.io/laude/task:2.0")
    assert sandbox.template_alias(Path("/tasks/t")) != original


def test_template_alias_tracks_build_resources(monkeypatch):
    """E2B sizes sandboxes at template-build time, so a task.toml resource
    bump must re-bake instead of warm-starting under-provisioned sandboxes."""
    _patch_recipe(monkeypatch)
    small = sandbox.template_alias(Path("/tasks/t"))
    _patch_recipe(monkeypatch, resources={"cpu_count": 4, "memory_mb": 8192})
    big = sandbox.template_alias(Path("/tasks/t"))
    assert small != big


class _FakeTemplateCls:
    """Stands in for e2b.Template: records build calls, controls alias_exists."""

    instances = None  # set per test

    def __init__(self):
        self.commands = []
        self.steps = []  # ordered build steps, to prove set_user precedes them

    def from_image(self, base):
        self.base = base
        return self

    def set_user(self, user):
        self.user = user
        self.steps.append(("set_user", user))
        return self

    def run_cmd(self, cmd):
        self.commands.append(cmd)
        self.steps.append(("run_cmd", cmd))
        return self

    @staticmethod
    def alias_exists(alias, **kwargs):
        return _FakeTemplateCls.exists

    @staticmethod
    def build(template, alias, **kwargs):
        _FakeTemplateCls.builds.append((template, alias, kwargs))


@pytest.fixture
def fake_e2b(monkeypatch):
    _FakeTemplateCls.exists = False
    _FakeTemplateCls.builds = []
    mod = types.ModuleType("e2b")
    mod.Template = _FakeTemplateCls
    monkeypatch.setitem(sys.modules, "e2b", mod)
    monkeypatch.setattr(sandbox, "resolve_api_key", lambda: "e2b_" + "0" * 40)
    return mod


def test_ensure_template_short_circuits_on_existing_alias(monkeypatch, fake_e2b):
    _patch_recipe(monkeypatch)
    _FakeTemplateCls.exists = True
    alias = sandbox.ensure_task_template(Path("/tasks/t"))
    assert alias == sandbox.template_alias(Path("/tasks/t"))
    assert _FakeTemplateCls.builds == []  # no rebuild when the alias exists


def test_ensure_template_builds_with_recipe_and_resources(monkeypatch, fake_e2b):
    _patch_recipe(monkeypatch, resources={"cpu_count": 3, "memory_mb": 4096})
    alias = sandbox.ensure_task_template(Path("/tasks/t"))
    ((template, built_alias, kwargs),) = _FakeTemplateCls.builds
    assert built_alias == alias
    assert template.base == "ghcr.io/laude/task:1.0"
    assert template.commands == ["apt-get install -y curl"]
    assert kwargs["cpu_count"] == 3 and kwargs["memory_mb"] == 4096
    assert kwargs["skip_cache"] is False
    assert kwargs["api_key"].startswith("e2b_")


def test_ensure_template_builds_as_root(monkeypatch, fake_e2b):
    """E2B runs build commands as a NON-root user by default, which fails every
    layer of the recipe (apt-get exits 100, /opt is not writable) — observed on
    E2B Cloud, invisible on a self-hosted AgentENV that builds as root. The user
    must be set BEFORE the first command, since set_user only affects what
    follows it."""
    _patch_recipe(monkeypatch, commands=("apt-get install -y curl", "mkdir -p /opt/envserver"))
    sandbox.ensure_task_template(Path("/tasks/t"))
    ((template, _, _),) = _FakeTemplateCls.builds
    assert template.user == "root"
    assert template.steps[0] == ("set_user", "root")


def test_template_alias_tracks_the_build_user(monkeypatch, fake_e2b):
    """The user is part of the baked artifact, so changing it must re-bake
    rather than serve a template built by whoever built it first."""
    _patch_recipe(monkeypatch)
    before = sandbox.template_alias(Path("/tasks/t"))
    monkeypatch.setattr(sandbox, "_BUILD_USER", "someone-else")
    assert sandbox.template_alias(Path("/tasks/t")) != before


def test_ensure_template_force_rebuilds_skipping_cache(monkeypatch, fake_e2b):
    _patch_recipe(monkeypatch)
    _FakeTemplateCls.exists = True  # force must rebuild anyway
    sandbox.ensure_task_template(Path("/tasks/t"), force=True)
    ((_, _, kwargs),) = _FakeTemplateCls.builds
    assert kwargs["skip_cache"] is True


def test_ensure_template_enforces_build_wall_clock(monkeypatch, fake_e2b):
    """build_timeout_s bounds the whole blocking build, not one HTTP request."""
    _patch_recipe(monkeypatch)

    def hanging_build(template, alias, **kwargs):
        time.sleep(2)

    monkeypatch.setattr(_FakeTemplateCls, "build", staticmethod(hanging_build))
    with pytest.raises(concurrent.futures.TimeoutError):
        sandbox.ensure_task_template(Path("/tasks/t"), build_timeout_s=0.05)


class _FakeSandbox:
    def __init__(self):
        self.killed = False
        self.ran = []
        self.commands = type("C", (), {"run": lambda _self, *a, **k: self.ran.append((a, k))})()

    def get_host(self, port):
        return f"{port}-id.example.test"

    def kill(self, **kwargs):
        self.killed = True


def test_create_task_sandbox_kills_on_partial_failure(monkeypatch, fake_e2b):
    """A sandbox whose server never becomes ready must not leak until TTL."""
    created = _FakeSandbox()
    fake_e2b.Sandbox = type("S", (), {"create": staticmethod(lambda **k: created)})
    monkeypatch.setattr(sandbox, "ensure_task_template", lambda task_dir: "tb2-t-abc")
    monkeypatch.setattr(sandbox, "sandbox_labels", lambda task_dir: {})
    monkeypatch.setattr(sandbox, "server_cmd", lambda *a, **k: "serve")

    def never_ready(url, timeout_s):
        raise TimeoutError("server not ready")

    monkeypatch.setattr(sandbox, "wait_server_ready", never_ready)
    with pytest.raises(TimeoutError):
        sandbox.create_task_sandbox(Path("/tasks/t"), ready_timeout_s=0.01)
    assert created.killed


def test_create_runs_the_env_server_as_root(monkeypatch, fake_e2b):
    """The env server executes the agent's commands, so the user it runs as IS
    the task environment's user. E2B defaults to a non-root user; a TB2 task
    image expects root (its own tests apt-install), so the server must be
    started as root or the agent silently loses privileges it should have."""
    created = _FakeSandbox()
    fake_e2b.Sandbox = type("S", (), {"create": staticmethod(lambda **k: created)})
    monkeypatch.setattr(sandbox, "ensure_task_template", lambda task_dir: "tb2-t-abc")
    monkeypatch.setattr(sandbox, "sandbox_labels", lambda task_dir: {})
    monkeypatch.setattr(sandbox, "server_cmd", lambda *a, **k: "serve")
    monkeypatch.setattr(sandbox, "wait_server_ready", lambda url, timeout_s: None)
    monkeypatch.setattr(sandbox, "_start_keepalive", lambda sb, task_id: None)

    sandbox.create_task_sandbox(Path("/tasks/t"))

    ((args, kwargs),) = created.ran
    assert kwargs["user"] == "root"
    assert kwargs["background"] is True
    assert "serve" in args[0]


def test_build_lock_is_per_alias():
    lock_a1 = sandbox._build_lock("tb2-a-123")
    lock_a2 = sandbox._build_lock("tb2-a-123")
    lock_b = sandbox._build_lock("tb2-b-456")
    assert lock_a1 is lock_a2
    assert lock_a1 is not lock_b


def test_task_build_resources_from_task_toml(tmp_path: Path):
    (tmp_path / "task.toml").write_text("[environment]\ncpus = 4\nmemory_mb = 8192\n")
    assert sandbox.task_build_resources(tmp_path) == {"cpu_count": 4, "memory_mb": 8192}


def test_task_build_resources_floors(tmp_path: Path):
    (tmp_path / "task.toml").write_text("[environment]\ncpus = 0\nmemory_mb = 128\n")
    assert sandbox.task_build_resources(tmp_path) == {"cpu_count": 1, "memory_mb": 2048}


# --- key supply --------------------------------------------------------------


def test_resolve_api_key_env_value_wins(monkeypatch, tmp_path: Path):
    key_file = tmp_path / "api_key"
    key_file.write_text("e2b_from_file\n")
    monkeypatch.setenv("E2B_API_KEY", "e2b_from_env")
    monkeypatch.setenv("E2B_API_KEY_FILE", str(key_file))
    assert sandbox.resolve_api_key() == "e2b_from_env"


def test_resolve_api_key_falls_back_to_file(monkeypatch, tmp_path: Path):
    # The launcher forwards only this path (never the value, which would be
    # echoed into driver logs via ray runtime_env); workers must read it here.
    key_file = tmp_path / "api_key"
    key_file.write_text("e2b_from_file\n")
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("E2B_API_KEY_FILE", str(key_file))
    assert sandbox.resolve_api_key() == "e2b_from_file"  # whitespace stripped


def test_resolve_api_key_default_path_under_home(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.delenv("E2B_API_KEY_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".config" / "e2b"
    cfg.mkdir(parents=True)
    (cfg / "api_key").write_text("e2b_default\n")
    assert sandbox.resolve_api_key() == "e2b_default"


def test_resolve_api_key_errors_when_absent(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("E2B_API_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="missing or empty"):
        sandbox.resolve_api_key()


# --- base_url ---------------------------------------------------------------


class _HostStub:
    def get_host(self, port):
        return f"{port}-sbx123.example.dev"


def test_base_url_default_https(monkeypatch):
    monkeypatch.delenv("OPENENV_E2B_URL_SCHEME", raising=False)
    assert sandbox.base_url(_HostStub()) == "https://8000-sbx123.example.dev"


def test_base_url_scheme_override(monkeypatch):
    # Self-hosted gateways (e.g. an AgentENV deployment inside the tailnet)
    # may terminate plain HTTP.
    monkeypatch.setenv("OPENENV_E2B_URL_SCHEME", "http")
    assert sandbox.base_url(_HostStub()) == "http://8000-sbx123.example.dev"


# --- TTL / keepalive ---------------------------------------------------------


def test_sandbox_ttl_armed_by_default():
    # The dead-man's-switch contract: sandboxes are created with a bounded
    # lifetime, or a hard-killed caller's orphans run forever.
    assert sandbox._SANDBOX_TTL_S > 0
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
