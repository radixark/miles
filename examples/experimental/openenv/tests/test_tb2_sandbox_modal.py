"""Tests for the Modal sandbox half: image layering, sandbox sizing, the
tunnel URL, and the create path's lifetime/diagnostic contract.

Not collected by the repo-level pytest run (testpaths = ./tests); run manually
when touching the recipe:

    pytest examples/experimental/openenv/tests/ -q
"""

import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tb2_sandbox_modal as sandbox  # noqa: E402

_COMMANDS = ("apt-get install -y curl", "curl -LsSf https://astral.sh/uv/install.sh | sh", "mkdir -p /opt/tb2-tasks")


# --- fake modal SDK ---------------------------------------------------------


class _FakeImage:
    """Records layering: every run_commands call is one layer."""

    def __init__(self, tag, layers=()):
        self.tag = tag
        self.layers = list(layers)

    def run_commands(self, *commands):
        return _FakeImage(self.tag, [*self.layers, commands])

    @staticmethod
    def from_registry(tag, **kwargs):
        _FakeImage.from_registry_kwargs = kwargs
        return _FakeImage(tag)


class _FakeTunnel:
    def __init__(self, url):
        self.url = url


class _FakeSandbox:
    """Stands in for one created sandbox."""

    def __init__(self, exit_code=None, output="", tunnel_url="https://abc.r5.modal.host"):
        self._exit_code = exit_code
        self.stdout = types.SimpleNamespace(read=lambda: output)
        self.terminated = False
        self.tunnel_timeout = None
        self._tunnel_url = tunnel_url

    def tunnels(self, timeout=50):
        self.tunnel_timeout = timeout
        return {8000: _FakeTunnel(self._tunnel_url)}

    def terminate(self):
        self.terminated = True

    def poll(self):
        return self._exit_code


class _FakeSandboxCls:
    created = None
    instance = None

    @staticmethod
    def create(*args, **kwargs):
        _FakeSandboxCls.created = {"args": args, "kwargs": kwargs}
        return _FakeSandboxCls.instance


class _FakeApp:
    lookups = None

    @staticmethod
    def lookup(name, **kwargs):
        _FakeApp.lookups.append((name, kwargs))
        return f"app:{name}"


@pytest.fixture
def fake_modal(monkeypatch):
    """Install a fake `modal` module and reset the backend's process-wide app cache."""
    _FakeApp.lookups = []
    _FakeSandboxCls.instance = _FakeSandbox()
    mod = types.ModuleType("modal")
    mod.Image = _FakeImage
    mod.Sandbox = _FakeSandboxCls
    mod.App = _FakeApp
    monkeypatch.setitem(sys.modules, "modal", mod)
    monkeypatch.setattr(sandbox, "_app", None)
    monkeypatch.setattr(sandbox, "resolve_docker_image", lambda task_dir, override=None: "ghcr.io/laude/task:1.0")
    monkeypatch.setattr(sandbox, "server_layer_commands", lambda task_dir: list(_COMMANDS))
    monkeypatch.setattr(sandbox, "sandbox_labels", lambda task_dir: {"openenv-tbench2-task": task_dir.name})
    monkeypatch.setattr(sandbox, "server_cmd", lambda timeout, default_task_id="": f"serve {default_task_id}")
    monkeypatch.setattr(sandbox, "wait_server_ready", lambda url, timeout_s=300.0: None)
    monkeypatch.setattr(sandbox, "task_resources", lambda task_dir: {"cpu": 2.0, "memory": 4096})
    return mod


# --- image layering ---------------------------------------------------------


def test_task_image_is_one_layer_per_recipe_command(fake_modal):
    """Per-command layers so that editing the recipe's tail re-runs only the
    layers below the edit; collapsing the recipe into one layer would reinstall
    apt/uv for every such change. (The saving is per task: a task's own base
    image makes its chain hash unique, so nothing is shared across tasks.)"""
    image = sandbox.task_image(Path("/tasks/regex-chess"))
    assert image.tag == "ghcr.io/laude/task:1.0"
    assert image.layers == [(c,) for c in _COMMANDS]


def test_task_image_pulls_anonymously_by_default(fake_modal):
    """No registry secret is passed: the TB2 task images are public, and a
    private one would have to be opted into explicitly."""
    sandbox.task_image(Path("/tasks/t"))
    assert _FakeImage.from_registry_kwargs == {}


# --- sandbox sizing ---------------------------------------------------------


def test_task_resources_floors_and_omits_disk(tmp_path):
    """Modal is billed on max(request, actual) so the request tracks task.toml
    (floored by the recipe); storage_mb has no Modal counterpart and must not
    leak in as an unexpected kwarg."""
    (tmp_path / "task.toml").write_text("[environment]\ncpus = 4\nmemory_mb = 8192\nstorage_mb = 20480\n")
    assert sandbox.task_resources(tmp_path) == {"cpu": 4.0, "memory": 8192}

    (tmp_path / "task.toml").write_text("[environment]\n")
    assert sandbox.task_resources(tmp_path) == {"cpu": 1.0, "memory": 2048}


# --- app handle -------------------------------------------------------------


def test_app_is_looked_up_once_per_process(fake_modal, monkeypatch):
    """App.lookup is a network round-trip and a rollout fans out many episodes."""
    monkeypatch.setenv("OPENENV_MODAL_APP", "openenv-tbench2")
    monkeypatch.setattr(sandbox, "_APP_NAME", "openenv-tbench2")
    assert sandbox.get_app() == "app:openenv-tbench2"
    assert sandbox.get_app() == "app:openenv-tbench2"
    assert _FakeApp.lookups == [("openenv-tbench2", {"create_if_missing": True})]


# --- create path ------------------------------------------------------------


def test_create_passes_the_lifetime_and_ownership_contract(fake_modal):
    sandbox_obj, url = sandbox.create_task_sandbox(
        Path("/tasks/regex-chess"), command_timeout_s=600, ttl_s=1200, idle_timeout_s=120
    )
    created = _FakeSandboxCls.created
    kwargs = created["kwargs"]

    assert url == "https://abc.r5.modal.host"
    assert sandbox_obj is _FakeSandboxCls.instance
    # The env server is the sandbox's entrypoint, not a follow-up exec: the task
    # image's own CMD must not decide what runs here.
    assert created["args"] == ("bash", "-c", "serve regex-chess")
    assert kwargs["encrypted_ports"] == [8000]
    assert kwargs["timeout"] == 1200
    assert kwargs["idle_timeout"] == 120
    assert kwargs["tags"] == {"openenv-tbench2-task": "regex-chess"}
    assert kwargs["cpu"] == 2.0 and kwargs["memory"] == 4096
    assert not sandbox_obj.terminated


def test_create_defaults_lifetime_from_the_env_knobs(fake_modal, monkeypatch):
    monkeypatch.setattr(sandbox, "_SANDBOX_TTL_S", 1800)
    monkeypatch.setattr(sandbox, "_IDLE_TIMEOUT_S", 300)
    sandbox.create_task_sandbox(Path("/tasks/t"))
    kwargs = _FakeSandboxCls.created["kwargs"]
    assert kwargs["timeout"] == 1800
    assert kwargs["idle_timeout"] == 300


def test_create_terminates_the_sandbox_when_the_server_never_comes_up(fake_modal, monkeypatch):
    """A ready-timeout must not leak the sandbox, and the failure should carry
    the exited server's output instead of only a timeout."""
    _FakeSandboxCls.instance = _FakeSandbox(exit_code=1, output="Traceback: no module named tbench2_env\n")

    def boom(url, timeout_s=300.0):
        raise RuntimeError("health check timed out")

    monkeypatch.setattr(sandbox, "wait_server_ready", boom)

    with pytest.raises(RuntimeError, match="no module named tbench2_env"):
        sandbox.create_task_sandbox(Path("/tasks/regex-chess"))
    assert _FakeSandboxCls.instance.terminated


def test_create_does_not_read_the_stream_of_a_live_sandbox(fake_modal, monkeypatch):
    """Reading stdout of a RUNNING sandbox would block on a stream that never
    closes, so a still-running sandbox contributes no diagnostics."""

    def exploding_read():
        raise AssertionError("stdout of a live sandbox must not be read")

    _FakeSandboxCls.instance = _FakeSandbox(exit_code=None)
    _FakeSandboxCls.instance.stdout = types.SimpleNamespace(read=exploding_read)

    def boom(url, timeout_s=300.0):
        raise RuntimeError("health check timed out")

    monkeypatch.setattr(sandbox, "wait_server_ready", boom)

    with pytest.raises(RuntimeError, match="health check timed out"):
        sandbox.create_task_sandbox(Path("/tasks/t"))
    assert _FakeSandboxCls.instance.terminated


def test_create_enforces_the_build_wall_clock(fake_modal, monkeypatch):
    """The first create for a task builds the image, which Modal does not
    deadline itself; a hung build must give the slot back rather than block the
    episode forever."""
    release = threading.Event()

    def slow_create(*args, **kwargs):
        release.wait(10)
        return _FakeSandboxCls.instance

    monkeypatch.setattr(_FakeSandboxCls, "create", staticmethod(slow_create))
    try:
        with pytest.raises(TimeoutError):
            sandbox.create_task_sandbox(Path("/tasks/t"), create_timeout_s=0.05)
    finally:
        release.set()


# --- tunnel ----------------------------------------------------------------


def test_base_url_is_the_tunnel_for_the_env_server_port(fake_modal, monkeypatch):
    monkeypatch.setattr(sandbox, "_TUNNEL_TIMEOUT_S", 60.0)
    sb = _FakeSandbox(tunnel_url="https://xyz.r5.modal.host")
    assert sandbox.base_url(sb) == "https://xyz.r5.modal.host"
    assert sb.tunnel_timeout == 60
