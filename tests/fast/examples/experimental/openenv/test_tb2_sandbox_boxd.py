"""Tests for the boxd sandbox half: machine sizing, the Dockerfile, snapshot
naming, the bake, the per-episode create and its cleanup, and key supply.

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import tb2_sandbox_boxd as sandbox

# --- sizing ------------------------------------------------------------------


@pytest.mark.parametrize(
    "resources, expected",
    [
        ((1, 2048, 10240), (1, 4096)),
        ((1, 4096, 10240), (1, 4096)),
        ((1, 8192, 10240), (2, 8192)),
        ((2, 2048, 10240), (2, 8192)),
        ((4, 4096, 10240), (4, 16384)),
        ((8, 32768, 10240), (8, 32768)),
    ],
)
def test_machine_size_is_the_smallest_covering_size(monkeypatch, resources, expected):
    monkeypatch.setattr(sandbox, "task_env_resources", lambda task_dir: resources)
    assert sandbox.machine_size(Path("/tasks/t")) == expected


def test_machine_size_refuses_a_task_above_the_largest_size(monkeypatch):
    monkeypatch.setattr(sandbox, "task_env_resources", lambda task_dir: (16, 4096, 10240))
    with pytest.raises(ValueError, match="largest boxd size"):
        sandbox.machine_size(Path("/tasks/t"))


# --- Dockerfile and naming ---------------------------------------------------


def _patch_recipe(monkeypatch, base="ghcr.io/laude/task:1.0", commands=("apt-get install -y curl",), size=(1, 4096)):
    monkeypatch.setattr(sandbox, "resolve_docker_image", lambda task_dir, override: base)
    monkeypatch.setattr(sandbox, "server_layer_commands", lambda task_dir: list(commands))
    monkeypatch.setattr(sandbox, "machine_size", lambda task_dir: size)


def test_dockerfile_is_the_task_image_plus_one_run_per_layer_as_root(monkeypatch):
    _patch_recipe(monkeypatch, commands=("step one", "step two"))
    lines = sandbox.dockerfile(Path("/tasks/t")).splitlines()
    assert lines[0] == "FROM ghcr.io/laude/task:1.0"
    # root before the first layer: the recipe apt-installs and writes /opt
    assert lines[1] == "USER root"
    assert lines[2:4] == ["RUN step one", "RUN step two"]
    # the task image's own entrypoint must not wrap the env server command
    assert "ENTRYPOINT []" in lines


def test_snapshot_name_is_deterministic_and_sanitized(monkeypatch):
    _patch_recipe(monkeypatch)
    n1 = sandbox.snapshot_name(Path("/tasks/Regex_Chess"))
    assert n1 == sandbox.snapshot_name(Path("/tasks/Regex_Chess"))
    assert n1.startswith("tb2-regex-chess-")


def test_snapshot_name_tracks_recipe_and_size(monkeypatch):
    """Editing the recipe or the size re-bakes instead of serving a stale snapshot."""
    _patch_recipe(monkeypatch)
    original = sandbox.snapshot_name(Path("/tasks/t"))
    _patch_recipe(monkeypatch, commands=("apt-get install -y curl", "extra step"))
    assert sandbox.snapshot_name(Path("/tasks/t")) != original
    _patch_recipe(monkeypatch, base="ghcr.io/laude/task:2.0")
    assert sandbox.snapshot_name(Path("/tasks/t")) != original
    _patch_recipe(monkeypatch, size=(2, 8192))
    assert sandbox.snapshot_name(Path("/tasks/t")) != original


@pytest.mark.parametrize("prefix", ["tb2-", "t-run-", "a-much-longer-prefix-"])
def test_names_honor_the_prefix_and_fit_a_dns_label(monkeypatch, prefix):
    """Every name is a DNS label (the machine URL is https://<name>.<domain>),
    and its random part survives a long task id: one task runs many episodes at
    once, and two processes may bake one snapshot at once."""
    _patch_recipe(monkeypatch)
    monkeypatch.setattr(sandbox, "_NAME_PREFIX", prefix)
    task = Path("/tasks/" + "x" * 80)
    snapshot = sandbox.snapshot_name(task)
    machines = {sandbox._machine_name(task) for _ in range(20)}
    builders = {sandbox._builder_name(task) for _ in range(20)}
    assert len(machines) == 20 and len(builders) == 20
    for name in {snapshot, *machines, *builders}:
        assert name.startswith(prefix) and len(name) <= 63 and not name.endswith("-"), name
    assert snapshot.endswith("-" + sandbox._digest(task))


def test_a_prefix_that_leaves_no_room_is_refused(monkeypatch):
    _patch_recipe(monkeypatch)
    monkeypatch.setattr(sandbox, "_NAME_PREFIX", "p" * 60)
    with pytest.raises(ValueError, match="OPENENV_BOXD_NAME_PREFIX"):
        sandbox._machine_name(Path("/tasks/t"))


# --- fake boxd client ---------------------------------------------------------


class _NotFound(Exception):
    pass


class _FakeMachines:
    def __init__(self, client):
        self.client = client
        self.files = SimpleNamespace(upload=lambda mid, path, data: client.calls.append(("upload", mid, path)))

    def create(self, name, **kwargs):
        self.client.calls.append(("create", name, kwargs))
        mid = f"id-{len(self.client.created)}"
        self.client.created.append(mid)
        return SimpleNamespace(id=mid, name=name)

    def wait_until_ready(self, mid, timeout):
        self.client.calls.append(("ready", mid))
        if "ready" in self.client.fail_on:
            raise TimeoutError("not ready")
        return SimpleNamespace(id=mid, access=SimpleNamespace(url=f"https://{mid}.nyc.boxd.sh"))

    def exec(self, mid, command, timeout):
        self.client.calls.append(("exec", mid, command, timeout))
        code = 1 if any(word in command for word in self.client.fail_on) else 0
        return SimpleNamespace(exit_code=code, stdout="#1 build\n", stderr="boom" if code else "")

    def set_bot_protection(self, mid, mode):
        self.client.calls.append(("bot", mid, mode))

    def set_auto_hibernate_timeout(self, mid, seconds):
        self.client.calls.append(("hibernate_after", mid, seconds))

    def delete(self, mid):
        self.client.calls.append(("delete", mid))


class _FakeSnapshots:
    """Each get pops one state: a status, read at the version saved before the
    test's own bake, or a ``(status, version)`` pair."""

    def __init__(self, client):
        self.client = client

    def get(self, name, org=None):
        if not self.client.snapshot_states:
            raise _NotFound(name)
        state = self.client.snapshot_states.pop(0)
        status, version = state if isinstance(state, tuple) else (state, self.client.saved_version)
        return SimpleNamespace(status=status, version=version)

    def create(self, mid, name):
        self.client.calls.append(("snapshot", mid, name))
        old = self.client.saved_version
        # a save over an existing name: the earlier version stays readable until the new one is ready
        stale = [("ready", old)] if old else []
        self.client.snapshot_states = [("pending", old), *stale, ("ready", old + 1)]
        return SimpleNamespace(version=old + 1)


class _CreateRefused:
    """The retrying client's machines: a create through it is the bug."""

    def __init__(self, machines):
        self._machines = machines

    def __getattr__(self, name):
        return getattr(self._machines, name)

    def create(self, name, **kwargs):
        raise AssertionError("machine creates go through make_client(retry=False)")


class _FakeClient:
    def __init__(self, snapshot_states=(), fail_on=()):
        self.calls = []
        self.created = []
        self.snapshot_states = list(snapshot_states)
        self.saved_version = 1 if snapshot_states else 0
        self.fail_on = fail_on
        self.no_retry = SimpleNamespace(machines=_FakeMachines(self))
        self.machines = _CreateRefused(self.no_retry.machines)
        self.snapshots = _FakeSnapshots(self)

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]


@pytest.fixture
def fake(monkeypatch):
    def install(**kwargs):
        client = _FakeClient(**kwargs)
        monkeypatch.setattr(sandbox, "make_client", lambda retry=True: client if retry else client.no_retry)
        monkeypatch.setattr(sandbox, "_get_snapshot", _snapshot_via(client))
        monkeypatch.setattr(sandbox, "_SNAPSHOT_POLL_S", 0.0)
        monkeypatch.setattr(
            sandbox,
            "sandbox_labels",
            lambda task_dir: {"openenv-tbench2-task": task_dir.name, "openenv-run-id": "run-7"},
        )
        monkeypatch.setattr(sandbox, "_max_lifetime_s", lambda: 5400)

        def health(url, timeout_s):
            client.calls.append(("health", url))
            if "health" in client.fail_on:
                raise TimeoutError("no /health")

        monkeypatch.setattr(sandbox, "wait_server_ready", health)
        _patch_recipe(monkeypatch)
        return client

    return install


def _snapshot_via(client):
    def get(_client, name):
        try:
            return client.snapshots.get(name)
        except _NotFound:
            return None

    return get


# --- bake ---------------------------------------------------------------------


def test_existing_ready_snapshot_is_reused_without_a_build(fake):
    client = fake(snapshot_states=["ready"])
    name = sandbox.ensure_task_snapshot(Path("/tasks/t"))
    assert name == sandbox.snapshot_name(Path("/tasks/t"))
    assert client.of("create") == []


def test_missing_snapshot_is_baked_on_an_isolated_builder_that_is_then_deleted(fake):
    client = fake()
    name = sandbox.ensure_task_snapshot(Path("/tasks/t"))

    (create,) = client.of("create")
    assert create[2]["isolated"] is True
    assert create[2]["vcpu"] == 1 and create[2]["memory"] == "4G"
    # no run id: one snapshot serves many runs, and its machines inherit these labels
    assert create[2]["labels"] == {"openenv-tbench2-task": "t"}
    # a builder whose caller dies mid-bake is reclaimed by its timer
    assert create[2]["delete_after"] > sandbox._BUILD_TIMEOUT_S
    assert client.of("upload") == [("upload", "id-0", f"{sandbox._BUILD_DIR}/Dockerfile")]
    (build,) = client.of("exec")
    assert "docker build" in build[2] and sandbox._image_tag(name) in build[2]
    assert build[3] == sandbox._BUILD_TIMEOUT_S
    assert client.of("snapshot") == [("snapshot", "id-0", name)]
    assert client.of("delete") == [("delete", "id-0")]


def test_a_pending_snapshot_is_awaited_not_rebuilt(fake):
    client = fake(snapshot_states=["pending", "pending", "ready"])
    sandbox.ensure_task_snapshot(Path("/tasks/t"))
    assert client.of("create") == []


def test_the_build_timeout_reaches_the_build(fake):
    client = fake()
    sandbox.ensure_task_snapshot(Path("/tasks/t"), build_timeout_s=42)
    (build,) = client.of("exec")
    assert build[3] == 42


def test_a_failed_snapshot_is_rebaked(fake):
    client = fake(snapshot_states=["failed"])
    sandbox.ensure_task_snapshot(Path("/tasks/t"))
    assert len(client.of("snapshot")) == 1


def test_a_pending_snapshot_that_fails_raises(fake):
    fake(snapshot_states=["pending", "failed"])
    with pytest.raises(RuntimeError, match="'failed'"):
        sandbox.ensure_task_snapshot(Path("/tasks/t"))


def test_force_rebuilds_an_existing_snapshot(fake):
    client = fake(snapshot_states=["ready"])
    sandbox.ensure_task_snapshot(Path("/tasks/t"), force=True)
    assert len(client.of("create")) == 1


def test_a_rebake_waits_for_the_version_it_saved(fake):
    """Saving a name again keeps the earlier version readable until the new one
    is ready: the bake must not take that earlier "ready" for its own."""
    client = fake(snapshot_states=["ready"])
    sandbox.ensure_task_snapshot(Path("/tasks/t"), force=True)
    # every state was read, the stale ("ready", 1) included, before the builder went
    assert client.snapshot_states == []
    assert client.of("delete") == [("delete", "id-0")]


def test_a_snapshot_wait_timeout_leaves_the_builder(fake, monkeypatch):
    """The save may still be running on the builder: its delete-after timer
    reclaims it, a delete here could cut the save short."""
    client = fake()

    def never_ready(_client, name, *, version=None):
        raise TimeoutError("not ready")

    monkeypatch.setattr(sandbox, "_wait_snapshot_ready", never_ready)
    with pytest.raises(TimeoutError):
        sandbox.ensure_task_snapshot(Path("/tasks/t"))
    assert len(client.of("snapshot")) == 1
    assert client.of("delete") == []


def test_a_snapshot_that_fails_to_save_deletes_the_builder(fake, monkeypatch):
    client = fake()
    snapshots = client.snapshots
    create = snapshots.create

    def create_then_fail(mid, name):
        saved = create(mid, name)
        client.snapshot_states = [("failed", 0)]
        return saved

    snapshots.create = create_then_fail
    with pytest.raises(RuntimeError, match="'failed'"):
        sandbox.ensure_task_snapshot(Path("/tasks/t"))
    assert client.of("delete") == [("delete", "id-0")]


def test_a_failed_build_deletes_the_builder_and_raises_with_the_log(fake):
    client = fake(fail_on=("docker build",))
    with pytest.raises(RuntimeError, match="boom"):
        sandbox.ensure_task_snapshot(Path("/tasks/t"))
    assert client.of("snapshot") == []
    assert client.of("delete") == [("delete", "id-0")]


# --- per-episode create -------------------------------------------------------


def test_create_starts_from_the_snapshot_and_serves_on_the_machine_url(fake):
    client = fake(snapshot_states=["ready"])
    machine_id, url = sandbox.create_task_sandbox(Path("/tasks/fix-git"), command_timeout_s=123)

    name = sandbox.snapshot_name(Path("/tasks/fix-git"))
    (create,) = client.of("create")
    assert create[2]["from_snapshot"] == name
    assert create[2]["isolated"] is True
    assert create[2]["auto_suspend_timeout"] == 0
    # this episode's own labels, run id included, merged over the snapshot's
    assert create[2]["labels"] == {"openenv-tbench2-task": "fix-git", "openenv-run-id": "run-7"}
    assert create[2]["delete_after"] == 5400
    assert machine_id == "id-0" and url == "https://id-0.nyc.boxd.sh"
    assert client.of("bot") == [("bot", "id-0", "off")]
    assert client.of("hibernate_after") == [("hibernate_after", "id-0", sandbox._IDLE_HIBERNATE_S)]
    (run,) = client.of("exec")
    assert run[2].startswith("docker run -d ")
    assert "-p 8000:8000" in run[2] and sandbox._image_tag(name) in run[2]
    assert "TB2_COMMAND_TIMEOUT_S=123" in run[2] and "TB2_DEFAULT_TASK_ID=fix-git" in run[2]
    assert client.of("health") == [("health", url)]
    assert client.of("delete") == []


@pytest.mark.parametrize("failing", ["ready", "docker run", "health"])
def test_a_failed_start_deletes_the_machine_and_raises(fake, failing):
    client = fake(snapshot_states=["ready"], fail_on=(failing,))
    with pytest.raises((RuntimeError, TimeoutError)):
        sandbox.create_task_sandbox(Path("/tasks/t"))
    assert client.of("delete") == [("delete", "id-0")]


def test_the_max_lifetime_is_the_episode_cap_plus_slack(monkeypatch):
    import openenv_agent_function

    monkeypatch.setattr(openenv_agent_function, "MAX_ROLLOUT_TIME_S", 1000.0)
    assert sandbox._max_lifetime_s() == 1000 + sandbox._MAX_LIFETIME_SLACK_S


# --- snapshot lookup and client against the SDK's shapes -----------------------


class _SdkNotFound(Exception):
    pass


@pytest.fixture
def fake_sdk(monkeypatch):
    """A stand-in `boxd` module: the status lookup imports NotFoundError from it."""
    module = types.ModuleType("boxd")
    module.NotFoundError = _SdkNotFound
    module.built = []

    def boxd_client(**kwargs):
        module.built.append(kwargs)
        return SimpleNamespace(options=kwargs)

    module.Boxd = boxd_client
    monkeypatch.setitem(sys.modules, "boxd", module)
    monkeypatch.setenv("BOXD_ORG", "my-org")


def _snapshots(get):
    return SimpleNamespace(snapshots=SimpleNamespace(get=get))


def test_get_snapshot_maps_not_found_to_none_and_passes_the_org(fake_sdk):
    seen = {}

    def get(name, org=None):
        seen.update(name=name, org=org)
        raise _SdkNotFound(name)

    assert sandbox._get_snapshot(_snapshots(get), "snap") is None
    assert seen == {"name": "snap", "org": "my-org"}


def test_get_snapshot_returns_the_snapshot(fake_sdk):
    snap = SimpleNamespace(status="ready", archived=False, storage="local")
    assert sandbox._get_snapshot(_snapshots(lambda name, org=None: snap), "snap") is snap


def test_an_archived_snapshot_is_a_clear_error(fake_sdk):
    snap = SimpleNamespace(status="ready", archived=True, storage="archived")
    with pytest.raises(RuntimeError, match="archived"):
        sandbox._get_snapshot(_snapshots(lambda name, org=None: snap), "snap")


@pytest.mark.parametrize("state", [("pending", 1), ("ready", 1)])
def test_waiting_on_a_snapshot_has_a_deadline(monkeypatch, state):
    """Also when the version waited for never shows up behind an earlier ready one."""
    snap = SimpleNamespace(status=state[0], version=state[1])
    monkeypatch.setattr(sandbox, "_get_snapshot", lambda client, name: snap)
    monkeypatch.setattr(sandbox, "_SNAPSHOT_POLL_S", 0.0)
    with pytest.raises(TimeoutError):
        sandbox._wait_snapshot_ready(object(), "snap", version=2, timeout_s=0.01)


def test_the_create_client_never_resends_and_both_clients_are_built_once(fake_sdk, monkeypatch):
    """A resent create whose first send arrived would come back as a name
    conflict, so the client used for creates has the SDK's retries off."""
    monkeypatch.setenv("BOXD_API_KEY", "key")
    monkeypatch.setattr(sandbox, "_clients", {})
    retrying, creating = sandbox.make_client(), sandbox.make_client(retry=False)
    assert retrying.options == {"api_key": "key"}
    assert creating.options == {"api_key": "key", "max_retries": 0}
    assert sandbox.make_client() is retrying and sandbox.make_client(retry=False) is creating
    assert len(sys.modules["boxd"].built) == 2


def test_delete_sandbox_deletes_by_id(fake):
    client = fake()
    sandbox.delete_sandbox("id-7")
    assert client.of("delete") == [("delete", "id-7")]


# --- key supply ---------------------------------------------------------------


def test_api_key_prefers_the_environment_then_the_file(monkeypatch, tmp_path):
    key_file = tmp_path / "api_key"
    key_file.write_text("key-from-file\n")
    monkeypatch.setenv("BOXD_API_KEY_FILE", str(key_file))
    monkeypatch.delenv("BOXD_API_KEY", raising=False)
    assert sandbox.resolve_api_key() == "key-from-file"
    monkeypatch.setenv("BOXD_API_KEY", "key-from-env")
    assert sandbox.resolve_api_key() == "key-from-env"
