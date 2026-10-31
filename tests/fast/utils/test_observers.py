"""Run observers hear the run in plain data, and nothing an observer does can break it."""

import logging
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from miles.rollout.agentic.sandbox_reports import report_sandbox
from miles.utils import observers as observers_module
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.misc import SingletonMeta
from miles.utils.observers import Observer, Observers, RunRefusedError
from miles.utils.timer import Timer


class _Recorder(Observer):
    def __init__(self, env: dict[str, str] | None = None):
        self.env = env or {}
        self.calls: list[tuple] = []

    def start_run(self, run):
        self.calls.append(("start_run", run))
        return self.env

    def start_process(self, process):
        self.calls.append(("start_process", process))

    def phase_start(self, name, t0):
        self.calls.append(("phase_start", name))

    def phase_end(self, name, t0, t1):
        self.calls.append(("phase_end", name))

    def trajectory_event(self, event):
        self.calls.append(("trajectory_event", event))

    def sandbox_report(self, report):
        self.calls.append(("sandbox_report", report))


class _Broken(Observer):
    def start_run(self, run):
        raise RuntimeError("observer bug")

    def phase_end(self, name, t0, t1):
        raise RuntimeError("observer bug")


@pytest.fixture
def install(monkeypatch):
    """Installs observers as if their packages registered entry points; undoes everything they hooked into."""
    monkeypatch.setattr(observers_module, "get_current_node_ip", lambda: "10.0.0.1")
    sinks_before = list(Timer().event_sinks)

    def _install(*observers: Observer) -> Observers:
        SingletonMeta._instances.pop(Observers, None)
        installed = [SimpleNamespace(name=type(o).__name__, load=lambda o=o: lambda: o) for o in observers]
        monkeypatch.setattr(observers_module, "entry_points", lambda group: installed)
        return Observers()

    yield _install
    SingletonMeta._instances.pop(Observers, None)
    Timer().event_sinks[:] = sinks_before
    TrajectoryLifecycle().sink = None


def test_without_observers_nothing_is_hooked(install):
    install()
    sinks_before = list(Timer().event_sinks)

    assert observers_module.start_run(run_uuid="0123456789abcdef", args=SimpleNamespace(lr=1e-6)) == {}
    observers_module.start_process("rollout_manager")

    assert Timer().event_sinks == sinks_before
    assert TrajectoryLifecycle().sink is None


def test_a_failing_observer_breaks_neither_the_run_nor_the_other_observers(install, caplog):
    recorder = _Recorder(env={"COLLECTOR": "http://10.0.0.1:9"})
    install(_Broken(), recorder)

    env = observers_module.start_run(run_uuid="0123456789abcdef", args=SimpleNamespace(lr=1e-6))
    observers_module.start_process("train", rank=3, gpus=lambda: ["5"])
    with caplog.at_level(logging.WARNING):
        Timer().start("actor_train")
        Timer().end("actor_train")
        Timer().start("actor_train")
        Timer().end("actor_train")

    assert env == {"COLLECTOR": "http://10.0.0.1:9"}
    assert ("start_process", dict(role="train", node="10.0.0.1", rank=3, gpus=[5])) in recorder.calls
    assert [name for kind, name in recorder.calls if kind == "phase_end"] == ["actor_train", "actor_train"]
    warnings = [r for r in caplog.records if "failed in phase_end" in r.message]
    assert len(warnings) == 1, "a repeating failure must warn once per interval, not once per event"


def test_only_start_run_can_refuse_the_run(install):
    """A refusal stops the run before it starts; raised any later, it is isolated like any other failure."""

    class _Refusing(Observer):
        def start_run(self, run):
            raise RunRefusedError("needs --dump-details")

        def phase_end(self, name, t0, t1):
            raise RunRefusedError("too late to refuse")

    install(_Refusing())

    with pytest.raises(RunRefusedError, match="needs --dump-details"):
        observers_module.start_run(run_uuid="0123456789abcdef", args=SimpleNamespace(lr=1e-6))
    observers_module.start_process("train", rank=0, gpus=lambda: [0])
    Timer().start("actor_train")
    Timer().end("actor_train")


def test_a_refused_run_still_ends_the_observers_that_started_it(install):
    """A run record an earlier observer opened must not stay open when a later observer refuses the run."""
    ended: list[bool] = []

    class _Starting(Observer):
        def end_run(self, *, succeeded):
            ended.append(succeeded)

    class _Refusing(Observer):
        def start_run(self, run):
            raise RunRefusedError("needs --dump-details")

    install(_Starting(), _Refusing())

    with pytest.raises(RunRefusedError):
        observers_module.start_run(run_uuid="0123456789abcdef", args=SimpleNamespace(lr=1e-6))
    assert ended == [False]


def test_start_run_reports_every_argument_as_plain_data_without_secrets(install):
    recorder = _Recorder()
    install(recorder)

    class _ServerConfig(BaseModel):
        api_key: str
        port: int

    server = _ServerConfig(api_key="server-secret", port=30000)
    args = SimpleNamespace(lr=1e-6, sglang=object(), wandb_key="wandb-secret", server=server)
    observers_module.start_run(run_uuid="0123456789abcdef", args=args)

    [(_, run)] = [call for call in recorder.calls if call[0] == "start_run"]
    assert run["run_uuid"] == "0123456789abcdef"
    assert run["args"]["lr"] == 1e-6
    assert isinstance(run["args"]["sglang"], str)
    assert run["args"]["server"]["port"] == 30000, "a nested config must arrive as data, not its repr"
    for secret in ("wandb-secret", "server-secret"):
        assert secret not in str(run["args"]), "observers must not receive the run's secrets"


def _sample(**overrides):
    fields = dict(
        index=7,
        group_index=2,
        metadata={},
        status=SimpleNamespace(value="completed"),
        all_weight_version_spans=[SimpleNamespace(version="3"), SimpleNamespace(version="4")],
    )
    return SimpleNamespace(**(fields | overrides))


def test_agentic_turns_become_generation_and_agent_gap_spans(install):
    recorder = _Recorder()
    install(recorder)
    observers_module.start_process("rollout_manager")
    turn_samples = [
        _sample(metadata={"lifecycle": dict(t0=10.0, t1=12.0, turn=1)}),
        _sample(metadata={"lifecycle": dict(t0=15.0, t1=18.0, turn=2, prev_t1=12.0, req_ts=14.0)}),
    ]

    TrajectoryLifecycle().sink.attempt_end(turn_samples)

    events = [call[1] for call in recorder.calls if call[0] == "trajectory_event"]
    assert [(e["kind"], e["ts"], e["turn"]) for e in events[:-1]] == [
        ("gen_start", 10.0, 1),
        ("gen_end", 12.0, 1),
        ("tool_start", 12.0, 2),
        ("tool_end", 14.0, 2),
        ("gen_start", 15.0, 2),
        ("gen_end", 18.0, 2),
    ]
    assert events[-1]["kind"] == "attempt_end"
    assert events[-1]["detail"] == "completed"
    assert all(e["sample_index"] == 7 and e["group_index"] == 2 and e["weight_version"] == "4" for e in events)


def test_the_lifecycle_seam_stays_empty_when_no_observer_takes_sample_lifecycles(install):
    """Lifecycle events cost work for every sample, so no reporter is installed when no observer takes them."""

    class _MetricsOnly(Observer):
        def metrics(self, metrics, *, step_key, step):
            pass

    install(_MetricsOnly())
    observers_module.start_process("rollout_manager")

    assert TrajectoryLifecycle().sink is None


def test_a_sample_the_reporter_cannot_read_never_reaches_the_rollout(install, caplog):
    recorder = _Recorder()
    install(recorder)
    observers_module.start_process("rollout_manager")

    with caplog.at_level(logging.WARNING):
        TrajectoryLifecycle().sink.attempt_start(SimpleNamespace(index=1))
        TrajectoryLifecycle().sink.attempt_end([SimpleNamespace(index=1, metadata=None)])

    assert not [call for call in recorder.calls if call[0] == "trajectory_event"]
    assert any("failed in trajectory_event" in r.message for r in caplog.records)


def test_a_sandbox_report_reaches_the_observers_as_plain_data(install):
    """What agent code reports about a sandbox reaches every observer as plain data."""
    recorder = _Recorder()
    install(recorder)

    report_sandbox("sb-1", task="t", provider="agentenv", status="error", error="exit 1")

    [report] = [args[0] for name, *args in recorder.calls if name == "sandbox_report"]
    assert report == dict(
        ts=report["ts"],
        sandbox_key="sb-1",
        status="error",
        task="t",
        provider="agentenv",
        sandbox_id=None,
        error="exit 1",
    )


def test_an_observer_that_does_not_take_part_is_dropped_as_if_never_installed(monkeypatch):
    """An installed package whose observer sits a run out must leave the run as if it were not installed."""

    class _Off(Observer):
        def enabled(self):
            return False

    installed = [SimpleNamespace(name="off", load=lambda: _Off), SimpleNamespace(name="on", load=lambda: _Recorder)]
    monkeypatch.setattr(observers_module, "entry_points", lambda group: installed)

    assert [name for name, _ in observers_module._load_observers()] == ["on"]


def test_only_events_some_observer_implements_are_heard(install):
    """Reports that cost real work to build, like each step's samples, are skipped when no observer implements them."""

    class _MetricsOnly(Observer):
        def metrics(self, metrics, *, step_key, step):
            pass

    observers = install(_MetricsOnly())

    assert observers.hears("metrics")
    assert not observers.hears("rollout_samples")
