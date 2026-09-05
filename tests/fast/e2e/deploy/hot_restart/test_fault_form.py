import threading
from pathlib import Path
from random import Random
from typing import Any

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import fault_form as fault_form_module
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import compute_hot_restart_workloads
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import HOT_RESTART_ARG, compute_release_of_config
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import RunProgress
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import (
    HOT_RESTART_FORM_NAME,
    HotRestartFaultForm,
    restamped_replaced_workloads,
)

from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.naming import ORCHESTRATOR_COMPONENT
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name

CELL: dict = {"metadata": {"name": "actor-0"}}
CONFIG: ExecuteTrainConfig = ExecuteTrainConfig(run_id="demo", namespace="rl")
RELEASE: str = compute_release_of_config(CONFIG)
STAMPED: frozenset[str] = compute_hot_restart_workloads(RELEASE)
ORCHESTRATOR: str = component_name(RELEASE, ORCHESTRATOR_COMPONENT)
ROLLOUT_EXECUTOR: str = component_name(RELEASE, ROLLOUT_EXECUTOR_POOL_ID)
TRAINER: str = "a-workload-a-take-over-leaves-alone"


def _form(launch, **overrides: Any) -> HotRestartFaultForm:
    kwargs: dict[str, Any] = dict(
        launch=launch,
        config=CONFIG,
        checkpoint_dir=Path("/dumps/checkpoints"),
        events_dir=Path("/dumps/events"),
        poll_interval_seconds=0.0,
        timeout_seconds=5.0,
    )
    kwargs.update(overrides)
    return HotRestartFaultForm(**kwargs)


def _install_run(
    monkeypatch,
    *,
    finished: int | None = 1,
    saved: int | None = 1,
    stamps: list[dict[str, str | None] | None] | None = None,
) -> None:
    reads = list(stamps) if stamps is not None else [_stamps("t1"), _stamps("t2")]
    monkeypatch.setattr(
        fault_form_module,
        "read_restart_stamp_of_workload",
        lambda **_kwargs: reads.pop(0) if len(reads) > 1 else reads[0],
    )
    monkeypatch.setattr(
        fault_form_module,
        "read_run_progress",
        lambda **_kwargs: RunProgress(last_saved_iteration=saved, last_finished_rollout_id=finished),
    )


class TestWhatCountsAsATakeOverThatLanded:
    def test_a_take_over_replaces_the_orchestrator_and_the_rollout_executor_of_the_release(self):
        """These two names are what every stamp assertion below is written against."""
        assert STAMPED == {ORCHESTRATOR, ROLLOUT_EXECUTOR}

    def test_a_run_whose_replaced_workloads_were_all_restamped_counts(self):
        """A take-over rewrites the stamp of each workload it replaces, whatever the run had trained."""
        assert restamped_replaced_workloads(
            before=_stamps(orchestrator=None, rollout_executor=None), after=_stamps(), workloads=STAMPED
        )

    def test_a_second_take_over_rewriting_the_first_ones_stamps_counts(self):
        """One object carries one stamp, rewritten each time, so a landing is a value that changed."""
        assert restamped_replaced_workloads(
            before=_stamps(orchestrator="t1", rollout_executor="t1"), after=_stamps(), workloads=STAMPED
        )

    def test_a_run_whose_workloads_still_carry_the_stamps_they_carried_does_not_count(self):
        """The relaunch is still installing, and the stamps it will rewrite are the ones drawn against."""
        assert not restamped_replaced_workloads(before=_stamps(), after=_stamps(), workloads=STAMPED)

    def test_a_run_only_half_of_whose_workloads_were_restamped_does_not_count(self):
        """The two workloads are rolled by one upgrade but observed apart, so one of them is a half-landing."""
        assert not restamped_replaced_workloads(
            before=_stamps(orchestrator=None, rollout_executor=None),
            after=_stamps(rollout_executor=None),
            workloads=STAMPED,
        )

    def test_a_workload_the_read_did_not_return_does_not_count(self):
        """A workload absent from the read has not been seen carrying anything, which is not evidence."""
        assert not restamped_replaced_workloads(before={}, after={ORCHESTRATOR: "t2"}, workloads=STAMPED)

    def test_a_stamp_the_run_never_carried_on_another_workload_does_not_count(self):
        """Only the two workloads a take-over replaces are stamped; anything else says nothing about it."""
        assert not restamped_replaced_workloads(
            before=_stamps(orchestrator=None, rollout_executor=None),
            after={**_stamps(orchestrator=None, rollout_executor=None), TRAINER: "t2"},
            workloads=STAMPED,
        )

    def test_a_stamp_read_that_failed_does_not_count(self):
        """A kubectl call that did not answer must not become a take-over that never landed."""
        assert not restamped_replaced_workloads(before=_stamps(), after=None, workloads=STAMPED)


class TestInject:
    def test_every_draw_relaunches_the_release_that_is_already_up(self, monkeypatch):
        """A relaunch under another release would leave the trainers this run is watching behind."""
        launched: list[ExecuteTrainConfig] = []
        _install_run(monkeypatch)

        _form(launched.append).inject(CELL, Random(0))

        assert [one.hot_restart for one in launched] == [HOT_RESTART_ARG]
        assert [one.run_id for one in launched] == ["demo"]

    def test_a_draw_before_the_first_save_fires_like_any_other(self, monkeypatch):
        """Taking a run over before it saved costs everything it trained, which is a path worth covering."""
        launched: list[ExecuteTrainConfig] = []
        _install_run(monkeypatch, saved=None)

        _form(launched.append).inject(CELL, Random(0))

        assert len(launched) == 1

    def test_a_draw_before_the_run_trained_a_step_lands_on_the_restart_stamps(self, monkeypatch):
        """A run holding nothing yet is a path this soak covers, and its take-over lands like any other."""
        _install_run(
            monkeypatch,
            finished=None,
            saved=None,
            stamps=[_stamps(orchestrator=None, rollout_executor=None), _stamps()],
        )

        form = _form(lambda _config: None)
        form.inject(CELL, Random(0))

        assert [one.frozen_rollout_id for one in form.records] == [-1]

    def test_the_step_a_record_carries_is_the_one_the_run_stood_at_when_it_was_drawn(self, monkeypatch):
        """The record is what the artifact reports a take-over cost, read where the draw fired."""
        _install_run(monkeypatch, finished=4, saved=3)

        form = _form(lambda _config: None)
        form.inject(CELL, Random(0))

        assert [(one.frozen_rollout_id, one.saved_iteration_at_trigger) for one in form.records] == [(4, 3)]

    def test_a_second_draw_before_the_run_trained_a_step_lands_on_the_rewritten_stamps(self, monkeypatch):
        """One object carries one stamp: a second take-over adds none, and only the rewrite says it landed."""
        _install_run(monkeypatch, finished=None, saved=None, stamps=[_stamps(), _stamps("t3")])

        form = _form(lambda _config: None)
        form.inject(CELL, Random(0))

        assert len(form.records) == 1

    def test_a_draw_before_the_run_trained_a_step_still_times_out_without_a_restamp(self, monkeypatch):
        """Without a rewritten stamp there is nothing saying the take-over installed, and that must not pass."""
        never_returns = threading.Event()
        _install_run(monkeypatch, finished=None, saved=None, stamps=[_stamps()])

        try:
            with pytest.raises(AssertionError, match="still carry the stamps they carried"):
                _form(lambda _config: never_returns.wait(timeout=30.0), timeout_seconds=0.2).inject(CELL, Random(0))
        finally:
            never_returns.set()

    def test_a_draw_that_cannot_read_the_workloads_it_would_compare_against_is_reported(self, monkeypatch):
        """Without a baseline nothing could tell a rewritten stamp from the one that was already there."""
        _install_run(monkeypatch, finished=None, saved=None, stamps=[None])

        with pytest.raises(AssertionError, match="has nothing to compare against"):
            _form(lambda _config: None, baseline_read_attempts=2).inject(CELL, Random(0))

    def test_the_injector_is_not_blocked_by_a_relaunch_that_drives_the_run_to_its_end(self, monkeypatch):
        """The relaunch installs a script that trains to the end, so a call that waits for it never returns."""
        driving = threading.Event()
        _install_run(monkeypatch)

        _form(lambda _config: driving.wait(timeout=30.0)).inject(CELL, Random(0))

        driving.set()

    def test_a_relaunch_that_never_reached_the_run_is_reported_rather_than_counted(self, monkeypatch):
        """An injection counted as landed would let this soak pass on a run nothing ever replaced."""
        _install_run(monkeypatch, stamps=[_stamps()])

        with pytest.raises(AssertionError, match="returned without restamping"):
            _form(lambda _config: None).inject(CELL, Random(0))

    def test_a_relaunch_the_cluster_refused_is_reported_rather_than_counted(self, monkeypatch):
        """A refused upgrade leaves the run training under the very script this injection meant to replace."""
        _install_run(monkeypatch, stamps=[_stamps()])

        with pytest.raises(AssertionError, match="refused rather than installed"):
            _form(_raise_refused).inject(CELL, Random(0))

    def test_a_later_draw_is_judged_on_its_own_relaunch_and_not_on_an_earlier_one(self, monkeypatch):
        """A refused upgrade may well be transient, and a soak that gives up on the first one proves less."""
        _install_run(monkeypatch, stamps=[_stamps()])
        form = _form(_raise_refused)
        with pytest.raises(AssertionError, match="refused rather than installed"):
            form.inject(CELL, Random(0))

        _install_run(monkeypatch)
        form._launch = lambda _config: None

        form.inject(CELL, Random(0))


def _raise_refused(_config: ExecuteTrainConfig) -> None:
    raise SystemExit("the relaunch would change more than the size of this run")


class TestForm:
    def test_the_form_is_named_after_what_it_does(self):
        """The name is what the injection log carries, and a form nobody can name cannot be read back."""
        assert _form(lambda _config: None).name == HOT_RESTART_FORM_NAME

    def test_the_form_declares_that_it_leaves_the_cell_it_was_drawn_for_running(self):
        """A cell counted as crashed is dropped from the live set forever, and no later draw would fire."""
        assert not _form(lambda _config: None).harms_cell


class TestTheClosingContract:
    def test_a_soak_whose_take_overs_all_installed_cleanly_passes(self, monkeypatch):
        """Nothing raised and nothing is still running, so what was collected can be read."""
        _install_run(monkeypatch)
        form = _form(lambda _config: None)
        form.inject(CELL, Random(0))

        form.join_relaunches(timeout_seconds=30.0)
        form.assert_take_overs_installed_cleanly()

    def test_the_run_verdict_raised_by_the_last_relaunch_is_not_lost(self, monkeypatch):
        """The last relaunch's launcher is what observes the run's own metric checker."""
        _install_run(monkeypatch)
        form = _form(_raise_run_verdict)
        form.inject(CELL, Random(0))

        form.join_relaunches(timeout_seconds=30.0)

        with pytest.raises(AssertionError, match="did not install cleanly"):
            form.assert_take_overs_installed_cleanly()

    def test_a_relaunch_still_running_at_the_end_is_reported(self, monkeypatch):
        """A run still being replaced under the dumps about to be read is not a finished soak."""
        never_returns = threading.Event()
        _install_run(monkeypatch)
        form = _form(lambda _config: never_returns.wait(timeout=30.0))
        form.inject(CELL, Random(0))

        try:
            form.join_relaunches(timeout_seconds=0.05)
            with pytest.raises(AssertionError, match="still installing a hot restart"):
                form.assert_take_overs_installed_cleanly()
        finally:
            never_returns.set()

    def test_a_failure_from_an_earlier_take_over_is_still_reported_at_the_end(self, monkeypatch):
        """Per-draw judgement must not throw away what an earlier draw already proved broken."""
        _install_run(monkeypatch, stamps=[_stamps()])
        form = _form(_raise_refused)
        with pytest.raises(AssertionError, match="refused rather than installed"):
            form.inject(CELL, Random(0))

        _install_run(monkeypatch)
        form._launch = lambda _config: None
        form.inject(CELL, Random(0))
        form.join_relaunches(timeout_seconds=30.0)

        with pytest.raises(AssertionError, match="take-over 0"):
            form.assert_take_overs_installed_cleanly()


def _raise_run_verdict(_config: ExecuteTrainConfig) -> None:
    raise SystemExit("eval/gsm8k 0.31 is below the required 0.55")


def _stamps(
    at: str = "t2", *, orchestrator: str | None = "", rollout_executor: str | None = ""
) -> dict[str, str | None]:
    return {
        ORCHESTRATOR: at if orchestrator == "" else orchestrator,
        ROLLOUT_EXECUTOR: at if rollout_executor == "" else rollout_executor,
    }
