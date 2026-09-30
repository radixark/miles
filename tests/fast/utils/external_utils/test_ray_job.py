import json
import os
import re
import shlex
import signal
import sys

import pytest
from ray.job_submission import JobStatus
from tests.fast.utils.external_utils.conftest import FakeJobSubmissionClient

from miles.utils.external_utils.ray_job import run_ray_job

_DEFAULT_ADDRESS = "http://127.0.0.1:8265"
_RUNTIME_ENV = {"env_vars": {"MILES_X": "1"}}


class TestAnIndependentJob:
    def test_an_explicit_submission_id_is_handed_to_ray_job_submit(
        self, recorded_ray_submit_commands: list[str]
    ) -> None:
        """Without the flag ray picks a random id, and the launcher can no longer name the job it started."""
        run_ray_job(entrypoint="python3 train.py", runtime_env=_RUNTIME_ENV, submission_id="miles-job-7")

        (command,) = recorded_ray_submit_commands
        tokens = _submit_tokens(command)
        assert tokens[tokens.index("--runtime-env-json=" + _json(_RUNTIME_ENV)) - 1] == "--submission-id=miles-job-7"
        assert tokens[-3:] == ["--", "python3", "train.py"]

    @pytest.mark.parametrize("submission_id", [None, ""])
    def test_a_launch_that_names_no_job_passes_no_submission_id(
        self, recorded_ray_submit_commands: list[str], submission_id: str | None
    ) -> None:
        """An empty flag would make ray reject the submission, so an unnamed job keeps the old command."""
        run_ray_job(entrypoint="python3 train.py", runtime_env=_RUNTIME_ENV, submission_id=submission_id)

        (command,) = recorded_ray_submit_commands
        assert not any(token.startswith("--submission-id") for token in _submit_tokens(command))
        assert f'--address="{_DEFAULT_ADDRESS}"' in command

    def test_a_submission_id_is_passed_to_the_shell_as_one_quoted_word(
        self, recorded_ray_submit_commands: list[str]
    ) -> None:
        """An unquoted id with shell syntax in it would split the submit command or run something else."""
        run_ray_job(entrypoint="python3 train.py", runtime_env=_RUNTIME_ENV, submission_id="a b;touch /tmp/x")

        (command,) = recorded_ray_submit_commands
        assert "--submission-id=a b;touch /tmp/x" in _submit_tokens(command)

    def test_an_unknown_lifetime_is_refused_without_submitting(self, recorded_ray_submit_commands: list[str]) -> None:
        """A typo in the lifetime must not silently fall back to one of the two real behaviours."""
        with pytest.raises(ValueError, match="Unknown job lifetime"):
            run_ray_job(entrypoint="python3 train.py", runtime_env=_RUNTIME_ENV, job_lifetime="forever")

        assert recorded_ray_submit_commands == []


class TestALauncherOwnedJob:
    def test_an_explicit_submission_id_names_the_submitted_followed_stopped_and_judged_job(
        self, fake_job_client: FakeJobSubmissionClient
    ) -> None:
        """Stopping or judging any other id would leave the started job running or report another job's result."""
        _run_launcher_owned(submission_id="miles-job-7")

        _assert_one_job_throughout(fake_job_client, "miles-job-7")

    def test_a_default_submission_id_is_minted_once_and_used_throughout(
        self, fake_job_client: FakeJobSubmissionClient
    ) -> None:
        """Minting the default twice would stop a job that was never submitted and leak the real one."""
        _run_launcher_owned(submission_id=None)

        (submission_id,) = {submitted["submission_id"] for submitted in fake_job_client.submitted}
        assert re.fullmatch(r"miles-[0-9a-f]{32}", submission_id)
        _assert_one_job_throughout(fake_job_client, submission_id)

    def test_two_unnamed_launches_get_two_different_ids(self, fake_job_client: FakeJobSubmissionClient) -> None:
        """A fixed default id would make a second launch collide with, and then stop, the first job."""
        _run_launcher_owned(submission_id=None)
        _run_launcher_owned(submission_id=None)

        assert len({submitted["submission_id"] for submitted in fake_job_client.submitted}) == 2

    def test_the_stop_command_waits_on_the_cli_for_the_same_address_and_id(
        self, fake_job_client: FakeJobSubmissionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stop must target the cluster the job was submitted to and fail loudly when it cannot stop it."""
        monkeypatch.setenv("RAY_ADDRESS", "http://ray-head:8265")

        _run_launcher_owned(submission_id="miles-job-7")

        assert fake_job_client.addresses == ["http://ray-head:8265"]
        assert fake_job_client.stop_commands == [
            (
                [
                    sys.executable,
                    "-m",
                    "ray.scripts.scripts",
                    "job",
                    "stop",
                    "--address",
                    "http://ray-head:8265",
                    "miles-job-7",
                ],
                {"check": True, "timeout": 45},
            )
        ]

    def test_a_submission_that_fails_still_stops_the_id_it_tried_to_submit(
        self, fake_job_client: FakeJobSubmissionClient
    ) -> None:
        """A submit that raised after the server accepted the job would otherwise leave it running unowned."""
        fake_job_client.submit_error = ConnectionError("the response was lost")

        with pytest.raises(ConnectionError):
            _run_launcher_owned(submission_id="miles-job-7")

        assert fake_job_client.stopped_ids == ["miles-job-7"]
        assert fake_job_client.status_queries == []

    def test_a_job_that_did_not_succeed_is_reported_with_its_id_after_it_was_stopped(
        self, fake_job_client: FakeJobSubmissionClient
    ) -> None:
        """A failed job must fail the launcher, naming the job, rather than pass as a finished run."""
        fake_job_client.status = JobStatus.FAILED

        with pytest.raises(RuntimeError, match="miles-job-7 ended with FAILED: the job miles-job-7 broke"):
            _run_launcher_owned(submission_id="miles-job-7")

        assert fake_job_client.stopped_ids == ["miles-job-7"]

    def test_a_termination_signal_stops_the_same_job_and_exits_with_the_signal_code(
        self, fake_job_client: FakeJobSubmissionClient
    ) -> None:
        """A killed launcher must take its job down with it and restore the handlers it replaced."""
        fake_job_client.signal_while_following = signal.SIGTERM
        previous = signal.getsignal(signal.SIGTERM)

        with pytest.raises(SystemExit) as raised:
            _run_launcher_owned(submission_id="miles-job-7")

        assert raised.value.code == 128 + signal.SIGTERM
        assert fake_job_client.stopped_ids == ["miles-job-7"]
        assert fake_job_client.status_queries == []
        assert signal.getsignal(signal.SIGTERM) is previous

    def test_the_job_env_gains_unbuffered_output_without_losing_or_mutating_the_callers(
        self, fake_job_client: FakeJobSubmissionClient
    ) -> None:
        """Buffered logs would reach the launcher only at exit, and the caller's env vars must all arrive."""
        runtime_env = {"env_vars": {"MILES_X": "1"}, "working_dir": "/w"}

        run_ray_job(entrypoint="python3 train.py", runtime_env=runtime_env, job_lifetime="launcher")

        assert fake_job_client.submitted[0]["runtime_env"] == {
            "env_vars": {"MILES_X": "1", "PYTHONUNBUFFERED": "1"},
            "working_dir": "/w",
        }
        assert runtime_env == {"env_vars": {"MILES_X": "1"}, "working_dir": "/w"}

    @pytest.mark.parametrize("previous", [None, "corp.example"])
    def test_no_proxy_covers_the_local_dashboard_only_while_the_job_runs(
        self, fake_job_client: FakeJobSubmissionClient, monkeypatch: pytest.MonkeyPatch, previous: str | None
    ) -> None:
        """A proxied dashboard request fails, and a leaked no_proxy would change every later request."""
        if previous is not None:
            monkeypatch.setenv("no_proxy", previous)

        _run_launcher_owned(submission_id="miles-job-7")

        expected_prefix = [] if previous is None else [previous]
        assert fake_job_client.no_proxy_at_submit == [",".join([*expected_prefix, "127.0.0.1", "localhost"])]
        assert os.environ.get("no_proxy") == previous


def _run_launcher_owned(*, submission_id: str | None) -> None:
    run_ray_job(
        entrypoint="python3 train.py", runtime_env=_RUNTIME_ENV, job_lifetime="launcher", submission_id=submission_id
    )


def _assert_one_job_throughout(client: FakeJobSubmissionClient, submission_id: str) -> None:
    assert [submitted["submission_id"] for submitted in client.submitted] == [submission_id]
    assert client.tailed == [submission_id]
    assert client.stopped_ids == [submission_id]
    assert client.status_queries == [submission_id]
    assert client.addresses == [_DEFAULT_ADDRESS]


def _submit_tokens(command: str) -> list[str]:
    return shlex.split(command.split("&& ray job submit", 1)[1])


def _json(value: dict) -> str:
    return json.dumps(value)
