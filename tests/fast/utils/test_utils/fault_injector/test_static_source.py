import json
import os
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from miles.utils.test_utils.fault_injector.models import FaultHookRequest
from miles.utils.test_utils.fault_injector.static_source import (
    compute_fault_hooks_arg,
    read_declared_fault_hooks,
    render_fault_hooks,
    write_fault_hooks,
)

_REQUEST = FaultHookRequest.model_validate({
    "request_id": "stop", "hook_name": "trainer_controller_step_end", "rollout_id": 2,
    "action": {"kind": "stop_cell", "cell_id": "trainer-engine-actor-0"},
})


class TestStaticSource:
    def test_launch_fragment_round_trips_the_complete_plan(self) -> None:
        """Shell parsing must preserve the flag and the complete structured request."""
        fragment = compute_fault_hooks_arg([_REQUEST])
        flag, raw = shlex.split(fragment)

        assert flag == "--ci-fault-hooks"
        assert fragment.endswith(" ")
        assert read_declared_fault_hooks(SimpleNamespace(ci_fault_hooks=raw, ci_fault_hooks_path=None)) == [_REQUEST]

    @pytest.mark.parametrize("raw", [None, "", "[]"])
    def test_absent_plan_loads_no_requests(self, raw: str | None) -> None:
        """Runs without declarations must leave fault hooks unarmed."""
        assert read_declared_fault_hooks(SimpleNamespace(ci_fault_hooks=raw, ci_fault_hooks_path=None)) == []

    def test_missing_registered_arguments_are_rejected(self) -> None:
        """A malformed arguments object must not silently disable configured faults."""
        with pytest.raises(AttributeError):
            read_declared_fault_hooks(SimpleNamespace())

    @pytest.mark.parametrize("change", [
        {"bogus": 5}, {"cell_index": -1}, {"rollout_id": -1}, {"request_id": ""},
        {"action": {"kind": "not_a_real_action"}}, {"action": {"kind": "stop_cell"}},
        {"action": {"kind": "stop_cell", "cell_id": "trainer-engine-actor-0", "bogus": 5}},
    ])
    def test_invalid_declarations_are_rejected_before_owner_filtering(self, change: dict[str, object]) -> None:
        """Invalid requests must fail even when their owner belongs to another process."""
        request = _REQUEST.model_dump(mode="json") | change
        with pytest.raises(ValidationError):
            read_declared_fault_hooks(SimpleNamespace(ci_fault_hooks=json.dumps([request]), ci_fault_hooks_path=None))

    def test_rewritten_file_rearms_the_next_launch(self, tmp_path: Path) -> None:
        """A fresh launch must load the latest plan from the same command-line path."""
        path = tmp_path / "nested" / "plan.json"
        args = SimpleNamespace(ci_fault_hooks=None, ci_fault_hooks_path=str(path))
        write_fault_hooks(path, [_REQUEST])
        assert read_declared_fault_hooks(args) == [_REQUEST]
        write_fault_hooks(path, [])
        assert read_declared_fault_hooks(args) == []

    def test_missing_file_fails_instead_of_disarming_the_run(self, tmp_path: Path) -> None:
        """A declared file must exist before a run starts."""
        with pytest.raises(AssertionError, match="does not exist"):
            read_declared_fault_hooks(SimpleNamespace(ci_fault_hooks=None, ci_fault_hooks_path=str(tmp_path / "absent")))

    def test_two_plan_sources_are_rejected(self, tmp_path: Path) -> None:
        """Two declarations must never silently choose one source."""
        with pytest.raises(AssertionError, match="both name the hooks"):
            read_declared_fault_hooks(SimpleNamespace(ci_fault_hooks=render_fault_hooks([_REQUEST]), ci_fault_hooks_path=str(tmp_path / "plan")))

    def test_process_request_defaults_preserve_wildcard_matching(self) -> None:
        """Omitted rank and attempt filters must keep the new request wildcard semantics."""
        raw = json.dumps([{"request_id": "crash", "hook_name": "trainer_step_before_allreduce", "action": {"kind": "exit_process"}, "rollout_id": 3, "target": {"kind": "declared", "cell_id": "trainer-engine-actor-00002"}}])
        [request] = read_declared_fault_hooks(SimpleNamespace(ci_fault_hooks=raw, ci_fault_hooks_path=None))
        assert request.target.cell_id == "trainer-engine-actor-00002"
        assert request.target.rank is None
        assert request.attempt is None
        assert request.rollout_id == 3

    def test_file_contents_are_read_even_when_the_timestamp_is_unchanged(self, tmp_path: Path) -> None:
        """The new static source must not reuse the retired file-stamp cache."""
        path = tmp_path / "plan.json"
        write_fault_hooks(path, [_REQUEST])
        args = SimpleNamespace(ci_fault_hooks=None, ci_fault_hooks_path=str(path))
        assert read_declared_fault_hooks(args) == [_REQUEST]
        stamp = path.stat()
        path.write_text("[]")
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert read_declared_fault_hooks(args) == []
