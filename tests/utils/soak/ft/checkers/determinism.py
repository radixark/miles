from collections.abc import Sequence
from typing import Any

from tests.utils.env_reports import report_ranks, trainer_env_reports

from miles.utils.audit_utils.event_analyzer.rules import engine_env_report_coverage
from miles.utils.audit_utils.event_logger.models import EngineEnvReportEvent, Event


def assert_deterministic_environment(
    events: Sequence[Event],
    *,
    trainer_ranks: set[tuple[int, int]],
    engine_count: int,
    trainer_env: dict[str, str],
    trainer_args: dict[str, Any],
    engine_env: dict[str, str],
    engine_args: dict[str, Any],
) -> None:
    trainer_reports = [report for report in trainer_env_reports(events) if report.source.component == "actor"]
    for report in trainer_reports:
        facts = report.report.process
        for key, value in trainer_env.items():
            assert facts.env_vars.get(key) == value, f"Trainer numerical prerequisite differs: {key}"
        for key, value in trainer_args.items():
            assert facts.args.values.get(key) == value, f"Trainer argument differs: {key}"
    assert (
        report_ranks(trainer_reports) == trainer_ranks
    ), "Trainer numerical reports do not cover the configured ranks"

    engine_reports = [event for event in events if isinstance(event, EngineEnvReportEvent)]
    for report in engine_reports:
        assert report.workers_hash, "Engine numerical report lacks its incarnation"
        info = report.server_info
        for key, value in engine_args.items():
            assert info.get(key) == value, f"Engine argument differs: {key}"
        states = info.get("internal_states")
        assert isinstance(states, list) and states, "Engine lacks worker numerical reports"
        for state in states:
            actual = state.get("env_vars", {})
            for key, value in engine_env.items():
                assert actual.get(key) == value, f"Engine numerical prerequisite differs: {key}"
    assert (
        len({report.cell_id for report in engine_reports}) == engine_count
    ), "Engine numerical reports do not cover the configured fleet"

    missing = engine_env_report_coverage.check(list(events), include_latest=True)
    assert not missing, f"Updated engine incarnations lack numerical reports: {missing}"
