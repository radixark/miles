from collections.abc import Sequence

from miles.utils.audit_utils.event_logger.models import (
    EngineEnvReportEvent,
    EnvReportEvent,
    Event,
    WeightUpdateResultEvent,
)
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


def assert_deterministic_environment(
    events: Sequence[Event],
    *,
    trainer_ranks: set[tuple[int, int]],
    engine_count: int,
    engine_env: dict[str, str],
    trainer_env: dict[str, str],
) -> None:
    observed_ranks: set[tuple[int, int]] = set()
    observed_engines: set[str] = set()
    reported_incarnations: set[tuple[str, str]] = set()
    updated_incarnations: set[tuple[str, str]] = set()
    for event in events:
        if isinstance(event, WeightUpdateResultEvent):
            for cell_id in event.updated_cell_ids:
                assert event.target_incarnations.get(cell_id), "Updated engine lacks its incarnation"
                updated_incarnations.add((cell_id, event.target_incarnations[cell_id]))
        if isinstance(event, EnvReportEvent) and isinstance(event.source, TrainProcessIdentity):
            if event.source.component != "actor":
                continue
            observed_ranks.add((event.source.cell_index, event.source.rank_within_cell))
            facts = event.report.process
            for key, value in trainer_env.items():
                assert facts.env_vars.get(key) == value, f"Trainer numerical prerequisite differs: {key}"
            for key, value in {
                "deterministic_mode": True,
                "debug_deterministic_collective": True,
                "ft_components": ["rollout"],
                "update_weight_transfer_mode": "p2p",
                "sglang_router_policy": "round_robin",
                "colocate": False,
            }.items():
                assert facts.args.values.get(key) == value, f"Trainer argument differs: {key}"
        elif isinstance(event, EngineEnvReportEvent):
            assert event.workers_hash, "Engine numerical report lacks its incarnation"
            observed_engines.add(event.cell_id)
            reported_incarnations.add((event.cell_id, event.workers_hash))
            info = event.server_info
            for key, value in {
                "enable_deterministic_inference": True,
                "attention_backend": "flashinfer",
                "disable_radix_cache": True,
            }.items():
                assert info.get(key) == value, f"Engine argument differs: {key}"
            states = info.get("internal_states")
            assert isinstance(states, list) and states, "Engine lacks worker numerical reports"
            for state in states:
                actual = state.get("env_vars", {})
                for key, value in engine_env.items():
                    assert actual.get(key) == value, f"Engine numerical prerequisite differs: {key}"
    assert observed_ranks == trainer_ranks, "Trainer numerical reports do not cover the configured ranks"
    assert len(observed_engines) == engine_count, "Engine numerical reports do not cover the configured fleet"
    missing = updated_incarnations - reported_incarnations
    assert not missing, f"Updated engine incarnations lack numerical reports: {sorted(missing)}"
