import statistics
from pathlib import Path
from typing import NamedTuple

from examples.multi_policy.run_solver_verifier_gsm8k import (
    SOLVER_MODEL_ID,
    VERIFIER_MODEL_ID,
    ScriptArgs,
    build_train_args,
    compute_events_dir,
    compute_megatron_config,
    launch_train,
)

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import EnvReportEvent, MetricEvent
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


class TrainRewardBounds(NamedTuple):
    initial_max: float
    final_min: float


TRAIN_REWARD_BOUNDS = {
    SOLVER_MODEL_ID: TrainRewardBounds(initial_max=0.9, final_min=0.01),
    VERIFIER_MODEL_ID: TrainRewardBounds(initial_max=0.9, final_min=0.01),
}

NUM_VERIFIED_ARGS_PER_POLICY = {SOLVER_MODEL_ID: 25, VERIFIER_MODEL_ID: 26}


def execute(
    args: ScriptArgs,
    *,
    wandb_args: str,
    train_reward_bounds: dict[str, TrainRewardBounds] | None = None,
) -> None:
    events_dir = compute_events_dir(args)
    megatron_config = compute_megatron_config(args)

    launch_train(build_train_args(args, wandb_args=wandb_args, megatron_config=megatron_config), args)

    assert_every_rank_trained_with_its_own_policy_args(
        events_dir, megatron_config=megatron_config, expected_num_ranks=args.actor_num_gpus_per_policy
    )
    assert_every_policy_reported_reward_in_bounds(events_dir, bounds=train_reward_bounds or TRAIN_REWARD_BOUNDS)


def assert_every_rank_trained_with_its_own_policy_args(
    events_dir: Path, *, megatron_config: dict, expected_num_ranks: int
) -> None:
    reports_by_model_id: dict[str, list[EnvReportEvent]] = {}
    for event in read_events(events_dir):
        if isinstance(event, EnvReportEvent) and isinstance(event.source, TrainProcessIdentity):
            reports_by_model_id.setdefault(event.source.model_id, []).append(event)

    expected_model_ids = sorted(trainer["model_id"] for trainer in megatron_config["trainers"])
    assert sorted(reports_by_model_id) == expected_model_ids, (
        f"the env reports under {events_dir} come from trainer ranks of {sorted(reports_by_model_id)}, but this "
        f"run trains {expected_model_ids}; a policy whose ranks reported nothing was never actually trained"
    )

    ranks_by_model_id: dict[str, list[tuple[int, int]]] = {
        model_id: sorted({(event.source.cell_index, event.source.rank_within_cell) for event in reports})
        for model_id, reports in reports_by_model_id.items()
    }
    every_rank = sorted({rank for ranks in ranks_by_model_id.values() for rank in ranks})
    for model_id, ranks in ranks_by_model_id.items():
        assert len(ranks) == expected_num_ranks, (
            f"policy {model_id!r} is reported from ranks {ranks}, while this run trains each policy on "
            f"{expected_num_ranks} rank(s); a rank that reported nothing leaves the arguments it was built with "
            f"unverified"
        )
        assert ranks == every_rank, (
            f"policy {model_id!r} is reported from ranks {ranks}, while this run's trainer ranks are {every_rank}; "
            f"a rank that reported nothing leaves the arguments it was built with unverified"
        )

    for trainer in megatron_config["trainers"]:
        model_id = trainer["model_id"]
        expected = dict(trainer["overrides"])
        assert len(expected) == NUM_VERIFIED_ARGS_PER_POLICY[model_id], (
            f"policy {model_id!r} overrides {sorted(expected)}, but this test claims to verify "
            f"{NUM_VERIFIED_ARGS_PER_POLICY[model_id]} arguments; a verification that quietly shrank proves nothing"
        )
        for report in reports_by_model_id[model_id]:
            values = report.report.process.args.values
            assert values["trainer_model_id"] == model_id, (
                f"rank {report.source.to_name()} reports trainer_model_id {values['trainer_model_id']!r} while its "
                f"process identity says {model_id!r}"
            )
            actual = {key: values[key] for key in expected}
            assert len(actual) == len(expected), f"{sorted(set(expected) - set(actual))} never reached the report"
            assert actual == expected, (
                f"rank {report.source.to_name()} of policy {model_id!r} was built with {actual}, but its "
                f"--megatron-config overrides prescribe {expected}"
            )


def assert_every_policy_reported_reward_in_bounds(events_dir: Path, *, bounds: dict[str, TrainRewardBounds]) -> None:
    for model_id, model_bounds in bounds.items():
        rewards = _read_train_reward_series(events_dir, model_id=model_id)
        assert rewards, (
            f"no {_compute_train_reward_key(model_id)} value was logged under {events_dir}, so policy "
            f"{model_id!r} never reported a training reward and nothing about its learning can be checked"
        )

        initial = rewards[0]
        final = statistics.mean(rewards[-max(1, len(rewards) // 3) :])
        assert initial <= model_bounds.initial_max, (
            f"policy {model_id!r} starts at training reward {initial}, above {model_bounds.initial_max}; a run "
            f"that starts already solved cannot show that training moved it"
        )
        assert final >= model_bounds.final_min, (
            f"policy {model_id!r} ends at training reward {final}, below {model_bounds.final_min}; either its "
            f"reward function never fires, or training destroyed the model"
        )


def _read_train_reward_series(events_dir: Path, *, model_id: str) -> list[float]:
    reward_key = _compute_train_reward_key(model_id)
    step_key = f"{model_id}/rollout/step"
    points = [
        (event.metrics[step_key], event.metrics[reward_key])
        for event in read_events(events_dir)
        if isinstance(event, MetricEvent) and reward_key in event.metrics
    ]
    return [reward for _, reward in sorted(points)]


def _compute_train_reward_key(model_id: str) -> str:
    return f"{model_id}/rollout/raw_reward"
