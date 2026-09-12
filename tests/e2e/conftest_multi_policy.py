from pathlib import Path
from typing import NamedTuple

from examples.multi_policy.run_solver_verifier_gsm8k import (
    EVAL_DATASET_NAME,
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


class EvalScoreBounds(NamedTuple):
    initial_max: float
    peak_min: float
    min_growth: float | None = None


NUM_VERIFIED_ARGS_PER_POLICY = {SOLVER_MODEL_ID: 25, VERIFIER_MODEL_ID: 26}


def execute(
    args: ScriptArgs,
    *,
    wandb_args: str,
    eval_score_bounds: dict[str, EvalScoreBounds] | None = None,
) -> None:
    events_dir = compute_events_dir(args)
    megatron_config = compute_megatron_config(args)

    launch_train(build_train_args(args, wandb_args=wandb_args, megatron_config=megatron_config), args)

    assert_ranks_trained_with_policy_args(
        events_dir, megatron_config=megatron_config, expected_num_ranks=args.actor_num_gpus_per_policy
    )
    assert_policies_reported_eval_points(
        events_dir,
        model_ids=[trainer["model_id"] for trainer in megatron_config["trainers"]],
        dataset_name=EVAL_DATASET_NAME,
    )
    if eval_score_bounds is not None:
        assert_policy_eval_scores_learned(events_dir, bounds=eval_score_bounds, dataset_name=EVAL_DATASET_NAME)


def assert_ranks_trained_with_policy_args(events_dir: Path, *, megatron_config: dict, expected_num_ranks: int) -> None:
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


def assert_policies_reported_eval_points(events_dir: Path, *, model_ids: list[str], dataset_name: str) -> None:
    for model_id in model_ids:
        eval_key = f"eval/{dataset_name}/{model_id}"
        points = [
            event.metrics[eval_key]
            for event in read_events(events_dir)
            if isinstance(event, MetricEvent) and eval_key in event.metrics
        ]
        assert points, (
            f"policy {model_id!r} logged no {eval_key} point under {events_dir}, but every run evaluates at least "
            f"once, so held-out eval never actually ran for this policy"
        )


def assert_policy_eval_scores_learned(
    events_dir: Path, *, bounds: dict[str, EvalScoreBounds], dataset_name: str
) -> None:
    for model_id, model_bounds in bounds.items():
        scores = _read_eval_score_series(events_dir, model_id=model_id, dataset_name=dataset_name)
        assert scores, f"policy {model_id!r}: no eval/{dataset_name}/{model_id} points under {events_dir}"

        first = scores[0]
        peak = max(scores)
        assert (
            first <= model_bounds.initial_max
        ), f"policy {model_id!r}: first eval score {first} > {model_bounds.initial_max} (starts already solved)"
        assert peak >= model_bounds.peak_min, f"policy {model_id!r}: peak eval score {peak} < {model_bounds.peak_min}"
        if model_bounds.min_growth is not None:
            assert peak - first >= model_bounds.min_growth, (
                f"policy {model_id!r}: eval growth {peak - first} < {model_bounds.min_growth} "
                f"(first {first}, peak {peak})"
            )


def _read_eval_score_series(events_dir: Path, *, model_id: str, dataset_name: str) -> list[float]:
    eval_key = f"eval/{dataset_name}/{model_id}"
    version_key = f"{eval_key}/weight_version/max"
    points = [
        (event.metrics.get(version_key, 0), event.metrics[eval_key])
        for event in read_events(events_dir)
        if isinstance(event, MetricEvent) and eval_key in event.metrics
    ]
    return [score for _, score in sorted(points)]
