import dataclasses
import math
import os

import typer

from examples.infra_features.split_deployment.run_solver_verifier_gsm8k_split import (
    build_deployment_train_args,
    compute_deployment_identities,
)
from examples.multi_policy.run_solver_verifier_gsm8k import (
    LEADER_MODEL_ID,
    MODEL_IDS,
    TRAIN_EXTRA_ENV_VARS,
    ScriptArgs,
    compute_events_dir,
    compute_megatron_config,
    launch_train,
    prepare,
)
from tests.e2e.conftest_multi_policy import (
    TRAIN_REWARD_BOUNDS,
    assert_every_policy_reported_reward_in_bounds,
    assert_every_rank_trained_with_its_own_policy_args,
)
from tests.e2e.deploy.conftest_deploy.common.utils import assert_the_cluster_can_deploy_runs
from tests.e2e.deploy.conftest_deploy.split.split_deployment import RunDeployment, run_split_training_into

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.run_uuid import generate_run_uuid
from miles.utils.test_utils.comparisons.metrics import assert_metric_was_finite_and_nonzero, read_metric_series

TEST_NAME: str = "split_multi_policy"
MIN_TRAINED_ROLLOUTS: int = 2
MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF: float = 0.1


def _run(args: ScriptArgs) -> None:
    prepare(args)

    config = dataclasses.replace(args, run_uuid=generate_run_uuid())
    run_split_training_into(
        deployments=_build_deployments(config),
        launch=launch_train,
        config=config,
        dump_dir=_compute_dump_dir(config),
    )

    _verify(args)


def _build_deployments(args: ScriptArgs) -> list[RunDeployment]:
    return [
        RunDeployment(
            deploy_component=deploy_component,
            deploy_instance_id=deploy_instance_id,
            train_args=build_deployment_train_args(
                dataclasses.replace(args, deploy_component=deploy_component, deploy_instance_id=deploy_instance_id)
            ),
        )
        for deploy_component, deploy_instance_id in compute_deployment_identities(args)
    ]


def _verify(args: ScriptArgs) -> None:
    events_dir = compute_events_dir(args)
    dump_dir = _compute_dump_dir(args)

    assert events_dir.is_dir(), (
        f"no run wrote anything under {events_dir}, so every assertion below would report missing metrics rather "
        f"than the missing run; verifying a previous run needs its id in MILES_SCRIPT_RUN_ID, and this process "
        f"was given {args.run_id!r}"
    )

    assert_every_rank_trained_with_its_own_policy_args(
        events_dir,
        megatron_config=compute_megatron_config(args),
        expected_num_ranks=args.actor_num_gpus_per_policy,
    )
    assert_every_policy_reported_reward_in_bounds(events_dir, bounds=TRAIN_REWARD_BOUNDS)
    _assert_the_leader_policy_ran_every_rollout(dump_dir, num_rollout=args.num_rollout)

    for model_id in MODEL_IDS:
        assert_metric_was_finite_and_nonzero(
            side=model_id,
            dump_dir=dump_dir,
            key=f"{model_id}/train/grad_norm",
            min_rollouts=MIN_TRAINED_ROLLOUTS,
        )
        assert_metric_was_finite_and_nonzero(
            side=model_id, dump_dir=dump_dir, key=f"{model_id}/train/loss", min_rollouts=MIN_TRAINED_ROLLOUTS
        )
        _assert_the_trainer_scores_what_its_engines_generated(dump_dir, model_id=model_id)

    print("Split multi policy deployment test PASSED")


def _assert_the_leader_policy_ran_every_rollout(dump_dir: str, *, num_rollout: int) -> None:
    series = read_metric_series(dump_dir, key=f"{LEADER_MODEL_ID}/train/grad_norm")
    trained_rollout_ids = sorted({rollout_id for rollout_id, _ in series})

    assert trained_rollout_ids == list(range(num_rollout)), (
        f"the leader policy {LEADER_MODEL_ID!r}, whose loop ends the run after {num_rollout}, trained on rollouts "
        f"{trained_rollout_ids} ({series}): a missing one means a deployment failed rather than the run finishing"
    )


def _assert_the_trainer_scores_what_its_engines_generated(dump_dir: str, *, model_id: str) -> None:
    key = f"{model_id}/train/train_rollout_logprob_abs_diff"
    series = read_metric_series(dump_dir, key=key)

    rollouts = {rollout_id for rollout_id, _ in series}
    assert len(rollouts) >= MIN_TRAINED_ROLLOUTS, (
        f"{key} was reported for only {len(rollouts)} rollout(s) ({series}), so nothing compares what the engines "
        f"of policy {model_id!r} generated against what its trainer scored"
    )

    unusable = [(rollout_id, value) for rollout_id, value in series if not math.isfinite(value)]
    assert not unusable, (
        f"policy {model_id!r} reported {key} as {unusable} ({series}), and max() passes over such a value rather "
        f"than reporting it, so nothing here says its trainer and its engines serve the same weights"
    )

    worst = max(value for _, value in series)
    assert worst <= MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF, (
        f"policy {model_id!r} scores the tokens its engines generated {worst} apart in log probability ({series}), "
        f"above {MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF}: its trainer and its engines serve different weights"
    )


def _compute_dump_dir(config: ExecuteTrainConfig) -> str:
    return str(compute_events_dir(config).parent)


# ================================= app wiring =================================


def _run_ci_where_a_run_can_be_deployed() -> None:
    config = command_utils.default_config(ScriptArgs)
    assert_the_cluster_can_deploy_runs(config)
    os.environ.update(TRAIN_EXTRA_ENV_VARS)
    _run(config)


def _verify_what_a_previous_run_left() -> None:
    _verify(command_utils.default_config(ScriptArgs))


app: typer.Typer = typer.Typer()
app.command(name="run")(_run_ci_where_a_run_can_be_deployed)
app.command(name="verify")(_verify_what_a_previous_run_left)
run_ci = _run_ci_where_a_run_can_be_deployed

if __name__ == "__main__":
    app()
