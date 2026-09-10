import dataclasses
import os

from examples.multi_policy.run_solver_verifier_gsm8k import (
    LEADER_MODEL_ID,
    ScriptArgs,
    build_train_args,
    compute_events_dir,
    compute_save_dir,
    launch_train,
    prepare,
)
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.conftest_fully_async_ckpt import assert_checkpoint_replayed, read_checkpoint_sample_indices

from miles.utils.external_utils import command_utils
from miles.utils.multi_policy.checkpoint_state import MultiPolicyCheckpointState

register_cuda_ci(est_time=3000, suite="stage-c-4-gpu-h200", labels=["ckpt", "multi-policy", "fully-async"])

NUM_ROLLOUT = 4
CHECKPOINT_ID = 1


def execute(args: ScriptArgs) -> None:
    save_dir = compute_save_dir(args)
    train_args = build_train_args(args, wandb_args=command_utils.get_default_wandb_args(__file__))
    launch_train(train_args=f"{train_args} --debug-exit-after-rollout 2", args=args)
    state = MultiPolicyCheckpointState.load(save_dir, leader_rollout_id=CHECKPOINT_ID)
    assert state is not None
    assert state.leader_model_id == LEADER_MODEL_ID
    saved_indices = read_checkpoint_sample_indices(save_dir=save_dir, rollout_id=CHECKPOINT_ID)

    launch_train(train_args=f"{train_args} --load {save_dir}", args=args)

    assert_checkpoint_replayed(
        event_dir=compute_events_dir(args),
        saved_indices=saved_indices,
        rollout_ids=state.rollout_ids,
        num_rollout=NUM_ROLLOUT,
        leader_model_id=LEADER_MODEL_ID,
    )


if __name__ == "__main__":
    args = dataclasses.replace(command_utils.default_config(ScriptArgs), num_rollout=NUM_ROLLOUT, save_interval=2)
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
