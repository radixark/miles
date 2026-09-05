"""Evaluate puzzle correctness while training uses teacher log-probabilities."""

from copy import copy

from miles.rollout.sglang_rollout import GenerateState
from miles.rollout.sglang_rollout import generate_rollout as sglang_generate_rollout


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if not evaluation:
        raise ValueError("Puzzle evaluation hook requires evaluation=True")
    # Legacy rollout state retains its first args. Initialize it with training
    # settings before the evaluation-only copy can disable OPD for later rollouts.
    GenerateState(args)
    eval_args = copy(args)
    eval_args.custom_rm_path = "examples.mopd_puzzles.tasks.reward_func"
    eval_args.custom_reward_post_process_path = None
    eval_args.opd_log_prob_top_k = 0
    eval_args.use_opd = False
    return sglang_generate_rollout(eval_args, rollout_id, data_source, evaluation=True)
