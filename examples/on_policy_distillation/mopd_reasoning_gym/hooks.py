"""Rollout hooks for multi-teacher OPD on reasoning_gym (scripts/run_qwen3_6_35b_a3b_mopd.py).

Training samples are scored by their routed teacher through
``miles.rollout.on_policy_distillation.reward_func``, which stores the teacher's sampled-token
log-probs in ``sample.teacher_log_probs``. The scalar reward is the reasoning_gym verifier score,
logged as ``rollout/raw_reward`` as a training-accuracy diagnostic; ``post_process_rewards`` keeps
it out of the objective by handing the advantage estimator the zero task reward of pure OPD.

Held-out samples carry ``metadata.mopd_eval`` (set by the launcher's eval config) and are scored
by the verifier the teachers were trained with: reasoning_gym ``score_answer == 1`` on the last
``<answer>`` after ``</think>``. A response that never closes ``</think>`` scores 0.

Every sample also records wall-clock stamps in its metadata, from which analyze.py rebuilds
rollout latency, teacher latency and batch completion out of the --dump-details rollout dumps:
    gen_t0 / gen_t1              generation request start / end
    opd_score_t0 / opd_score_t1  teacher scoring request start / end
"""

import re
import time
from functools import cache

from miles.rollout import on_policy_distillation as opd
from miles.rollout.generate_hub.single_turn import generate as _single_turn_generate

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@cache
def _dataset(task: str):
    # reasoning_gym is a dependency of this example only
    import reasoning_gym

    # score_answer reads only the entry it is given, so any instance of the task scores any entry
    return reasoning_gym.create_dataset(task, size=1, seed=0)


def task_score(sample) -> float:
    response = sample.response or ""
    if "</think>" not in response:
        return 0.0
    answers = _ANSWER.findall(response.rsplit("</think>", 1)[1])
    if not answers:
        return 0.0
    metadata = sample.metadata
    return 1.0 if _dataset(metadata["task"]).score_answer(answers[-1].strip(), metadata["entry"]) == 1 else 0.0


async def reward_func(args, sample, **kwargs) -> float:
    if sample.metadata.get("mopd_eval"):
        return task_score(sample)
    assert args.opd_log_prob_top_k == 0, "this example runs sampled-token OPD"
    started = time.time()
    await opd.reward_func(args, sample, **kwargs)
    sample.metadata.update(opd_score_t0=started, opd_score_t1=time.time())
    return task_score(sample)


def post_process_rewards(args, samples, **kwargs) -> tuple[list[float], list[float]]:
    # the stock hook checks every sample carries its teacher log-probs and returns zero task rewards
    _, rewards = opd.post_process_rewards(args, samples, **kwargs)
    return [sample.get_reward_value(args) for sample in samples], rewards


async def generate(input, url=None):
    started = time.time()
    output = await _single_turn_generate(input, url)
    finished = time.time()
    for sample in output.samples if isinstance(output.samples, list) else [output.samples]:
        sample.metadata.update(gen_t0=started, gen_t1=finished)
    return output
