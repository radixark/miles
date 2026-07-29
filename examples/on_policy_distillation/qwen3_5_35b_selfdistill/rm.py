"""Reward functions for the Qwen3.5-35B-A3B self-distillation example.

Two reward functions are provided:

* ``reward_func`` — format-agnostic boxed-answer correctness (0/1). Used as the
  task reward for Phase 1 (RLVR teacher training) AND as the eval scorer for both
  phases. It accepts EITHER answer format the DAPO prompt may elicit:
  ``\\boxed{ANS}`` (via ``grade_answer_verl``) OR ``Answer: ANS`` (via the DAPO
  ``compute_score``). A clean 0/1 == true accuracy, so ``rollout/raw_reward`` and
  ``eval/<name>`` are directly interpretable.

* ``reward_func_pure_opd`` — for *pure* on-policy distillation (Phase 2, pure
  variant). EVAL samples (tagged ``opd_reward_mode=eval_math`` via the eval
  config's ``metadata_overrides``) are scored for accuracy so ``eval/<name>``
  still measures real held-out accuracy; TRAINING samples return a constant
  ``0.0`` so the GRPO task advantage is ~0 and the ONLY learning signal is the
  teacher's reverse-KL. This isolates OPD's effect for clean attribution.

Why a custom reward instead of ``--rm-type``:
  - ``--rm-type math`` only extracts ``\\boxed{}``; ``--rm-type dapo`` only the
    ``Answer:`` form; ``--rm-type deepscaler`` additionally requires a
    ``</think>`` tag in the response. Qwen3.5 reasons inline (no ``</think>``),
    so ``deepscaler`` would score every response 0. This format-agnostic reward
    avoids all three traps.

Place ``examples/on_policy_distillation/qwen3_5_35b_selfdistill`` on PYTHONPATH (or
run from the miles repo root) and pass e.g.
``--custom-rm-path examples.on_policy_distillation.qwen3_5_35b_selfdistill.rm.reward_func``.
"""

from miles.rollout.rm_hub.math_dapo_utils import compute_score as _dapo_score
from miles.rollout.rm_hub.math_utils import grade_answer_verl


def _is_eval_sample(sample) -> bool:
    md = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    return md.get("opd_reward_mode") == "eval_math"


def _is_correct(sample) -> bool:
    response, label = sample.response or "", sample.label
    if label is None:
        return False
    # boxed form: \boxed{ANS}
    try:
        if grade_answer_verl(response, label):
            return True
    except Exception:
        pass
    # DAPO prompt form: "Answer: ANS"
    try:
        result = _dapo_score(response, label)
        if isinstance(result, dict) and result.get("acc"):
            return True
    except Exception:
        pass
    return False


async def reward_func(args, sample, **kwargs):
    """Format-agnostic correctness (1.0 / 0.0). Phase-1 task reward + eval scorer."""
    return 1.0 if _is_correct(sample) else 0.0


async def reward_func_pure_opd(args, sample, **kwargs):
    """Pure-OPD reward: accuracy for eval samples, constant 0.0 for training.

    Training reward is constant so the GRPO task advantage vanishes and the only
    learning signal is the OPD reverse-KL toward the teacher.
    """
    if _is_eval_sample(sample):
        return 1.0 if _is_correct(sample) else 0.0
    return 0.0
